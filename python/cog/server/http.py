import argparse
import asyncio
import functools
import logging
import os
import signal
import socket
import sys
import textwrap
import threading
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum, auto, unique
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Type,
)

import sentry_sdk
import structlog
import uvicorn
from fastapi import Body, FastAPI, Header, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .. import schema
from ..config import Config
from ..logging import setup_logging
from ..mode import Mode
from ..types import PYDANTIC_V2

# [cb] `..files.upload_file` / `..json.upload_files` are intentionally NOT
# imported: file upload is disabled in this fork (see json.upload_files).

try:
    from .._version import __version__
except ImportError:
    __version__ = "dev"

if PYDANTIC_V2:
    from .helpers import (
        unwrap_pydantic_serialization_iterators,
        update_openapi_schema_for_pydantic_2,
    )
else:
    from .helpers import update_nullable_optional

from .probes import ProbeHelper
from .runner import (
    PredictionRunner,
    RunnerBusyError,
    SetupResult,
)
from .telemetry import make_trace_context, trace_context
from .worker import make_worker

if TYPE_CHECKING:
    from typing import ParamSpec, TypeVar  # pylint: disable=import-outside-toplevel

    P = ParamSpec("P")  # pylint: disable=invalid-name
    T = TypeVar("T")  # pylint: disable=invalid-name

log = structlog.get_logger("cog.server.http")


sentry_dsn = os.getenv("SENTRY_DSN", None)
traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0"))
profiles_sample_rate = float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.0"))
environment = os.getenv("SENTRY_ENVIRONMENT", "production")
if sentry_dsn is not None:
    sentry_sdk.init(
        dsn=sentry_dsn,
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        traces_sample_rate=traces_sample_rate,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=profiles_sample_rate,
        enable_tracing=True,
        environment=environment,
    )


@unique
class Health(Enum):
    UNKNOWN = auto()
    STARTING = auto()
    READY = auto()
    BUSY = auto()
    SETUP_FAILED = auto()
    DEFUNCT = auto()
    UNHEALTHY = auto()


class MyState:
    health: Health
    setup_result: Optional[SetupResult]


class MyFastAPI(FastAPI):
    # TODO: not, strictly speaking, legal
    # https://github.com/microsoft/pyright/issues/5933
    # but it'd need a FastAPI patch to fix
    state: MyState  # type: ignore


def add_setup_failed_routes(
    app: MyFastAPI,  # pylint: disable=redefined-outer-name
    started_at: datetime,
    msg: str,
) -> None:
    print(msg)
    result = SetupResult(
        started_at=started_at,
        completed_at=datetime.now(tz=timezone.utc),
        logs=[msg],
        status=schema.Status.FAILED,
    )
    app.state.setup_result = result
    app.state.health = Health.SETUP_FAILED

    @app.get("/health-check")
    async def healthcheck_startup_failed() -> Any:
        assert app.state.setup_result
        return jsonable_encoder(
            {
                "status": app.state.health.name,
                "setup": app.state.setup_result.to_dict(),
            }
        )


def create_app(  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    cog_config: Config,
    shutdown_event: Optional[threading.Event],  # pylint: disable=redefined-outer-name
    app_threads: Optional[int] = None,
    upload_url: Optional[str] = None,
    mode: Mode = Mode.PREDICT,
    is_build: bool = False,
    await_explicit_shutdown: bool = False,  # pylint: disable=redefined-outer-name
) -> MyFastAPI:
    started_at = datetime.now(tz=timezone.utc)

    @asynccontextmanager
    async def lifespan(app: MyFastAPI) -> AsyncGenerator[None, None]:
        # Startup code (was previously in @app.on_event("startup"))
        # check for early setup failures
        if (
            app.state.setup_result
            and app.state.setup_result.status == schema.Status.FAILED
        ):
            # signal shutdown if interactive run
            if shutdown_event and not await_explicit_shutdown:
                shutdown_event.set()
        else:
            setup_task = runner.setup()
            setup_task.add_done_callback(_handle_setup_done)

        yield

        # Shutdown code (was previously in @app.on_event("shutdown"))
        worker.terminate()

    app = MyFastAPI(  # pylint: disable=redefined-outer-name
        title="Cog",  # TODO: mention model name?
        # version=None # TODO
        lifespan=lifespan,
    )

    def custom_openapi() -> Dict[str, Any]:
        if not app.openapi_schema:
            openapi_schema = get_openapi(
                title="Cog",
                openapi_version="3.0.2",
                version="0.1.0",
                routes=app.routes,
            )

            # Pydantic 2 changes how optional fields are represented in OpenAPI schema.
            # See: https://github.com/tiangolo/fastapi/pull/9873#issuecomment-1997105091
            if PYDANTIC_V2:
                update_openapi_schema_for_pydantic_2(openapi_schema)
            else:
                update_nullable_optional(openapi_schema, app)

            app.openapi_schema = openapi_schema

        return app.openapi_schema

    app.openapi = custom_openapi

    app.state.health = Health.STARTING
    app.state.setup_result = None

    # shutdown is needed no matter what happens
    @app.post("/shutdown")
    async def start_shutdown() -> Any:
        log.info("shutdown requested via http")
        if shutdown_event:
            shutdown_event.set()
        return JSONResponse({}, status_code=200)

    try:
        predictor_info = cog_config.get_predictor_info(mode=Mode.PREDICT)
    except Exception:  # pylint: disable=broad-exception-caught
        msg = "Error while loading predictor:\n\n" + traceback.format_exc()
        add_setup_failed_routes(app, started_at, msg)
        return app

    InputType = predictor_info.input_type
    OutputType = predictor_info.output_type
    is_async = predictor_info.is_async

    worker = make_worker(
        predictor_ref=cog_config.get_predictor_ref(mode=mode),
        is_async=is_async,
        is_train=False if mode == Mode.PREDICT else True,
        max_concurrency=cog_config.max_concurrency,
        has_user_healthcheck=predictor_info.has_healthcheck,
    )
    runner = PredictionRunner(worker=worker, max_concurrency=cog_config.max_concurrency)

    class PredictionRequest(schema.PredictionRequest.with_types(input_type=InputType)):
        pass

    class PredictionResponse(
        schema.PredictionResponse.with_types(
            input_type=InputType, output_type=OutputType
        )
    ):
        pass

    NewPredictionRequest = schema.NewPredictionRequest.with_types(input_type=InputType)
    NewPredictionResponse = schema.NewPredictionResponse.with_types(
        output_type=OutputType
    )

    if app_threads is None:
        app_threads = 1 if cog_config.requires_gpu else _cpu_count()
    http_semaphore = asyncio.Semaphore(app_threads)

    def limited(f: "Callable[P, Awaitable[T]]") -> "Callable[P, Awaitable[T]]":
        @functools.wraps(f)
        async def wrapped(*args: "P.args", **kwargs: "P.kwargs") -> "T":  # pylint: disable=redefined-outer-name
            async with http_semaphore:
                return await f(*args, **kwargs)

        return wrapped

    index_document = {
        "cog_version": __version__,
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "shutdown_url": "/shutdown",
        "healthcheck_url": "/health-check",
        "readiness_url": "/health/ready",
        "liveness_url": "/health/live",
        "predictions_url": "/predictions",
    }

    @app.get("/")
    async def root() -> Any:
        return index_document

    @app.get("/health-check")
    async def healthcheck() -> Any:
        if app.state.health == Health.READY:
            health = Health.BUSY if runner.is_busy() else Health.READY

            # Run custom healthcheck. If it doesn't exist, this will
            # always return healthy (healthcheck_result.error = False)
            healthcheck_result = await runner.healthcheck()
            custom_health_ok = not healthcheck_result.error
            custom_health_error = healthcheck_result.error_detail

            if not custom_health_ok:
                health = Health.UNHEALTHY
        else:
            health = app.state.health
            custom_health_ok = True
            custom_health_error = None

        setup = app.state.setup_result.to_dict() if app.state.setup_result else {}

        response = {
            "status": health.name,
            "setup": setup,
        }

        if not custom_health_ok:
            response["user_healthcheck_error"] = custom_health_error

        return jsonable_encoder(response)

    @app.get("/health/ready")
    def healthcheck_readiness() -> Any:
        health = app.state.health

        if health == Health.UNKNOWN:
            return JSONResponse({"detail": "Unknown server status"}, status_code=500)
        if health == Health.SETUP_FAILED:
            return JSONResponse({"detail": "Error starting server"}, status_code=500)
        if health == Health.STARTING:
            return JSONResponse({"detail": "Server is starting"}, status_code=503)
        return jsonable_encoder(
            {
                "status": health.name,
                "setup": app.state.setup_result.to_dict(),
            }
        )

    @app.get("/health/live")
    def healthcheck_liveliness() -> Any:
        if app.state.health == Health.READY:
            health = Health.BUSY if runner.is_busy() else Health.READY
        else:
            health = app.state.health

        if health == Health.UNKNOWN:
            return JSONResponse({"detail": "Unknown server status"}, status_code=500)
        if health == Health.SETUP_FAILED:
            return JSONResponse({"detail": "Error starting server"}, status_code=500)
        return jsonable_encoder({"status": health.name})

    @limited
    @app.post(
        "/predictions",
        response_model=NewPredictionResponse,
        response_model_exclude_unset=True,
    )
    async def predict(
        request: NewPredictionRequest = Body(default=None),
        # prefer: Optional[str] = Header(default=None),
        traceparent: Optional[str] = Header(default=None, include_in_schema=False),
        tracestate: Optional[str] = Header(default=None, include_in_schema=False),
    ) -> Any:  # type: ignore
        """
        Run a single prediction on the model
        """
        # TODO: spec-compliant parsing of Prefer header.
        # respond_async = prefer == "respond-async"
        respond_async = False

        with trace_context(make_trace_context(traceparent, tracestate)):
            return await _predict(
                request=request,
                response_type=NewPredictionResponse,
                respond_async=respond_async,
            )

    async def _predict(
        *,
        request: NewPredictionRequest,
        response_type: Type[schema.NewPredictionResponse],
        respond_async: bool = False,
        is_train: bool = False,
    ) -> Response:
        # [compat] If no body is supplied, assume that this model can be run
        # with empty input. This will throw a ValidationError if that's not
        # possible.
        # if request is None:
        #     request = PredictionRequest(input={})
        # [compat] If body is supplied but input is None, set it to an empty
        # dictionary so that later code can be simpler.
        # if request.input is None:
        #     request.input = {}
        all_results = []
        for instance in request.instances:
            instance_request = PredictionRequest(input=instance)
            # CB: From Cog origin and we don't do the async mechanism in this way.
            # task_kwargs = {}
            # if respond_async:
            #     # For now, we only ask PredictionService to handle file uploads for
            #     # async predictions. This is unfortunate but required to ensure
            #     # backwards-compatible behaviour for synchronous predictions.
            #     task_kwargs["upload_url"] = upload_url

            try:
                # predict_task = runner.predict(instance_request, is_train, task_kwargs=task_kwargs)
                predict_task = runner.predict(instance_request, is_train)
            except RunnerBusyError:
                return JSONResponse(
                    {"detail": "Already running a prediction"}, status_code=409
                )

            # CB: From Cog origin and we don't do the async mechanism in this way.
            # if hasattr(instance_request.input, "cleanup"):
            #     predict_task.add_done_callback(
            #         lambda _: instance_request.input.cleanup()
            #     )

            predict_task.add_done_callback(_handle_predict_done)

            # CB: From Cog origin and we don't do the async mechanism in this way.
            # if respond_async:
            #     return JSONResponse(
            #         jsonable_encoder(predict_task.result), status_code=202
            #     )

            # Otherwise, wait for the prediction to complete...
            predict_task.wait()

            # ...and return the result.
            if PYDANTIC_V2:
                response_object = unwrap_pydantic_serialization_iterators(
                    predict_task.result.model_dump()
                )
            else:
                response_object = predict_task.result.dict()

            if response_object.get("status") == "failed":
                # use error_status_code if it exists, otherwise default to 500
                status_code = response_object.get("http_status_code") or 500
                detail_item = {
                    "msg": response_object.get("error"),
                    "error_type": response_object.get("error_type", None),
                }
                body = {"detail": [detail_item]}
                return JSONResponse(body, status_code=status_code)
            try:
                PredictionResponse(**response_object)
            except ValidationError as e:
                _log_invalid_output(e, mode)
                raise HTTPException(status_code=500, detail=str(e)) from e

            all_results.append(response_object["output"])

        try:
            response = NewPredictionResponse(predictions=all_results)
        except ValidationError as e:
            _log_invalid_output(e, mode)
            raise HTTPException(status_code=500, detail=str(e)) from e

        response_object = response.dict()
        # response_object["output"] = upload_files(
        #     response_object["output"],
        #     upload_file=lambda fh: upload_file(fh, request.output_file_prefix),  # type: ignore
        # )

        # FIXME: clean up output files
        encoded_response = jsonable_encoder(response_object)
        return JSONResponse(content=encoded_response)

    def _handle_predict_done(response: schema.PredictionResponse) -> None:
        if response._fatal_exception:
            _maybe_shutdown(response._fatal_exception)

    def _handle_setup_done(setup_result: SetupResult) -> None:
        app.state.setup_result = setup_result

        if app.state.setup_result.status == schema.Status.SUCCEEDED:
            app.state.health = Health.READY

            # In kubernetes, mark the pod as ready now setup has completed.
            probes = ProbeHelper()
            probes.ready()
        else:
            _maybe_shutdown(Exception("setup failed"), status=Health.SETUP_FAILED)

    def _maybe_shutdown(exc: BaseException, *, status: Health = Health.DEFUNCT) -> None:
        log.error("encountered fatal error", exc_info=exc)
        app.state.health = status
        if shutdown_event and not await_explicit_shutdown:
            log.error("shutting down immediately")
            shutdown_event.set()
        else:
            log.error("awaiting explicit shutdown")

    return app


def _log_invalid_output(error: Any, mode: Mode) -> None:
    function_name = "predict()"
    if mode == Mode.TRAIN:
        function_name = "train()"
    log.error(
        textwrap.dedent(
            f"""\
            The return value of {function_name} was not valid:

            {error}

            Check that your predict function is in this form, where `output_type` is the same as the type you are returning (e.g. `str`):

                def {function_name} -> output_type:
                    ...
           """
        )
    )


class Server(uvicorn.Server):
    def start(self) -> None:
        self._thread = threading.Thread(target=self.run)  # pylint: disable=attribute-defined-outside-init
        self._thread.start()

    def stop(self) -> None:
        log.info("stopping server")
        self.should_exit = True  # pylint: disable=attribute-defined-outside-init

        self._thread.join(timeout=5)
        if not self._thread.is_alive():
            return

        log.warn("failed to exit after 5 seconds, setting force_exit")
        self.force_exit = True  # pylint: disable=attribute-defined-outside-init
        self._thread.join(timeout=5)
        if not self._thread.is_alive():
            return

        log.warn("failed to exit after another 5 seconds, sending SIGKILL")
        os.kill(os.getpid(), signal.SIGKILL)


def is_port_in_use(port: int) -> bool:  # pylint: disable=redefined-outer-name
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("localhost", port)) == 0


def signal_ignore(signum: Any, frame: Any) -> None:  # pylint: disable=unused-argument
    log.warn("Got a signal to exit, ignoring it...", signal=signal.Signals(signum).name)


def signal_set_event(event: threading.Event) -> Callable[[Any, Any], None]:
    def _signal_set_event(signum: Any, frame: Any) -> None:  # pylint: disable=unused-argument
        event.set()

    return _signal_set_event


def _cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0)) or 1  # type: ignore
    except AttributeError:  # not available on every platform
        return os.cpu_count() or 1


if __name__ == "__main__":
    # [cb] Upstream v0.16.10 added an opt-in delegation to the Rust coglet
    # server here: it tried `import coglet` and, when that succeeded, handed the
    # whole process to `coglet.serve()` and exited.
    #
    # That is removed in this fork on purpose. The Rust server implements
    # upstream's API, not ours -- delegating to it would silently drop the
    # `instances`/`predictions` batch API, Sentry reporting, the /health/ready
    # and /health/live probes, and the 400-vs-500 error semantics, while still
    # starting up and answering requests as if nothing were wrong. The fork must
    # always serve its own Python app, so there is no coglet branch at all.
    parser = argparse.ArgumentParser(description="Cog HTTP server")
    parser.add_argument(
        "-v", "--version", action="store_true", help="Show version and exit"
    )
    parser.add_argument(
        "--host",
        dest="host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--threads",
        dest="threads",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to number of CPUs, or 1 if using a GPU.",
    )
    parser.add_argument(
        "--upload-url",
        dest="upload_url",
        type=str,
        default=None,
        help="An endpoint for Cog to PUT output files to",
    )
    parser.add_argument(
        "--await-explicit-shutdown",
        dest="await_explicit_shutdown",
        type=bool,
        default=False,
        help="Ignore SIGTERM and wait for a request to /shutdown (or a SIGINT) before exiting",
    )
    parser.add_argument(
        "--x-mode",
        dest="mode",
        type=Mode,
        default=Mode.PREDICT,
        choices=list(Mode),
        help="Experimental: Run in 'predict' or 'train' mode",
    )
    args = parser.parse_args()

    if args.version:
        print(f"cog.server.http {__version__}")
        sys.exit(0)

    # log level is configurable so we can make it quiet or verbose for `cog predict`
    # cog predict --debug       # -> debug
    # cog predict               # -> warning
    # docker run <image-name>   # -> info (default)
    log_level = logging.getLevelName(os.environ.get("COG_LOG_LEVEL", "INFO").upper())
    setup_logging(log_level=log_level)

    shutdown_event = threading.Event()

    await_explicit_shutdown = args.await_explicit_shutdown
    if await_explicit_shutdown:
        signal.signal(signal.SIGTERM, signal_ignore)
    else:
        signal.signal(signal.SIGTERM, signal_set_event(shutdown_event))

    app = create_app(
        cog_config=Config(),
        shutdown_event=shutdown_event,
        app_threads=args.threads,
        upload_url=args.upload_url,
        mode=args.mode,
        await_explicit_shutdown=await_explicit_shutdown,
    )

    host: str = args.host

    port = int(os.getenv("PORT", "5000"))
    if is_port_in_use(port):
        log.error(f"Port {port} is already in use")
        sys.exit(1)

    server_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_config=None,
        # This is the default, but to be explicit: only run a single worker
        workers=1,
    )

    s = Server(config=server_config)
    s.start()

    try:
        shutdown_event.wait()
    except KeyboardInterrupt:
        pass

    s.stop()

    # return error exit code when setup failed and cog is running in interactive mode (not k8s)
    if (
        app.state.setup_result
        and app.state.setup_result.status == schema.Status.FAILED
        and not await_explicit_shutdown
    ):
        sys.exit(-1)
