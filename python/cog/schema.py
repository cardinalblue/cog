import typing as t
from datetime import datetime
from enum import Enum

import pydantic


class Status(str, Enum):
    STARTING = "starting"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    CANCELED = "canceled"
    FAILED = "failed"

    @staticmethod
    def is_terminal(status: t.Optional["Status"]) -> bool:
        return status in {Status.SUCCEEDED, Status.CANCELED, Status.FAILED}


class WebhookEvent(str, Enum):
    START = "start"
    OUTPUT = "output"
    LOGS = "logs"
    COMPLETED = "completed"

    @classmethod
    def default_events(cls) -> t.List["WebhookEvent"]:
        # if this is a set, it gets serialized to an array with an unstable ordering
        # so even though it's logically a set, have it as a list for deterministic schemas
        # note: this change removes "uniqueItems":true
        return [cls.START, cls.OUTPUT, cls.LOGS, cls.COMPLETED]


class PredictionBaseModel(pydantic.BaseModel, extra=pydantic.Extra.allow):
    pass


class PredictionRequest(PredictionBaseModel):
    instances: t.List[t.Any]

    @classmethod
    def with_types(cls, input_type: t.Type) -> t.Any:
        # [compat] Input is implicitly optional -- previous versions of the
        # Cog HTTP API allowed input to be omitted (e.g. for models that don't
        # have any inputs). We should consider changing this in future.
        return pydantic.create_model(
            cls.__name__, __base__=cls, instances=(t.List[input_type], ...)
        )


class PredictionResponse(PredictionBaseModel):
    predictions: t.List[t.Any]

    @classmethod
    def with_types(cls, output_type: t.Type) -> t.Any:
        # [compat] Input is implicitly optional -- previous versions of the
        # Cog HTTP API allowed input to be omitted (e.g. for models that don't
        # have any inputs). We should consider changing this in future.
        return pydantic.create_model(
            cls.__name__,
            __base__=cls,
            predictions=(t.List[output_type], None)
        )
