from typing import Optional


class CogError(Exception):
    """Base class for all Cog errors."""


class ConfigDoesNotExist(CogError):
    """Exception raised when a cog.yaml does not exist."""


class PredictorNotSet(CogError):
    """Exception raised when 'predict' is not set in cog.yaml when it needs to be."""


class PredictorBaseError(Exception):
    """Base class for all predictor errors."""

    def __init__(
        self, message: str, type_: Optional[str] = None, status_code: int = 500
    ) -> None:
        self.message = message
        self.type = type_
        self._status_code = status_code

    def __str__(self) -> str:
        if self.type:
            return f"{self.type}: {self.message}"

        return self.message


class PredictorInputError(PredictorBaseError):
    """Exception raised when the input to the predictor is invalid."""

    def __init__(self, message: str, type_: Optional[str] = None) -> None:
        super().__init__(message, type_, status_code=400)


class PredictorInternalError(PredictorBaseError):
    """Exception raised when the predictor encounters an internal error."""

    def __init__(self, message: str, type_: Optional[str] = None) -> None:
        super().__init__(message, type_, status_code=500)
