from cog import BasePredictor
from cog.errors import PredictorInternalError


class SubClass(PredictorInternalError):
    pass


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    # The arguments and types the model takes as input
    def predict(self, use_subclass: bool, with_type: bool)-> str:
        """Run a single prediction on the model"""


        kwargs = {"type_": "error type"} if with_type else {}
        if use_subclass:
            raise SubClass("subclass error", **kwargs)
        else:
            raise PredictorInternalError("predictor internal error", **kwargs)
