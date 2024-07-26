from cog import BasePredictor


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    # The arguments and types the model takes as input
    def predict(self, error_message: str)-> str:
        """Run a single prediction on the model"""
        raise ValueError(error_message)
