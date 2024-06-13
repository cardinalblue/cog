from typing import List

from pydantic import BaseModel

from cog import BasePredictor


class TestDict(BaseModel):
    text: str
    numbers: List[int]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    # The arguments and types the model takes as input
    def predict(self, test_dict: TestDict, list_test_dict: List[TestDict])-> str:
        """Run a single prediction on the model"""
        return "test"
