from typing import List

from pydantic import BaseModel

from cog import BasePredictor


class TestDict(BaseModel):
    text: str
    numbers: List[int]


class Output(BaseModel):
    test_dict: TestDict


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    # The arguments and types the model takes as input
    def predict(self, text: str, numbers: List[int])-> Output:
        """Run a single prediction on the model"""
        return Output(test_dict=TestDict(text=text, numbers=numbers))
