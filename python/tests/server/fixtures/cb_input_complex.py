from typing import List
from enum import Enum

from pydantic import BaseModel, HttpUrl

from cog import BasePredictor
# To make custom class TestEnum be pickleable, it needs to be separated from the predictor file.
# Hence, when it's sent to the worker, it's referenced by the full module path.
from tests.server.fixtures.cb_input_complex_schema import TestEnum


class SubDict(BaseModel):
    text: str
    numbers: List[int]
    enum: TestEnum
    url: HttpUrl


class TestDict(BaseModel):
    text: str
    numbers: List[int]
    sub_dict: SubDict


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    # The arguments and types the model takes as input
    def predict(self, test_dict: TestDict, list_test_dict: List[TestDict])-> str:
        """Run a single prediction on the model"""
        return "test"
