import pandas as pd
import pytest

from jmspack.utils import flatten


@pytest.fixture
def list_of_lists_test():
    """example set of list of lists"""
    return [
        [f"p_{x}" for x in range(10)],
        [f"p_{x}" for x in range(10, 20)],
        [f"p_{x}" for x in range(20, 30)],
    ]
