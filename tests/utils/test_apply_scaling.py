import pandas as pd
import pytest
import seaborn as sns

from jmspack.utils import apply_scaling


@pytest.fixture
def df_test():
    """test dataset from seaborn"""
    return sns.load_dataset("iris")
