import pandas as pd
import pytest
import seaborn as sns

from jmspack.utils import apply_scaling


@pytest.fixture
def df_test():
    """test dataset from seaborn"""
    return sns.load_dataset("iris").head(5)


class TestApplyScaling:
    """Testing class to test the apply_scaling function."""

    def test_output_matches_list(self, df_test):
        """Check if the function output matches what is expected."""

        output_df = df_test.select_dtypes("number").pipe(apply_scaling)

        output_comparison_df = pd.DataFrame(
            {
                "sepal_length": {
                    0: 1.0,
                    1: 0.6000000000000014,
                    2: 0.20000000000000107,
                    3: 0.0,
                    4: 0.8000000000000007,
                },
                "sepal_width": {
                    0: 0.833333333333333,
                    1: 0.0,
                    2: 0.33333333333333304,
                    3: 0.16666666666666607,
                    4: 1.0,
                },
                "petal_length": {
                    0: 0.4999999999999991,
                    1: 0.4999999999999991,
                    2: 0.0,
                    3: 1.0,
                    4: 0.4999999999999991,
                },
                "petal_width": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
            }
        )

        assert output_df.equals(output_comparison_df)
