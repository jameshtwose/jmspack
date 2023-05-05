import pandas as pd
import pytest

from jmspack.NLTSA import ts_levels


@pytest.fixture
def df_test():
    """200 rows of time series data"""
    return pd.read_csv(
        "https://raw.githubusercontent.com/jameshtwose/jmspack/main/datasets/time_series_dataset.csv",
        index_col=0,
    ).loc[0:200, :]


class TestTsLevels:
    """Testing class to test the ts_levels function."""

    def test_if_dataframe_not_affected(self, df_test):
        """Check if the function leaves the data frame the same."""
        df_original = df_test.copy()
        ts = df_test["lorenz"]
        ts_levels_df, fig, ax = ts_levels(
            ts,
            ts_x=None,
            criterion="squared_error",
            max_depth=10,
            min_samples_leaf=1,
            min_samples_split=2,
            max_leaf_nodes=30,
            plot=False,
            equal_spaced=True,
            n_x_ticks=10,
        )
        assert df_original.equals(df_test)

    def test_returns_expected_objects(self, df_test):
        """Check if the function returns the expected output objects."""
        ts = df_test["lorenz"]
        ts_levels_df, fig, ax = ts_levels(
            ts,
            ts_x=None,
            criterion="squared_error",
            max_depth=10,
            min_samples_leaf=1,
            min_samples_split=2,
            max_leaf_nodes=30,
            plot=False,
            equal_spaced=True,
            n_x_ticks=10,
        )

        assert isinstance(ts_levels_df, pd.DataFrame)

    # for this test to run you will need to make sure that
    # pytest-mpl is installed and that you have the baseline images
    # by running `pytest --mpl-generate-path=baseline` in the same
    # directory as the test files
    @pytest.mark.mpl_image_compare
    def test_ts_levels_plot(self, df_test):
        """Check if the function returns the expected figure (compared to a baseline plot)."""
        ts = df_test["lorenz"]
        ts_levels_df, fig, ax = ts_levels(
            ts,
            ts_x=None,
            criterion="squared_error",
            max_depth=10,
            min_samples_leaf=1,
            min_samples_split=2,
            max_leaf_nodes=30,
            plot=True,
            equal_spaced=True,
            n_x_ticks=10,
        )
        return fig
