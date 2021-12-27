import numpy as np
import pandas as pd
import pytest

from jmspack.NLTSA import distribution_uniformity


@pytest.fixture
def df_test():
    """200 rows of time series data"""
    return pd.read_csv(
        "https://raw.githubusercontent.com/jameshtwose/jmspack/main/datasets/time_series_dataset.csv",
        index_col=0,
        nrows=200,
    )


@pytest.fixture
def df_casnet_du():
    """200 rows of casnet data"""
    return pd.read_csv(
        "https://raw.githubusercontent.com/jameshtwose/jmspack/develop/datasets/casnet_du.csv",
        index_col=0,
    )


class TestDistributionUniformity:
    """Testing class to test the ts_levels function."""

    def test_if_dataframe_not_affected(self, df_test):
        """Check if the function leaves the data frame the same."""
        df_original = df_test.copy()
        _ = distribution_uniformity(
            df=df_test,
            win=7,
            xmin=df_test.min().min(),
            xmax=df_test.max().max(),
            col_first=1,
            col_last=df_test.shape[1],
        )
        assert df_original.equals(df_test)

    def test_returns_expected_objects(self, df_test):
        """Check if the function returns the expected output objects."""
        distribution_uniformity_df = distribution_uniformity(
            df=df_test,
            win=7,
            xmin=df_test.min().min(),
            xmax=df_test.max().max(),
            col_first=1,
            col_last=df_test.shape[1],
        )
        assert isinstance(distribution_uniformity_df, pd.DataFrame)

    def test_compare_du_df_casnet_du_df(self, df_test, df_casnet_du):
        """Check if the python version of the df is sufficiently similar to the casnet R version."""
        distribution_uniformity_df = distribution_uniformity(
            df=df_test,
            win=7,
            xmin=df_test.min().min(),
            xmax=df_test.max().max(),
            col_first=1,
            col_last=df_test.shape[1],
        )

        np.testing.assert_allclose(
            distribution_uniformity_df.replace(0, np.nan).values,
            df_casnet_du.values,
            rtol=0.05,
        )
