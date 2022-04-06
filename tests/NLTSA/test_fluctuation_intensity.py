import numpy as np
import pandas as pd
import pytest

from jmspack.NLTSA import fluctuation_intensity


@pytest.fixture
def df_test():
    """200 rows of time series data"""
    return pd.read_csv(
        "https://raw.githubusercontent.com/jameshtwose/jmspack/main/datasets/time_series_dataset.csv",
        index_col=0,
        nrows=200,
    )


@pytest.fixture
def df_casnet_fi():
    """200 rows of casnet data"""
    return pd.read_csv(
        "https://raw.githubusercontent.com/jameshtwose/jmspack/develop/datasets/casnet_fi.csv",
        index_col=0,
    )


class TestFluctuationIntensity:
    """Testing class to test the ts_levels function."""

    def test_if_dataframe_not_affected(self, df_test):
        """Check if the function leaves the data frame the same."""
        df_original = df_test.copy()
        _ = fluctuation_intensity(
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
        fluctuation_intensity_df = fluctuation_intensity(
            df=df_test,
            win=7,
            xmin=df_test.min().min(),
            xmax=df_test.max().max(),
            col_first=1,
            col_last=df_test.shape[1],
        )
        assert isinstance(fluctuation_intensity_df, pd.DataFrame)

    def test_compare_fi_df_casnet_fi_df(self, df_test, df_casnet_fi):
        """Check if the python version of the df is sufficiently similar to the casnet R version."""
        fluctuation_intensity_df = fluctuation_intensity(
            df=df_test,
            win=7,
            xmin=df_test.min().min(),
            xmax=df_test.max().max(),
            col_first=1,
            col_last=df_test.shape[1],
        )

        np.testing.assert_allclose(
            fluctuation_intensity_df.replace(0, np.nan).values,
            df_casnet_fi.values,
            rtol=0.05,
        )
