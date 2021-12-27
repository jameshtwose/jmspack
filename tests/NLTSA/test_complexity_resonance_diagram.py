import numpy as np
import pandas as pd
import pytest

from jmspack.NLTSA import complexity_resonance
from jmspack.NLTSA import distribution_uniformity
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
def df_casnet_crd():
    """200 rows of casnet data"""
    return pd.read_csv(
        "https://raw.githubusercontent.com/jameshtwose/jmspack/develop/datasets/casnet_crd.csv",
        index_col=0,
    )


class TestComplexityResonance:
    """Testing class to test the ts_levels function."""

    def test_if_dataframe_not_affected(self, df_test):
        """Check if the function leaves the data frame the same."""
        df_original = df_test.copy()
        fluctuation_intensity_df = fluctuation_intensity(
            df=df_test,
            win=7,
            xmin=df_test.min().min(),
            xmax=df_test.max().max(),
            col_first=1,
            col_last=df_test.shape[1],
        )
        distribution_uniformity_df = distribution_uniformity(
            df=df_test,
            win=7,
            xmin=df_test.min().min(),
            xmax=df_test.max().max(),
            col_first=1,
            col_last=df_test.shape[1],
        )

        _ = complexity_resonance(
            distribution_uniformity_df=distribution_uniformity_df,
            fluctuation_intensity_df=fluctuation_intensity_df,
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
        distribution_uniformity_df = distribution_uniformity(
            df=df_test,
            win=7,
            xmin=df_test.min().min(),
            xmax=df_test.max().max(),
            col_first=1,
            col_last=df_test.shape[1],
        )

        complexity_resonance_df = complexity_resonance(
            distribution_uniformity_df=distribution_uniformity_df,
            fluctuation_intensity_df=fluctuation_intensity_df,
        )
        assert isinstance(complexity_resonance_df, pd.DataFrame)

    def test_compare_crd_df_casnet_crd_df(self, df_test, df_casnet_crd):
        """Check if the python version of the df is sufficiently similar to the casnet R version."""
        fluctuation_intensity_df = fluctuation_intensity(
            df=df_test,
            win=7,
            xmin=df_test.min().min(),
            xmax=df_test.max().max(),
            col_first=1,
            col_last=df_test.shape[1],
        )
        distribution_uniformity_df = distribution_uniformity(
            df=df_test,
            win=7,
            xmin=df_test.min().min(),
            xmax=df_test.max().max(),
            col_first=1,
            col_last=df_test.shape[1],
        )

        complexity_resonance_df = complexity_resonance(
            distribution_uniformity_df=distribution_uniformity_df,
            fluctuation_intensity_df=fluctuation_intensity_df,
        )

        np.testing.assert_allclose(
            complexity_resonance_df.replace(0, np.nan).values,
            df_casnet_crd.values,
            rtol=0.06,
        )
