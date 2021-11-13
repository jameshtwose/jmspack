import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from jmspack.frequentist_statistics import potential_for_change_index


@pytest.fixture
def df_test():
    """test dataset from seaborn"""
    return sns.load_dataset("iris")


@pytest.fixture
def output_comparison_df():
    """test output dataset calculated using the R version of the potential_for_change_index"""
    return pd.DataFrame(
        {
            "p_delta": {
                "Petal.Length": 2.98186702428436,
                "Sepal.Length": 1.6094676000894,
                "Petal.Width": 1.24414923096124,
            },
            "lower": {"Petal.Length": 1.0, "Sepal.Length": 4.3, "Petal.Width": 0.1},
            "current": {
                "Petal.Length": 3.758,
                "Sepal.Length": 5.84333333333333,
                "Petal.Width": 1.19933333333333,
            },
            "upper": {"Petal.Length": 6.9, "Sepal.Length": 7.9, "Petal.Width": 2.5},
            "weight": {
                "Petal.Length": 0.949034699008388,
                "Sepal.Length": 0.782561231810081,
                "Petal.Width": 0.956547332876403,
            },
        }
    )


class TestPotentialForChangeIndex:
    """Testing class to test the potential_for_change_index function."""

    def test_if_dataframe_not_affected(self, df_test):
        """Check if the function leaves the data frame the same."""

        target = "species_cat_codes"
        df_original = df_test.assign(
            **{target: lambda x: x["species"].astype("category").cat.codes}
        )
        features_list = df_original.select_dtypes(float).columns.tolist()

        _ = potential_for_change_index(
            data=df_original.select_dtypes("number"),
            features_list=features_list,
            target=target,
            minimum_measure="min",
            centrality_measure="mean",
            maximum_measure="max",
            weight_measure="r-value",
            scale_data=False,
            pci_heatmap=False,
        )
        assert df_original.drop(target, axis=1).equals(df_test)

    def test_output_matches_r(self, df_test, output_comparison_df):
        """Check if the function output matches the R version."""

        target = "species_cat_codes"
        df_original = df_test.assign(
            **{target: lambda x: x["species"].astype("category").cat.codes}
        )
        features_list = df_original.select_dtypes(float).columns.tolist()

        pot_df = potential_for_change_index(
            data=df_original.select_dtypes("number"),
            features_list=features_list,
            target=target,
            minimum_measure="min",
            centrality_measure="mean",
            maximum_measure="max",
            weight_measure="r-value",
            scale_data=False,
            pci_heatmap=False,
        )
        python_result_array = (
            pot_df.sort_values(by="PCI", ascending=False)
            .drop("p-value", axis=1)
            .head(3)
            .to_numpy()
        )
        r_result_array = output_comparison_df.to_numpy()

        np.testing.assert_array_almost_equal(
            python_result_array, r_result_array, decimal=8
        )
