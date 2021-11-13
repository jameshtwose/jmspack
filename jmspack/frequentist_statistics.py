r"""Submodule frequentist_statistics.py includes the following functions: <br>
- **normal_check():** compare the distribution of numeric variables to a normal distribution using the
    Kolmogrov-Smirnov test <br>
- **correlation_analysis():** Run correlations for numerical features and return output in different formats <br>
- **correlations_as_sample_increases():** Run correlations for subparts of the data to check robustness <br>
- **multiple_univariate_OLSs():** Tmp <br>
- **potential_for_change_index():** Calculate the potential for change index based on either variants of the r-squared
    (from linear regression) or the r-value (pearson correlation) <br>
"""
from itertools import combinations
from itertools import product
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

from .utils import apply_scaling


def normal_check(data: pd.DataFrame) -> pd.DataFrame:
    r"""Compare the distribution of numeric variables to a normal distribution using the Kolmogrov-Smirnov test
    Wrapper for `scipy.stats.kstest`: the empircal data is compared to a normally distributed variable with the
    same mean and standard deviation. A significant result (p < 0.05) in the goodness of fit test means that the
    data is not normally distributed.
    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe including the columns of interest
    Returns
    ----------
    df_normality_check: pd.DataFrame
        Dataframe with column names, p-values and an indication of normality
    Examples
    ----------
    >>> tips = sns.load_dataset("tips")
    >>> df_normality_check = normal_check(tips)
    """
    # Select numeric columns only
    num_features = data.select_dtypes(include="number").columns.tolist()
    # Compare distribution of each feature to a normal distribution with given mean and std
    df_normality_check = data[num_features].apply(
        lambda x: stats.kstest(
            x.dropna(), stats.norm.cdf, args=(np.nanmean(x), np.nanstd(x)), N=len(x)
        )[1],
        axis=0,
    )

    # create a label that indicates whether a feature has a normal distribution or not
    df_normality_check = pd.DataFrame(df_normality_check).reset_index()
    df_normality_check.columns = ["feature", "p-value"]
    df_normality_check["normality"] = df_normality_check["p-value"] >= 0.05

    return df_normality_check


def permute_test(a, test_type, test, **kwargs):
    r"""Helper function to run tests for permutations
    Parameters
    ----------
    a : np.array
    test_type: str {'correlation', 'independent_t_test'}
        Type of the test to be used
    test:
        e.g. `scipy.stats.pearsonr` or `statsmodels.stats.weightstats.ttest_ind`
    **kwargs:
        Additional keywords to be added to `test`
        - `a2` for the second feature if test_type = 'correlation'
    Returns
    ----------
    float:
        p value for permutation
    """
    if test_type == "correlation":
        a2 = kwargs["a2"]
        _, p = test(a, a2)

    else:
        raise ValueError("Unknown test_type provided")


def correlation_analysis(
    data: pd.DataFrame,
    col_list=None,
    row_list=None,
    check_norm=False,
    method: str = "pearson",
    dropna: str = "pairwise",
    permutation_test: bool = False,
    n_permutations: int = 1000,
    random_state=None,
):
    r"""Run correlations for numerical features and return output in different formats
    Different methods to compute correlations and to handle missing values are implemented.
    Inspired by `researchpy.corr_case` and `researchpy.corr_pair`.
    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with variables in columns, cases in rows
    row_list: list or None (default: None)
        List with names of columns in `data` that should be in the rows of the correlogram.
        If None, all columns are used but only every unique combination.
    col_list: list or None (default: None)
        List with names of columns in `data` that should be in the columns of the correlogram.
        If None, all columns are used and only every unique combination.
    check_norm: bool (default: False)
        If True, normality will be checked for columns in `data` using `normal_check`. This influences the used method
        for correlations, i.e. Pearson or Spearman. Note: normality check ignores missing values.
    method: {'pearson', 'kendall', 'spearman'}, default 'pearson'
        Type of correlation, either Pearson's r, Spearman's rho, or Kendall's tau, implemented via respectively
        `scipy.stats.pearsonr`, `scipy.stats.spearmanr`, and `scipy.stats.kendalltau`
        Will be ignored if check_norm=True. Instead, Person's r is used for every combination of normally distributed
        columns and Spearman's rho is used for all other combinations.
    dropna : {'listwise', 'pairwise'}, default 'pairwise'
        Should rows with missing values be dropped over the complete `data` ('listwise') or for every correlation
        separately ('pairwise')
    permutation_test: bool (default: False)
        If true, a permutation test will added
    n_permutations: int (default: 1000)
        Number of permutations in the permutation test
    random_state: None or int (default: None)
        Random state for permutation_test. If not None, random_state will be updated for every permutation
    plot_permutation: bool (default: False)
        Whether to plot the results of the permutation test
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches
    Returns
    ----------
    result_dict: dict
        Dictionary containing with the following keys:
        info : pandas.DataFrame
            Description of correlation method, missing values handling and number of observations
        r-values : pandas.DataFrame
            Dataframe with correlation coefficients. Indices and columns are column names from `data`. Only lower
            triangle is filled.
        p-values : pandas.DataFrame
            Dataframe with p-values. Indices and columns are column names from `data`. Only lower triangle is filled.
        N        : pandas.DataFrame
            Dataframe with numbers of observations. Indices and columns are column names from `data`. Only lower
            triangle is filled. If dropna ='listwise', every correlation will have the same number of observations.
        summary : pandas.DataFrame
            Dataframe with columns ['analysis', 'feature1', 'feature2', 'r-value', 'p-value', 'N', 'stat-sign']
            which indicate the type of test used for the correlation, the pair of columns, the correlation coefficient,
            the p-value, the number of observations for each combination of columns in `data` and whether the r-value is
            statistically significant.
    plotted_permuations: Figure
    Examples
    ----------
    >>> from jmspack.frequentist_statistics import correlation_analysis
    >>> import seaborn as sns
    >>> iris = sns.load_dataset('iris')
    >>> dict_results = correlation_analysis(iris, method='pearson', dropna='listwise', permutation_test=True,
    >>>                                        n_permutations=100, check_norm=True)
    >>> dict_results['summary']
    References
    ----------
    Bryant, C (2018). researchpy's documentation [Revision 9ae5ed63]. Retrieved from
    https://researchpy.readthedocs.io/en/latest/
    """

    # Settings test
    if method == "pearson":
        test, test_name = stats.pearsonr, "Pearson"
    elif method == "spearman":
        test, test_name = stats.spearmanr, "Spearman Rank"
    elif method == "kendall":
        test, test_name = stats.kendalltau, "Kendall's Tau-b"
    else:
        raise ValueError("method not in {'pearson', 'kendall', 'spearman'}")

    # Copy numerical data from the original data
    data = data.copy().select_dtypes("number")

    # Get correct lists
    if col_list and not row_list:
        row_list = data.select_dtypes("number").drop(col_list, axis=1).columns.tolist()
    elif row_list and not col_list:
        col_list = data.select_dtypes("number").drop(row_list, axis=1).columns.tolist()

    # Initializing dataframes to store results
    info = pd.DataFrame()
    summary = pd.DataFrame()
    if not col_list and not row_list:
        r_vals = pd.DataFrame(columns=data.columns, index=data.columns)
        p_vals = pd.DataFrame(columns=data.columns, index=data.columns)
        n_vals = pd.DataFrame(columns=data.columns, index=data.columns)
        iterator = combinations(data.columns, 2)
    else:
        r_vals = pd.DataFrame(columns=col_list, index=row_list)
        p_vals = pd.DataFrame(columns=col_list, index=row_list)
        n_vals = pd.DataFrame(columns=col_list, index=row_list)
        iterator = product(col_list, row_list)

    if dropna == "listwise":
        # Remove rows with missing values
        data = data.dropna(how="any", axis="index")
        info = info.append(
            {
                f"{test_name} correlation test using {dropna} deletion": f"Total observations used = {len(data)}"
            },
            ignore_index=True,
        )
    elif dropna == "pairwise":
        info = info.append(
            {
                f"{test_name} correlation test using {dropna} deletion": f"Observations in the data = {len(data)}"
            },
            ignore_index=True,
        )
    else:
        raise ValueError("dropna not in {'listwise', 'pairwise'}")

    if check_norm:
        # Check normality of all columns in the data
        df_normality = normal_check(data)
        norm_names = df_normality.loc[df_normality["normality"], "feature"].tolist()

    # Iterating through the Pandas series and performing the correlation
    for col1, col2 in iterator:
        if dropna == "pairwise":
            # Remove rows with missing values in the pair of columns
            test_data = data[[col1, col2]].dropna()
        else:
            test_data = data

        if check_norm:
            # Select Pearson's r only if both columns are normally distributed
            if (col1 in norm_names) and (col2 in norm_names):
                test, test_name = stats.pearsonr, "Pearson"
            else:
                test, test_name = stats.spearmanr, "Spearman Rank"

        # Run correlations
        r_value, p_value = test(test_data.loc[:, col1], test_data.loc[:, col2])
        n_value = len(test_data)

        # Store output in matrix format
        try:
            r_vals.loc[col2, col1] = r_value
            p_vals.loc[col2, col1] = p_value
            n_vals.loc[col2, col1] = n_value
        except KeyError:
            r_vals.loc[col1, col2] = r_value
            p_vals.loc[col1, col2] = p_value
            n_vals.loc[col1, col2] = n_value

        # Store output in dataframe format
        dict_summary = {
            "analysis": test_name,
            "feature1": col1,
            "feature2": col2,
            "r-value": r_value,
            "p-value": p_value,
            "stat-sign": (p_value < 0.05),
            "N": n_value,
        }

        if permutation_test:
            raise ValueError("permutation_test has yet to be implemented")

            # # Copy the complete data
            # col2_shuffle = np.array(test_data.loc[:, col2])
            # col2_shuffle = np.repeat(
            #     col2_shuffle[:, np.newaxis], n_permutations, axis=1
            # )
            # # Shuffle within the columns
            # np.random.seed(random_state)
            # ix_i = np.random.sample(col2_shuffle.shape).argsort(axis=0)
            # ix_j = np.tile(np.arange(col2_shuffle.shape[1]), (col2_shuffle.shape[0], 1))
            # col2_shuffle = col2_shuffle[ix_i, ix_j]
            # permutations = np.apply_along_axis(
            #     permute_test,
            #     axis=0,
            #     arr=col2_shuffle,
            #     test_type="correlation",
            #     test=test,
            #     a2=np.array(test_data.loc[:, col1]),
            # )
            #
            # extreme_permutation = np.where(permutations < p_value, 1, 0)
            # p_permutation = extreme_permutation.sum() / len(permutations)
            # dict_summary["permutation-p-value"] = p_permutation
            #
            # # Reset random seed numpy
            # np.random.seed(None)

        summary = pd.concat(
            [summary, pd.DataFrame(data=dict_summary, index=[0])],
            axis=0,
            ignore_index=True,
            sort=False,
        )

    # Embed results within a dictionary
    result_dict = {
        "r-value": r_vals,
        "p-value": p_vals,
        "N": n_vals,
        "info": info,
        "summary": summary,
    }

    return result_dict


def correlations_as_sample_increases(
    data: pd.DataFrame,
    feature1: str,
    feature2: str,
    starting_N: int = 10,
    step: int = 1,
    method="pearson",
    random_state=42,
    bootstrap: bool = False,
    bootstrap_per_N: int = 2,
    plot: bool = True,
    addition_to_title: str = "",
    figsize: Tuple[float, float] = (9.0, 4.0),
    alpha: float = 0.05,
):
    r"""Plot changes in r-value and p-value from correlation between two features when sample size increases.
    Different methods to compute correlations are implemented. Data is shuffled first, to prevent any order effects.
    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with variables in columns, cases in rows
    feature1: str
        Name of column with first feature to be included in correlation
    feature2: str
        Name of column with second feature to be included in correlation
    starting_N: int (default: 10)
        Number of cases that should be used for first correlation
    step: int (default: 1)
        Step for increasing the number of cases for the correlations
    method: {'pearson', 'kendall', 'spearman'}, default 'pearson'
        Type of correlation, either Pearson's r, Spearman's rho, or Kendall's tau, implemented via respectively
        `scipy.stats.pearsonr`, `scipy.stats.spearmanr`, and `scipy.stats.kendalltau`.
    random_state: int (default: 42)
        Random state for reordering the data
    bootstrap: bool
        Whether to bootstrap the data at each N
    bootstrap_per_N: int
        If bootstrap is True then how many bootstraps per each sample size should be performed i.e if bootstrap_per_N
        is 2 then at sample size N=20, 2 bootstraps will be performed. This will continue until starting_N == N.
    plot: bool (default: True)
        Whether to plot the results
    addition_to_title: str (default: '')
        The title of the plot will be "The absolute r-value between {feature1} and {feature2} as N increases" and
        followed by the addition (e.g. to describe a dataset).
    alpha: float (default: 0.05)
        Threshold for p-value that should be shown in the plot
    Returns
    ----------
    cor_results: pd.DataFrame
        Dataframe with the results for all ran analyses
    fig: Figure
        Figure will be returned if plot=True, otherwise None. This allows you to change properties of the figure
        afterwards, e.g. fig.axes[0].set_title('This is my new title')
    Examples
    ----------
    >>> import seaborn as sns
    >>> from jmspack.frequentist_statistics import correlations_as_sample_increases
    >>> iris = sns.load_dataset('iris')
    >>> summary,  fig = correlations_as_sample_increases(data=iris,feature1='petal_width',feature2='sepal_length',
    >>> starting_N=20)
    """

    data = (
        data[[feature1, feature2]].copy()
        # Remove rows with np.nans
        .dropna()
        # Randomize order of the data
        .sample(frac=1, random_state=random_state)
    )
    if data.shape[0] < starting_N:
        raise ValueError("Number of valid cases is smaller than the starting_N")
    if data.shape[0] < starting_N + step:
        raise ValueError(
            "Number of valid cases is smaller than the starting_N + step (only one correlation possible)"
        )

    # Initiate data frame for results
    corr_results = pd.DataFrame()

    # Loop through all possible number of rows from starting N till number of rows
    for i in range(starting_N, data.shape[0] + 1, step):
        boot_corr_results = pd.DataFrame()
        if bootstrap:
            for boot_num in range(0, bootstrap_per_N):
                boot_data = data.sample(frac=1, random_state=boot_num)
                current_boot_corr = correlation_analysis(
                    boot_data.iloc[0:i],
                    method=method,
                    check_norm=False,
                    permutation_test=False,
                )["summary"][["r-value", "p-value", "N"]]
                boot_corr_results = pd.concat(
                    [boot_corr_results, current_boot_corr], ignore_index=True
                )
            corr_results = pd.concat(
                [corr_results, boot_corr_results], ignore_index=True
            )
        else:
            # Run correlation with all data from first row until row i
            current_corr = correlation_analysis(
                data.iloc[0:i], method=method, check_norm=False, permutation_test=False
            )["summary"][["r-value", "p-value", "N"]]
            corr_results = pd.concat([corr_results, current_corr], ignore_index=True)

    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        # Add r-value and p-value
        _ = sns.lineplot(
            x=corr_results["N"],
            y=abs(corr_results["r-value"]),
            label="absolute r-value",
            ax=ax,
        ).set_title(
            f"The absolute r-value between {feature1} and {feature2}\nas N increases {addition_to_title}"
        )
        _ = sns.lineplot(
            x=corr_results["N"], y=corr_results["p-value"], label="p-value", ax=ax
        )
        # Add alpha level (threshold for p-value)
        _ = ax.axhline(
            y=alpha, color="black", alpha=0.5, linestyle="--", label=f">= {alpha}"
        )

        _ = ax.set_ylabel("")
        _ = ax.set_ylim(0, 1)
        _ = plt.legend()
    return corr_results, fig


def multiple_univariate_OLSs(
    X: pd.DataFrame,
    y: pd.Series,
    features_list: list,
):
    all_coefs_df = pd.DataFrame()
    for feature in features_list:
        mod = sm.OLS(endog=y, exog=sm.add_constant(X[[feature]]))
        res = mod.fit()
        coef_df = pd.read_html(
            res.summary().tables[1].as_html(), header=0, index_col=0
        )[0].drop("const")
        coef_df = coef_df.assign(
            **{"rsquared": res.rsquared, "rsquared_adj": res.rsquared_adj}
        )
        all_coefs_df = pd.concat([all_coefs_df, coef_df])
    return all_coefs_df


def potential_for_change_index(
    data: pd.DataFrame,
    features_list: list,
    target: str,
    minimum_measure: str = "min",
    centrality_measure: str = "mean",
    maximum_measure: str = "max",
    weight_measure: str = "rsquared_adj",
    scale_data: bool = True,
    pci_heatmap: bool = True,
    pci_heatmap_figsize: Tuple[float, float] = (1.0, 4.0),
):
    if scale_data:
        data = data.pipe(apply_scaling)

    if weight_measure == "rsquared_adj" or weight_measure == "rsquared":
        tmp_X = data[features_list]
        tmp_y = data[target]
        weight_df = multiple_univariate_OLSs(
            X=tmp_X, y=tmp_y, features_list=features_list
        )

        negative_list = weight_df[weight_df["coef"] < 0].index.tolist()

    else:
        output_dict = correlation_analysis(
            data=data,
            col_list=features_list,
            row_list=[target],
            method="pearson",
            check_norm=False,
            dropna="pairwise",
            permutation_test=False,
            n_permutations=10,
            random_state=69420,
        )
        weight_df = output_dict["summary"].set_index("feature1")
        negative_list = weight_df[weight_df["r-value"] < 0].index.tolist()

    if len(negative_list) < 0:

        pci_df = (
            # room for improvement calculation (series)
            (
                data[features_list].agg(centrality_measure)
                - (data[features_list].agg(maximum_measure))
            ).abs()
            * weight_df[weight_measure]  # weight (based on weight_measure series)
        ).to_frame("PCI")
    else:
        neg_pci_df = (
            # room for improvement calculation (series)
            (
                data[negative_list].agg(centrality_measure)
                - (data[negative_list].agg(minimum_measure))
            ).abs()
            * weight_df.loc[
                negative_list, weight_measure
            ]  # weight (based on weight_measure series)
        ).to_frame("PCI")

        pos_pci_df = (
            # room for improvement calculation (series)
            (
                data[features_list].drop(negative_list, axis=1).agg(centrality_measure)
                - (data[features_list].drop(negative_list, axis=1).agg(maximum_measure))
            ).abs()
            * weight_df[weight_measure].drop(
                negative_list, axis=0
            )  # weight (based on weight_measure series)
        ).to_frame("PCI")

        pci_df = pd.concat([pos_pci_df, neg_pci_df])

    if pci_heatmap:
        _ = plt.figure(figsize=pci_heatmap_figsize)
        _ = sns.heatmap(
            data=pci_df.sort_values(by="PCI", ascending=False),
            annot=True,
            fmt=".3g",
        )

    if weight_measure == "rsquared_adj" or weight_measure == "rsquared":
        return pci_df.merge(
            data[features_list]
            .agg([minimum_measure, centrality_measure, maximum_measure])
            .T,
            left_index=True,
            right_index=True,
        ).merge(weight_df[[weight_measure, "P>|t|"]], left_index=True, right_index=True)

    else:
        return pci_df.merge(
            data[features_list]
            .agg([minimum_measure, centrality_measure, maximum_measure])
            .T,
            left_index=True,
            right_index=True,
        ).merge(
            weight_df.loc[:, [weight_measure, "p-value"]],
            left_index=True,
            right_index=True,
        )