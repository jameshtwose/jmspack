r"""Submodule imputation_utils.py includes the following functions:

- mice_forest(): using miceforest package to impute missing values.

- mice_forest_tune(): using miceforest package to tune parameters.

- simple_impute(): using sklearn.impute.SimpleImputer to impute missing values.

"""
import miceforest as mf
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer


def mice_forest(data, params=None, mi_datasets=10, random_state=42, iterations=10):
    """Impute missing values using miceforest package.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with missing values.
    params : dict, optional
        Parameters for miceforest package, by default None.
    mi_datasets : int, optional
        Number of datasets to create, by default 10.
    random_state : int, optional
        Random state for reproducibility, by default 42.
    iterations : int, optional
        Number of iterations to run, by default 10.

    Returns
    -------
    completed_data : pandas.DataFrame
        Dataframe with imputed missing values.

    Examples
    --------
    >>> from jmspack.imputation_utils import mice_forest
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    >>> data.iloc[0, 0] = np.nan
    >>> data.iloc[1, 1] = np.nan
    >>> data.iloc[2, 0] = np.nan
    >>> completed_data = mice_forest(data)

    """

    # Create kernel.
    kds = mf.ImputationKernel(
        data, datasets=mi_datasets, save_all_iterations=True, random_state=random_state
    )

    # Run the MICE algorithm for N amount of iterations
    kds.mice(iterations=iterations, variable_parameters=params)

    # Return the completed kernel data
    completed_data = kds.complete_data(dataset=0, inplace=False)

    completed_data = completed_data.set_index(data.index)
    return completed_data


def mice_forest_tune(data, mi_datasets=10, random_state=42, optimization_steps=5):
    """Tune parameters for miceforest package.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with missing values.
    mi_datasets : int
        Number of datasets to create, by default 10.
    random_state : int
        Random state for reproducibility, by default 42.
    optimization_steps : int
        Number of optimization steps, by default 5.

    Returns
    -------
    optimal_parameters : dict
        Optimal parameters for miceforest package.
    losses : list
        Losses for each optimization step.

    Examples
    --------
    >>> from jmspack.imputation_utils import mice_forest_tune
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    >>> data.iloc[0, 0] = np.nan
    >>> data.iloc[1, 1] = np.nan
    >>> data.iloc[2, 0] = np.nan
    >>> optimal_parameters, losses = mice_forest_tune(data)

    """

    # Create kernel.
    kds = mf.ImputationKernel(
        data, datasets=mi_datasets, save_all_iterations=True, random_state=random_state
    )

    optimal_parameters, losses = kds.tune_parameters(
        dataset=0, optimization_steps=optimization_steps
    )

    return optimal_parameters, losses


def simple_impute(data, strategy="mean"):
    """Impute missing values using sklearn.impute.SimpleImputer.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with missing values.
    strategy : str
        Strategy for imputation, by default "mean".

    Returns
    -------
    imputed_data : pandas.DataFrame
        Dataframe with imputed missing values.

    Examples
    --------
    >>> from jmspack.imputation_utils import simple_impute
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    >>> data.iloc[0, 0] = np.nan
    >>> data.iloc[1, 1] = np.nan
    >>> data.iloc[2, 0] = np.nan
    >>> imputed_data = data.pipe(simple_impute, strategy="median")

    """

    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputed_values = imp.fit_transform(data.values)
    return pd.DataFrame(imputed_values, columns=data.columns)
