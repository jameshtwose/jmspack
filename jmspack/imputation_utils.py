r"""Submodule imputation_utils.py includes the following functions:

- RMSE(): Root Mean Squared Error.

- mice_forest(): tmp.

- mice_forest_tune(): tmp.

- groupby_mice(): tmp.

- simple_impute(): tmp.


"""
import miceforest as mf
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error


def RMSE(true, pred):
    return np.sqrt(mean_squared_error(y_true=true, y_pred=pred))


def mice_forest(data, params=None, mi_datasets=10, random_state=42, iterations=10):
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
    # Create kernel.
    kds = mf.ImputationKernel(
        data, datasets=mi_datasets, save_all_iterations=True, random_state=random_state
    )

    optimal_parameters, losses = kds.tune_parameters(
        dataset=0, optimization_steps=optimization_steps
    )

    return optimal_parameters, losses


def groupby_mice(data, id_column, index_column, ids):
    return pd.concat(
        [
            mice_forest(
                data[data[id_column] == user]
                .drop(id_column, axis=1)
                .set_index(index_column)
            )
            .assign(**{id_column: user})
            .reset_index()
            for user in ids
        ]
    ).set_index([id_column, index_column])


def simple_impute(data, strategy="mean"):
    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    return imp.fit_transform(data.values.reshape(-1, 1))
