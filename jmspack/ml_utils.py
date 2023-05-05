r"""Submodule ml_utils.py includes the following functions:

- plot_decision_boundary(): Generate a simple plot of the decision boundary of a classifier.

- plot_cv_indices(): Visualise the inputted cross validation method in chunks.

- plot_learning_curve(): Plot the learning curve of an estimator as samples increase to evaluate overfitting.

- dict_of_models: A dictionary of useful models.

- multi_roc_auc_plot(): A utility to plot the ROC curves of multiple classifiers (suggested to use in conjunction with the dict_of_models).

- optimize_model(): A utility to run gridsearch and Recursive Feature Elimination on a classifier to return a model with the best parameters.

- plot_confusion_matrix(): Visualise a confusion matrix.

- summary_performance_metrics_classification(): A utility to return a selection of regularly used classification performance metrics.

- RMSE(): Root Mean Squared Error.

"""
import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.linear_model
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from jmspack.utils import JmsColors


def plot_decision_boundary(
    X: pd.DataFrame,
    y: pd.Series,
    clf: ClassifierMixin = sklearn.linear_model.LogisticRegression(),
    title: str = "Decision Boundary Logistic Regression",
    legend_title: str = "Legend",
    h: float = 0.05,
    figsize: tuple = (11.7, 8.27),
):
    """Generate a simple plot of the decision boundary of a classifier.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Classifier vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples)
        Target relative to X for classification. Datatype should be integers.
    clf : scikit-learn algorithm
        An object that has the `predict` and `predict_proba` methods
    h : int (default: 0.05)
        Step size in the mesh
    title : string
        Title for the plot.
    legend_title : string
        Legend title for the plot.
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches

    Returns
    -------
    boundaries: Figure
        Properties of the figure can be changed later, e.g. use `boundaries.axes[0].set_ylim(0,100)` to change ylim
    ax: Axes
        The axes associated with the boundaries Figure.

    Examples
    --------
    >>> import seaborn as sns
    >>> from sklearn.svm import SVC
    >>> data = sns.load_dataset("iris")
    >>> # convert the target from string to category to numeric as sklearn cannot handle strings as target
    >>> y = data["species"]
    >>> X = data[["sepal_length", "sepal_width"]]
    >>> clf = SVC(kernel="rbf", gamma=2, C=1, probability=True)
    >>> _ = plot_decision_boundary(X=X, y=y, clf=clf, title = 'Decision Boundary', legend_title = "Species")

    """

    if X.shape[1] != 2:
        raise ValueError("X must contains only two features.")

    if not (
        pd.api.types.is_integer_dtype(y)
        or pd.api.types.is_object_dtype(y)
        or pd.api.types.is_categorical_dtype(y)
    ):
        raise TypeError(
            "The target variable y can only have the following dtype: [int, object, category]."
        )

    label_0 = X.columns.tolist()[0]
    label_1 = X.columns.tolist()[1]

    X = X.copy()
    y = y.copy()

    X = X.values
    y = y.astype("category").cat.codes.values

    #     full_col_list = list(sns.color_palette("husl", len(np.unique(y))))
    full_col_list = list(sns.color_palette())

    if len(np.unique(y)) > len(full_col_list):
        raise ValueError(
            "More labels in the data then colors in the color list. Either reduce the number of labels or expend the color list"
        )

    sub_col_list = full_col_list[0 : len(np.unique(y))]
    cmap_bold = ListedColormap(sub_col_list)

    # Try to include a mapping in a later release (+ show categorical labels in the legend)

    _ = clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    Z_proba = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z_max = Z_proba.max(axis=1)  # Take the class with highest probability
    Z_max = Z_max.reshape(xx.shape)

    # Put the result into a color plot
    boundaries, ax = plt.subplots(figsize=figsize)
    _ = ax.contour(xx, yy, Z, cmap=cmap_bold)
    _ = ax.scatter(
        xx, yy, s=(Z_max**2 / h), c=Z, cmap=cmap_bold, alpha=1, edgecolors="none"
    )

    # Plot also the training points
    training = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors="black")
    _ = plt.xlim(xx.min(), xx.max())
    _ = plt.ylim(yy.min(), yy.max())
    _ = plt.title(title)
    _ = plt.subplots_adjust(right=0.8)
    _ = plt.xlabel(label_0)
    _ = plt.ylabel(label_1)

    # Add legend colors
    leg1 = plt.legend(
        *training.legend_elements(),
        frameon=False,
        fontsize=12,
        borderaxespad=0,
        bbox_to_anchor=(1, 0.5),
        handlelength=2,
        handletextpad=1,
        title=legend_title,
    )

    # Add legend sizes
    l1 = plt.scatter([], [], c="black", s=0.4**2 / h, edgecolors="none")
    l2 = plt.scatter([], [], c="black", s=0.6**2 / h, edgecolors="none")
    l3 = plt.scatter([], [], c="black", s=0.8**2 / h, edgecolors="none")
    l4 = plt.scatter([], [], c="black", s=1**2 / h, edgecolors="none")

    labels = ["0.4", "0.6", "0.8", "1"]
    _ = plt.legend(
        [l1, l2, l3, l4],
        labels,
        frameon=False,
        fontsize=12,
        borderaxespad=0,
        bbox_to_anchor=(1, 1),
        handlelength=2,
        handletextpad=1,
        title="Probabilities",
        scatterpoints=1,
    )
    _ = plt.gca().add_artist(leg1)

    return boundaries, ax


def plot_cv_indices(cv, X, y, group, n_splits, lw=10, figsize=(6, 3)):
    """Create an example plot for indices of a cross-validation object.

    Parameters
    ----------
    cv : cross-validation generator
        A scikit-learn cross-validation object with a split method.
    X : array-like
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like
        Target relative to X for classification or regression.
    group : array-like
        Group relative to X for classification or regression.
    n_splits : int
        Number of splits in the cross-validation object.
    lw : int
        Line width for the plots.
    figsize : tuple
        Width and height of the figure in inches

    Returns
    -------
    fig: matplotlib.figure.Figure
        Properties of the figure can be changed later, e.g. use `fig.axes[0].set_ylim(0,100)` to change ylim
    ax: matplotlib.axes._subplots.AxesSubplot
        The axes associated with the fig Figure.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GroupKFold
    >>> import matplotlib.pyplot as plt
    >>> from jmspack.ml_utils import plot_cv_indices
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 2, 1, 2])
    >>> groups = np.array([0, 0, 2, 2])
    >>> group_kfold = GroupKFold(n_splits=2)
    >>> _ = plot_cv_indices(cv=group_kfold, X=X, y=y, group=groups, n_splits=2, lw=10, figsize=(6, 3))
    >>> _ = plt.show()

    """

    # set plotting options
    cmap_data = plt.cm.Paired
    cmap_cv = plt.cm.coolwarm

    fig, ax = plt.subplots(figsize=figsize)

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, len(X)],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)

    ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Testing set", "Training set"],
        loc=(1.02, 0.8),
    )
    # Make the legend fit
    plt.tight_layout()
    fig.subplots_adjust(right=0.7)

    return fig, ax


def plot_learning_curve(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: BaseEstimator = sklearn.linear_model.LogisticRegression(),
    title: str = "Learning Curve Logistic Regression",
    groups: Union[None, np.array] = None,
    cross_color: str = JmsColors.PURPLE,
    test_color: str = JmsColors.YELLOW,
    scoring: str = "accuracy",
    ylim: Union[None, tuple] = None,
    cv: Union[None, int] = None,
    n_jobs: int = -1,
    train_sizes: np.array = np.linspace(0.1, 1.0, 40),
    figsize: tuple = (10, 5),
):
    """Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    cross_color : string
        Signifies the color of the cross validation in the plot
    test_color : string
        Signifies the color of the test set in the plot
    scoring : string
        Signifies a scoring to evaluate the cross validation
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :param groups:
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    """

    fig, ax = plt.subplots(figsize=figsize)
    _ = plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    _ = plt.xlabel("Training examples")
    _ = plt.ylabel(scoring)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        random_state=42,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    _ = plt.grid()

    _ = plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color=test_color,
    )
    _ = plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color=cross_color,
    )
    _ = plt.plot(
        train_sizes, train_scores_mean, "o-", color=test_color, label="Training score"
    )
    _ = plt.plot(
        train_sizes,
        test_scores_mean,
        "o-",
        color=cross_color,
        label="Cross-validation score",
    )

    _ = plt.legend(loc="best")
    return fig, ax


# create a dictionary of models
dict_of_models = [
    {
        "label": "Logistic Regression",
        "model": LogisticRegression(solver="lbfgs"),
    },
    {
        "label": "Gradient Boosting",
        "model": GradientBoostingClassifier(),
    },
    {
        "label": "K_Neighbors Classifier",
        "model": KNeighborsClassifier(3),
    },
    {
        "label": "SVM Classifier (linear)",
        "model": SVC(kernel="linear", C=0.025, probability=True),
    },
    {
        "label": "SVM Classifier (Radial Basis Function; RBF)",
        "model": SVC(kernel="rbf", gamma=2, C=1, probability=True),
    },
    {
        "label": "Gaussian Process Classifier",
        "model": GaussianProcessClassifier(1.0 * RBF(1.0)),
    },
    {
        "label": "Decision Tree (depth=5)",
        "model": DecisionTreeClassifier(max_depth=5),
    },
    {
        "label": "Random Forest Classifier(depth=5)",
        "model": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    },
    {
        "label": "Multilayer Perceptron (MLP) Classifier",
        "model": MLPClassifier(alpha=1, max_iter=1000),
    },
    {
        "label": "AdaBoost Classifier",
        "model": AdaBoostClassifier(),
    },
    {
        "label": "Naive Bayes (Gaussian) Classifier",
        "model": GaussianNB(),
    },
    {
        "label": "Quadratic Discriminant Analysis Classifier",
        "model": QuadraticDiscriminantAnalysis(),
    },
]


def multi_roc_auc_plot(
    X: pd.DataFrame,
    y: pd.Series,
    models: list = dict_of_models,
    figsize: tuple = (7, 7),
):

    """Plot the ROC curves of multiple classifiers.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Classifier vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples)
        Target relative to X for classification. Datatype should be integers.
    models : list
        A list of dictionaries containing the model and the label to be used in the plot.
    figsize: tuple (default: (7, 7))
        Width and height of the figure in inches

    Returns
    -------
    fig: matplotlib.figure.Figure
        Properties of the figure can be changed later, e.g. use `fig.axes[0].set_ylim(0,100)` to change ylim
    ax: matplotlib.axes._subplots.AxesSubplot
        The axes associated with the fig Figure.

    Examples
    --------
    >>> import seaborn as sns
    >>> from jmspack.ml_utils import multi_roc_auc_plot, dict_of_models
    >>> data = (
    ...     sns.load_dataset("iris")
    ...     .loc[lambda df: df["species"].isin(["setosa", "virginica"])]
    ...     .replace({"virginica": 0, "setosa": 1})
    ... )
    >>> y = data["species"]
    >>> X = data[["sepal_length", "sepal_width"]]
    >>> _ = multi_roc_auc_plot(X=X, y=y, models=dict_of_models, figsize=(7, 7))

    """

    # scale the data and create training and test sets of the data
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    fig, ax = plt.subplots(figsize=figsize)
    # Below for loop iterates through your models list
    for m in models:
        model = m["model"]  # select the model
        model.fit(X_train, y_train)  # train the model
        # Compute False postive rate, and True positive rate
        fpr, tpr, thresholds = metrics.roc_curve(
            y_test, model.predict_proba(X_test)[:, 1]
        )
        # Calculate Area under the curve to display on the plot
        auc_score = metrics.roc_auc_score(
            y_test, model.predict(X_test), average="macro"
        )
        # Now, plot the computed values
        plt.plot(fpr, tpr, label="%s ROC (area = %0.2f)" % (m["label"], auc_score))
    # Custom settings for the plot
    _ = plt.plot([0, 1], [0, 1], c="grey", ls="--")
    _ = plt.xlim([0.0, 1.0])
    _ = plt.ylim([0.0, 1.05])
    _ = plt.xlabel("1-Specificity (False Positive Rate)")
    _ = plt.ylabel("Sensitivity (True Positive Rate)")
    _ = plt.title("Receiver Operating Characteristics")
    _ = plt.legend(loc="lower right")
    # plt.show()  # Display

    return fig, ax


def optimize_model(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: BaseEstimator = sklearn.ensemble.RandomForestClassifier(),
    grid_params_dict: dict = {
        "max_depth": [1, 2, 3, 4, 5, 10],
        "n_estimators": [10, 20, 30, 40, 50],
        "max_features": ["log2", "sqrt"],
        "criterion": ["gini", "entropy"],
    },
    gridsearch_kwargs: dict = {"scoring": "roc_auc", "cv": 3, "n_jobs": -2},
    rfe_kwargs: dict = {"n_features_to_select": 2, "verbose": 1},
):

    """A utility to run gridsearch and Recursive Feature Elimination on a classifier to return a model with the best parameters.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Classifier vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples)
        Target relative to X for classification. Datatype should be integers.
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    grid_params_dict : dict
        A dictionary of parameters to be used in the gridsearch.
    gridsearch_kwargs : dict
        A dictionary of parameters to be used in the gridsearch.
    rfe_kwargs : dict
        A dictionary of parameters to be used in the Recursive Feature Elimination.

    Returns
    -------
    optimized_estimator: sklearn estimator
        The optimized estimator.
    feature_ranking: pandas DataFrame
        A dataframe with features ranking (high = dropped early on).
    feature_selected: list
        A list of features selected.
    feature_importance: pandas DataFrame
        A dataframe with importances per feature.
    optimal_parameters: pandas DataFrame
        A dataframe with the optimal parameters.

    Examples
    --------
    >>> import seaborn as sns
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from jmspack.ml_utils import optimize_model
    >>> data = (
    ...     sns.load_dataset("iris")
    ...     .loc[lambda df: df["species"].isin(["setosa", "virginica"])]
    ...     .replace({"virginica": 0, "setosa": 1})
    ... )
    >>> y = data["species"]
    >>> X = data[["sepal_length", "sepal_width"]]
    >>> model = RandomForestClassifier()
    >>> (
    ...    optimized_estimator,
    ...    feature_ranking,
    ...    feature_selected,
    ...    feature_importance,
    ...    optimal_parameters,
    ... ) = optimize_model(X=X, y=y, estimator=model)

    """
    # Perform a 75% training and 25% test data split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        # stratify=y,
        random_state=42,
    )

    # Instantiate grid_dt
    grid_dt = GridSearchCV(
        estimator=estimator, param_grid=grid_params_dict, **gridsearch_kwargs
    )

    # Optimize hyperparameter
    _ = grid_dt.fit(X_train, y_train)

    # Extract the best estimator
    optimized_estimator = grid_dt.best_estimator_

    # Create the RFE with a optimized random forest
    rfe = RFE(estimator=optimized_estimator, **rfe_kwargs)

    # Fit the eliminator to the data
    _ = rfe.fit(X_train, y_train)

    # create dataframe with features ranking (high = dropped early on)
    feature_ranking = pd.DataFrame(
        data=dict(zip(X.columns, rfe.ranking_)), index=np.arange(0, len(X.columns))
    )
    feature_ranking = feature_ranking.loc[0, :].sort_values()

    # create dataframe with feature selected
    feature_selected = X.columns[rfe.support_].to_list()

    # create dataframe with importances per feature
    feature_importance = pd.Series(
        dict(zip(X.columns, optimized_estimator.feature_importances_.round(2)))
    )

    # Calculates the test set accuracy
    # acc = metrics.accuracy_score(y_test, rfe.predict(X_test))

    print("\n- Sizes :")
    print(f"- X shape = {X.shape}")
    print(f"- y shape = {y.shape}")
    print(f"- X_train shape = {X_train.shape}")
    print(f"- X_test shape = {X_test.shape}")
    print(f"- y_train shape = {y_train.shape}")
    print(f"- y_test shape = {y_test.shape}")

    print("\n- Model info :")
    print(f"- Optimal Parameters = {optimized_estimator.get_params()}")
    print(f"- Selected feature list = {feature_selected}")
    # print("- Accuracy score on test set = {0:.1%}".format(acc))

    return (
        optimized_estimator,
        feature_ranking,
        feature_selected,
        feature_importance,
        pd.DataFrame(optimized_estimator.get_params(), index=["optimal_parameters"]),
    )


def plot_confusion_matrix(
    cf,
    group_names=None,
    categories="auto",
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize: tuple = (7, 5),
    cmap="Blues",
    title=None,
):
    """This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Parameters
    ----------
    cf:
        confusion matrix to be passed in
    group_names:
        List of strings that represent the labels row by row to be shown in each square.
    categories:
        List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:
        If True, show the raw number in the confusion matrix. Default is True.
    normalize:
        If True, show the proportions for each category. Default is True.
    cbar:
        If True, show the color bar. The cbar values are based off the values in the confusion matrix. Default is True.
    xyticks:
        If True, show x and y ticks. Default is True.
    xyplotlabels:
        If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:
        If True, display summary statistics below the figure. Default is True.
    figsize:
        Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:
        Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
        See http://matplotlib.org/examples/color/colormaps_reference.html
    title:
        Title for the heatmap. Default is None.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Properties of the figure can be changed later, e.g. use `fig.axes[0].set_ylim(0,100)` to change ylim
    ax: matplotlib.axes._subplots.AxesSubplot
        The axes associated with the fig Figure.

    Examples
    --------
    >>> import seaborn as sns
    >>> from sklearn.metrics import confusion_matrix
    >>> from jmspack.ml_utils import plot_confusion_matrix
    >>> y_true = ["cat", "dog", "cat", "cat", "dog", "bird"]
    >>> y_pred = ["cat", "cat", "cat", "dog", "bird", "bird"]
    >>> cf = confusion_matrix(y_true, y_pred, labels=["cat", "dog", "bird"])
    >>> _ = plot_confusion_matrix(cf, figsize=(7, 5))

    """

    fig, ax = plt.subplots(figsize=figsize)

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [
            "{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)
        ]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    _ = sns.heatmap(
        cf,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )

    if xyplotlabels:
        _ = plt.ylabel("True label")
        _ = plt.xlabel("Predicted label" + stats_text)
    else:
        _ = plt.xlabel(stats_text)

    if title:
        _ = plt.title(title)

    return fig, ax


def _bootstrap_auc(
    model, X_test, y_true, use_probabilities, bootstraps, fold_size, random_state
):
    """Internal function to bootstrap auc.
    Originates from the AI in healthcare specialization of coursera. https://www.coursera.org/specializations/ai-healthcare

    Parameters
    ----------
    model:
        The fitted sklearn model.
    X_test: pd.Series
        The predictors used to match to y_true.
    y_true: pd.Series
        The actual binary targets.
    classes: list(str)
        List with the name of the classes in string format.
    bootstraps: int
        The number of bootstraps.
    fold_size: int
        The number of folds.

    Returns
    -------
    list

    """

    if use_probabilities:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        df = pd.DataFrame({"y": y_true, "pred": y_pred_proba})
    else:
        y_pred = model.predict(X_test)
        df = pd.DataFrame({"y": y_true, "pred": y_pred})

    statistics = np.zeros(bootstraps)

    df_pos = df[df.y == 1]
    df_neg = df[df.y == 0]
    prevalence = len(df_pos) / len(df)

    # get positive examples for stratified sampling
    for i in range(bootstraps):
        # stratified sampling of positive and negative examples
        pos_sample = df_pos.sample(
            n=int(fold_size * prevalence), replace=True, random_state=random_state
        )
        neg_sample = df_neg.sample(
            n=int(fold_size * (1 - prevalence)),
            replace=True,
            random_state=random_state + 1,
        )

        y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
        pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])

        if use_probabilities:
            fpr, tpr, thresholds = metrics.roc_curve(y_sample, pred_sample, pos_label=1)
            score = metrics.auc(fpr, tpr)
        else:
            score = metrics.roc_auc_score(y_sample, pred_sample)

        statistics[i] = score

    mean = statistics.mean()
    max_ = np.quantile(statistics, 0.95)
    min_ = np.quantile(statistics, 0.05)

    return [f"{mean:.3f} (95% CI {min_:.3f}-{max_:.3f})"]


def summary_performance_metrics_classification(
    model, X_test, y_true, bootstraps=100, fold_size=1000, random_state=69420
):
    """Summary of different evaluation metrics specific to a single class classification learning problem.

    Parameters
    ----------
    model: sklearn.model
        A fitted sklearn model with predict() and predict_proba() methods.
    X_test: pd.DataFrame
        A data frame used to run predict the target values (y_pred).
    y_true: pd.Series or np.arrays
        Binary true values.
    bootstraps: int
    fold_size: int

    Returns
    -------
    summary_df: pd.DataFrame
        A dataframe with the summary of the metrics.

    Notes
    -----
    The function returns the following metrics:
        - true positive (TP): The model classifies the example as positive, and the actual label also positive.
        - false positive (FP): The model classifies the example as positive, but the actual label is negative.
        - true negative (TN): The model classifies the example as negative, and the actual label is also negative.
        - false negative (FN): The model classifies the example as negative, but the label is actually positive.
        - accuracy: The fractions of predictions the model got right.
        - prevalance: The proportion of positive examples. Where y=1.
        - sensitivity: The probability that our test outputs positive given that the case is actually positive.
        - specificity: The probability that the test outputs negative given that the case is actually negative.
        - positive predictive value: The proportion of positive predictions that are true positives.
        - negative predictive value: The proportion of negative predictions that are true negatives.
        - auc: A measure of goodness of fit.
        - bootstrapped auc: The bootstrap estimates the uncertainty by resampling the dataset with replacement.
        - F1: The harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.

    Examples
    --------
    >>> import seaborn as sns
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from jmspack.ml_utils import summary_performance_metrics_classification
    >>> data = (
    ...     sns.load_dataset("iris")
    ...    .loc[lambda df: df["species"].isin(["setosa", "virginica"])]
    ...    .replace({"virginica": 0, "setosa": 1})
    ... )
    >>> y = data["species"]
    >>> X = data[["sepal_length", "sepal_width"]]
    >>> model = RandomForestClassifier()
    >>> model.fit(X=X, y=y)
    >>> summary_df = summary_performance_metrics_classification(model=model, X_test=X, y_true=y)

    """

    y_pred = model.predict(X_test)

    # check if the fitted model has the "predict_proba" attribute
    if "predict_proba" in dir(model):
        # check that the fitted model has the "probability" attribute
        if "probability" in dir(model):
            # and that it is set to True (this can be the case for SVC)
            if model.probability:
                predict_proba_bool = True
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                # auc
                fpr, tpr, thresholds = metrics.roc_curve(
                    y_true, y_pred_proba, pos_label=1
                )
                auc_score = metrics.auc(fpr, tpr)
            else:
                predict_proba_bool = False
                warnings.warn(
                    f"The classifier {model.__class__} does have the 'predict_proba' method, however it does not have"
                    f" the 'probability' parameter set to True, hence model evaluation metrics will be based on "
                    f"binary predictions"
                )
                auc_score = metrics.roc_auc_score(y_true, y_pred)
        else:
            # the model has "predict_proba and no "probability" boolean so it 100% has "predict_proba"
            predict_proba_bool = True
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            # auc
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_proba, pos_label=1)
            auc_score = metrics.auc(fpr, tpr)
    else:
        # the model has no "predict_proba" attribute so probabilities are not used
        predict_proba_bool = False
        warnings.warn(
            f"The classifier {model.__class__} does not have the 'predict_proba' method, hence "
            f"model evaluation metrics will be based on binary predictions"
        )
        auc_score = metrics.roc_auc_score(y_true, y_pred)

    # bootstrapped auc
    bootstrap_auc_metric = _bootstrap_auc(
        model,
        X_test,
        y_true,
        use_probabilities=predict_proba_bool,
        bootstraps=bootstraps,
        fold_size=fold_size,
        random_state=random_state,
    )

    # TP, TN, FP, FN
    confusion_matrix_metric = metrics.confusion_matrix(y_true, y_pred)
    TN = confusion_matrix_metric[0][0]
    FP = confusion_matrix_metric[0][1]
    FN = confusion_matrix_metric[1][0]
    TP = confusion_matrix_metric[1][1]

    # accuracy
    accuracy_score_metric = metrics.accuracy_score(y_true, y_pred)

    # balanced accuracy
    balanced_accuracy_score_metric = metrics.balanced_accuracy_score(y_true, y_pred)

    # prevalance
    prevalence = np.mean(y_true == 1)

    # sensitivity
    sensitivity = TP / (TP + FN)

    # specificity
    specificity = TN / (TN + FP)

    # positive predictive value
    PPV = TP / (TP + FP)

    # negative predictive value
    NPV = TN / (TN + FN)

    # F1
    f1 = metrics.f1_score(y_true, y_pred)

    df_metrics = pd.DataFrame(
        {
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "TP": TP,
            "Accuracy": accuracy_score_metric,
            "Balanced Accuracy": balanced_accuracy_score_metric,
            "Prevalence": prevalence,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "PPV": PPV,
            "NPV": NPV,
            "auc": auc_score,
            "Mean AUC (CI 5%-95%)": bootstrap_auc_metric,
            "F1": f1,
        },
        index=["scores"],
    )

    return df_metrics.round(3)


def RMSE(true, pred):
    """Root Mean Squared Error.

    Parameters
    ----------
    true: pd.Series
        The actual values.
    pred: pd.Series
        The predicted values.

    Returns
    -------
    float

    Examples
    --------
    >>> import pandas as pd
    >>> from jmspack.ml_utils import RMSE
    >>> true = pd.Series([1, 2, 5, 4, 5])
    >>> pred = pd.Series([1, 2, 3, 4, 5])
    >>> RMSE(true, pred)

    """
    return np.sqrt(mean_squared_error(y_true=true, y_pred=pred))
