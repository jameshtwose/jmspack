"""Submodule NLTSA.py includes the following functions: <br>
- fluctuation_intensity(): run fluctuation intensity on a time series to detect non linear change <br>
- distribution_uniformity(): run distribution uniformity on a time series to detect non linear change <br>
- complexity_resonance(): the product of fluctuation_intensity and distribution_uniformity <br>
- complexity_resonance_diagram(): plots a heatmap of the complexity_resonance <br>
- ts_levels(): defines distinct levels in a time series based on decision tree regressor <br>
- cmaps_options[]: a list of possible colour maps that may be used when plotting <br>
- flatten(): a utils function which flattens a list of lists into one list <br>
- cumulative_complexity_peaks(): a function which will calculate the significant peaks in the dynamic
complexity of a set of time series (these peaks are known as cumulative complexity peaks; CCPs) <br>
- cumulative_complexity_peaks_plot(): plots a heatmap of the cumulative_complexity_peaks <br>
"""
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# create flatten function to use when you have lists within lists
flatten = lambda l: [item for sublist in l for item in sublist]

cmaps_options = [
    "flag",
    "prism",
    "ocean",
    "gist_earth",
    "terrain",
    "gist_stern",
    "gnuplot",
    "gnuplot2",
    "CMRmap",
    "cubehelix",
    "brg",
    "gist_rainbow",
    "rainbow",
    "jet",
    "nipy_spectral",
    "gist_ncar",
]


def ts_levels(
    ts,
    ts_x=None,
    criterion="mse",
    max_depth=2,
    min_samples_leaf=1,
    min_samples_split=2,
    max_leaf_nodes=30,
    plot=True,
    equal_spaced=True,
    n_x_ticks=10,
    figsize=(20, 5),
):
    r"""Use recursive partitioning (DecisionTreeRegressor) to perform a 'classification' of relatively stable levels in a timeseries.
    Parameters
    ---------
    ts: pandas DataFrame (column)
        A dataframe column containing a univariate time series from 1 person.
        Rows should indicate time, column should indicate the time series variable.
    ts_x: pandas DataFrame (column; Default=None)
        A dataframe column containing the corresponding timestamps to the aforementioned time series.
        If None is passed, the index of the time series will be used (Default = None).
    criterion: str (Default="mse")
        The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error, which
        is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of each
        terminal node, “friedman_mse”, which uses mean squared error with Friedman’s improvement score for potential
        splits, and “mae” for the mean absolute error, which minimizes the L1 loss using the median of each terminal node.
    max_depth: int or None, optional (default=2)
        The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves
        contain less than min_samples_split samples.
    min_samples_leaf: int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered
        if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have
        the effect of smoothing the model, especially in regression.
        If int, then consider min_samples_leaf as the minimum number.
        If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are
        the minimum number of samples for each node.
    min_samples_split: int, float, optional (default=2)
        The minimum number of samples required to split an internal node.
        If int, then consider min_samples_split as the minimum number.
        If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum
        number of samples for each split.
    max_leaf_nodes: int or None, optional (default=30)
        Identify max_leaf_nodes amount of time series levels in the time series in best-first fashion. Best splits are
        defined as relative reduction in impurity. If None then unlimited number of splits.
    plot: boolean (Default=True)
        A boolean to define whether to plot the time series and it's time series levels.
    equal_spaced: boolean (Default=True)
        A boolean to define whether or not the time series is continuously measured or not. If False this will be taken
        into account when plotting the X-axis of the plot.
    n_x_ticks: int (Default=10)
        The amount of x-ticks you wish to show when plotting.
    figsize: tuple (Default=(20,5))
        The tuple used to specify the size of the plot if plot = True.

    Examples
    ---------
    Demonstration of the function using time series data
    >>> ts_df = pd.read_csv("time_series_dataset.csv", index_col=0)
    >>> ts = ts_df["lorenz"]
    >>> ts_levels_df, fig, ax = ts_levels(ts, ts_x=None, criterion="mse", max_depth=10, min_samples_leaf=1,
    >>>                          min_samples_split=2, max_leaf_nodes=30, plot=True, equal_spaced=True, n_x_ticks=10)
    """
    # Change ts to a numpy array
    if not isinstance(ts, np.ndarray):
        ts = ts.to_numpy()
    # Check whether ts has only one dimension
    if len(ts.shape) != 1:
        raise ValueError("ts is not one-dimensional")

    # Make sure the ts_x matches ts
    if ts_x is not None:
        # Change ts_x to a numpy array
        if not isinstance(ts_x, np.ndarray):
            ts_x = ts_x.to_numpy()
        # Check whether ts_x has only one dimension
        if len(ts.shape) != 1:
            raise ValueError("ts_x is not one-dimensional")
        # Check whether ts and tx_x have the same length
        if not len(ts) == len(ts_x):
            raise ValueError("ts and ts_x have different lengths")

    # predictor for the tree
    x = np.array(np.arange(len(ts)))
    x = x.reshape(-1, 1)

    dtr = DecisionTreeRegressor(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_leaf_nodes=max_leaf_nodes,
    )
    tree = dtr.fit(x, ts)
    p = tree.predict(x)

    fig = None
    ax = None

    if plot:
        # Select n indices
        idx = np.round(np.linspace(0, len(x) - 1, n_x_ticks)).astype(int)

        # X ticks and labels based on indices
        if ts_x is None:
            xticks = flatten(x[idx].tolist())
            xlabs = flatten(x[idx].tolist())
            x_plot = x

        # X labels based on ts_x
        else:
            # Plot using x indices and ts_x labels
            if equal_spaced:
                x_plot = x
                xticks = flatten(x[idx].tolist())
                xlabs = flatten(ts_x[idx].tolist())

            # Plot using ts_x indices and ts_x labels
            else:
                x_plot = ts_x
                xticks = flatten(ts_x[idx].tolist())
                xlabs = flatten(ts_x[idx].tolist())

        # _ = plt.figure(figsize=(20, 7))
        fig, ax = plt.subplots(figsize=figsize)
        _ = plt.scatter(
            x_plot,
            ts,
            s=20,
            edgecolor="#2167C5",
            c="#EB5E23",
            label="Original Time Series",
        )
        _ = plt.plot(
            x_plot,
            p,
            c="white",
            path_effects=[pe.Stroke(linewidth=5, foreground="#2167C5"), pe.Normal()],
            label="Time Series Levels",
        )
        _ = plt.xticks(xticks, xlabs)
        _ = plt.ylabel("Amount")
        _ = plt.xlabel("Time")
        _ = plt.legend()
        _ = plt.title("Time Series Levels Plot")
        # _ = plt.show()

    # Store t_steps, original ts and ts_levels to a dataframe
    df_result = pd.DataFrame(
        {
            "t_steps": flatten(x.tolist()),
            "original_ts": ts.tolist(),
            "ts_levels": p.tolist(),
        }
    )

    # Add the additional ts_x
    if ts_x is not None:
        df_result["ts_x"] = ts_x

    return df_result, fig, ax


def distribution_uniformity(df, win, xmin, xmax, col_first, col_last):
    r"""Run distribution uniformity on a time series to detect non linear change
    Parameters
    ---------
    df: pandas DataFrame
        A dataframe containing multivariate time series data from 1 person.
        Rows should indicate time, columns should indicate the time series variables.
        All time series in df should be on the same scale. Otherwise the comparisons across
        time series will make no sense.
    win: int
        Size of sliding window in which to calculate distribution uniformity
        (amount of data considered in each evaluation of change).
    xmin: int
        The theoretical minimum that the values in the time series can take (scaling?)
    xmax: int
        The theoretical maximum that the values in the time series can take (scaling?)
    col_first: int
        The first column index you wish to be included in the calculation (index starts at 1!)
    col_last: int
        The last column index you wish to be included in the calculation (index starts at 1!)

    Examples
    ---------
    Demonstration of the function using time series data
    >>> ts_df = pd.read_csv("time_series_dataset.csv", index_col=0)
    >>> scaler = MinMaxScaler()
    >>> scaled_ts_df = pd.DataFrame(scaler.fit_transform(ts_df), columns=ts_df.columns.tolist())
    >>> distribution_uniformity_df = pd.DataFrame(distribution_uniformity(scaled_ts_df, win=7, xmin=0, xmax=1, col_first=1, col_last=7))
    >>> distribution_uniformity_df.columns=scaled_ts_df.columns.tolist()
    """

    col_first = int(col_first)
    col_last = int(col_last)
    win = int(win)
    xmin = int(xmin)
    xmax = int(xmax)

    nrow = len(df) + 2
    ncol = col_last - col_first + 1

    ew_data_D = np.zeros((nrow, ncol))
    ew_data_D = pd.DataFrame(ew_data_D)

    for column in range(col_first - 1, col_last):
        ts = df.iloc[:, column : column + 1].values
        # s = xmax - xmin
        y = np.linspace(xmin, xmax, win)

        for i in range(0, len(ts) - win + 1):
            x = ts[i : i + win]
            x = np.sort(x, axis=None)
            r = 0
            g = 0

            for e in range(0, win - 1):
                for d in range(e + 1, win):
                    for a in range(e, d):
                        for b in range(a + 1, d + 1):
                            h = np.heaviside((y[b] - y[a]) - (x[b] - x[a]), 0)
                            if h == 1:
                                r += (y[b] - y[a]) - (x[b] - x[a])
                                g += y[b] - y[a]

            ew_data_D.iloc[i + win - 1, column - col_first + 1] = 1.0 - r / g

        distribution_uniformity_df = pd.DataFrame(ew_data_D.iloc[0 : len(df), :])

        distribution_uniformity_df.columns = df.columns.tolist()
        distribution_uniformity_df.index = df.index.tolist()

    return distribution_uniformity_df


def fluctuation_intensity(df, win, xmin, xmax, col_first, col_last):
    r"""Run fluctuation intensity on a time series to detect non linear change
    Parameters
    ---------
    df: pandas DataFrame
        A dataframe containing multivariate time series data from 1 person.
        Rows should indicate time, columns should indicate the time series variables.
        All time series in df should be on the same scale. Otherwise the comparisons across
        time series will make no sense.
    win: int
        Size of sliding window in which to calculate fluctuation intensity
        (amount of data considered in each evaluation of change).
    xmin: int
        The theoretical minimum that the values in the time series can take (scaling?)
    xmax: int
        The theoretical maximum that the values in the time series can take (scaling?)
    col_first: int
        The first column index you wish to be included in the calculation (index starts at 1!)
    col_last: int
        The last column index you wish to be included in the calculation (index starts at 1!)

    Examples
    ---------
    Demonstration of the function using time series data
    >>> ts_df = pd.read_csv("time_series_dataset.csv", index_col=0)
    >>> scaler = MinMaxScaler()
    >>> scaled_ts_df = pd.DataFrame(scaler.fit_transform(ts_df), columns=ts_df.columns.tolist())
    >>> fluctuation_intensity_df = pd.DataFrame(fluctuation_intensity(scaled_ts_df, win=7, xmin=0, xmax=1, col_first=1, col_last=7))
    >>> fluctuation_intensity_df.columns=scaled_ts_df.columns.tolist()
    """

    col_first = int(col_first)
    col_last = int(col_last)
    win = int(win)
    xmin = int(xmin)
    xmax = int(xmax)

    nrow = len(df)
    ncol = col_last - col_first + 1

    ew_data_F = np.zeros((nrow, ncol))
    ew_data_F = pd.DataFrame(ew_data_F)

    newrows = df.iloc[0:1, :].copy()
    newrows.iloc[:, col_first - 1 : col_last] = 0.0

    data = df.append(newrows)
    data = data.append(newrows)
    data.reset_index(inplace=True, drop=True)

    s = xmax - xmin
    length_ts = len(data)

    for column in range(col_first - 1, col_last):
        distance = 1
        ts = data.iloc[:, column : column + 1].values

        for i in range(0, length_ts - win - 1):
            y = [0] * (win - 1)
            fluct = [0] * (win - 1)
            k = [0] * (win - 1)
            dist_next = 1

            for j in range(0, win - 1):
                if (ts[i + j + 1] >= ts[i + j]) and (ts[i + j + 1] > ts[i + j + 2]):
                    k[j] = 1
                elif (ts[i + j + 1] <= ts[i + j]) and (ts[i + j + 1] < ts[i + j + 2]):
                    k[j] = 1
                elif (ts[i + j + 1] > ts[i + j]) and (ts[i + j + 1] == ts[i + j + 2]):
                    k[j] = 1
                elif (ts[i + j + 1] < ts[i + j]) and (ts[i + j + 1] == ts[i + j + 2]):
                    k[j] = 1
                elif (ts[i + j + 1] == ts[i + j]) and (ts[i + j + 1] > ts[i + j + 2]):
                    k[j] = 1
                elif (ts[i + j + 1] == ts[i + j]) and (ts[i + j + 1] < ts[i + j + 2]):
                    k[j] = 1
                else:
                    k[j] = 0

            k[win - 2] = 1

            for g in range(0, len(k)):
                if k[g] == 1:
                    y[g] = abs(ts[i + g + 1] - ts[i + g + 1 - dist_next])
                    fluct[g] = y[g] / ((i + g + 2) - (i + g + 2 - dist_next))
                    dist_next = distance
                else:
                    y[g] = 0
                    fluct[g] = 0
                    dist_next += 1

            summation = 0.0
            for num in fluct:
                summation = summation + num / (s * (win - 1))

            ew_data_F.iloc[i + win - 1, column - col_first + 1] = summation

        fluctuation_intensity_df = pd.DataFrame(ew_data_F)

        fluctuation_intensity_df.columns = df.columns.tolist()
        fluctuation_intensity_df.index = df.index.tolist()

    return fluctuation_intensity_df


def complexity_resonance(distribution_uniformity_df, fluctuation_intensity_df):
    r"""Create a complexity resonance data frame based on the product of the distribution uniformity and the fluctuation intensity
    Parameters
    ---------
    distribution_uniformity_df: pandas DataFrame
        A dataframe containing distribution uniformity values from multivariate time series data from 1 person.
        Rows should indicate time, columns should indicate the distribution uniformity.
    fluctuation_intensity_df: pandas DataFrame
        A dataframe containing fluctuation intensity values from multivariate time series data from 1 person.
        Rows should indicate time, columns should indicate the fluctuation intensity.

    Examples
    ---------
    Demonstration of the function using time series data
    >>> ts_df = pd.read_csv("time_series_dataset.csv", index_col=0)
    >>> scaler = MinMaxScaler()
    >>> scaled_ts_df = pd.DataFrame(scaler.fit_transform(ts_df), columns=ts_df.columns.tolist())
    >>> distribution_uniformity_df = pd.DataFrame(distribution_uniformity(scaled_ts_df, win=7, xmin=0, xmax=1, col_first=1, col_last=7))
    >>> distribution_uniformity_df.columns=scaled_ts_df.columns.tolist()
    >>> fluctuation_intensity_df = pd.DataFrame(fluctuation_intensity(scaled_ts_df, win=7, xmin=0, xmax=1, col_first=1, col_last=7))
    >>> fluctuation_intensity_df.columns=scaled_ts_df.columns.tolist()
    >>> complexity_resonance_df = complexity_resonance(distribution_uniformity_df, fluctuation_intensity_df)
    """
    complexity_resonance_df = distribution_uniformity_df * fluctuation_intensity_df
    return complexity_resonance_df


def complexity_resonance_diagram(
    df,
    cmap_n: int = 12,
    plot_title="Complexity Resonance Diagram",
    labels_n=10,
    figsize=(20, 7),
):
    r"""Create a complexity resonance data frame based on the product of the distribution uniformity and the fluctuation intensity
    Parameters
    ---------
    df: pandas DataFrame
        A dataframe containing complexity resonance values from multivariate time series data from 1 person.
        Rows should indicate time, columns should indicate the complexity resonance.
    cmap_n: int (Default=12)
        An integer indicating which colour map to use when plotting the heatmap. These values correspond to the index
        of the cmaps_options list (['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
        'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral',
        'gist_ncar']). Index=12 corresponds to 'rainbow.
    plot_title: str
        A string indicating the title to be used at the top of the plot
    labels_n: int (Default=10)
        An integer indicating the nth value to be taken for the x-axis of the plot. So if the x-axis consists of
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, and the labels_n value is set to 2, then 2, 4, 6, 8, 10, will be shown on the
        x-axis of the plot.

    Examples
    ---------
    Demonstration of the function using time series data
    >>> ts_df = pd.read_csv("time_series_dataset.csv", index_col=0)
    >>> scaler = MinMaxScaler()
    >>> scaled_ts_df = pd.DataFrame(scaler.fit_transform(ts_df), columns=ts_df.columns.tolist())
    >>> distribution_uniformity_df = pd.DataFrame(distribution_uniformity(scaled_ts_df, win=7, xmin=0, xmax=1, col_first=1, col_last=7))
    >>> distribution_uniformity_df.columns=scaled_ts_df.columns.tolist()
    >>> fluctuation_intensity_df = pd.DataFrame(fluctuation_intensity(scaled_ts_df, win=7, xmin=0, xmax=1, col_first=1, col_last=7))
    >>> fluctuation_intensity_df.columns=scaled_ts_df.columns.tolist()
    >>> complexity_resonance_df = complexity_resonance(distribution_uniformity_df, fluctuation_intensity_df)
    >>> complexity_resonance_diagram(complexity_resonance_df, cmap_n=12, labels_n=30)
    """

    df_for_plot = df.copy()
    # df_for_plot.insert(loc=0, column="", value=np.nan)

    # plot the complexity resonance diagram
    fig, ax = plt.subplots(figsize=figsize)

    plot_comp = plt.imshow(df_for_plot.T, cmap=cmaps_options[cmap_n])
    plt.gca().set_aspect(aspect="auto")

    # set the color bar on the right, based on the values from the data
    _ = fig.colorbar(plot_comp)

    # Show all ticks
    ax.set_xticks(np.arange(0, len(df_for_plot)))
    ax.set_yticks(np.arange(0, len(list(df_for_plot))))

    # and label them with the respective list entries
    ax.set_xticklabels(list(df_for_plot.index))
    ax.set_yticklabels(list(df_for_plot))
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")

    # select the amount of labels on the X-axis you want visible (default=10)
    # Keeps every nth label
    [
        l.set_visible(False)
        for (i, l) in enumerate(ax.xaxis.get_ticklabels())
        if i % labels_n != 0
    ]

    # set the axis title
    ax.set_title(plot_title)
    # plt.show()

    return ax


def cumulative_complexity_peaks(
    df: pd.DataFrame,
    significant_level_item: float = 0.05,
    significant_level_time: float = 0.05,
):
    r"""Create a complexity resonance data frame based on the product of the distribution uniformity and the fluctuation intensity
    Parameters
    ---------
    df: pandas DataFrame
        A dataframe containing complexity resonance values from multivariate time series data from 1 person.
        Rows should indicate time, columns should indicate the complexity resonance.
    significant_level_item: float (Default=0.05)
        A float indicating the cutoff of when a point in time is significantly different than the rest on an individual item level (i.e.
        is this time point different than all the other time points for this item/ feature).
    significant_level_time: float (Default=0.05)
        A float indicating the cutoff of when a point in time is significantly different than the rest on a timepoint level
        (i.e. is this day different than all the other days).

    Examples
    ---------
    Demonstration of the function using time series data
    >>> ts_df = pd.read_csv("time_series_dataset.csv", index_col=0)
    >>> scaler = MinMaxScaler()
    >>> scaled_ts_df = pd.DataFrame(scaler.fit_transform(ts_df), columns=ts_df.columns.tolist())
    >>> distribution_uniformity_df = pd.DataFrame(distribution_uniformity(scaled_ts_df, win=7, xmin=0, xmax=1, col_first=1, col_last=7))
    >>> distribution_uniformity_df.columns=scaled_ts_df.columns.tolist()
    >>> fluctuation_intensity_df = pd.DataFrame(fluctuation_intensity(scaled_ts_df, win=7, xmin=0, xmax=1, col_first=1, col_last=7))
    >>> fluctuation_intensity_df.columns=scaled_ts_df.columns.tolist()
    >>> complexity_resonance_df = complexity_resonance(distribution_uniformity_df, fluctuation_intensity_df)
    >>> cumulative_complexity_peaks_df, significant_peaks_df = cumulative_complexity_peaks(df=complexity_resonance_df)
    """

    ## Creating CCP data frame
    z_scale_cr_df = pd.DataFrame(
        StandardScaler().fit_transform(df), columns=df.columns.tolist()
    )
    ccp_df = z_scale_cr_df.mask(
        z_scale_cr_df > norm.ppf(1 - significant_level_item), 1
    ).mask(z_scale_cr_df <= norm.ppf(1 - significant_level_item), 0)

    ## Creating significant CCP time points data frame (1 column)
    z_scale_sum_ccp_df = pd.DataFrame(
        StandardScaler().fit_transform(ccp_df.sum(axis=1).values.reshape(-1, 1)),
        columns=["Significant CCPs"],
    )
    sig_peaks_df = z_scale_sum_ccp_df.mask(
        z_scale_sum_ccp_df > norm.ppf(1 - significant_level_time), 1
    ).mask(z_scale_sum_ccp_df <= norm.ppf(1 - significant_level_time), 0)

    ccp_df.columns = df.columns.tolist()
    ccp_df.index = df.index.tolist()

    sig_peaks_df.index = df.index.tolist()

    return ccp_df, sig_peaks_df


def cumulative_complexity_peaks_plot(
    cumulative_complexity_peaks_df: pd.DataFrame,
    significant_peaks_df: pd.DataFrame,
    plot_title: str = "Cumulative Complexity Peaks Plot",
    figsize: tuple = (20, 5),
    height_ratios: list = [1, 3],
    labels_n: int = 10,
):
    r"""Create a cumulative complexity peaks plot based on the cumulative_complexity_peaks_df and the significant_peaks_df
    Parameters
    ---------
    cumulative_complexity_peaks_df: pandas DataFrame
        A dataframe containing cumulative complexity peaks values from multivariate time series data from 1 person.
        Rows should indicate time, columns should indicate the cumulative complexity peaks.
    significant_peaks_df: pandas DataFrame
        A dataframe containing one column of significant complexity peaks values from multivariate time series data from 1 person.
        Rows should indicate time, columns should indicate the significant cumulative complexity peaks.
    plot_title: str
        A string indicating the title to be used at the top of the plot
    figsize: tuple (Default=(20,5))
        The tuple used to specify the size of the plot.
    height_ratios: list (Default=[1,3])
        The tuple used to specify the size of the plot.
    labels_n: int (Default=10)
        An integer indicating the nth value to be taken for the x-axis of the plot. So if the x-axis consists of
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, and the labels_n value is set to 2, then 2, 4, 6, 8, 10, will be shown on the
        x-axis of the plot.

    Examples
    ---------
    Demonstration of the function using time series data
    >>> ts_df = pd.read_csv("datasets/time_series_dataset.csv", index_col=0)
    >>> scaler = MinMaxScaler()
    >>> scaled_ts_df = pd.DataFrame(scaler.fit_transform(ts_df), columns=ts_df.columns.tolist())
    >>> distribution_uniformity_df = pd.DataFrame(distribution_uniformity(scaled_ts_df, win=7, xmin=0, xmax=1, col_first=1, col_last=7))
    >>> distribution_uniformity_df.columns=scaled_ts_df.columns.tolist()
    >>> fluctuation_intensity_df = pd.DataFrame(fluctuation_intensity(scaled_ts_df, win=7, xmin=0, xmax=1, col_first=1, col_last=7))
    >>> fluctuation_intensity_df.columns=scaled_ts_df.columns.tolist()
    >>> complexity_resonance_df = complexity_resonance(distribution_uniformity_df, fluctuation_intensity_df)
    >>> cumulative_complexity_peaks_df, significant_peaks_df = cumulative_complexity_peaks(df=complexity_resonance_df)
    >>> _ = cumulative_complexity_peaks_plot(cumulative_complexity_peaks_df=cumulative_complexity_peaks_df, significant_peaks_df=significant_peaks_df)
    """
    custom_cmap = sns.color_palette(["#FFFFFF", "#000000"])

    fig, axes = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": height_ratios}
    )

    axe = sns.heatmap(significant_peaks_df.T, cmap=custom_cmap, cbar=False, ax=axes[0])
    _ = axe.set_xticks([])
    _ = axe.get_yticklabels()[0].set_rotation(0)

    _ = axe.set_title(plot_title)

    ax = sns.heatmap(
        cumulative_complexity_peaks_df.T, cmap=custom_cmap, cbar=False, ax=axes[1]
    )

    # Show all ticks
    ax.set_xticks(np.arange(0, len(cumulative_complexity_peaks_df)))
    ax.set_yticks(np.arange(0, len(list(cumulative_complexity_peaks_df))))

    # and label them with the respective list entries
    ax.set_xticklabels(list(cumulative_complexity_peaks_df.index))
    ax.set_yticklabels(list(cumulative_complexity_peaks_df))
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")

    # select the amount of labels on the X-axis you want visible (default=10)
    # Keeps every nth label
    [
        l.set_visible(False)
        for (i, l) in enumerate(ax.xaxis.get_ticklabels())
        if i % labels_n != 0
    ]

    return fig, axes
