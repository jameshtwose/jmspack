r"""Submodule networks.py includes the following functions and classes:

- **apply_scaling():** a utility function to be used in conjunction with pandas pipe() to scale columns of a data frame seperately.

"""
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx

from .utils import JmsColors


def network_plot(
    data,
    feature_column_1="feature1",
    feature_column_2="feature2",
    figsize=(15, 10),
    edge_drop_cutoff=0.4,
    edge_feature: Union[None, str] = "r-value",
    edge_color_cutoff=0.5,
    layout="kamada_kawai",
    node_color=JmsColors.LIGHTGREY,
    node_shape="o",
    node_size=500,
    node_font_size=12,
    edge_positive_color=JmsColors.PURPLE,
    edge_negative_color=JmsColors.GREENYELLOW,
    edge_positive_width=1,
    edge_negative_width=1,
    edge_positive_alpha=1,
    edge_negative_alpha=1,
    edge_positive_style="solid",
    edge_negative_style="dashed",
    random_state=42,
    plot_title="Correlations - edges are r-values",
):
    """Plot a network graph of the data.

    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe containing the data to be plotted.
    feature_column_1: str
        Name of the column containing the first feature.
    feature_column_2: str
        Name of the column containing the second feature.
    figsize: tuple
        Size of the figure to be plotted.
    edge_drop_cutoff: float
        Cutoff value for dropping edges from the plot.
    edge_feature: str
        Name of the column containing the edge feature.
    edge_color_cutoff: float
        Cutoff value for coloring edges.
    layout: str
        Layout of the network graph.
    node_color: str
        Color of the nodes.
    node_shape: str
        Shape of the nodes.
    node_size: int
        Size of the nodes.
    node_font_size: int
        Font size of the node labels.
    edge_positive_color: str
        Color of the positive edges.
    edge_negative_color: str
        Color of the negative edges.
    edge_positive_width: int
        Width of the positive edges.
    edge_negative_width: int
        Width of the negative edges.
    edge_positive_alpha: float
        Alpha value of the positive edges.
    edge_negative_alpha: float
        Alpha value of the negative edges.
    edge_positive_style: str
        Style of the positive edges.
    edge_negative_style: str
        Style of the negative edges.
    random_state: int
        Random state for the layout function.
    plot_title: str
        Title of the plot.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure object of the plot.
    ax: matplotlib.axes._subplots.AxesSubplot
        Axes object of the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from jmspack.networks import network_plot
    >>> np.random.seed(42)
    >>> data = pd.DataFrame({"feature1": np.random.rand(10),
    ...                     "feature2": np.random.rand(10),
    ...                     "r-value": np.random.rand(10)}).round(3)
    >>> fig, ax = network_plot(data, figsize=(7,7))
    """

    if edge_feature:
        plot_df = data[data[edge_feature].abs() > edge_drop_cutoff]
    else:
        plot_df = data

    layout_fn_dict = {
        "spring": nx.spring_layout,
        "random": nx.random_layout,
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "shell": nx.shell_layout,
        "spiral": nx.spiral_layout,
    }

    layout_fn = layout_fn_dict[layout]

    fig, ax = plt.subplots(figsize=figsize)
    G = nx.from_pandas_edgelist(
        df=plot_df,
        source=feature_column_1,
        target=feature_column_2,
        edge_attr=edge_feature,
    )

    elarge = [
        (u, v)
        for (u, v, d) in G.edges(data=True)
        if d[edge_feature] > edge_color_cutoff
    ]
    esmall = [
        (u, v)
        for (u, v, d) in G.edges(data=True)
        if d[edge_feature] <= edge_color_cutoff
    ]

    if layout_fn.__name__ == "spring_layout" or layout_fn.__name__ == "random_layout":
        pos = layout_fn(G, seed=random_state)
    else:
        pos = layout_fn(G)

    # nodes
    nx.draw_networkx_nodes(
        G, pos, node_size=node_size, node_color=node_color, node_shape=node_shape
    )

    # edges
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=elarge,
        width=edge_positive_width,
        alpha=edge_positive_alpha,
        edge_color=edge_positive_color,
        style=edge_positive_style,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=esmall,
        width=edge_negative_width,
        alpha=edge_negative_alpha,
        edge_color=edge_negative_color,
        style=edge_negative_style,
    )

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=node_font_size)
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, edge_feature)
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax.margins(0.0)
    plt.axis("off")
    plt.title(plot_title)
    fig.tight_layout()
    return fig, ax
