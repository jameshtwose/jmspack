"""Submodule utils.py includes the following functions and classes: <br>
- JmsColors: a class containing useful colours according to Jms and functions to show these colors in various forms. <br>
- apply_scaling(): a utility function to be used in conjunction with pandas pipe() to scale columns of a data frame seperately. <br>
- flatten(): a utility function used to flatten a list of lists to a single list. <br>
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Callable, Dict, Union

class JmsColors:
    r"""Utility class for James Twose's color codes.
    Available Functions
    --------
    get_names(): returns a list of the color names e.g. [PURPLE, DARKBLUE, etc.]
    to_dict(): returns a dictionary of format {color name: hexcode}
    to_list(): returns a list of hexcodes
    plot_colors(): returns a lineplot of all the available colours (like a color swatch)

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from jmspack.utils import JmsColors
    >>> x = np.linspace(0, 10, 100)
    >>> fig = plt.figure()
    >>> _ = plt.plot(x, np.sin(x), color=JmsColors.ORANGE)
    >>> _ = plt.plot(x, np.cos(x), color=JmsColors.LIGHTBLUE)
    >>> # plt.show()
    """

    PURPLE = "#8f0fd4"
    DARKBLUE = "#0072e8"
    BLUEGREEN = "#009cdc"
    GREENBLUE = "#00c7b1"
    GREENYELLOW = "#71db5c"
    YELLOW = "#fcdd14"

    DARKGREY = "#282d32"
    MEDIUMGREY = "#808080"
    LIGHTGREY = "#b1b1b1"
    OFFWHITE = "#d5d5d5"

    @staticmethod
    def get_names():
        return [
            k
            for k in JmsColors.__dict__.keys()
            if not k.startswith("__") and not callable(getattr(JmsColors, k))
        ]

    @staticmethod
    def to_dict():
        return {
            k: v
            for k, v in JmsColors.__dict__.items()
            if not k.startswith("__") and not callable(getattr(JmsColors, k))
        }

    @staticmethod
    def to_list():
        return [
            v
            for k, v in JmsColors.__dict__.items()
            if not k.startswith("__") and not callable(getattr(JmsColors, k))
        ]

    @staticmethod
    def plot_colors():
        for i, c in enumerate(JmsColors.to_list()):
            _ = plt.title("Available Jms Colors")
            _ = plt.plot([1, 5], [i, i], color=c, linewidth=5)


def apply_scaling(df: pd.DataFrame,
                  method: Union[Callable, str] = "MinMax",
                  kwargs: Dict = {}):

    r"""Utility function to be used in conjunction with pandas pipe()
    to scale columns of a data frame seperately.

    Examples
    --------
    >>> import seaborn as sns
    >>> import pandas as pd
    >>> df = sns.load_dataset("iris")
    >>> scaled_df = (df
    ...             .select_dtypes("number")
    ...             .pipe(apply_scaling)
    ...             )
    """

    if method == "MinMax":
        scal_df = pd.DataFrame(MinMaxScaler(**kwargs).fit_transform(df),
             index = df.index,
            columns = df.columns)
    elif method == "Standard":
        scal_df = pd.DataFrame(StandardScaler(**kwargs).fit_transform(df),
             index = df.index,
            columns = df.columns)
    else:
        scal_df = pd.DataFrame(method(**kwargs).fit_transform(df),
             index = df.index,
            columns = df.columns)
    return scal_df


def flatten(l):
    r"""Utility function used to flatten a list of list into a single list.

    Examples
    --------
    >>>
    >>>

    """
    return [item for sublist in l for item in sublist]
