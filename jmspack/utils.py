import matplotlib.pyplot as plt

class JmsColors:
    """Utility class for James Twose color codes.
    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>>
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
