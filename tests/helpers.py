"""Data and helpers shared by the test files: from helpers import X, Y, Z."""
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

import kx

X = np.linspace(1, 10, 20)
Y = np.sin(X) + 2
Z = np.cos(X) + 2
DF = pd.DataFrame({"t": X, "speed": Y, "accel": Z, "group": np.where(X > 5, "late", "early")})
PASSENGERS = kx.load("passengers")


def _dashed(ax: Axes) -> list:
    return [ln for ln in ax.lines if ln.get_linestyle() == "--"]
