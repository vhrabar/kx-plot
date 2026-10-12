"""kx.grid: a figure of panels as one flat array of axes."""
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def grid(n: int, cols: int = 3, **kw: Any) -> np.ndarray:
    """A figure with n panels, `cols` per row, as a flat array of n axes:  for ax in kx.grid(5): ...

    Each panel gets the theme's figure.figsize, so panels look like single plots. Cells past n in the
    last row are removed. Other keywords go to plt.subplots (sharex=True, figsize=..., layout=...).
    The figure is axs[0].figure.
    """
    if n < 1 or cols < 1:
        raise ValueError(f"kx.grid needs n >= 1 and cols >= 1, got n={n}, cols={cols}")
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    w, h = plt.rcParams["figure.figsize"]
    kw = {"figsize": (cols * w, rows * h), "layout": "constrained", **kw}   # constrained: room for colorbars
    fig, axs = plt.subplots(rows, cols, squeeze=False, **kw)
    for ax in axs.flat[n:]:
        fig.delaxes(ax)
    return axs.flat[:n]
