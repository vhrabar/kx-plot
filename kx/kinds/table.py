"""
Table kinds: a whole DataFrame (or 2-D array) as the data.
"""
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .._spec import kind
from .._util import COLOUR, colorbar, norm


def _ink(rgba: tuple[float, ...]) -> str:
    """Black on a light cell, white on a dark one (relative luminance of the cell colour)."""
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3])
    return "black" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.18 else "white"


@kind("heat", flags=("corr", "ann", *COLOUR), table=True)
def heat(ax: Axes, x: pd.DataFrame, y: None, z: None, flags: set[str], **kw: Any) -> None:
    """The table's values as coloured cells, columns along x and index labels down y.
    corr = correlations of the numeric columns (rdbu, ±1), ann = the value in each cell,
    nocb = no colorbar, sym = colours symmetric around 0, logc = log colour scale. kw: ax.imshow."""
    frame = x.corr(numeric_only=True) if "corr" in flags else x
    text = [str(c) for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if text:
        raise ValueError(f"kind 'heat' colours numbers; columns {text} are not. "
                         "Pass df.select_dtypes('number'), or use 'corr'")
    values = frame.to_numpy(dtype=float)
    corr = {"cmap": "rdbu", "norm": mpl.colors.Normalize(-1, 1)}  # the scale of r, whatever this data's range
    style = corr if "corr" in flags else {"norm": norm(values, flags)}
    im = ax.imshow(values, **{**style, **kw})
    ax.set_xticks(range(values.shape[1]), [str(c) for c in frame.columns])
    ax.set_yticks(range(values.shape[0]), [str(i) for i in frame.index])
    if frame.columns.name is not None: ax.set_xlabel(str(frame.columns.name))
    if frame.index.name is not None:   ax.set_ylabel(str(frame.index.name))
    colorbar(ax, im, "r" if "corr" in flags else "", flags)
    if "ann" in flags:
        for (i, j), v in np.ndenumerate(values):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2g}", ha="center", va="center", color=_ink(im.cmap(im.norm(v))))
