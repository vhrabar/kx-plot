"""
Table kinds: a whole DataFrame (or 2-D array) as the data.
"""
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from .._spec import kind
from .._util import COLOUR, colorbar, label, norm


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
    corr = {"cmap": "rdbu", "norm": mpl.colors.Normalize(-1, 1)}  # the scale of r, whatever this data's range
    style = corr if "corr" in flags else {"norm": norm(frame.to_numpy(dtype=float), flags)}
    _matrix(ax, frame, flags, "r" if "corr" in flags else "", **{**style, **kw})


def _check_crosstab(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if z is not None:
        raise ValueError("kind 'crosstab' counts pairs of x and y; it takes no z")


@kind("crosstab", flags=("norm", "ann", "nocb", "logc"), check=_check_crosstab)
def crosstab(ax: Axes, x: ArrayLike, y: ArrayLike, z: None, flags: set[str], **kw: Any) -> None:
    """How often each pair of categories occurs: x along x, y down y, like pd.crosstab(y, x).
    norm = shares within each x category (each column sums to 100%), ann = the value in each cell,
    nocb = no colorbar, logc = log colour scale (empty cells stay blank). kw: ax.imshow."""
    counts = pd.crosstab(pd.Series(np.asarray(y, dtype=object), name=label(y, "y")),
                         pd.Series(np.asarray(x, dtype=object), name=label(x, "x")))
    if "norm" in flags:
        counts = counts / counts.sum()
    if "logc" in flags:                         # log(0) has no colour: leave those cells empty
        counts = counts.where(counts > 0)
    share = "norm" in flags
    _matrix(ax, counts, flags, "share" if share else "count", "{:.0%}" if share else "{:g}",
            **{"norm": norm(counts.to_numpy(dtype=float), flags), "aspect": "auto", **kw})


def _matrix(ax: Axes, frame: pd.DataFrame, flags: set[str], cblabel: str, fmt: str = "{:.2g}", **kw: Any) -> None:
    """frame as an image: column labels along x, index labels down y, a colorbar, ann = fmt in each cell."""
    values = frame.to_numpy(dtype=float)
    im = ax.imshow(values, **kw)
    ax.set_xticks(range(values.shape[1]), [str(c) for c in frame.columns])
    ax.set_yticks(range(values.shape[0]), [str(i) for i in frame.index])
    if frame.columns.name is not None: ax.set_xlabel(str(frame.columns.name))
    if frame.index.name is not None:   ax.set_ylabel(str(frame.index.name))
    cb = colorbar(ax, im, cblabel, flags)
    if cb is not None and "%" in fmt:
        cb.formatter = mpl.ticker.PercentFormatter(1, decimals=0)
    if "ann" in flags:
        for (i, j), v in np.ndenumerate(values):
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", color=_ink(im.cmap(float(im.norm(v)))))
