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
from .._util import COLOUR, colorbar, diagonal, label, norm
from .basic import _bars


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
    _matrix(ax, frame, flags, "r" if "corr" in flags else "", "{:.2f}" if "corr" in flags else "{:.3g}",
            **{**style, **kw})


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


@kind("missing", flags=("ann", "all"), table=True, value_axis="x")
def missing(ax: Axes, x: pd.DataFrame, y: None, z: None, flags: set[str], **kw: Any) -> None:
    """The share of missing values in each column, as bars lying down, the most missing on top.
    Only columns with missing values, unless all; ann = the share on each bar. kw: ax.barh."""
    share = x.isna().mean()
    if "all" not in flags:
        share = share[share > 0]
    if share.empty:                             # nothing to draw is an answer too, not an error
        ax.text(0.5, 0.5, "no missing values", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    _bars(ax, np.array([str(c) for c in share.index], dtype=object), [("missing", share.to_numpy(dtype=float))],
          {"sort", *(flags & {"ann"})}, True, fmt="{:.1%}", **kw)
    ax.invert_yaxis()                           # largest on top
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    ax.set_xlabel("missing")


@kind("nullmap", flags=("sort",), table=True)
def nullmap(ax: Axes, x: pd.DataFrame, y: None, z: None, flags: set[str], **kw: Any) -> None:
    """Where the missing values are: one row per row of the table, columns along x, missing cells in the
    palette's first colour on the background. sort = the rows with the most missing on top. kw: ax.imshow."""
    gaps = x.isna()
    if "sort" in flags:                         # most missing first; ties by pattern, so the same gaps form a block
        g = gaps.to_numpy()
        gaps = gaps.iloc[np.lexsort([*(~g[:, j] for j in reversed(range(g.shape[1]))), -g.sum(axis=1)])]
    colours = mpl.colors.ListedColormap([ax.get_facecolor(), mpl.colors.to_rgba("C0")])
    ax.imshow(gaps.to_numpy(dtype=float), **{"cmap": colours, "vmin": 0, "vmax": 1, "interpolation": "nearest",
                                             "aspect": "auto", **kw})
    ax.set_xticks(range(gaps.shape[1]), [str(c) for c in gaps.columns])
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(10, integer=True))    # row positions, not every row
    ax.set_ylabel("row, most missing first" if "sort" in flags else "row")


def _check_hexbin(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if z is None and "sym" in flags:
        raise ValueError("flag 'sym' centres the colours of a z on 0; counts are never below 0")


@kind("hexbin", flags=("diag", *COLOUR), check=_check_hexbin)
def hexbin(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """How many points fall in each hexagon, or the mean of z there. Rows with a NaN are left out.
    nocb = no colorbar, sym = colours symmetric around 0 (z only), logc = log colour scale, diag = a dashed y = x line.
    kw: ax.hexbin (gridsize= sets the number of hexagons across)."""
    cols = [np.asarray(a, dtype=float).ravel() for a in (x, y, z) if a is not None]
    keep = np.logical_and.reduce([np.isfinite(c) for c in cols])
    xs, ys, *zs = (c[keep] for c in cols)
    if z is None:                               # counts; LogNorm, as bins="log" stores log10 counts before 3.8
        style = {"mincnt": 1, "norm": mpl.colors.LogNorm() if "logc" in flags else None}
    else:                                       # mincnt left out: cells with any point, on every matplotlib
        style = {"C": zs[0], "reduce_C_function": np.mean, "norm": norm(zs[0], flags)}
    hb = ax.hexbin(xs, ys, **{**style, **kw})
    colorbar(ax, hb, "count" if z is None else label(z, "z"), flags)
    if "diag" in flags:
        diagonal(ax, xs, ys, flags)


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
