"""Small helpers shared by the kinds."""
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from numpy.typing import ArrayLike

COLOUR = ("nocb", "sym", "logc")               # flags for numeric values on a colour scale


def label(arr: ArrayLike, default: str) -> str:
    """Use a pandas Series name as legend label when there is one."""
    name = getattr(arr, "name", None)
    return str(name) if name is not None else default


def categories(z: ArrayLike) -> list[Any] | None:
    """Category values of z if it is text, bool or categorical; None if z is numeric."""
    s = z if isinstance(z, pd.Series) else pd.Series(np.asarray(z))
    if isinstance(s.dtype, pd.CategoricalDtype):
        return list(s.cat.categories)
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return None
    return sorted(s.dropna().unique(), key=str)


def norm(values: ArrayLike, flags: set[str]) -> mpl.colors.Normalize | None:
    """sym = limits ±max|z| so 0 sits mid-colormap, logc = log colour scale. None = matplotlib's linear default."""
    v = np.asarray(values, dtype=float)
    if "sym" in flags:
        top = np.nanmax(np.abs(v))
        return mpl.colors.Normalize(-top, top)
    if "logc" in flags:
        if np.nanmin(v) <= 0:
            raise ValueError("flag 'logc' needs z > 0; use 'sym' for values around 0")
        return mpl.colors.LogNorm()
    return None


def colorbar(ax: Axes, mappable: ScalarMappable, label: str, flags: set[str]) -> Colorbar | None:
    """A colorbar for mappable next to ax, unless nocb."""
    return None if "nocb" in flags else ax.figure.colorbar(mappable, ax=ax, label=label)


def diagonal(ax: Axes, x: ArrayLike, y: ArrayLike, flags: set[str]) -> None:
    """diag: a dashed y = x line across the range x and y share, from the lower minimum to the higher maximum."""
    xs, ys = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    lo = min(np.nanmin(xs), np.nanmin(ys))
    hi = max(np.nanmax(xs), np.nanmax(ys))
    logs = flags & {"logx", "logy", "logxy"}
    ends = np.geomspace(lo, hi, 100) if logs and lo > 0 else np.linspace(lo, hi, 100)   # straight on any scale
    ax.plot(ends, ends, color=mpl.rcParams["text.color"], linestyle="--", linewidth=1, marker="", alpha=0.6,
            zorder=3)
