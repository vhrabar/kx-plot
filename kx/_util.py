"""Small helpers shared by the kinds."""
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
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


def colorbar(ax: Axes, mappable: ScalarMappable, label: str, flags: set[str]) -> None:
    """A colorbar for mappable next to ax, unless nocb."""
    if "nocb" not in flags:
        ax.figure.colorbar(mappable, ax=ax, label=label)
