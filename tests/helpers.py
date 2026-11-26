"""Data and helpers shared by the test files: from helpers import X, Y, Z."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

import kx

X = np.linspace(1, 10, 20)
Y = np.sin(X) + 2
Z = np.cos(X) + 2
DF = pd.DataFrame({"t": X, "speed": Y, "accel": Z, "group": np.where(X > 5, "late", "early")})
PASSENGERS = kx.load("passengers")


def _dashed(ax: Axes) -> list:
    return [ln for ln in ax.lines if ln.get_linestyle() == "--"]


def _nan_free(v: object) -> object:
    """v with every NaN as None, so two drawings with gaps compare equal."""
    if isinstance(v, list):
        return [_nan_free(i) for i in v]
    return None if isinstance(v, float) and v != v else v


def _drawn(axs: object) -> list:
    """What a plot draws, comparable across runs: line and patch geometry, points, images, tick labels, axis
    labels and titles, texts and legends, of the axes given (a plot's return)."""
    out = []
    for ax in np.atleast_1d(axs):
        leg = ax.get_legend()
        out.append([ln.get_xydata().tolist() for ln in ax.lines])
        out.append([p.get_path().vertices.tolist() for p in ax.patches])
        out.append([np.asarray(c.get_offsets()).tolist() for c in ax.collections])
        out.append([np.asarray(im.get_array()).tolist() for im in ax.images])
        out.append([t.get_text() for t in [*ax.get_xticklabels(), *ax.get_yticklabels()]])
        out.append([ax.get_xlabel(), ax.get_ylabel(), ax.get_title()])
        out.append([t.get_text() for t in ax.texts])
        out.append(None if leg is None else [leg.get_title().get_text(), [t.get_text() for t in leg.texts]])
    return _nan_free(out)


def _polars_series(s: pd.Series) -> object:
    pl = pytest.importorskip("polars")
    name = "" if s.name is None else str(s.name)
    if isinstance(s.dtype, pd.CategoricalDtype):
        values = s.astype(object).where(s.notna(), None).tolist()
        return pl.Series(name, values, dtype=pl.Enum([str(c) for c in s.cat.categories]))
    if pd.api.types.is_float_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return pl.Series(name, s.to_numpy(), nan_to_null=True)
    return pl.Series(name, s.astype(object).where(s.notna(), None).tolist())


def _to_polars(obj: object) -> object:
    """A pandas frame or Series as polars: the gaps as nulls, a categorical as an Enum of its categories."""
    pl = pytest.importorskip("polars")
    if isinstance(obj, pd.Series):
        return _polars_series(obj)
    return pl.DataFrame([_polars_series(obj[c].rename(c)) for c in obj.columns])


def _same(obj: object) -> object:
    return obj
