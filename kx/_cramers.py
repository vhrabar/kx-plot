"""kx.cramers: how strongly each pair of categorical columns goes together, as a heat map."""
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from . import _stats
from ._plot import plot as draw

MAX_LEVELS = 20


def cramers(df: pd.DataFrame, cols: list[str] | None = None, spec: str = "", cluster: bool = False,
            plot: bool = True, **kw: Any) -> Axes | pd.DataFrame:
    """Cramér's V (bias-corrected, 0 = independent, 1 = one fixes the other) for every pair of categorical
    columns, drawn with heat-ann on 0..1: kx.cramers(df). The columns are cols=, else every column that is not
    a float with at most 20 values (text, bool, small integers such as a class). cluster=True puts the columns
    that go together side by side; the names are turned (rot, tight). plot=False returns the matrix. spec= adds
    a theme and flags; other keywords go to kx.plot (title=, ax=)."""
    if cols is None:
        cols = [c for c in df.columns
                if not pd.api.types.is_float_dtype(df[c]) and df[c].nunique(dropna=True) <= MAX_LEVELS]
    else:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"cols= names {missing}, which the table does not have. Columns: {list(df.columns)}")
    if len(cols) < 2:
        raise ValueError(f"Cramér's V compares pairs of categorical columns; found {list(cols)}. Pass cols=[...]")
    v = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        for b in cols[i:]:
            v.loc[a, b] = v.loc[b, a] = _stats.cramers_v(df[a], df[b])
    if cluster:
        far = (1 - v.fillna(0)).to_numpy(copy=True)
        np.fill_diagonal(far, 0)
        order = _stats.leaves(_stats.linkage(far, distances=True))
        v = v.iloc[order, order]
    if not plot:
        return v
    ax = draw(v, "-".join(["heat", "ann", "rot", "tight", *([spec] if spec else [])]),
              **{"cmap": "blues", "norm": mpl.colors.Normalize(0, 1), "fmt": "{:.2f}", **kw})
    bar = ax.images[0].colorbar
    if bar is not None:
        bar.set_label("Cramér's V")
    return ax
