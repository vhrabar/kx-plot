"""kx.plot: the one-liner. Kinds live in kx/kinds/; this file checks input and applies common flags."""
import contextlib
import difflib
import textwrap
from typing import Any

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.category import StrCategoryFormatter
from matplotlib.layout_engine import ConstrainedLayoutEngine
from numpy.typing import ArrayLike

from . import kinds  # noqa: F401  (registers every kind)
from ._spec import KINDS, Spec, parse
from ._theme import style_path

OUTSIDE = {"loc": "center left", "bbox_to_anchor": (1, 0.5)}       # legend right of the axes
WRAP = 12                                                           # wrap: characters per tick label line


def plot(*data: ArrayLike | pd.DataFrame | str, spec: str | Spec | None = None, title: str | None = None,
         ax: Axes | None = None, by: ArrayLike | str | None = None, **kwargs: Any) -> Axes:
    """kx.plot(x, y, z, 'scatter-dark-grid-leg')   or   kx.plot(x, y, 'bar-light')
    kx.plot(df, 'month', 'y2025', 'y2026', 'bar')   names columns of a DataFrame.
    kx.plot(df, 'heat')   a table kind takes the whole DataFrame (or a 2-D array) instead.

    Up to three arrays x, y, z, then the spec (or spec=). by= gives a group label per x value,
    one curve, box or violin per group: kx.plot(df, 'value', spec='kde', by='group').
    Other keywords go to the kind's matplotlib call and win over kx's defaults:
    kx.plot(x, y, 'line', linewidth=3), kx.plot(x, spec='hist', bins=50). kx.grammar(kind) says which call.

    A theme in the spec applies to this plot only; kx.use() sets the notebook default.
    """
    x, y, z, spec, by = _arrays(data, spec, by)
    name, theme, flags = parse(spec or "line")
    kind = KINDS[name]
    if kind.takes_table and not (isinstance(x, pd.DataFrame) and y is None and by is None):
        raise TypeError(f"kind {name!r} takes a whole table and no columns, y, z or by=: kx.plot(df, {name!r}), "
                        f"or kx.plot(df[cols], {name!r}) for some columns")
    if kind.needs_y and y is None:
        raise ValueError(f"kind {name!r} needs both x and y")
    if by is not None and not kind.takes_by:
        raise ValueError(f"by= only works with kind {sorted(k.name for k in KINDS.values() if k.takes_by)}, "
                         f"not {name!r}")
    if kind.check:
        kind.check(y, z, flags)
    if "date" in flags:                         # strings and numbers become dates; a Series keeps its name
        x = pd.to_datetime(x)

    style = plt.style.context(["default", style_path(theme)]) if theme else contextlib.nullcontext()
    with style:
        ax = ax or plt.subplots()[1]
        kind.draw(ax, x, y, z, flags, **({"by": by} if by is not None else {}), **kwargs)
        _common(ax, name, kind.value_axis, flags)
        if title:            ax.set_title(title)
        if "tight" in flags and not isinstance(ax.figure.get_layout_engine(), ConstrainedLayoutEngine):
            ax.figure.tight_layout()            # a constrained figure (kx.grid) is laid out already
    return ax


def _arrays(data: tuple, spec: str | Spec | None, by: ArrayLike | str | None) -> tuple:
    """(x, y, z, spec, by) from kx.plot's positional arguments: arrays, or a DataFrame and column names."""
    data = list(data)
    df = data.pop(0) if data and isinstance(data[0], pd.DataFrame) else None
    last = data[-1] if data else None
    if isinstance(last, Spec) or (isinstance(last, str) and (df is None or last not in df.columns)):
        if spec is not None:                    # a trailing string that is not a column is the spec
            raise TypeError(f"spec given twice ({data[-1]!r} and spec={spec!r})")
        if df is not None and isinstance(last, str):    # not a spec either, but close to a column: a typo
            try:
                parse(data[-1])
            except ValueError:
                if difflib.get_close_matches(data[-1], [str(c) for c in df.columns]):
                    _column(df, data[-1])
                raise
        spec = data.pop()
    table = KINDS[parse(spec or "line").kind].takes_table
    if table and df is None and len(data) == 1 and np.ndim(data[0]) == 2:
        df, data = pd.DataFrame(data[0]), []    # a 2-D array is a table with 0..n labels
    if df is not None:
        if not data and table:
            return df, None, None, spec, by
        if not data:
            tables = ", ".join(sorted(k.name for k in KINDS.values() if k.takes_table))
            raise TypeError("kx.plot(df, ...) needs the column names to plot: kx.plot(df, 'x', 'y')"
                            + (f", or a table kind: {tables}" if tables else ""))
        data = [_column(df, c) for c in data]
        by = _column(df, by) if isinstance(by, str) else by
    elif isinstance(by, str):
        raise TypeError(f"by={by!r} names a column, which needs a DataFrame first: kx.plot(df, 'x', by={by!r})")
    if not 1 <= len(data) <= 3:
        raise TypeError(f"kx.plot takes x, y and z: up to 3 arrays, got {len(data)}")
    return *data, *[None] * (3 - len(data)), spec, by


def _column(df: pd.DataFrame, name: object) -> pd.Series:
    """df[name], or an error naming the columns that are close."""
    if not isinstance(name, str):
        raise TypeError(f"after a DataFrame, give column names, not {type(name).__name__}")
    if name not in df.columns:
        close = difflib.get_close_matches(name, [str(c) for c in df.columns], n=3)
        hint = f" Did you mean {close}?" if close else f" Columns: {list(df.columns)}"
        raise ValueError(f"no column {name!r} in the DataFrame.{hint}")
    return df[name]


def _common(ax: Axes, name: str, value_axis: str, flags: set[str]) -> None:
    """Flags every kind takes: legend, scales, value-axis format, ticks."""
    values = ax.xaxis if value_axis == "x" else ax.yaxis
    if "grid" in flags:  ax.grid(True)
    if flags & {"leg", "legout"} and ax.get_legend_handles_labels()[0]:   # nothing labelled (scatter z): none
        old = ax.get_legend()
        outside = "legout" in flags or (name == "area" and "norm" in flags)  # a 100% stack has no free corner
        if old is None or outside:              # moving a kind's own legend keeps its title
            ax.legend(title=old.get_title().get_text() if old else None, **(OUTSIDE if outside else {}))
    if flags & {"logx", "logxy"}: ax.set_xscale("log")
    if flags & {"logy", "logxy"}: ax.set_yscale("log")
    if "symlogy" in flags:        ax.set_yscale("symlog")
    if "zero" in flags:                         # stretch the value axis to include 0
        lo, hi = values.get_view_interval()
        (ax.set_xlim if value_axis == "x" else ax.set_ylim)(min(lo, 0), max(hi, 0))
    if "eq" in flags:    ax.set_aspect("equal")
    if "pct" in flags and not isinstance(values.get_major_formatter(), mpl.ticker.PercentFormatter):
        values.set_major_formatter(mpl.ticker.PercentFormatter(1))  # 0.25 -> 25%; area-norm is already %
    if "si" in flags:    values.set_major_formatter(mpl.ticker.EngFormatter(sep=""))   # 1500 -> 1.5k
    if "date" in flags:
        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    if "wrap" in flags:
        for axis in (ax.xaxis, ax.yaxis):
            _wrap(axis)
    if "rot" in flags:                          # rotation stays for ticks matplotlib adds later
        ax.tick_params(axis="x", labelrotation=45)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")


def _wrap(axis: mpl.axis.Axis) -> None:
    """Text tick labels broken into lines of WRAP characters at spaces; numbers and dates are left alone."""
    def fill(text: str) -> str:
        return textwrap.fill(text, WRAP, break_long_words=False)
    old = axis.get_major_formatter()
    if isinstance(old, StrCategoryFormatter):               # text plotted directly: matplotlib's categories
        axis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, pos: fill(old(v, pos))))
    elif isinstance(axis.get_major_locator(), mpl.ticker.FixedLocator):    # set_xticks(ticks, labels): stay fixed
        axis.set_ticks(axis.get_majorticklocs(), [fill(t.get_text()) for t in axis.get_majorticklabels()])
