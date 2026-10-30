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
from matplotlib.cm import ScalarMappable
from matplotlib.layout_engine import ConstrainedLayoutEngine
from numpy.typing import ArrayLike

from . import kinds  # noqa: F401  (registers every kind)
from ._spec import KINDS, Spec, parse
from ._theme import style_path
from ._util import categories, label, split
from .kinds.basic import _levels

OUTSIDE = {"loc": "center left", "bbox_to_anchor": (1, 0.5)}       # legend right of the axes
WRAP = 12                                                           # wrap: characters per tick label line


def plot(*data: ArrayLike | pd.DataFrame | str, spec: str | Spec | None = None, title: str | None = None,
         ax: Axes | None = None, by: ArrayLike | str | None = None, facet: ArrayLike | str | None = None,
         **kwargs: Any) -> Axes | np.ndarray:
    """kx.plot(x, y, z, 'scatter-dark-grid-leg')   or   kx.plot(x, y, 'bar-light')
    kx.plot(df, 'month', 'y2025', 'y2026', 'bar')   names columns of a DataFrame.
    kx.plot(df, 'heat')   a table kind takes the whole DataFrame (or a 2-D array) instead.

    Up to three arrays x, y, z, then the spec (or spec=). by= gives a group label per x value,
    one curve, box or violin per group: kx.plot(df, 'value', spec='kde', by='group').
    Other keywords go to the kind's matplotlib call and win over kx's defaults:
    kx.plot(x, y, 'line', linewidth=3), kx.plot(x, spec='hist', bins=50). kx.grammar(kind) says which call.

    The facet flag draws the plot once per value of facet= (a column or one label per row), side by side with
    shared axes and one colour scale: kx.plot(df, 'pclass', 'port', 'crosstab-facet', facet='sex').

    A theme in the spec applies to this plot only; kx.use() sets the notebook default.
    Returns the axes; a kind with several panels (or facet) returns them as a flat array, in ax's place when ax=
    is given.
    """
    x, y, z, spec, by, facet = _arrays(data, spec, by, facet)
    name, theme, flags = parse(spec or "line")
    kind = KINDS[name]
    if kind.takes_table and not (isinstance(x, pd.DataFrame) and y is None and (by is None or kind.takes_by)):
        raise TypeError(f"kind {name!r} takes a whole table and no columns, y, z"
                        f"{'' if kind.takes_by else ' or by='}: kx.plot(df, {name!r}), "
                        f"or kx.plot(df[cols], {name!r}) for some columns")
    if kind.needs_y and y is None:
        raise ValueError(f"kind {name!r} needs both x and y")
    if by is not None and not kind.takes_by:
        raise ValueError(f"by= only works with kind {sorted(k.name for k in KINDS.values() if k.takes_by)}, "
                         f"not {name!r}")
    if "facet" in flags and facet is None:
        raise ValueError("flag 'facet' needs facet=: a column name or one label per row, one panel per value")
    if facet is not None and "facet" not in flags:
        raise ValueError(f"facet= splits the plot into panels; add the flag: {name}-facet")
    if "facet" in flags and kind.panels:
        raise ValueError(f"kind {name!r} draws several panels already; facet splits a single plot")
    if kind.check:
        kind.check(y, z, flags)
    if "date" in flags:                         # strings and numbers become dates; a Series keeps its name
        x = pd.to_datetime(x)

    style = plt.style.context(["default", style_path(theme)]) if theme else contextlib.nullcontext()
    with style:
        ax = ax or plt.subplots(subplot_kw={"projection": kind.projection})[1]
        if kind.projection:
            ax = _projected(ax, kind.projection)
        fig, alone = ax.figure, len(ax.figure.axes) == 1
        if "facet" in flags:
            return _facet(ax, kind, x, y, z, by, facet, flags, title, alone, **kwargs)
        out = kind.draw(ax, x, y, z, flags, **({"by": by} if by is not None else {}), **kwargs)
        axs = _panels(name, out) if kind.panels else [ax]
        for a in axs:
            _common(a, name, kind.value_axis, flags)
        if title and kind.panels and alone:
            fig.suptitle(title)                 # the panels fill the figure: one title over them all
        elif title:
            axs[0].set_title(title)
        if "tight" in flags and not isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine):
            fig.tight_layout()                  # a constrained figure (kx.grid) is laid out already
    return axs if kind.panels else ax


def _projected(ax: Axes, projection: str) -> Axes:
    """ax, or new axes with that projection in its place (ax is removed): radar on a cell of kx.grid."""
    if ax.name == projection:
        return ax
    fig, spec, box = ax.figure, ax.get_subplotspec(), ax.get_position()
    ax.remove()
    return fig.add_subplot(spec, projection=projection) if spec else fig.add_axes(box, projection=projection)


def _panels(name: str, out: object) -> np.ndarray:
    """The axes a kind with several panels returned, as a flat array."""
    axs = np.asarray(out if out is not None else [], dtype=object).ravel()
    if not len(axs) or not all(isinstance(a, Axes) for a in axs):
        raise TypeError(f"kind {name!r} draws several panels and must return their axes, got {out!r}")
    return axs


def _arrays(data: tuple, spec: str | Spec | None, by: ArrayLike | str | None,
            facet: ArrayLike | str | None = None) -> tuple:
    """(x, y, z, spec, by, facet) from kx.plot's positional arguments: arrays, or a DataFrame and column names.
    by= and facet= may name columns; a table kind's table then leaves them out."""
    data = list(data)
    df = data.pop(0) if data and isinstance(data[0], pd.DataFrame) else None
    last = data[-1] if data else None
    if isinstance(last, Spec) or (isinstance(last, str) and (df is None or last not in df.columns
                                                             or (spec is None and _table_spec(last)))):
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
    if table and df is None and len(data) == 1 and _history(data[0]) is not None:
        df, data = pd.DataFrame(_history(data[0])), []    # a dict, or a Keras History: kx.plot(model.fit(...), 'loss')
    elif table and df is None and len(data) == 1 and np.ndim(data[0]) == 2:
        df, data = pd.DataFrame(data[0]), []    # a 2-D array is a table with 0..n labels
    if df is not None:
        if not data and table:                  # by= and facet= name columns: taken out of the table
            named = [c for c in (by, facet) if isinstance(c, str)]
            by = _column(df, by) if isinstance(by, str) else by
            facet = _column(df, facet) if isinstance(facet, str) else facet
            return (df.drop(columns=named) if named else df), None, None, spec, by, facet
        if not data:
            tables = ", ".join(sorted(k.name for k in KINDS.values() if k.takes_table))
            raise TypeError("kx.plot(df, ...) needs the column names to plot: kx.plot(df, 'x', 'y')"
                            + (f", or a table kind: {tables}" if tables else ""))
        data = [_column(df, c) for c in data]
        by = _column(df, by) if isinstance(by, str) else by
        facet = _column(df, facet) if isinstance(facet, str) else facet
    else:
        for key, v in (("by", by), ("facet", facet)):
            if isinstance(v, str):
                raise TypeError(f"{key}={v!r} names a column, which needs a DataFrame first: "
                                f"kx.plot(df, 'x', {key}={v!r})")
    if not 1 <= len(data) <= 3:
        raise TypeError(f"kx.plot takes x, y and z: up to 3 arrays, got {len(data)}")
    return *data, *[None] * (3 - len(data)), spec, by, facet


def _table_spec(text: str) -> bool:
    """text is a spec for a table kind. Table kinds take no columns, so a column named 'loss' (as in every Keras
    history) does not hide kx.plot(h, 'loss')."""
    try:
        return KINDS[parse(text).kind].takes_table
    except ValueError:
        return False


def _history(obj: object) -> dict | None:
    """A dict of columns: obj itself, or its .history (a Keras History), so Keras need not be imported."""
    if isinstance(obj, dict):
        return obj
    history = getattr(obj, "history", None)
    return history if isinstance(history, dict) else None


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


def _full_categories(v: Any) -> Any:
    """Text as a categorical with every category of the whole column, so each facet panel shows the same
    categories in the same places and colours, even where one of them is missing. Numbers, bools, tables pass."""
    if v is None or isinstance(v, pd.DataFrame) or np.ndim(v) != 1:
        return v
    s = v if isinstance(v, pd.Series) else pd.Series(np.asarray(v, dtype=object))
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return v
    cats = categories(s)
    if cats is None or s.dropna().map(type).isin([bool]).all():
        return v
    return pd.Series(pd.Categorical(s.to_numpy(dtype=object), categories=cats), name=getattr(v, "name", None))


def _rows(v: Any, mask: np.ndarray) -> Any:
    """The rows of v where mask is True, keeping a Series' name or a table's columns."""
    if v is None:
        return None
    if isinstance(v, (pd.Series, pd.DataFrame)):
        return v[mask]
    return np.asarray(v)[mask]


def _axis_key(axis: mpl.axis.Axis) -> tuple | None:
    """What an axis shows: None for numbers (any panels can share it), the tick labels for categories."""
    if isinstance(axis.get_major_locator(), mpl.ticker.FixedLocator):
        return tuple(t.get_text() for t in axis.get_majorticklabels())
    if isinstance(axis.get_major_formatter(), StrCategoryFormatter):
        return tuple(axis.units._mapping) if axis.units is not None else ()
    return None


def _share(axs: list[Axes], which: str) -> None:
    """One range for an axis across the panels and the panels linked, when they all show the same thing."""
    axes = [getattr(a, f"{which}axis") for a in axs]
    if len({_axis_key(axis) for axis in axes}) > 1:
        return                                  # different categories: sharing would mislabel them
    get, put = f"get_{which}lim", f"set_{which}lim"
    limits = [getattr(a, get)() for a in axs]
    lo, hi = min(min(lim) for lim in limits), max(max(lim) for lim in limits)
    flipped = limits[0][0] > limits[0][1]       # barh and images run top to bottom
    for a in axs:
        getattr(a, put)((hi, lo) if flipped else (lo, hi))
        if a is not axs[0]:
            getattr(a, f"share{which}")(axs[0])


def _mappable(ax: Axes) -> ScalarMappable | None:
    """The panel's colour-scaled artist (an image, or points or hexagons coloured by a value), if any."""
    for m in [*ax.images, *ax.collections]:
        if m.get_array() is not None and getattr(m, "norm", None) is not None:
            return m
    return None


def _one_colour_scale(axs: list[Axes]) -> None:
    """The colour kinds' panels on one scale, so a colour means the same value in each, and their colorbars
    replaced by one beside them all."""
    found = [_mappable(a) for a in axs]
    if any(m is None for m in found):
        return
    for m in found:
        m.autoscale_None()
    lo = min(m.norm.vmin for m in found)
    hi = max(m.norm.vmax for m in found)
    for m in found:
        m.set_clim(lo, hi)
    bars = [m.colorbar for m in found if m.colorbar is not None]
    if not bars:
        return
    name, ticks = bars[0].ax.get_ylabel(), bars[0].formatter
    for bar in bars:
        bar.remove()
    axs[0].figure.colorbar(found[0], ax=list(axs), label=name).formatter = ticks


def _facet(ax: Axes, kind: Any, x: Any, y: Any, z: Any, by: Any, facet: Any, flags: frozenset[str],
           title: str | None, alone: bool, **kwargs: Any) -> np.ndarray:
    """The kind drawn once per value of facet, in ax's place: up to 4 in a row, else 3 per row."""
    groups, masks = _levels(facet)
    n = len(x) if isinstance(x, pd.DataFrame) else np.size(x) if np.ndim(x) == 1 else len(np.asarray(x))
    if np.size(facet) != n:
        raise ValueError(f"facet= has {np.size(facet)} values but the data has {n} rows")
    x, y, z, by = (_full_categories(v) for v in (x, y, z, by))
    cols = len(groups) if len(groups) <= 4 else 3
    rows = -(-len(groups) // cols)
    fig = ax.figure
    if alone:                                   # a figure of its own: each panel the size of a single plot
        w, h = plt.rcParams["figure.figsize"]
        fig.set_size_inches(cols * w, rows * h)
        if not isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine):
            fig.set_layout_engine("constrained")    # room for the titles and one colorbar for all
    axs = list(split(ax, rows, cols, projection=kind.projection))
    for a in axs[len(groups):]:
        a.remove()
    axs = axs[:len(groups)]
    flags = flags - {"facet"}
    for a, g, m in zip(axs, groups, masks, strict=True):
        m = np.asarray(m)
        kind.draw(a, *(_rows(v, m) for v in (x, y, z)), flags,
                  **({"by": _rows(by, m)} if by is not None else {}), **kwargs)
        _common(a, kind.name, kind.value_axis, flags)
        a.set_title(f"{label(facet, 'facet')} = {g}")
    if not kind.projection:
        _share(axs, "x")
        _share(axs, "y")
    _one_colour_scale(axs)
    if title and alone:
        fig.suptitle(title)
    elif title:
        axs[0].set_title(f"{title}\n{axs[0].get_title()}")
    if "tight" in flags and not isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine):
        fig.tight_layout()
    return np.array(axs)
