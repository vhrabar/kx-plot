"""Basic kinds: y and z against x."""
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from .._spec import kind
from .._util import COLOUR, categories, colorbar, label, norm

LINES = ("date", "mk")                          # flags of the kinds that draw lines against x


def _mk(flags: set[str]) -> dict[str, str]:
    """mk: a marker on every data point."""
    return {"marker": "o"} if "mk" in flags else {}


@kind("line", flags=LINES)
def line(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """y and z against x. date = x as dates, mk = a marker on every point. kw: ax.plot."""
    ax.plot(x, y, **{"label": label(y, "y"), **_mk(flags), **kw})
    if z is not None:
        ax.plot(x, z, **{"label": label(z, "z"), **_mk(flags), **kw})


@kind("step", flags=LINES)
def step(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """Like line, drawn as steps centred on each x. kw: ax.step."""
    ax.step(x, y, **{"where": "mid", "label": label(y, "y"), **_mk(flags), **kw})
    if z is not None:
        ax.step(x, z, **{"where": "mid", "label": label(z, "z"), **_mk(flags), **kw})


def _check_area(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if "norm" in flags and z is None:
        raise ValueError("flag 'norm' needs two series (y and z) to split 100% between")
    if "mk" in flags and flags & {"stack", "norm"}:
        raise ValueError("flag 'mk' marks the lines of an overlapping area; a stack has no lines")


@kind("area", flags=("stack", "norm", *LINES), check=_check_area)
def area(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """Filled y and z; stack = on top of each other, norm = 100% stack. kw: the fills (stackplot or fill_between)."""
    series = [(np.asarray(a, dtype=float), label(a, n)) for a, n in ((y, "y"), (z, "z")) if a is not None]
    if "norm" in flags:
        total = sum(v for v, _ in series)
        with np.errstate(invalid="ignore", divide="ignore"):
            series = [(100 * v / total, n) for v, n in series]
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(100))
        ax.set_ylim(0, 100)
    if "stack" in flags or "norm" in flags:
        ax.stackplot(x, *(v for v, _ in series), **{"labels": [n for _, n in series], **kw})
    else:
        for v, n in series:
            ln, = ax.plot(x, v, label=n, **_mk(flags))
            ax.fill_between(x, v, **{"color": ln.get_color(), "alpha": 0.3, "linewidth": 0, **kw})


def _check_band(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if z is None:
        raise ValueError("kind 'band' needs z: the half-width of the band (y - z to y + z)")


@kind("band", flags=LINES, check=_check_band)
def band(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike, flags: set[str], **kw: Any) -> None:
    """y line with a y ± z band in the line's colour. kw: the line (ax.plot)."""
    mid, half = np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    ln, = ax.plot(x, mid, **{"label": label(y, "y"), **_mk(flags), **kw})
    ax.fill_between(x, mid - half, mid + half, color=ln.get_color(), alpha=0.25,
                    linewidth=0, label=f"± {label(z, 'z')}")


def _check_scatter(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if flags & set(COLOUR) and (z is None or categories(z) is not None):
        raise ValueError(f"flags {sorted(flags & set(COLOUR))} colour a numeric z; this scatter has none")


@kind("scatter", flags=("date", *COLOUR), check=_check_scatter)
def scatter(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """y against x; z becomes a colorbar (numbers) or a legend (categories).
    Numeric z: nocb = no colorbar, sym = colours symmetric around 0, logc = log colour scale. kw: ax.scatter."""
    size = 0.5 * plt.rcParams["lines.markersize"] ** 2
    cats = None if z is None else categories(z)
    if cats is None:
        sc = ax.scatter(x, y, **{"c": z, "s": size, "norm": None if z is None else norm(z, flags), **kw})
        if z is not None:
            colorbar(ax, sc, label(z, "z"), flags)
    else:
        xs, ys, zs = np.asarray(x), np.asarray(y), np.asarray(z, dtype=object)
        for c in cats:
            m = zs == c
            ax.scatter(xs[m], ys[m], **{"s": size, "label": str(c), **kw})
        ax.legend(title=label(z, "z"))


@kind("bar", flags=("stack", "sort", "ann"))
def bar(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str],
        horizontal: bool = False, **kw: Any) -> None:
    """y and z grouped per category, x = category labels; stack = z on top of y.
    sort = largest first (by y, or by the total with stack), ann = the value on each bar.
    kw: ax.bar (ax.barh); width= (height= for barh) is the space a category's bars fill, 0.8 by default."""
    series = [(label(a, n), np.asarray(a, dtype=float)) for a, n in ((y, "y"), (z, "z")) if a is not None]
    _bars(ax, np.asarray(x, dtype=object), series, flags, horizontal, **kw)


def _bars(ax: Axes, cats: np.ndarray, series: list[tuple[str, np.ndarray]], flags: set[str], horizontal: bool,
          fmt: str = "{:.3g}", **kw: Any) -> None:
    """One bar per category for each series, side by side (or stacked), with the category labels as ticks.
    sort = largest first (by the first series, or by the total with stack), ann = fmt on each bar."""
    w = kw.pop("height" if horizontal else "width", 0.8)
    stack = "stack" in flags and len(series) > 1
    if "sort" in flags:
        order = np.argsort(-(sum(v for _, v in series) if stack else series[0][1]), kind="stable")
        cats, series = cats[order], [(n, v[order]) for n, v in series]
    idx = np.arange(len(cats))
    draw = ax.barh if horizontal else ax.bar
    bottom = None
    for i, (name, v) in enumerate(series):
        if stack:                               # each series starts where the last ended: the end is the total
            draw(idx, v, w, bottom, **{"label": name, **kw})
            bottom = v if bottom is None else bottom + v
        else:                                   # side by side, together filling w
            draw(idx + (i + 0.5 - len(series) / 2) * w / len(series), v, w / len(series), **{"label": name, **kw})
    (ax.set_yticks if horizontal else ax.set_xticks)(idx, [str(c) for c in cats])
    if "ann" in flags:                          # stacked segments: value inside; otherwise past the end
        for c in ax.containers:
            ax.bar_label(c, fmt=fmt, label_type="center" if stack else "edge", padding=0 if stack else 2)


def _levels(v: ArrayLike) -> tuple[list[str], list[np.ndarray]]:
    """Labels and masks of v's categories in order, with missing values as a last 'NaN' category."""
    arr = np.asarray(v, dtype=object).ravel()
    missing = pd.isna(arr)
    cats = categories(v)
    if cats is None:
        cats = sorted(pd.unique(arr[~missing]))
    labels, masks = [str(c) for c in cats], [(arr == c) & ~missing for c in cats]
    if missing.any():
        labels.append("NaN")
        masks.append(missing)
    return labels, masks


def _check_count(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if y is not None:
        raise ValueError("kind 'count' counts the categories of x; split them with by=, not y")


@kind("count", flags=("stack", "sort", "ann", "norm"), needs_y=False, takes_by=True, check=_check_count)
def count(ax: Axes, x: ArrayLike, y: None, z: None, flags: set[str], by: ArrayLike | None = None,
          **kw: Any) -> None:
    """How often each category of x occurs, one bar per by= group; missing values count as 'NaN'.
    stack = groups on top of each other, sort = most common first, ann = the count on each bar,
    norm = shares instead of counts (within each category with by= and stack, else within each group).
    kw: ax.bar; width= is the space a category's bars fill, 0.8 by default."""
    cats, xs = _levels(x)
    groups, gs = _levels(by) if by is not None else (["count"], [np.ones(np.size(x), dtype=bool)])
    counts = np.array([[np.sum(xm & gm) for xm in xs] for gm in gs], dtype=float)    # groups x categories
    if "norm" in flags:
        with np.errstate(invalid="ignore", divide="ignore"):
            counts = counts / (counts.sum(axis=0) if by is not None and "stack" in flags
                               else counts.sum(axis=1, keepdims=True))
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    _bars(ax, np.array(cats, dtype=object), list(zip(groups, counts, strict=True)), flags, False,
          fmt="{:.0%}" if "norm" in flags else "{:g}", **kw)
    ax.set_xlabel(label(x, ""))
    ax.set_ylabel("share" if "norm" in flags else "count")
    if by is not None:                          # a 100% stack has no free corner: legend outside, as area-norm
        full = "norm" in flags and "stack" in flags
        ax.legend(title=label(by, "by"), **({"loc": "center left", "bbox_to_anchor": (1, 0.5)} if full else {}))


@kind("barh", flags=("stack", "sort", "ann"), value_axis="x")
def barh(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """Like bar, lying down: categories (x) top to bottom, bar lengths along the x axis."""
    bar(ax, x, y, z, flags, horizontal=True, **kw)
    ax.invert_yaxis()                           # first category on top, read like a table
