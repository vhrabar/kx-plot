"""Basic kinds: y and z against x."""
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from .._spec import kind
from .._util import categories, label

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


COLOUR = ("nocb", "sym", "logc")               # flags for a numeric z on a colour scale


def _check_scatter(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if flags & set(COLOUR) and (z is None or categories(z) is not None):
        raise ValueError(f"flags {sorted(flags & set(COLOUR))} colour a numeric z; this scatter has none")


def _norm(z: ArrayLike, flags: set[str]) -> mpl.colors.Normalize | None:
    """sym = limits ±max|z| so 0 sits mid-colormap, logc = log colour scale. None = matplotlib's linear default."""
    v = np.asarray(z, dtype=float)
    if "sym" in flags:
        top = np.nanmax(np.abs(v))
        return mpl.colors.Normalize(-top, top)
    if "logc" in flags:
        if np.nanmin(v) <= 0:
            raise ValueError("flag 'logc' needs z > 0; use 'sym' for values around 0")
        return mpl.colors.LogNorm()
    return None


@kind("scatter", flags=("date", *COLOUR), check=_check_scatter)
def scatter(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """y against x; z becomes a colorbar (numbers) or a legend (categories).
    Numeric z: nocb = no colorbar, sym = colours symmetric around 0, logc = log colour scale. kw: ax.scatter."""
    size = 0.5 * plt.rcParams["lines.markersize"] ** 2
    cats = None if z is None else categories(z)
    if cats is None:
        sc = ax.scatter(x, y, **{"c": z, "s": size, "norm": None if z is None else _norm(z, flags), **kw})
        if z is not None and "nocb" not in flags:
            ax.figure.colorbar(sc, ax=ax, label=label(z, "z"))
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
    w = kw.pop("height" if horizontal else "width", 0.8) / 2
    names = label(y, "y"), None if z is None else label(z, "z")
    x, y = np.asarray(x, dtype=object), np.asarray(y, dtype=float)
    z = None if z is None else np.asarray(z, dtype=float)
    if "sort" in flags:
        key = y + z if z is not None and "stack" in flags else y
        order = np.argsort(-key, kind="stable")
        x, y, z = x[order], y[order], None if z is None else z[order]
    idx = np.arange(len(x))
    draw = ax.barh if horizontal else ax.bar
    if z is None:
        draw(idx, y, 2 * w, **{"label": names[0], **kw})
    elif "stack" in flags:                      # z starts where y ends, so the bar's end is the total
        draw(idx, y, 2 * w, **{"label": names[0], **kw})
        draw(idx, z, 2 * w, y, **{"label": names[1], **kw})
    else:
        draw(idx - w / 2, y, w, **{"label": names[0], **kw})
        draw(idx + w / 2, z, w, **{"label": names[1], **kw})
    (ax.set_yticks if horizontal else ax.set_xticks)(idx, [str(v) for v in x])
    if "ann" in flags:                          # stacked segments: value inside; otherwise past the end
        inside = z is not None and "stack" in flags
        for c in ax.containers:
            ax.bar_label(c, fmt="{:.3g}", label_type="center" if inside else "edge", padding=0 if inside else 2)


@kind("barh", flags=("stack", "sort", "ann"), value_axis="x")
def barh(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """Like bar, lying down: categories (x) top to bottom, bar lengths along the x axis."""
    bar(ax, x, y, z, flags, horizontal=True, **kw)
    ax.invert_yaxis()                           # first category on top, read like a table
