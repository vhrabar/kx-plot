"""Basic kinds: y and z against x."""
import warnings
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.container import BarContainer
from matplotlib.patheffects import withStroke
from numpy.typing import ArrayLike

from .._frame import column, wrap
from .._spec import kind
from .._stats import bootstrap
from .._util import COLOUR, categories, colorbar, diagonal, endlabels, ink, label, norm

LINES = ("date", "mk", "endlab")                # flags of the kinds that draw lines against x


def _mk(flags: set[str]) -> dict[str, str]:
    """mk: a marker on every data point."""
    return {"marker": "o"} if "mk" in flags else {}


def _boot(flags: set[str], n_boot: int | None, seed: int | None) -> tuple[int, int]:
    """n_boot= and seed= for the flag ci, which they only go with."""
    if "ci" not in flags and (n_boot is not None or seed is not None):
        raise ValueError("n_boot= and seed= set the bootstrap of the flag ci; add it: line-ci")
    return (1000 if n_boot is None else n_boot), (0 if seed is None else seed)


def _mean_ci(x: ArrayLike, v: ArrayLike, n_boot: int, seed: int) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """ci: x's distinct values (sorted for numbers, dates and categoricals, in order of appearance for text), the
    mean of v at each, and the bootstrap interval of that mean."""
    s = column(x)
    xs = np.asarray(s.to_numpy())
    missing = np.asarray(pd.isna(xs), dtype=bool)
    present = xs[~missing]
    if isinstance(s.dtype, nw.Enum) or s.dtype == nw.Categorical:    # the categories there are, in their order
        seen = set(present.tolist())
        keys = np.array([c for c in categories(s) if c in seen], dtype=object)
    elif s.dtype.is_numeric() or s.dtype in (nw.Boolean, nw.Date, nw.Datetime, nw.Duration):
        keys = np.unique(present)
    else:
        keys = np.array(list(dict.fromkeys(present.tolist())), dtype=object)
    lookup = {k: i for i, k in enumerate(keys.tolist())}
    codes = np.full(len(xs), -1)
    codes[~missing] = [lookup[k] for k in present.tolist()]
    vs = np.asarray(v, dtype=float).ravel()
    if len(keys) == np.count_nonzero(codes >= 0):
        raise ValueError("ci bootstraps the y values at each x, and each x has one here. "
                         "For a spread you know, use band: kx.plot(x, y, err, 'band')")
    groups = [vs[codes == i] for i in range(len(keys))]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.array([np.nanmean(g) for g in groups])
    lo, hi = np.array([bootstrap(g, n_boot, seed) for g in groups]).T
    return keys, mean, lo, hi


def _lines(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], drawn: bool,
           n_boot: int | None, seed: int | None, **kw: Any) -> None:
    """line and step: y and z against x, as given or (ci) as the mean at each x with its interval as a band."""
    n_boot, seed = _boot(flags, n_boot, seed)
    draw = ax.step if drawn else ax.plot
    where = {"where": "mid"} if drawn else {}
    for v, name in ((y, "y"), (z, "z")):
        if v is None:
            continue
        if "ci" not in flags:
            draw(x, v, **{**where, "label": label(v, name), **_mk(flags), **kw})
            continue
        keys, mean, lo, hi = _mean_ci(x, v, n_boot, seed)
        ln, = draw(keys, mean, **{**where, "label": label(v, name), **_mk(flags), **kw})
        ax.fill_between(keys, lo, hi, color=ln.get_color(), alpha=0.25, linewidth=0,
                        **({"step": "mid"} if drawn else {}))
    if "endlab" in flags:
        endlabels(ax)


@kind("line", flags=(*LINES, "ci"), frames=True)
def line(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], n_boot: int | None = None,
         seed: int | None = None, **kw: Any) -> None:
    """y and z against x. date = x as dates, mk = a marker on every point, endlab = each line's name at its
    right end instead of a legend.
    ci = the mean of y at each x with a bootstrap 95% interval as a band (n_boot= resamples, 1000; seed=, 0).
    kw: ax.plot."""
    _lines(ax, x, y, z, flags, False, n_boot, seed, **kw)


@kind("step", flags=(*LINES, "ci"), frames=True)
def step(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], n_boot: int | None = None,
         seed: int | None = None, **kw: Any) -> None:
    """Like line, drawn as steps centred on each x; ci as for line, the band in steps too. kw: ax.step."""
    _lines(ax, x, y, z, flags, True, n_boot, seed, **kw)


def _check_area(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if "norm" in flags and z is None:
        raise ValueError("flag 'norm' needs two series (y and z) to split 100% between")
    if "mk" in flags and flags & {"stack", "norm"}:
        raise ValueError("flag 'mk' marks the lines of an overlapping area; a stack has no lines")
    if "endlab" in flags and flags & {"stack", "norm"}:
        raise ValueError("flag 'endlab' labels the lines of an overlapping area; a stack ends in bands, not points")


@kind("area", flags=("stack", "norm", *LINES), check=_check_area, frames=True)
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
        if "endlab" in flags:
            endlabels(ax)


def _check_band(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if z is None:
        raise ValueError("kind 'band' needs z: the half-width of the band (y - z to y + z)")


@kind("band", flags=LINES, check=_check_band, frames=True)
def band(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike, flags: set[str], **kw: Any) -> None:
    """y line with a y ± z band in the line's colour. kw: the line (ax.plot)."""
    mid, half = np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    ln, = ax.plot(x, mid, **{"label": label(y, "y"), **_mk(flags), **kw})
    ax.fill_between(x, mid - half, mid + half, color=ln.get_color(), alpha=0.25,
                    linewidth=0, label=f"± {label(z, 'z')}")
    if "endlab" in flags:
        endlabels(ax)


def _check_scatter(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if flags & set(COLOUR) and (z is None or categories(z) is not None):
        raise ValueError(f"flags {sorted(flags & set(COLOUR))} colour a numeric z; this scatter has none")
    if "fit" in flags and (logs := sorted(flags & {"logx", "logy", "logxy", "symlogy"})):
        raise ValueError(f"flag 'fit' draws a straight line, which is a curve on a {logs[0]} axis; "
                         "fit the logs instead: kx.plot(np.log(x), y, 'scatter-fit')")
    if flags & {"fit", "diag"} and "date" in flags:
        raise ValueError(f"flag {sorted(flags & {'fit', 'diag'})[0]!r} needs numbers on x, not dates")


def _fit(ax: Axes, x: ArrayLike, y: ArrayLike, **kw: Any) -> str | None:
    """fit: the least-squares line through the finite points. Returns 'slope …, R² …', None under 2 x values."""
    xs, ys = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    keep = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[keep], ys[keep]
    if len(np.unique(xs)) < 2:
        return None
    slope, icept = np.polyfit(xs, ys, 1)
    total = np.sum((ys - ys.mean()) ** 2)
    r2 = 1 - np.sum((ys - (slope * xs + icept)) ** 2) / total if total else 1.0
    ends = np.array([xs.min(), xs.max()])
    ax.plot(ends, slope * ends + icept, **{"linewidth": 1.5, "marker": "", "zorder": 3, **kw})
    return f"slope {slope:.3g}, R² {r2:.2f}"


@kind("scatter", flags=("date", "diag", "fit", *COLOUR), check=_check_scatter, frames=True)
def scatter(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """y against x; z becomes a colorbar (numbers) or a legend (categories).
    Numeric z: nocb = no colorbar, sym = colours symmetric around 0, logc = log colour scale.
    diag = a dashed y = x line, fit = a least-squares line with its slope and R² in the legend
    (one per group with a categorical z). kw: ax.scatter."""
    size = 0.5 * plt.rcParams["lines.markersize"] ** 2
    cats = None if z is None else categories(z)
    if cats is None:
        sc = ax.scatter(x, y, **{"c": z, "s": size, "norm": None if z is None else norm(z, flags), **kw})
        if z is not None:
            colorbar(ax, sc, label(z, "z"), flags)
        if "fit" in flags:
            text = _fit(ax, x, y, color=plt.rcParams["text.color"])
            if text is None:
                raise ValueError("flag 'fit' needs at least 2 distinct finite x values")
            ax.lines[-1].set_label(text)
            ax.legend()
    else:
        xs, ys, zs = np.asarray(x), np.asarray(y), np.asarray(z, dtype=object)
        for c in cats:
            m = zs == c
            pts = ax.scatter(xs[m], ys[m], **{"s": size, "label": str(c), **kw})
            if "fit" in flags and (text := _fit(ax, xs[m], ys[m], color=pts.get_facecolor()[0])):
                pts.set_label(f"{c} ({text})")  # one legend entry per group; under 2 x values, no line
        ax.legend(title=label(z, "z"))
    if "diag" in flags:
        diagonal(ax, x, y, flags)


@kind("bar", flags=("stack", "sort", "ann", "ci"), frames=True)
def bar(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str],
        horizontal: bool = False, n_boot: int | None = None, seed: int | None = None, **kw: Any) -> None:
    """y and z grouped per category, x = category labels; stack = z on top of y.
    sort = largest first (by y, or by the total with stack), ann = the value on each bar.
    ci = one bar per category at the mean of its rows, with a bootstrap 95% interval as an error bar
    (n_boot= resamples, 1000; seed=, 0); missing x values are a 'NaN' category, as in count.
    kw: ax.bar (ax.barh); width= (height= for barh) is the space a category's bars fill, 0.8 by default."""
    n_boot, seed = _boot(flags, n_boot, seed)
    named = [(label(a, n), a) for a, n in ((y, "y"), (z, "z")) if a is not None]
    if "ci" not in flags:
        series = [(n, np.asarray(a, dtype=float)) for n, a in named]
        _bars(ax, np.asarray(x, dtype=object), series, flags, horizontal, **kw)
        return
    cats, masks = _levels(x)
    series, errs = [], []
    for n, a in named:
        v = np.asarray(a, dtype=float).ravel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a category whose values are all NaN: no bar
            mean = np.array([np.nanmean(v[m]) for m in masks])
        lo, hi = np.array([bootstrap(v[m], n_boot, seed) for m in masks]).T
        series.append((n, mean))
        errs.append(np.array([mean - lo, hi - mean]))
    _bars(ax, np.array(cats, dtype=object), series, flags, horizontal, errs=errs, **kw)


def _bars(ax: Axes, cats: np.ndarray, series: list[tuple[str, np.ndarray]], flags: set[str], horizontal: bool,
          fmt: str = "{:.3g}", errs: list[np.ndarray] | None = None, **kw: Any) -> None:
    """One bar per category for each series, side by side (or stacked), with the category labels as ticks.
    sort = largest first (by the first series, or by the total with stack), ann = fmt on each bar (past the
    error bar's end with errs). errs: per series, a (2, n) array of the distances below and above each bar's end."""
    w = kw.pop("height" if horizontal else "width", 0.8)
    stack = "stack" in flags and len(series) > 1
    order = _order(ax, cats, sum(v for _, v in series) if stack else series[0][1], flags, horizontal)
    series = [(n, v[order]) for n, v in series]
    errs = None if errs is None else [e[:, order] for e in errs]
    idx = np.arange(len(cats))
    draw = ax.barh if horizontal else ax.bar
    bottom = None
    for i, (name, v) in enumerate(series):
        if errs is not None:
            kw = {"xerr" if horizontal else "yerr": errs[i], "capsize": 3,
                  "error_kw": {"ecolor": plt.rcParams["text.color"], "elinewidth": 1}, **kw}
        if stack:                               # each series starts where the last ended: the end is the total
            draw(idx, v, w, bottom, **{"label": name, **kw})
            bottom = v if bottom is None else bottom + v
        else:                                   # side by side, together filling w
            draw(idx + (i + 0.5 - len(series) / 2) * w / len(series), v, w / len(series), **{"label": name, **kw})
        if errs is not None:
            kw = {k: a for k, a in kw.items() if k not in ("xerr", "yerr")}
    if "ann" in flags:                          # stacked segments: value inside; otherwise past the end
        for c in ax.containers:
            if isinstance(c, BarContainer):
                ax.bar_label(c, fmt=fmt, label_type="center" if stack else "edge", padding=0 if stack else 2)


def _order(ax: Axes, cats: np.ndarray, key: np.ndarray, flags: set[str], horizontal: bool) -> np.ndarray:
    """The order to draw the categories in, as given or largest key first with sort (NaN last), and their
    labels as ticks at 0, 1, ... in that order: one rule for bar, barh and lollipop."""
    order = np.argsort(-np.asarray(key, dtype=float), kind="stable") if "sort" in flags else np.arange(len(cats))
    (ax.set_yticks if horizontal else ax.set_xticks)(np.arange(len(cats)), [str(c) for c in cats[order]])
    return order


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


@kind("count", flags=("stack", "sort", "ann", "norm"), needs_y=False, takes_by=True, check=_check_count, frames=True)
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


@kind("barh", flags=("stack", "sort", "ann", "ci"), value_axis="x", frames=True)
def barh(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """Like bar, lying down: categories (x) top to bottom, bar lengths along the x axis; ci as for bar."""
    bar(ax, x, y, z, flags, horizontal=True, **kw)
    ax.invert_yaxis()                           # first category on top, read like a table


def _check_lollipop(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if np.size(y) and np.ndim(y) != 1:
        raise ValueError("kind 'lollipop' takes one value per category: y (and z for a dumbbell) 1-D")


@kind("lollipop", flags=("sort", "ann"), value_axis="x", check=_check_lollipop, frames=True)
def lollipop(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str],
             fmt: str = "{:.3g}", **kw: Any) -> None:
    """A dot per category (x) at its value (y) on a thin stem from 0, lying down like barh, first category on top.
    With z, a dumbbell: a line from y to z and a dot at each end, no stems. sort = largest first (by z - y with z),
    ann = the value beside each dot (fmt=, '{:.3g}'). kw: ax.plot for the dots."""
    cats = np.asarray(x, dtype=object).ravel()
    ends = [(label(a, n), np.asarray(a, dtype=float).ravel()) for a, n in ((y, "y"), (z, "z")) if a is not None]
    if any(len(v) != len(cats) for _, v in ends):
        raise ValueError(f"kind 'lollipop' takes one value per category: {len(cats)} categories, "
                         f"{' and '.join(str(len(v)) for _, v in ends)} values")
    order = _order(ax, cats, ends[1][1] - ends[0][1] if z is not None else ends[0][1], flags, True)
    ends = [(n, v[order]) for n, v in ends]
    idx = np.arange(len(cats))
    if z is None:
        ax.hlines(idx, 0, ends[0][1], color="C0", lw=1.5, zorder=2)
    else:
        ax.hlines(idx, ends[0][1], ends[1][1], color=plt.rcParams["text.color"], alpha=0.3, lw=3, zorder=2)
    for i, (name, v) in enumerate(ends):
        ax.plot(v, idx, **{"ls": "none", "marker": "o", "ms": 8, "color": f"C{i}", "zorder": 3,
                           **({"label": name} if z is not None else {}), **kw})
    if "ann" in flags:
        for j, (_, v) in enumerate(ends):
            other = ends[1 - j][1] if z is not None else np.zeros(len(cats))
            for i, a in enumerate(v):
                if np.isfinite(a):
                    left = a < other[i] or (a == other[i] and j == 0 and z is not None)
                    ax.annotate(fmt.format(a), (a, i), xytext=(-8 if left else 8, 0), textcoords="offset points",
                                ha="right" if left else "left", va="center")
        ax.margins(x=0.12)
    ax.set_ylim(len(cats) - 0.5, -0.5)
    if z is not None:
        ax.legend()


def _legend_right(ax: Axes, title: str | None, gap: float = 0.02) -> None:
    """A legend beside the axes, for kinds that fill them (pie, radar, parallel)."""
    ax.legend(title=title, loc="upper left", bbox_to_anchor=(1 + gap, 1))


def _check_pie(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if z is not None:
        raise ValueError("kind 'pie' takes the categories (x) and maybe their values (y), no z")
    if y is not None and (np.asarray(y, dtype=float) < 0).any():
        raise ValueError("a pie's values are shares of a whole and cannot be negative")


def _shares(x: ArrayLike, y: ArrayLike | None, flags: set[str], top: int) -> tuple[list[str], np.ndarray, bool]:
    """Slice labels and sizes: the counts of x's categories, or y summed per category of x, and whether any were
    folded. sort = largest first; past `top` slices the smallest fold into one 'other' slice, kept last."""
    if y is None:
        cats, masks = _levels(x)
        sizes = np.array([m.sum() for m in masks], dtype=float)
    else:
        xs, ys = np.asarray(x, dtype=object).ravel(), np.asarray(y, dtype=float).ravel()
        if len(xs) != len(ys):
            raise ValueError(f"x has {len(xs)} categories but y has {len(ys)} values")
        sums = pd.Series(ys).groupby(pd.Series(xs).astype(str), sort=False).sum(min_count=1).dropna()
        cats, sizes = list(sums.index), sums.to_numpy()
    if not sizes.sum() > 0:
        raise ValueError("a pie needs a positive total to split")
    order = np.argsort(-sizes, kind="stable")
    if len(sizes) > top:                        # keep the top - 1 largest, in their order, and fold the rest
        keep = np.sort(order[:top - 1]) if "sort" not in flags else order[:top - 1]
        rest = sizes.sum() - sizes[keep].sum()
        return [cats[i] for i in keep] + ["other"], np.r_[sizes[keep], rest], True
    if "sort" in flags:
        return [cats[i] for i in order], sizes[order], False
    return cats, sizes, False


def _pie(ax: Axes, x: ArrayLike, y: ArrayLike | None, flags: set[str], hole: float, top: int, **kw: Any) -> None:
    """A pie (hole 0) or a ring (hole 0..1) starting at 12 o'clock, clockwise."""
    cats, sizes, folded = _shares(x, y, flags, top)
    colors = [c["color"] for c, _ in zip(plt.rcParams["axes.prop_cycle"], cats, strict=False)]
    colors = (colors * (len(cats) // len(colors) + 1))[:len(cats)]
    if folded:
        colors[-1] = "0.6"                      # the folded rest in neutral grey, not a category colour
    named = flags & {"leg", "legout"}           # a legend names the slices, so their labels go
    face = plt.rcParams["axes.facecolor"]
    wedges, *_ = ax.pie(sizes, **{"labels": None if named else cats, "colors": colors, "startangle": 90,
                                  "counterclock": False,
                                  "wedgeprops": {"width": 1 - hole, "edgecolor": face, "linewidth": 1}, **kw})
    for w, c in zip(wedges, cats, strict=True):
        w.set_label(c)
    if "ann" in flags:                          # the share in each slice, black or white for its colour
        for w, s in zip(wedges, sizes, strict=True):
            angle = np.deg2rad((w.theta1 + w.theta2) / 2)
            r = 1 - (1 - hole) / 2 if hole else 0.62
            ax.text(r * np.cos(angle), r * np.sin(angle), f"{s / sizes.sum():.0%}", ha="center", va="center",
                    color=ink(w.get_facecolor()))
    if hole:
        ax.text(0, 0, f"{sizes.sum():g}", ha="center", va="center", fontsize="x-large")
    if named:                                   # the pie fills the axes: its legend goes beside it
        _legend_right(ax, label(x, "") or None)


@kind("pie", flags=("ann", "sort"), needs_y=False, check=_check_pie, frames=True)
def pie(ax: Axes, x: ArrayLike, y: ArrayLike | None, z: None, flags: set[str], top: int = 7, **kw: Any) -> None:
    """Shares of a whole: the counts of x's categories (missing as 'NaN'), or with y their summed values.
    ann = the percent in each slice, sort = largest first. Past top= slices (7) the smallest fold into 'other'.
    Starts at 12 o'clock, clockwise. kw: ax.pie."""
    _pie(ax, x, y, flags, 0, top, **kw)


@kind("donut", flags=("ann", "sort"), needs_y=False, check=_check_pie, frames=True)
def donut(ax: Axes, x: ArrayLike, y: ArrayLike | None, z: None, flags: set[str], top: int = 7,
          hole: float = 0.55, **kw: Any) -> None:
    """A pie with a hole and the total in it. ann = the percent on each segment, sort = largest first.
    Past top= slices (7) the smallest fold into 'other'; hole= is the hole's share of the radius (0.55).
    kw: ax.pie."""
    if not 0 < hole < 1:
        raise ValueError(f"hole= is a share of the radius between 0 and 1, got {hole}")
    _pie(ax, x, y, flags, hole, top, **kw)


MAX_PROFILES = 10                               # more polygons than this on one radar is a scribble


def _numeric(frame: Any, name: str, least: int) -> tuple[list[str], np.ndarray, list[Any]]:
    """The table's column names, its values as floats (rows x columns, NaN for gaps) and its row labels (a pandas
    index, else 0..n-1), the columns checked to be numbers (or bools), at least `least` of them."""
    df = wrap(frame)
    cols = [df.get_column(c) for c in df.columns]      # get_column: a 2-D array's columns are named 0, 1, ...
    text = [str(s.name) for s in cols if not (s.dtype.is_numeric() or s.dtype == nw.Boolean)]
    if text:
        raise ValueError(f"kind {name!r} draws numbers; columns {text} are not. Pass the numeric columns only, "
                         "or name a column of groups with by=")
    if len(cols) < least:
        raise ValueError(f"kind {name!r} needs at least {least} numeric columns, got {len(cols)}")
    values = np.column_stack([np.asarray(s.cast(nw.Float64).to_numpy(), dtype=float) for s in cols])
    index = nw.maybe_get_index(df)
    return [str(c) for c in df.columns], values, list(index) if index is not None else list(range(len(df)))


def _range(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each column's min and max over its finite values; NaN for a column with none."""
    if not len(values):
        return np.full(values.shape[1], np.nan), np.full(values.shape[1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # an all-NaN column: NaN, as pandas gives
        return np.nanmin(values, axis=0), np.nanmax(values, axis=0)


def _scaled(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each column scaled to 0..1 over its min and max (a constant column sits at 0.5), and the mins and maxes."""
    lo, hi = _range(values)
    span = np.where(hi > lo, hi - lo, np.nan)
    scaled = (values - lo) / span
    scaled[:, np.isnan(span)] = 0.5
    return scaled, lo, hi


@kind("radar", flags=("raw",), table=True, takes_by=True, projection="polar", frames=True)
def radar(ax: Axes, x: Any, y: None, z: None, flags: set[str], by: ArrayLike | None = None,
          **kw: Any) -> None:
    """One polygon per row over the numeric columns, or with by= the mean of each group's rows:
    kx.plot(df, 'radar', by='class'). Each column is scaled to 0..1 over the polygons shown, with its range under
    its name; raw = the values as they are, for columns in one unit. kw: the outlines (ax.plot)."""
    names, frame, rows = _numeric(x, "radar", 3)
    if by is not None:
        rows, masks = _levels(by)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a group with a column all NaN: NaN, as pandas
            frame = np.array([np.nanmean(frame[np.asarray(m)], axis=0) for m in masks]).reshape(-1, len(names))
    if len(frame) > MAX_PROFILES:
        raise ValueError(f"kind 'radar' draws one polygon per row: {len(frame)} rows are too many to read. "
                         f"Pass by= to average groups, or at most {MAX_PROFILES} rows (df.head())")
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    values, lo, hi = (frame, *_range(frame)) if "raw" in flags else _scaled(frame)
    for name, row in zip(rows, values, strict=True):
        v = np.r_[row, row[0]]                                  # closed: back to the first corner
        ln, = ax.plot(np.r_[angles, 0], v, **{"label": str(name), **kw})
        ax.fill(np.r_[angles, 0], v, color=ln.get_color(), alpha=0.15)
    ax.set_theta_offset(np.pi / 2)              # the first column at 12 o'clock, then clockwise
    ax.set_theta_direction(-1)
    ax.set_xticks(angles, names if "raw" in flags else [f"{n}\n{a:.3g} to {b:.3g}" for n, a, b in zip(names, lo, hi,
                                                                                                strict=True)])
    ax.tick_params(axis="x", pad=8)
    for a, text in zip(angles, ax.get_xticklabels(), strict=True):   # names grow away from the ring
        side = np.sin(a)                        # clockwise from 12 o'clock: > 0 is the right half
        text.set_horizontalalignment("left" if side > 0.1 else "right" if side < -0.1 else "center")
    if "raw" not in flags:                      # rings at quarters of each column's range: no single scale
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1], [])
    if len(frame) > 1:
        _legend_right(ax, label(by, "") if by is not None else None, gap=0.3)   # past the right-hand names


@kind("parallel", flags=("mean",), table=True, takes_by=True, frames=True)
def parallel(ax: Axes, x: Any, y: None, z: None, flags: set[str], by: ArrayLike | None = None,
             **kw: Any) -> None:
    """Parallel coordinates: one line per row across the numeric columns, each column scaled to 0..1 with its
    real min and max at the ends; by= colours the lines by group: kx.plot(df, 'parallel', by='class').
    Lines fade as rows grow, mean = each group's mean as a thick line on top. kw: the lines (LineCollection)."""
    names, frame, _ = _numeric(x, "parallel", 2)
    values, lo, hi = _scaled(frame)
    cols = np.arange(len(names))
    groups, masks = _levels(by) if by is not None else ([None], [np.ones(len(frame), dtype=bool)])
    alpha = float(np.clip(40 / max(len(frame), 1), 0.05, 0.8))
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (g, m) in enumerate(zip(groups, masks, strict=True)):
        color = cycle[i % len(cycle)]
        rows = values[np.asarray(m)]
        lines = LineCollection([np.c_[cols, r] for r in rows],
                                               **{"color": color, "alpha": alpha, "linewidth": 1, **kw})
        ax.add_collection(lines)
        if g is not None:                       # a legend handle at full colour, not the faded lines
            ax.plot([], [], color=color, label=g)
        if "mean" in flags and len(rows):
            ax.plot(cols, np.nanmean(rows, axis=0), color=color, linewidth=3, zorder=3,
                    path_effects=[withStroke(linewidth=5, foreground=plt.rcParams["axes.facecolor"])])
    edge = plt.rcParams["text.color"]
    ax.vlines(cols, 0, 1, color=edge, linewidth=0.8, zorder=2)
    for c, a, b in zip(cols, lo, hi, strict=True):
        ax.text(c, 1.02, f"{b:.3g}", ha="center", va="bottom", fontsize="small")
        ax.text(c, -0.02, f"{a:.3g}", ha="center", va="top", fontsize="small")
    ax.set_xticks(cols, names)
    ax.set_xlim(-0.2, cols[-1] + 0.2)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=14)  # room for the min under each axis
    if by is not None:
        _legend_right(ax, label(by, "by"))
