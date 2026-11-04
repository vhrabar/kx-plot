"""Basic kinds: y and z against x."""
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.patheffects import withStroke
from numpy.typing import ArrayLike

from .._spec import kind
from .._util import COLOUR, categories, colorbar, diagonal, ink, label, norm

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


@kind("scatter", flags=("date", "diag", "fit", *COLOUR), check=_check_scatter)
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


@kind("pie", flags=("ann", "sort"), needs_y=False, check=_check_pie)
def pie(ax: Axes, x: ArrayLike, y: ArrayLike | None, z: None, flags: set[str], top: int = 7, **kw: Any) -> None:
    """Shares of a whole: the counts of x's categories (missing as 'NaN'), or with y their summed values.
    ann = the percent in each slice, sort = largest first. Past top= slices (7) the smallest fold into 'other'.
    Starts at 12 o'clock, clockwise. kw: ax.pie."""
    _pie(ax, x, y, flags, 0, top, **kw)


@kind("donut", flags=("ann", "sort"), needs_y=False, check=_check_pie)
def donut(ax: Axes, x: ArrayLike, y: ArrayLike | None, z: None, flags: set[str], top: int = 7,
          hole: float = 0.55, **kw: Any) -> None:
    """A pie with a hole and the total in it. ann = the percent on each segment, sort = largest first.
    Past top= slices (7) the smallest fold into 'other'; hole= is the hole's share of the radius (0.55).
    kw: ax.pie."""
    if not 0 < hole < 1:
        raise ValueError(f"hole= is a share of the radius between 0 and 1, got {hole}")
    _pie(ax, x, y, flags, hole, top, **kw)


MAX_PROFILES = 10                               # more polygons than this on one radar is a scribble


def _numeric(frame: pd.DataFrame, name: str, least: int) -> pd.DataFrame:
    """The table's columns, checked to be numbers, at least `least` of them."""
    text = [str(c) for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if text:
        raise ValueError(f"kind {name!r} draws numbers; columns {text} are not. Pass df.select_dtypes('number'), "
                         "or name a column of groups with by=")
    if frame.shape[1] < least:
        raise ValueError(f"kind {name!r} needs at least {least} numeric columns, got {frame.shape[1]}")
    return frame.astype(float)


def _scaled(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Each column scaled to 0..1 over its min and max (a constant column sits at 0.5), and the mins and maxes."""
    lo, hi = frame.min(), frame.max()
    span = (hi - lo).where(hi > lo)
    return ((frame - lo) / span).fillna({c: 0.5 for c in frame.columns[span.isna()]}), lo, hi


@kind("radar", flags=("raw",), table=True, takes_by=True, projection="polar")
def radar(ax: Axes, x: pd.DataFrame, y: None, z: None, flags: set[str], by: ArrayLike | None = None,
          **kw: Any) -> None:
    """One polygon per row over the numeric columns, or with by= the mean of each group's rows:
    kx.plot(df, 'radar', by='class'). Each column is scaled to 0..1 over the polygons shown, with its range under
    its name; raw = the values as they are, for columns in one unit. kw: the outlines (ax.plot)."""
    frame = _numeric(x, "radar", 3)
    if by is not None:
        groups, masks = _levels(by)
        frame = pd.DataFrame([frame[m].mean() for m in masks], index=groups)
    if len(frame) > MAX_PROFILES:
        raise ValueError(f"kind 'radar' draws one polygon per row: {len(frame)} rows are too many to read. "
                         f"Pass by= to average groups, or at most {MAX_PROFILES} rows (df.head())")
    angles = np.linspace(0, 2 * np.pi, frame.shape[1], endpoint=False)
    values, lo, hi = (frame, frame.min(), frame.max()) if "raw" in flags else _scaled(frame)
    for name, row in values.iterrows():
        v = np.r_[row.to_numpy(), row.iloc[0]]                  # closed: back to the first corner
        ln, = ax.plot(np.r_[angles, 0], v, **{"label": str(name), **kw})
        ax.fill(np.r_[angles, 0], v, color=ln.get_color(), alpha=0.15)
    ax.set_theta_offset(np.pi / 2)              # the first column at 12 o'clock, then clockwise
    ax.set_theta_direction(-1)
    names = [str(c) for c in frame.columns]
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


@kind("parallel", flags=("mean",), table=True, takes_by=True)
def parallel(ax: Axes, x: pd.DataFrame, y: None, z: None, flags: set[str], by: ArrayLike | None = None,
             **kw: Any) -> None:
    """Parallel coordinates: one line per row across the numeric columns, each column scaled to 0..1 with its
    real min and max at the ends; by= colours the lines by group: kx.plot(df, 'parallel', by='class').
    Lines fade as rows grow, mean = each group's mean as a thick line on top. kw: the lines (LineCollection)."""
    frame = _numeric(x, "parallel", 2)
    values, lo, hi = _scaled(frame)
    cols = np.arange(frame.shape[1])
    groups, masks = _levels(by) if by is not None else ([None], [np.ones(len(frame), dtype=bool)])
    alpha = float(np.clip(40 / max(len(frame), 1), 0.05, 0.8))
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (g, m) in enumerate(zip(groups, masks, strict=True)):
        color = cycle[i % len(cycle)]
        rows = values.to_numpy()[np.asarray(m)]
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
    ax.set_xticks(cols, [str(c) for c in frame.columns])
    ax.set_xlim(-0.2, cols[-1] + 0.2)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=14)  # room for the min under each axis
    if by is not None:
        _legend_right(ax, label(by, "by"))
