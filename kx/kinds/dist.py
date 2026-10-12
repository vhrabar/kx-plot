"""Distribution kinds."""
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from .._spec import kind
from .._util import categories, label


def _finite(arr: ArrayLike) -> np.ndarray:
    """Values as floats, NaN and inf dropped: a gap in the data is not a value."""
    v = np.asarray(arr, dtype=float).ravel()
    return v[np.isfinite(v)]


def _series(x: ArrayLike, y: ArrayLike | None, z: ArrayLike | None) -> list[tuple[np.ndarray, str]]:
    return [(_finite(a), label(a, n)) for a, n in ((x, "x"), (y, "y"), (z, "z")) if a is not None]


def _split(x: ArrayLike, y: ArrayLike | None, z: ArrayLike | None,
           by: ArrayLike | None) -> list[tuple[np.ndarray, str]]:
    """x, y, z as separate series, or with by= x split into one series per group (categories in order)."""
    if by is None:
        return _series(x, y, z)
    if y is not None or z is not None:
        raise ValueError("by= splits x into groups; pass only x, not y or z")
    xs, groups = np.asarray(x, dtype=float).ravel(), np.asarray(by, dtype=object).ravel()
    if len(xs) != len(groups):
        raise ValueError(f"by= has {len(groups)} values but x has {len(xs)}")
    cats = categories(by)
    if cats is None:
        cats = sorted(pd.unique(groups[pd.notna(groups)]))
    return [(_finite(xs[groups == c]), str(c)) for c in cats]


def _need(series: list[tuple[np.ndarray, str]], kind: str, distinct: int) -> None:
    for v, n in series:
        if len(np.unique(v)) < distinct:
            what = "a finite value" if distinct == 1 else f"at least {distinct} distinct finite values"
            raise ValueError(f"{kind} of {n!r} needs {what}")


STATS = ("mean", "median", "ann")              # mark each series' mean / median; ann writes the value


def _check_stats(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if "ann" in flags and not flags & {"mean", "median"}:
        raise ValueError("flag 'ann' writes the mean or median; add 'mean' or 'median'")


def _vlines(ax: Axes, series: list[tuple[np.ndarray, str]], flags: set[str]) -> None:
    """A vertical line at each series' mean (dashed) and median (dotted), in the series' colour."""
    for stat, fn, style in (("mean", np.mean, "--"), ("median", np.median, ":")):
        if stat not in flags:
            continue
        for i, (v, _) in enumerate(series):
            at = fn(v)
            ax.axvline(at, color=f"C{i}", linestyle=style, linewidth=1.2)
            if "ann" in flags:
                ax.annotate(f"{at:.3g}", (at, 1), xycoords=("data", "axes fraction"), xytext=(-2, -4),
                            textcoords="offset points", rotation=90, ha="right", va="top",
                            color=f"C{i}", fontsize="small")


@kind("hist", flags=("stack", "norm", "cum", *STATS), needs_y=False, check=_check_stats)
def hist(ax: Axes, x: ArrayLike, y: ArrayLike | None, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """Up to three distributions x, y, z on the same 30 bins, overlaid.
    stack = on top of each other, norm = density (area 1), cum = cumulative outlines (with norm: the CDF).
    mean / median = dashed / dotted line at each series' mean / median, ann = with its value.
    kw: ax.hist; bins= (a count, edges or a numpy rule like 'auto') replaces the 30 shared bins."""
    series = _series(x, y, z)
    edges = np.histogram_bin_edges(np.concatenate([v for v, _ in series]), bins=kw.pop("bins", 30))
    values, names = [v for v, _ in series], [n for _, n in series]
    if "stack" in flags:
        ax.hist(values, edges, **{"histtype": "barstacked", "label": names,
                                  "density": "norm" in flags, "cumulative": "cum" in flags, **kw})
    else:                                       # overlaid cumulative fills hide each other: outlines instead
        look = {"histtype": "step", "linewidth": 1.5} if "cum" in flags else {"alpha": 0.6}
        for v, n in zip(values, names, strict=True):
            ax.hist(v, edges, **{"label": n, "density": "norm" in flags, "cumulative": "cum" in flags, **look, **kw})
    _vlines(ax, series, flags)


def density(v: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    """Gaussian KDE of v on grid with bandwidth bw, in chunks so big v does not need a len(v) x len(grid) array."""
    out = np.zeros_like(grid)
    for i in range(0, len(v), 4096):
        out += np.exp(-0.5 * ((grid[:, None] - v[None, i:i + 4096]) / bw) ** 2).sum(axis=1)
    return out / (len(v) * bw * np.sqrt(2 * np.pi))


def bandwidth(v: np.ndarray) -> float:
    """Silverman's rule of thumb, the IQR form: one outlier does not flatten the curve."""
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    spread = min(v.std(ddof=1), iqr / 1.349) if iqr > 0 else v.std(ddof=1)
    return 0.9 * spread * len(v) ** -0.2


@kind("kde", flags=STATS, needs_y=False, takes_by=True, check=_check_stats)
def kde(ax: Axes, x: ArrayLike, y: ArrayLike | None, z: ArrayLike | None, flags: set[str],
        by: ArrayLike | None = None, **kw: Any) -> None:
    """Smooth density curves of x, y, z (Gaussian KDE, Silverman bandwidth).
    by= splits x into one curve per group: kx.plot(df.value, spec='kde', by=df.group). kw: the curves (ax.plot)."""
    series = _split(x, y, z, by)
    _need(series, "kde", 2)
    bws = [bandwidth(v) for v, _ in series]
    lo = min(v.min() - 3 * bw for (v, _), bw in zip(series, bws, strict=True))
    hi = max(v.max() + 3 * bw for (v, _), bw in zip(series, bws, strict=True))
    grid = np.linspace(lo, hi, 256)
    for (v, n), bw in zip(series, bws, strict=True):
        d = density(v, grid, bw)
        ln, = ax.plot(grid, d, **{"label": n, **kw})
        ax.fill_between(grid, d, color=ln.get_color(), alpha=0.15, linewidth=0)
    ax.set_ylim(bottom=0)
    _vlines(ax, series, flags)
    if by is not None:
        ax.legend(title=label(by, "by"))


@kind("ecdf", flags=STATS, needs_y=False, takes_by=True, check=_check_stats)
def ecdf(ax: Axes, x: ArrayLike, y: ArrayLike | None, z: ArrayLike | None, flags: set[str],
         by: ArrayLike | None = None, **kw: Any) -> None:
    """Empirical CDF of x, y, z: the share of values <= each x, a step up of 1/n at every value.
    by= splits x into one curve per group. kw: ax.step."""
    series = _split(x, y, z, by)
    _need(series, "ecdf", 1)
    for v, n in series:
        v = np.sort(v)
        ax.step(np.r_[v[0], v], np.arange(len(v) + 1) / len(v), **{"where": "post", "label": n, **kw})
    ax.set_ylim(0, 1.02)
    _vlines(ax, series, flags)
    if by is not None:
        ax.legend(title=label(by, "by"))


def _per_group(ax: Axes, series: list[tuple[np.ndarray, str]], flags: set[str], by: ArrayLike | None) -> None:
    """One tick per series, named; with by= the axis is named after the groups.
    mean = a diamond at each mean, ann = the median (and mean) written beside each box or violin."""
    pos = np.arange(1, len(series) + 1)
    ax.set_xticks(pos, [n for _, n in series])
    if by is not None:
        ax.set_xlabel(label(by, "by"))
    ink = plt.rcParams["text.color"]
    means = [np.mean(v) for v, _ in series]
    if "mean" in flags:
        ax.scatter(pos, means, marker="D", color=ink, s=16, zorder=4)
    if "ann" in flags:
        for p, (v, _), m in zip(pos, series, means, strict=True):
            text = f"{np.median(v):.3g}" + (f"\nμ {m:.3g}" if "mean" in flags else "")
            ax.annotate(text, (p + 0.3, np.median(v)), va="center", color=ink, fontsize="small")


@kind("box", flags=("mean", "ann"), needs_y=False, takes_by=True)
def box(ax: Axes, x: ArrayLike, y: ArrayLike | None, z: ArrayLike | None, flags: set[str],
        by: ArrayLike | None = None, **kw: Any) -> None:
    """Box plots of x, y, z side by side: median, quartile box, whiskers to 1.5 IQR, outliers as points.
    by= splits x into one box per group. mean = a diamond at the mean, ann = the median (and mean) as text.
    kw: ax.boxplot, e.g. showfliers=False, widths=0.3."""
    series = _split(x, y, z, by)
    _need(series, "box", 1)
    ink = plt.rcParams["text.color"]           # whiskers and medians in the theme's ink, not black on dark
    line = {"color": ink}
    parts = ax.boxplot([v for v, _ in series], **{"patch_artist": True, "boxprops": {"edgecolor": ink},
                                                  "medianprops": line, "whiskerprops": line, "capprops": line,
                                                  "flierprops": {"markeredgecolor": ink}, **kw})
    for i, patch in enumerate(parts["boxes"]):
        patch.set_facecolor((*mpl.colors.to_rgb(f"C{i}"), 0.7))      # alpha on the fill only, edge stays solid
    _per_group(ax, series, flags, by)


@kind("violin", flags=("mean", "ann"), needs_y=False, takes_by=True)
def violin(ax: Axes, x: ArrayLike, y: ArrayLike | None, z: ArrayLike | None, flags: set[str],
           by: ArrayLike | None = None, **kw: Any) -> None:
    """Violin plots of x, y, z side by side: a mirrored KDE with the median and quartiles marked.
    by= splits x into one violin per group. mean = a diamond at the mean, ann = the median (and mean) as text.
    kw: ax.violinplot, e.g. widths=0.5."""
    series = _split(x, y, z, by)
    _need(series, "violin", 2)
    values = [v for v, _ in series]
    parts = ax.violinplot(values, **{"showextrema": False, **kw})
    for i, body in enumerate(parts["bodies"]):
        body.set(facecolor=f"C{i}", edgecolor=f"C{i}", alpha=0.6)
    pos = np.arange(1, len(values) + 1)
    q1, med, q3 = np.array([np.percentile(v, [25, 50, 75]) for v in values]).T
    ink = plt.rcParams["text.color"]
    ax.vlines(pos, q1, q3, color=ink, linewidth=3)
    ax.scatter(pos, med, color=plt.rcParams["axes.facecolor"], edgecolor=ink, zorder=3)
    _per_group(ax, series, flags, by)


@kind("strip", flags=("mean", "ann"), needs_y=False, takes_by=True)
def strip(ax: Axes, x: ArrayLike, y: ArrayLike | None, z: ArrayLike | None, flags: set[str],
          by: ArrayLike | None = None, **kw: Any) -> None:
    """Every value as a point, one column per x, y, z, jittered sideways so ties stay visible.
    by= splits x into one column per group. mean = a diamond at the mean, ann = the median (and mean) as text.
    kw: ax.scatter; jitter= is the half-width of the sideways spread, 0.15 by default."""
    series = _split(x, y, z, by)
    _need(series, "strip", 1)
    jitter = kw.pop("jitter", 0.15)
    rng = np.random.default_rng(0)              # the same data always gets the same picture
    size = 0.5 * plt.rcParams["lines.markersize"] ** 2
    for i, (v, n) in enumerate(series):
        ax.scatter(i + 1 + rng.uniform(-jitter, jitter, len(v)), v,
                   **{"s": size, "color": f"C{i}", "alpha": 0.6, "linewidth": 0, "label": n, **kw})
    _per_group(ax, series, flags, by)
