"""
Table kinds: a whole DataFrame (or 2-D array) as the data.
"""
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.image import AxesImage
from numpy.typing import ArrayLike

from .. import _stats
from .._spec import kind
from .._util import COLOUR, colorbar, diagonal, ink, label, norm, split
from .basic import _bars, _levels
from .dist import bandwidth, density

MAX_PAIRS = 8


@kind("heat", flags=("corr", "ann", *COLOUR), table=True)
def heat(ax: Axes, x: pd.DataFrame, y: None, z: None, flags: set[str], **kw: Any) -> None:
    """The table's values as coloured cells, columns along x and index labels down y.
    corr = correlations of the numeric columns (rdbu, ±1), ann = the value in each cell,
    nocb = no colorbar, sym = colours symmetric around 0, logc = log colour scale. kw: ax.imshow."""
    frame, style = _heat_frame(x, flags, "heat")
    _matrix(ax, frame, flags, "r" if "corr" in flags else "", "{:.2f}" if "corr" in flags else "{:.3g}",
            **{**style, **kw})


def _heat_frame(x: pd.DataFrame, flags: set[str], name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The numbers heat and cluster colour (corr = the correlations) and their colour scale."""
    frame = x.corr(numeric_only=True) if "corr" in flags else x
    text = [str(c) for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if text:
        raise ValueError(f"kind {name!r} colours numbers; columns {text} are not. "
                         "Pass df.select_dtypes('number'), or use 'corr'")
    corr = {"cmap": "rdbu", "norm": mpl.colors.Normalize(-1, 1)}  # the scale of r, whatever this data's range
    return frame, corr if "corr" in flags else {"norm": norm(frame.to_numpy(dtype=float), flags)}


def _check_crosstab(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if z is not None:
        raise ValueError("kind 'crosstab' counts pairs of x and y; it takes no z")


@kind("crosstab", flags=("norm", "ann", "nocb", "logc"), check=_check_crosstab)
def crosstab(ax: Axes, x: ArrayLike, y: ArrayLike, z: None, flags: set[str], **kw: Any) -> None:
    """How often each pair of categories occurs: x along x, y down y, like pd.crosstab(y, x).
    norm = shares within each x category (each column sums to 100%), ann = the value in each cell,
    nocb = no colorbar, logc = log colour scale (empty cells stay blank). kw: ax.imshow."""
    counts = pd.crosstab(pd.Series(np.asarray(y, dtype=object), name=label(y, "y")),
                         pd.Series(np.asarray(x, dtype=object), name=label(x, "x")))
    if "norm" in flags:
        counts = counts / counts.sum()
    if "logc" in flags:                         # log(0) has no colour: leave those cells empty
        counts = counts.where(counts > 0)
    share = "norm" in flags
    _matrix(ax, counts, flags, "share" if share else "count", "{:.0%}" if share else "{:g}",
            **{"norm": norm(counts.to_numpy(dtype=float), flags), "aspect": "auto", **kw})


@kind("missing", flags=("ann", "all"), table=True, value_axis="x")
def missing(ax: Axes, x: pd.DataFrame, y: None, z: None, flags: set[str], **kw: Any) -> None:
    """The share of missing values in each column, as bars lying down, the most missing on top.
    Only columns with missing values, unless all; ann = the share on each bar. kw: ax.barh."""
    share = x.isna().mean()
    if "all" not in flags:
        share = share[share > 0]
    if share.empty:                             # nothing to draw is an answer too, not an error
        ax.text(0.5, 0.5, "no missing values", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    _bars(ax, np.array([str(c) for c in share.index], dtype=object), [("missing", share.to_numpy(dtype=float))],
          {"sort", *(flags & {"ann"})}, True, fmt="{:.1%}", **kw)
    ax.invert_yaxis()                           # largest on top
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    ax.set_xlabel("missing")


@kind("nullmap", flags=("sort",), table=True)
def nullmap(ax: Axes, x: pd.DataFrame, y: None, z: None, flags: set[str], **kw: Any) -> None:
    """Where the missing values are: one row per row of the table, columns along x, missing cells in the
    palette's first colour on the background. sort = the rows with the most missing on top. kw: ax.imshow."""
    gaps = x.isna()
    if "sort" in flags:                         # most missing first; ties by pattern, so the same gaps form a block
        g = gaps.to_numpy()
        gaps = gaps.iloc[np.lexsort([*(~g[:, j] for j in reversed(range(g.shape[1]))), -g.sum(axis=1)])]
    colours = mpl.colors.ListedColormap([ax.get_facecolor(), mpl.colors.to_rgba("C0")])
    ax.imshow(gaps.to_numpy(dtype=float), **{"cmap": colours, "vmin": 0, "vmax": 1, "interpolation": "nearest",
                                             "aspect": "auto", **kw})
    ax.set_xticks(range(gaps.shape[1]), [str(c) for c in gaps.columns])
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(10, integer=True))    # row positions, not every row
    ax.set_ylabel("row, most missing first" if "sort" in flags else "row")


def _check_hexbin(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if z is None and "sym" in flags:
        raise ValueError("flag 'sym' centres the colours of a z on 0; counts are never below 0")


@kind("hexbin", flags=("diag", *COLOUR), check=_check_hexbin)
def hexbin(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """How many points fall in each hexagon, or the mean of z there. Rows with a NaN are left out.
    nocb = no colorbar, sym = colours symmetric around 0 (z only), logc = log colour scale, diag = a dashed y = x line.
    kw: ax.hexbin (gridsize= sets the number of hexagons across)."""
    cols = [np.asarray(a, dtype=float).ravel() for a in (x, y, z) if a is not None]
    keep = np.logical_and.reduce([np.isfinite(c) for c in cols])
    xs, ys, *zs = (c[keep] for c in cols)
    if z is None:                               # counts; LogNorm, as bins="log" stores log10 counts before 3.8
        style = {"mincnt": 1, "norm": mpl.colors.LogNorm() if "logc" in flags else None}
    else:                                       # mincnt left out: cells with any point, on every matplotlib
        style = {"C": zs[0], "reduce_C_function": np.mean, "norm": norm(zs[0], flags)}
    hb = ax.hexbin(xs, ys, **{**style, **kw})
    colorbar(ax, hb, "count" if z is None else label(z, "z"), flags)
    if "diag" in flags:
        diagonal(ax, xs, ys, flags)


def _matrix(ax: Axes, frame: pd.DataFrame, flags: set[str], cblabel: str, fmt: str = "{:.2g}",
            **kw: Any) -> AxesImage:
    """frame as an image: column labels along x, index labels down y, a colorbar, ann = fmt in each cell."""
    values = frame.to_numpy(dtype=float)
    im = ax.imshow(values, **kw)
    ax.set_xticks(range(values.shape[1]), [str(c) for c in frame.columns])
    ax.set_yticks(range(values.shape[0]), [str(i) for i in frame.index])
    if frame.columns.name is not None: ax.set_xlabel(str(frame.columns.name))
    if frame.index.name is not None:   ax.set_ylabel(str(frame.index.name))
    cb = colorbar(ax, im, cblabel, flags)
    if cb is not None and "%" in fmt:
        cb.formatter = mpl.ticker.PercentFormatter(1, decimals=0)
    if "ann" in flags:
        for (i, j), v in np.ndenumerate(values):
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", color=ink(im.cmap(float(im.norm(v)))))
    return im


def _pair_columns(x: pd.DataFrame, cols: list[str] | None) -> pd.DataFrame:
    """The columns of a scatter matrix: cols= in its order, else every numeric (not bool) column; 2 to 8 of them."""
    if cols is not None:
        missing = [c for c in cols if c not in x.columns]
        if missing:
            raise ValueError(f"cols= names {missing}, which the table does not have. Columns: {list(x.columns)}")
        text = [str(c) for c in cols if not pd.api.types.is_numeric_dtype(x[c]) or pd.api.types.is_bool_dtype(x[c])]
        if text:
            raise ValueError(f"kind 'pair' plots numbers; cols= {text} are not. Pass a category as by=")
        frame = x[list(cols)]
    else:
        frame = x[[c for c in x.columns
                   if pd.api.types.is_numeric_dtype(x[c]) and not pd.api.types.is_bool_dtype(x[c])]]
    if frame.shape[1] < 2:
        raise ValueError(f"kind 'pair' needs at least 2 numeric columns, got {frame.shape[1]}")
    if frame.shape[1] > MAX_PAIRS:
        raise ValueError(f"kind 'pair' with {frame.shape[1]} numeric columns is {frame.shape[1] ** 2} panels; "
                         f"pick at most {MAX_PAIRS} with cols=[...]")
    return frame.astype(float)


@kind("pair", table=True, takes_by=True, panels=True)
def pair(ax: Axes, x: pd.DataFrame, y: None, z: None, flags: set[str], by: ArrayLike | None = None,
         cols: list[str] | None = None, **kw: Any) -> np.ndarray:
    """A scatter matrix: every numeric column against every other, and on the diagonal each column's density
    (a KDE per group), x shared down each column and y along each row: kx.plot(df, 'pair', by='class').
    cols= picks and orders the columns (at most 8). Returns the k x k panels, row by row. kw: ax.scatter."""
    frame = _pair_columns(x, cols)
    k, n = frame.shape[1], len(frame)
    groups, masks = _levels(by) if by is not None else ([None], [np.ones(n, dtype=bool)])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = ax.figure
    if len(fig.axes) == 1:                      # a figure of its own: sized to the matrix, legend kept inside
        side = max(plt.rcParams["figure.figsize"][1], 1.8 * k)
        fig.set_size_inches(side + (1.5 if by is not None else 0), side)
        if fig.get_layout_engine() is None:
            fig.set_layout_engine("constrained")
    axs = split(ax, k, k, wspace=0.08, hspace=0.08).reshape(k, k)
    for r in range(k):
        for c in range(k):
            a = axs[r, c]
            if r:
                a.sharex(axs[0, c])
            if c != (1 if r == 0 else 0):       # y shared along the row; the diagonal's density has its own inset
                a.sharey(axs[r, 1 if r == 0 else 0])
            a.tick_params(labelbottom=r == k - 1, labelleft=c == 0)
            if r == k - 1:
                a.set_xlabel(str(frame.columns[c]))
            if c == 0:
                a.set_ylabel(str(frame.columns[r]))
    values = frame.to_numpy()
    size = 0.3 * plt.rcParams["lines.markersize"] ** 2
    alpha = float(np.clip(300 / max(n, 1), 0.2, 0.8))
    for i, (g, m) in enumerate(zip(groups, masks, strict=True)):
        color = colors[i % len(colors)]
        rows = values[np.asarray(m)]
        for r in range(k):
            for c in range(k):
                if r != c:
                    axs[r, c].scatter(rows[:, c], rows[:, r],
                                      **{"s": size, "color": color, "alpha": alpha, "linewidths": 0,
                                         "label": g, **kw})
    for c in range(k):                          # the density on an inset sharing x only, so the row keeps its y
        dens = axs[c, c].inset_axes([0, 0, 1, 1], sharex=axs[c, c])
        dens.set_axis_off()
        col = values[:, c]
        finite = col[np.isfinite(col)]
        if len(finite) < 2 or finite.min() == finite.max():
            continue
        limits = axs[c, c].get_xlim()
        grid = np.linspace(*limits, 256)
        for i, m in enumerate(masks):
            v = col[np.asarray(m)]
            v = v[np.isfinite(v)]
            if len(v) < 2 or v.min() == v.max():
                continue                        # one value has no spread to smooth
            d = density(v, grid, bandwidth(v))
            dens.plot(grid, d, color=colors[i % len(colors)], linewidth=1.5)
            dens.fill_between(grid, d, color=colors[i % len(colors)], alpha=0.15, linewidth=0)
        dens.set_ylim(bottom=0)
        axs[c, c].set_xlim(limits)              # the curve's margins must not widen the shared column
    if by is not None:                          # one legend for the matrix, beside its top-right panel
        legend = axs[0, -1].legend(*axs[0, 1].get_legend_handles_labels(), title=label(by, "by"),
                                   loc="upper left", bbox_to_anchor=(1.05, 1), markerscale=2)
        for h in legend.legend_handles:
            h.set_alpha(1)                      # the faded points at full colour
    return axs.ravel()


def _dendrogram(ax: Axes, z: np.ndarray, side: str) -> None:
    """The linkage z as a tree over leaves at 0, 1, 2, ... (the heat map's cells), root away from the cells:
    side 'top' grows up from the columns, 'left' grows left from the rows."""
    n = len(z) + 1
    place = {leaf: float(i) for i, leaf in enumerate(_stats.leaves(z))}
    height = dict.fromkeys(range(n), 0.0)
    segments = []
    for k, (a, b, h, _) in enumerate(z):
        a, b = int(a), int(b)
        xa, xb = place[a], place[b]
        segments.append([(xa, height[a]), (xa, h), (xb, h), (xb, height[b])])   # the U joining a and b
        place[n + k], height[n + k] = (xa + xb) / 2, h
    if side == "left":                          # rows run down y, heights grow to the left
        segments = [[(-hh, pos) for pos, hh in seg] for seg in segments]
    ax.add_collection(LineCollection(segments, color=plt.rcParams["text.color"], linewidth=0.8))
    top = float(z[:, 2].max()) or 1.0
    if side == "top":
        ax.set_ylim(0, top * 1.05)
    else:
        ax.set_xlim(-top * 1.05, 0)
    ax.set_axis_off()


@kind("cluster", flags=("corr", "ann", *COLOUR), table=True, panels=True)
def cluster(ax: Axes, x: pd.DataFrame, y: None, z: None, flags: set[str], method: str = "average",
            **kw: Any) -> np.ndarray:
    """heat with the rows and the columns reordered so that similar ones sit together, and the dendrograms that
    order comes from above and left: kx.plot(df, 'cluster-corr') groups features that move together.
    Rows are compared by their values (euclidean), columns likewise; with corr both by 1 - |r|, so strongly
    negative correlation counts as similar. method= single, complete, average (default) or ward.
    ann, nocb, sym and logc as in heat. Returns the panels: top tree, left tree, heat map. kw: ax.imshow."""
    frame, style = _heat_frame(x, flags, "cluster")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("kind 'cluster' needs every cell; fill or drop the NaNs first (df.dropna())")
    trees = {}
    if "corr" in flags:                         # one order for both: the matrix stays symmetric
        if len(frame) > 1:
            trees["rows"] = trees["cols"] = _stats.linkage(1 - np.abs(values), method, distances=True)
    else:
        if values.shape[0] > 1:
            trees["rows"] = _stats.linkage(values, method)
        if values.shape[1] > 1:
            trees["cols"] = _stats.linkage(values.T, method)
    rows = _stats.leaves(trees["rows"]) if "rows" in trees else list(range(values.shape[0]))
    cols = _stats.leaves(trees["cols"]) if "cols" in trees else list(range(values.shape[1]))
    corner, top, left, cells = split(ax, 2, 2, width_ratios=[1, 5], height_ratios=[1, 5], wspace=0.02,
                                     hspace=0.02)
    top.sharex(cells)
    left.sharey(cells)
    im = _matrix(cells, frame.iloc[rows, cols], flags | {"nocb"}, "", "{:.2f}" if "corr" in flags else "{:.3g}",
                 **{**style, "aspect": "auto", **kw})
    cells.yaxis.tick_right()                    # the left side belongs to the tree
    cells.yaxis.set_label_position("right")
    if "cols" in trees:
        _dendrogram(top, trees["cols"], "top")
    else:
        top.set_axis_off()
    if "rows" in trees:
        _dendrogram(left, trees["rows"], "left")
    else:
        left.set_axis_off()
    if "nocb" in flags:
        corner.remove()
    else:                                       # the empty corner holds the colorbar
        corner.set_axis_off()
        cax = corner.inset_axes([0.1, 0.1, 0.12, 0.8])
        corner.figure.colorbar(im, cax=cax, label="r" if "corr" in flags else "")
    return np.array([top, left, cells])
