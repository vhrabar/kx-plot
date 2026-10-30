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
    for v, axis in ((y, "index"), (x, "columns")):   # a categorical keeps its unseen categories (facet panels)
        if isinstance(getattr(v, "dtype", None), pd.CategoricalDtype):
            counts = counts.reindex(**{axis: list(v.cat.categories)}, fill_value=0)
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


GAP = 0.02                                      # mosaic: space between columns and between tiles, axes fraction


def _check_mosaic(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if z is not None and np.size(z) != np.size(y):
        raise ValueError(f"y has {np.size(y)} values but z has {np.size(z)}")


def _shade(color: str, step: int, steps: int) -> tuple[float, ...]:
    """color mixed towards the background: step 0 is the colour itself, later steps lighter (darker on dark)."""
    back = np.array(mpl.colors.to_rgb(plt.rcParams["axes.facecolor"]))
    t = 0.6 * step / max(steps - 1, 1)
    return tuple((1 - t) * np.array(mpl.colors.to_rgb(color)) + t * back)


@kind("mosaic", flags=("ann",), check=_check_mosaic)
def mosaic(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """Two or three categoricals as tiles whose areas are their counts: one column per category of x, as wide as
    its share; in each, one tile per category of y, as tall as its share within that x; z splits each tile again
    side by side. Colours follow y (z in lighter shades of it); missing values count as 'NaN'.
    ann = the count in each tile big enough to hold it. kw: the tiles (ax.add_patch Rectangle)."""
    if np.size(x) != np.size(y):
        raise ValueError(f"x has {np.size(x)} values but y has {np.size(y)}")
    xs, xm = _levels(x)
    ys, ym = _levels(y)
    zs, zm = _levels(z) if z is not None else ([None], [np.ones(np.size(x), dtype=bool)])
    counts = np.array([[[np.sum(a & b & c) for c in zm] for b in ym] for a in xm], dtype=float)   # x, y, z
    total = counts.sum()
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    width = 1 - GAP * (len(xs) - 1)
    left = 0.0
    centres = []
    for i in range(len(xs)):
        w = width * counts[i].sum() / total
        centres.append(left + w / 2)
        top = 1.0                               # the first y category on top, read like a table
        column = counts[i].sum()
        for j in range(len(ys)):
            h = (1 - GAP * (len(ys) - 1)) * counts[i, j].sum() / column
            gap = min(GAP / 2, 0.2 * w / max(len(zs) - 1, 1))    # a thin column gets thin gaps, never negative
            inner = w - gap * (len(zs) - 1)
            x0 = left
            for k in range(len(zs)):
                n = counts[i, j, k]
                part = inner * n / counts[i, j].sum() if counts[i, j].sum() else 0
                color = _shade(cycle[j % len(cycle)], k, len(zs))
                ax.add_patch(mpl.patches.Rectangle((x0, top - h), part, h, **{"facecolor": color,
                                                                                "linewidth": 0, **kw}))
                if "ann" in flags and n and part > 0.018 and h > 0.04:   # a sliver has no room for a number
                    ax.text(x0 + part / 2, top - h / 2, f"{n:g}", ha="center", va="center", color=ink(color),
                            fontsize="small")
                x0 += part + gap
            top -= h + GAP
        left += w + GAP
    for j, name in enumerate(ys):               # one legend entry per colour: y, or y / z with a z
        for k, zname in enumerate(zs):
            ax.add_patch(mpl.patches.Rectangle((0, 0), 0, 0, facecolor=_shade(cycle[j % len(cycle)], k, len(zs)),
                                               label=name if zname is None else f"{name} / {zname}"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(centres, xs)
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel(label(x, "x"))
    _legend_beside(ax, label(y, "y") if z is None else f"{label(y, 'y')} / {label(z, 'z')}")


def _legend_beside(ax: Axes, title: str) -> None:
    """A legend right of axes the tiles fill; a figure of its own gets constrained layout, so saving keeps it."""
    ax.legend(title=title, loc="center left", bbox_to_anchor=(1, 0.5))
    if len(ax.figure.axes) == 1 and ax.figure.get_layout_engine() is None:
        ax.figure.set_layout_engine("constrained")


Node = tuple[str, int, list]                    # a category, its count and its children, for treemap and sunburst


def _check_tree(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if z is not None and y is None:
        raise ValueError("a hierarchy goes x, then y, then z: give y before z")


def _hierarchy(x: ArrayLike, y: ArrayLike | None, z: ArrayLike | None) -> list[Node]:
    """The rows counted down the levels x > y > z, largest first at each level; empty branches are left out."""
    levels = [v for v in (x, y, z) if v is not None]
    if len({np.size(v) for v in levels}) > 1:
        raise ValueError(f"the levels have different lengths: {[np.size(v) for v in levels]}")
    split_levels = [_levels(v) for v in levels]

    def grow(mask: np.ndarray, depth: int) -> list[Node]:
        if depth == len(levels):
            return []
        names, masks = split_levels[depth]
        nodes = [(n, int((mask & m).sum()), grow(mask & m, depth + 1)) for n, m in zip(names, masks, strict=True)]
        return sorted((nd for nd in nodes if nd[1]), key=lambda nd: -nd[1])
    return grow(np.ones(np.size(x), dtype=bool), 0)


def _depth(nodes: list[Node]) -> int:
    return 1 + max(_depth(c) for _, _, c in nodes) if nodes else 0


def _worst(row: list[float], side: float) -> float:
    """The worst aspect ratio of a row of areas laid along a side: what squarify keeps low."""
    total = sum(row)
    return max(max(side ** 2 * a / total ** 2, total ** 2 / (side ** 2 * a)) for a in row)


def squarify(sizes: list[float], x: float, y: float, w: float, h: float) -> list[tuple[float, float, float, float]]:
    """Rectangles (x, y, w, h) tiling the box, one per size (largest first) with areas in proportion, laid in rows
    along the shorter side as long as that keeps them closer to squares (Bruls, Huizing and van Wijk, 2000)."""
    total = sum(sizes)
    areas = [s * w * h / total for s in sizes]
    rects = []
    i = 0
    while i < len(areas):
        side = min(w, h)
        row = [areas[i]]
        while i + len(row) < len(areas) and _worst([*row, areas[i + len(row)]], side) <= _worst(row, side):
            row.append(areas[i + len(row)])
        thick = sum(row) / side
        pos = 0.0
        for a in row:                           # the row fills the short side; it eats into the long one
            long = a / thick
            rects.append((x, y + pos, thick, long) if w >= h else (x + pos, y + h - thick, long, thick))
            pos += long
        if w >= h:
            x, w = x + thick, w - thick
        else:
            h -= thick
        i += len(row)
    return rects


def _tree_names(x: ArrayLike, y: ArrayLike | None, z: ArrayLike | None) -> str:
    return " > ".join(label(v, n) for v, n in ((x, "x"), (y, "y"), (z, "z")) if v is not None)


@kind("treemap", flags=("ann",), needs_y=False, check=_check_tree)
def treemap(ax: Axes, x: ArrayLike, y: ArrayLike | None, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """A hierarchy of up to three categoricals as nested rectangles with areas by count: x's categories tile the
    axes, each split by y, and by z: kx.plot(df, 'pclass', 'sex', 'survived', spec='treemap'). Colours follow x,
    lighter at each level down; missing values count as 'NaN'. ann = each tile's path below x (its colour) and
    count, where there is room.
    kw: the tiles (Rectangle)."""
    nodes = _hierarchy(x, y, z)
    depth = _depth(nodes)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    face = plt.rcParams["axes.facecolor"]

    def draw(level: list[Node], box: tuple[float, float, float, float], d: int, base: str | None,
             path: tuple[str, ...]) -> None:
        for i, ((name, n, children), rect) in enumerate(zip(level, squarify([nd[1] for nd in level], *box),
                                                              strict=True)):
            colour = cycle[i % len(cycle)] if base is None else base
            fill = _shade(colour, d, depth)
            ax.add_patch(mpl.patches.Rectangle(rect[:2], *rect[2:], **{"facecolor": fill, "edgecolor": face,
                                                                       "linewidth": 2.5 - d, **kw}))
            if d == 0:
                ax.add_patch(mpl.patches.Rectangle((0, 0), 0, 0, facecolor=fill, label=name))   # legend handle
            if children:
                draw(children, rect, d + 1, colour, (*path, name))
            elif "ann" in flags and rect[2] > 0.08 and rect[3] > 0.07:   # the path below x: colour says x
                text = " / ".join((*path, name)[1:] if d else (name,))
                ax.text(rect[0] + rect[2] / 2, rect[1] + rect[3] / 2, f"{text}\n{n:g}", ha="center",
                        va="center", color=ink(fill), fontsize="small")
    draw(nodes, (0.0, 0.0, 1.0, 1.0), 0, None, ())
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    _legend_beside(ax, _tree_names(x, y, z))


@kind("sunburst", flags=("ann",), needs_y=False, check=_check_tree, projection="polar")
def sunburst(ax: Axes, x: ArrayLike, y: ArrayLike | None, z: ArrayLike | None, flags: set[str],
             hole: float = 0.3, **kw: Any) -> None:
    """The treemap's hierarchy as rings: x innermost, each segment's angle its count, y and z split it outward,
    starting at 12 o'clock and going clockwise, with the total in the hole (hole=, 0.3 of the radius).
    ann = the name in each segment with room. kw: the segments (ax.bar)."""
    if not 0 <= hole < 1:
        raise ValueError(f"hole= is a share of the radius between 0 and 1, got {hole}")
    nodes = _hierarchy(x, y, z)
    depth = _depth(nodes)
    ring = (1 - hole) / depth
    total = sum(nd[1] for nd in nodes)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    face = plt.rcParams["axes.facecolor"]

    def draw(level: list[Node], start: float, d: int, base: str | None) -> None:
        for i, (name, n, children) in enumerate(level):
            span = 2 * np.pi * n / total
            colour = cycle[i % len(cycle)] if base is None else base
            fill = _shade(colour, d, depth)
            bottom = hole + d * ring
            ax.bar(start + span / 2, ring, width=span, bottom=bottom, **{"color": fill, "edgecolor": face,
                                                                         "linewidth": 1, **kw})
            if d == 0:
                ax.bar(0, 0, width=0, color=fill, label=name)                   # legend handle
            if "ann" in flags and span * (bottom + ring / 2) > 0.25:
                ax.text(start + span / 2, bottom + ring / 2, str(name), ha="center", va="center",
                        color=ink(fill), fontsize="small")
            draw(children, start, d + 1, colour)
            start += span
    draw(nodes, 0.0, 0, None)
    ax.set_theta_offset(np.pi / 2)              # from 12 o'clock, clockwise
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    if hole:
        ax.text(0, 0, f"{total:g}", ha="center", va="center", fontsize="x-large")
    _legend_beside(ax, _tree_names(x, y, z))
