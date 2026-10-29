"""Table and 2-D kinds: heat, crosstab, missing, nullmap, hexbin, pair, cluster, mosaic."""
from collections.abc import Callable

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from helpers import PASSENGERS
from matplotlib.axes import Axes
from matplotlib.colors import to_hex

import kx
import kx._stats

# ---------- table kinds ----------
HEAT = pd.DataFrame({"a": [1.0, 2, 3], "b": [4.0, 5, 6], "c": [9.0, 7, 8]}, index=["r1", "r2", "r3"])


def test_heat_draws_the_frame_with_its_labels() -> None:
    ax = kx.plot(HEAT, "heat")
    im = ax.images[0]
    assert im.get_array().shape == HEAT.shape
    assert (im.get_array() == HEAT.to_numpy()).all()
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["r1", "r2", "r3"]
    assert len(ax.figure.axes) == 2             # the colorbar


def test_heat_takes_a_2d_array() -> None:
    ax = kx.plot(np.arange(6).reshape(2, 3), "heat")
    assert ax.images[0].get_array().shape == (2, 3)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "1", "2"]


def test_heat_corr_is_symmetric_from_minus_one_to_one() -> None:
    p = kx.load("passengers")
    ax = kx.plot(p, "heat-corr")
    r = ax.images[0].get_array()
    assert r.shape == (4, 4)                    # the numeric columns only
    assert np.allclose(r, r.T)
    assert np.allclose(np.diag(r), 1)
    assert ax.images[0].get_clim() == (-1, 1)
    assert ax.images[0].get_cmap().name == "rdbu"
    assert ax.figure.axes[1].get_ylabel() == "r"
    assert kx.plot(p, "heat-corr-sym").images[0].get_clim() == (-1, 1)


def test_heat_ann_writes_every_cell_in_contrasting_ink() -> None:
    ax = kx.plot(HEAT, "heat-ann", cmap="greys")
    texts = ax.texts
    assert len(texts) == HEAT.size
    assert texts[0].get_text() == "1"
    assert texts[0].get_color() == "black"      # lowest value: white cell on greys
    assert texts[2].get_color() == "white"      # 9, the highest: black cell


def test_heat_ann_keeps_three_digits_and_corr_two_decimals() -> None:
    texts = [t.get_text() for t in kx.plot(HEAT * 40, "heat-ann").texts]
    assert texts[:3] == ["40", "160", "360"]    # not 1.6e+02
    corr = [t.get_text() for t in kx.plot(HEAT, "heat-corr-ann").texts]
    assert corr[0] == "1.00"
    assert all(len(t.lstrip("-")) == 4 for t in corr)


def test_heat_colour_flags() -> None:
    lo, hi = kx.plot(HEAT - 5, "heat-sym").images[0].get_clim()
    assert lo == -hi == -4
    assert isinstance(kx.plot(HEAT, "heat-logc").images[0].norm, mpl.colors.LogNorm)
    assert len(kx.plot(HEAT, "heat-nocb").figure.axes) == 1


@pytest.mark.parametrize(("spec", "data", "match"), [
    ("heat-corr-logc", HEAT, "do not go together"),
    ("heat-logc", HEAT - 5, "needs z > 0"),
    ("heat", HEAT.assign(g=["x", "y", "z"]), r"columns \['g'\] are not"),
])
def test_heat_errors(spec: str, data: pd.DataFrame, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        kx.plot(data, spec)


def test_crosstab_counts_match_pandas() -> None:
    ax = kx.plot(PASSENGERS, "sex", "pclass", "crosstab-ann")
    want = pd.crosstab(PASSENGERS.pclass, PASSENGERS.sex)
    assert (ax.images[0].get_array() == want.to_numpy()).all()
    assert [t.get_text() for t in ax.get_xticklabels()] == ["female", "male"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["1", "2", "3"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("sex", "pclass")
    assert [t.get_text() for t in ax.texts] == [str(v) for v in want.to_numpy().ravel()]
    assert ax.figure.axes[1].get_ylabel() == "count"


def test_crosstab_norm_gives_shares_within_each_x() -> None:
    ax = kx.plot(PASSENGERS, "pclass", "deck", "crosstab-norm-ann")
    shares = ax.images[0].get_array()
    assert np.allclose(shares.sum(axis=0), 1)
    assert all(t.get_text().endswith("%") for t in ax.texts)
    cb = ax.figure.axes[1]
    assert cb.get_ylabel() == "share"
    assert isinstance(cb.yaxis.get_major_formatter(), mpl.ticker.PercentFormatter)


def test_crosstab_logc_leaves_empty_pairs_blank() -> None:
    x, y = ["a", "a", "b"], ["u", "v", "u"]     # (b, v) never occurs
    ax = kx.plot(x, y, "crosstab-logc-ann-nocb")
    assert np.ma.is_masked(ax.images[0].get_array()[1, 1])
    assert len(ax.texts) == 3
    assert len(ax.figure.axes) == 1


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot(PASSENGERS, "sex", spec="crosstab"), "needs both x and y"),
    (lambda: kx.plot(PASSENGERS, "sex", "pclass", "port", "crosstab"), "takes no z"),
])
def test_crosstab_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_missing_shows_columns_with_nans_most_missing_on_top() -> None:
    ax = kx.plot(PASSENGERS, "missing")
    assert [t.get_text() for t in ax.get_yticklabels()] == ["deck", "age", "port"]
    assert ax.yaxis_inverted()                      # deck, the first label, is drawn on top
    want = PASSENGERS.isna().mean()
    assert [p.get_width() for p in ax.patches] == pytest.approx([want.deck, want.age, want.port])
    assert isinstance(ax.xaxis.get_major_formatter(), mpl.ticker.PercentFormatter)


def test_missing_all_and_ann() -> None:
    ax = kx.plot(PASSENGERS, "missing-all-ann")
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == PASSENGERS.shape[1]
    assert labels[:3] == ["deck", "age", "port"]
    assert labels[3:] == ["fare", "sex", "pclass", "ticket_type", "survived"]   # ties keep column order
    texts = [t.get_text() for t in ax.texts]
    assert texts[0] == f"{PASSENGERS.deck.isna().mean():.1%}"
    assert all(t.endswith("%") for t in texts)


def test_missing_without_nans_says_so() -> None:
    ax = kx.plot(HEAT, "missing-ann")
    assert not ax.patches
    assert [t.get_text() for t in ax.texts] == ["no missing values"]


def _bars_and_texts(ax: Axes) -> tuple:
    return ([t.get_text() for t in ax.get_yticklabels()], [p.get_width() for p in ax.patches],
            [t.get_text() for t in ax.texts])


def test_missing_shortcut_is_the_long_form() -> None:
    assert _bars_and_texts(kx.missing(PASSENGERS)) == _bars_and_texts(kx.plot(PASSENGERS, "missing-ann"))
    assert "missing" in kx.__all__


def test_missing_shortcut_passes_theme_flags_and_keywords() -> None:
    ax = kx.missing(PASSENGERS, "dark-all-wrap", title="gaps", color="red")
    assert len(ax.patches) == PASSENGERS.shape[1]
    assert ax.get_title() == "gaps"
    assert to_hex(ax.patches[0].get_facecolor()) == "#ff0000"
    assert to_hex(ax.get_facecolor()) == "#111418"          # the dark theme, for this plot only
    with pytest.raises(ValueError, match="unknown tokens"):
        kx.missing(PASSENGERS, "dakr")


def test_nullmap_image_is_where_the_nans_are() -> None:
    ax = kx.plot(PASSENGERS, "nullmap")
    im = ax.images[0]
    assert (im.get_array() == PASSENGERS.isna().to_numpy()).all()
    assert [t.get_text() for t in ax.get_xticklabels()] == list(PASSENGERS.columns)
    assert len(ax.get_yticks()) <= 12           # row positions, not 500 ticks
    assert im.get_interpolation() == "nearest"
    assert ax.get_aspect() == "auto"


def test_nullmap_sort_puts_most_missing_rows_first_and_groups_patterns() -> None:
    gaps = kx.plot(PASSENGERS, "nullmap-sort").images[0].get_array().astype(bool)
    counts = gaps.sum(axis=1)
    assert (np.diff(counts) <= 0).all()
    assert sorted(map(tuple, gaps)) == sorted(map(tuple, PASSENGERS.isna().to_numpy()))   # the same rows
    patterns = [tuple(r) for r in gaps]
    runs = [p for i, p in enumerate(patterns) if i == 0 or p != patterns[i - 1]]
    assert len(runs) == len(set(patterns))      # each pattern is one block


def test_nullmap_colours_follow_theme_and_palette() -> None:
    with kx.context("dark", palette=["#e9c46a", "#264653"]):
        ax = kx.plot(PASSENGERS, "nullmap")
        cmap = ax.images[0].get_cmap()
        assert to_hex(cmap(0.0)) == to_hex(ax.get_facecolor())     # present: the background
        assert to_hex(cmap(1.0)) == "#e9c46a"                       # missing: the first palette colour
    assert ax.images[0].get_clim() == (0, 1)


def test_nullmap_without_nans_is_all_background() -> None:
    im = kx.plot(HEAT, "nullmap").images[0]
    assert not im.get_array().any()
    assert to_hex(im.to_rgba(im.get_array())[0, 0]) == to_hex(im.get_cmap()(0.0))


CLUMPS = pd.DataFrame({"x": [0, 0, 0, 10, 10, np.nan], "y": [0, 0, 0, 10, 10, 5],
                       "heat": [1, 2, 3, 4, 6, 100]})       # two clumps of points, and a row with a NaN


def test_hexbin_counts_points_per_hexagon() -> None:
    cloud = kx.load("cloud")
    ax = kx.plot(cloud, "x", "y", "hexbin", gridsize=10)
    assert ax.collections[0].get_array().sum() == len(cloud)
    assert ax.figure.axes[1].get_ylabel() == "count"
    counts = kx.plot(CLUMPS, "x", "y", "hexbin").collections[0].get_array()
    assert sorted(counts) == [2, 3]                 # the NaN row is left out, empty cells are not drawn


def test_hexbin_z_is_the_mean_per_hexagon() -> None:
    ax = kx.plot(CLUMPS, "x", "y", "heat", "hexbin")
    assert sorted(ax.collections[0].get_array()) == [2, 5]
    assert ax.figure.axes[1].get_ylabel() == "heat"


def test_hexbin_colour_flags() -> None:
    cloud = kx.load("cloud")
    ax = kx.plot(cloud, "x", "y", "hexbin-logc-nocb")
    assert isinstance(ax.collections[0].norm, mpl.colors.LogNorm)
    assert ax.collections[0].get_array().sum() == len(cloud)       # counts, not log10 counts
    assert len(ax.figure.axes) == 1
    lo, hi = kx.plot(cloud, "x", "y", "z", "hexbin-sym").collections[0].get_clim()
    assert lo == -hi


@pytest.mark.parametrize(("spec", "cols", "match"), [
    ("hexbin-logc", ("x", "y", "z"), "needs z > 0"),
    ("hexbin-sym", ("x", "y"), "counts are never below 0"),
    ("hexbin", ("x",), "needs both x and y"),
])
def test_hexbin_errors(spec: str, cols: tuple[str, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        kx.plot(kx.load("cloud"), *cols, spec)


# ---------- pair ----------
EMBEDDING = kx.load("embedding")
FOUR = ["x", "y", "size", "f0"]


def _offsets(ax: Axes) -> list[np.ndarray]:
    return [c.get_offsets() for c in ax.collections]


def test_pair_scatters_off_the_diagonal_one_colour_per_group() -> None:
    axs = kx.plot(EMBEDDING, "pair", by="label", cols=FOUR)
    assert isinstance(axs, np.ndarray)
    assert axs.shape == (16,)
    groups = sorted(EMBEDDING.label.unique())
    below = axs[2 * 4 + 1]                                   # row size, column y
    assert len(below.collections) == len(groups)
    alpha = EMBEDDING[EMBEDDING.label == groups[0]]
    assert np.array_equal(_offsets(below)[0], alpha[["y", "size"]].to_numpy())
    assert len({to_hex(c.get_facecolor()[0]) for c in below.collections}) == len(groups)
    for i in range(4):
        assert not axs[i * 5].collections                     # the diagonal holds densities, not points


def test_pair_diagonal_density_per_group_on_an_inset_sharing_x() -> None:
    axs = kx.plot(EMBEDDING, "pair", by="label", cols=FOUR)
    diag = axs[0]
    inset = diag.child_axes[0]
    assert len(inset.lines) == EMBEDDING.label.nunique()
    assert inset.get_shared_x_axes().joined(inset, diag)
    lo, hi = diag.get_xlim()
    assert all(lo <= ln.get_xdata().min() and ln.get_xdata().max() <= hi for ln in inset.lines)
    assert inset.get_ylim()[0] == 0
    assert diag.get_ylim() == axs[1].get_ylim()                # the diagonal keeps its row's y, for the labels


def test_pair_shares_x_down_columns_and_y_along_rows_labels_outside() -> None:
    axs = kx.plot(EMBEDDING, "pair", cols=FOUR).reshape(4, 4)
    for r in range(4):
        for c in range(4):
            assert axs[r, c].get_shared_x_axes().joined(axs[r, c], axs[0, c])
            assert axs[r, c].get_shared_y_axes().joined(axs[r, c], axs[r, 0 if r else 1])
    assert [a.get_xlabel() for a in axs[-1]] == FOUR
    assert [a.get_ylabel() for a in axs[:, 0]] == FOUR
    assert not any(a.get_xlabel() or a.get_ylabel() for a in axs[:-1, 1:].ravel())


def test_pair_takes_numeric_columns_and_legends_groups_beside() -> None:
    axs = kx.plot(PASSENGERS, "pair", by="sex", title="passengers")
    assert [a.get_xlabel() for a in axs[-4:]] == ["age", "fare", "pclass", "survived"]
    legend = axs[3].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["female", "male"]
    assert legend.get_title().get_text() == "sex"
    assert all(h.get_alpha() == 1 for h in legend.legend_handles)
    fig = axs[0].figure
    assert fig._suptitle.get_text() == "passengers"
    assert fig.get_figwidth() > fig.get_figheight() >= 4 * 1.8    # sized to the matrix, room for the legend
    assert kx.plot(PASSENGERS, "pair")[3].get_legend() is None


def test_pair_in_a_grid_cell_keeps_the_figure() -> None:
    left, right = kx.grid(2)
    fig = left.figure
    size = tuple(fig.get_size_inches())
    axs = kx.plot(EMBEDDING, "pair", cols=["x", "y"], ax=right)
    assert tuple(fig.get_size_inches()) == size
    assert len(fig.axes) == 1 + 4
    assert all(a.get_subplotspec().get_topmost_subplotspec() == right.get_subplotspec() for a in axs)


def test_pair_constant_column_draws_no_density() -> None:
    frame = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [5.0, 5, 5, 5]})
    axs = kx.plot(frame, "pair")
    assert len(axs[0].child_axes[0].lines) == 1
    assert not axs[3].child_axes[0].lines


@pytest.mark.parametrize(("frame", "kw", "match"), [
    (EMBEDDING, {"cols": ["x", "nope"]}, r"cols= names \['nope'\]"),
    (EMBEDDING, {"cols": ["x", "label"]}, r"cols= \['label'\] are not"),
    (EMBEDDING[["x", "label"]], {}, "at least 2 numeric columns"),
    (EMBEDDING, {}, "13 numeric columns is 169 panels"),
])
def test_pair_errors(frame: pd.DataFrame, kw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        kx.plot(frame, "pair", **kw)


# ---------- cluster ----------
MEANS = EMBEDDING.groupby("label")[[f"f{k}" for k in range(6)]].mean()


def _labels(axis: mpl.axis.Axis) -> list[str]:
    return [t.get_text() for t in axis.get_majorticklabels()]


def test_cluster_orders_rows_and_columns_by_their_linkage() -> None:
    *_, cells = kx.plot(MEANS, "cluster")
    values = MEANS.to_numpy()
    rows = kx._stats.leaves(kx._stats.linkage(values))
    cols = kx._stats.leaves(kx._stats.linkage(values.T))
    assert _labels(cells.yaxis) == [MEANS.index[i] for i in rows]
    assert _labels(cells.xaxis) == [MEANS.columns[j] for j in cols]
    assert np.array_equal(cells.images[0].get_array(), values[np.ix_(rows, cols)])
    assert cells.yaxis.get_ticks_position() == "right"           # the left side is the tree's


def test_cluster_trees_have_one_join_per_merge_on_the_cells() -> None:
    top, left, cells = kx.plot(MEANS, "cluster")
    (cols,), (rows,) = top.collections, left.collections
    assert len(cols.get_segments()) == MEANS.shape[1] - 1
    assert len(rows.get_segments()) == len(MEANS) - 1
    ends = np.concatenate(cols.get_segments())
    assert set(ends[ends[:, 1] == 0, 0]) == set(range(MEANS.shape[1]))   # every leaf sits on a column centre
    ends = np.concatenate(rows.get_segments())
    assert set(ends[ends[:, 0] == 0, 1]) == set(range(len(MEANS)))
    assert top.get_shared_x_axes().joined(top, cells)
    assert left.get_shared_y_axes().joined(left, cells)
    assert not top.axison
    assert not left.axison


def test_cluster_corr_one_symmetric_order_with_r_colorbar() -> None:
    frame = EMBEDDING.select_dtypes("number")
    *_, cells = kx.plot(frame, "cluster-corr-ann")
    r = frame.corr().to_numpy()
    order = kx._stats.leaves(kx._stats.linkage(1 - np.abs(r), distances=True))
    assert _labels(cells.xaxis) == _labels(cells.yaxis) == [frame.columns[i] for i in order]
    shown = cells.images[0].get_array()
    assert np.allclose(shown, shown.T)
    assert cells.images[0].norm.vmin == -1
    assert len(cells.texts) == frame.shape[1] ** 2
    bar = cells.images[0].colorbar
    assert bar.ax.get_ylabel() == "r"
    assert bar.ax.get_position().x1 < cells.get_position().x0   # in the corner, left of the cells


def test_cluster_nocb_and_method_and_title_over_the_block() -> None:
    top, _, cells = kx.plot(MEANS, "cluster-nocb", method="ward", title="means")
    assert len(cells.figure.axes) == 3
    assert top.get_title() == ""
    assert cells.figure._suptitle.get_text() == "means"
    ward = kx._stats.leaves(kx._stats.linkage(MEANS.to_numpy(), "ward"))
    assert _labels(cells.yaxis) == [MEANS.index[i] for i in ward]
    _, grid_cell = kx.grid(2)
    axs = kx.plot(MEANS, "cluster", title="cell", ax=grid_cell)
    assert axs[0].get_title() == "cell"                          # in a bigger figure: over the top tree


def test_cluster_one_row_keeps_its_place() -> None:
    top, left, cells = kx.plot(MEANS.iloc[:1], "cluster")
    assert _labels(cells.yaxis) == [MEANS.index[0]]
    assert not left.collections
    assert len(top.collections[0].get_segments()) == MEANS.shape[1] - 1


@pytest.mark.parametrize(("frame", "kw", "match"), [
    (EMBEDDING, {}, r"columns \['label', 'group'\] are not"),
    (MEANS.where(MEANS > 0), {}, "fill or drop the NaNs"),
    (MEANS, {"method": "median"}, "unknown method"),
])
def test_cluster_errors(frame: pd.DataFrame, kw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        kx.plot(frame, "cluster", **kw)


# ---------- mosaic ----------
def _tiles(ax: Axes) -> list[mpl.patches.Rectangle]:
    """The drawn tiles, without the empty legend handles."""
    return [p for p in ax.patches if p.get_height() > 0 or p.get_width() > 0]


def test_mosaic_widths_follow_x_and_heights_y_within_x() -> None:
    x = ["a"] * 6 + ["b"] * 2
    y = ["u", "u", "v", "v", "v", "v", "u", "v"]
    ax = kx.plot(x, y, "mosaic")
    a_u, a_v, b_u, b_v = _tiles(ax)
    gap = kx.kinds.table.GAP
    assert a_u.get_width() / b_u.get_width() == pytest.approx(3)          # a is 6 rows, b is 2
    assert a_u.get_width() + b_u.get_width() == pytest.approx(1 - gap)
    assert a_v.get_height() / a_u.get_height() == pytest.approx(2)        # within a: 4 v, 2 u
    assert b_v.get_height() == pytest.approx(b_u.get_height())
    assert a_u.get_y() > a_v.get_y()                                      # the first y on top
    assert to_hex(a_u.get_facecolor()) == to_hex(b_u.get_facecolor()) != to_hex(a_v.get_facecolor())
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["u", "v"]


def test_mosaic_tile_areas_are_counts() -> None:
    ax = kx.plot(PASSENGERS, "pclass", "sex", "mosaic-ann")
    counts = pd.crosstab(PASSENGERS.pclass, PASSENGERS.sex).to_numpy().ravel()
    areas = np.array([p.get_width() * p.get_height() for p in _tiles(ax)])
    assert areas / areas.sum() == pytest.approx(counts / counts.sum(), rel=0.1)   # gaps bend it a little
    assert [t.get_text() for t in ax.texts] == [f"{c:g}" for c in counts]
    assert all(t.get_color() in ("black", "white") for t in ax.texts)
    assert ax.get_xlabel() == "pclass"
    assert ax.get_legend().get_title().get_text() == "sex"


def test_mosaic_z_splits_tiles_in_shades_of_the_y_colour() -> None:
    ax = kx.plot(PASSENGERS, "pclass", "sex", "survived", spec="mosaic")
    tiles = _tiles(ax)
    assert len(tiles) == 3 * 2 * 2
    first, second = tiles[:2]                                             # pclass 1, female, survived 0 and 1
    assert first.get_y() == second.get_y()
    assert second.get_x() > first.get_x()
    want = pd.crosstab([PASSENGERS.pclass, PASSENGERS.sex], PASSENGERS.survived).iloc[0]
    assert first.get_width() / second.get_width() == pytest.approx(want[0] / want[1])
    assert first.get_facecolor() != second.get_facecolor()
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["female / 0", "female / 1", "male / 0",
                                                                   "male / 1"]
    assert ax.get_legend().get_title().get_text() == "sex / survived"


def test_mosaic_counts_nan_and_leaves_slivers_unlabelled() -> None:
    ax = kx.plot(PASSENGERS, "port", "sex", "survived", spec="mosaic-ann")
    assert [t.get_text() for t in ax.get_xticklabels()][-1] == "NaN"
    assert all(p.get_width() >= 0 for p in ax.patches)                   # a thin column never goes negative
    labelled = {t.get_text() for t in ax.texts}
    assert {"144", "73", "10"} <= labelled
    filled = [p for p in _tiles(ax) if p.get_width() * p.get_height() > 0]
    assert len(ax.texts) < len(filled)                                   # the NaN column's slivers stay bare


def test_mosaic_alone_keeps_its_legend_and_in_a_grid_leaves_the_layout() -> None:
    ax = kx.plot(["a", "b"], ["u", "v"], "mosaic")
    assert ax.figure.get_layout_engine() is not None
    left, _ = kx.grid(2)
    engine = left.figure.get_layout_engine()
    kx.plot(["a", "b"], ["u", "v"], "mosaic", ax=left)
    assert left.figure.get_layout_engine() is engine


def test_mosaic_errors() -> None:
    with pytest.raises(ValueError, match="x has 2 values but y has 3"):
        kx.plot(["a", "b"], ["u", "v", "w"], "mosaic")
    with pytest.raises(ValueError, match="y has 2 values but z has 1"):
        kx.plot(["a", "b"], ["u", "v"], ["p"], "mosaic")
