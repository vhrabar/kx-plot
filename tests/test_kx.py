import re
from collections.abc import Callable, Iterator
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from numpy.typing import ArrayLike

import kx
import kx._stats

X = np.linspace(1, 10, 20)
Y = np.sin(X) + 2
Z = np.cos(X) + 2


# ---------- spec parsing ----------
def test_parse_order_does_not_matter() -> None:
    assert kx.parse("scatter-dark-grid-leg") == kx.parse("leg-grid-dark-scatter")


def test_parse_defaults_to_line_and_no_theme() -> None:
    assert kx.parse("grid") == ("line", None, {"grid"})


def test_spec_fields_and_canonical_slug() -> None:
    spec = kx.parse("leg-grid-dark-scatter")
    assert (spec.kind, spec.theme, spec.flags) == ("scatter", "dark", frozenset({"grid", "leg"}))
    kind, theme, flags = spec
    assert (kind, theme, flags) == spec
    assert str(spec) == spec.slug == "scatter-dark-grid-leg"
    assert str(kx.parse("grid")) == "line-grid"
    assert kx.parse(str(spec)) == spec
    assert kx.parse(spec) is spec


def test_plot_takes_a_spec() -> None:
    assert len(kx.plot(X, Y, kx.parse("bar")).patches) == len(X)
    assert len(kx.plot(X, Y, spec=kx.parse("bar")).patches) == len(X)
    assert len(kx.plot(DF, "t", "speed", kx.parse("bar")).patches) == len(X)


@pytest.mark.parametrize(("spec", "hint"), [
    ("line-lgo", "Did you mean 'leg' or 'legout' for 'lgo'?"),
    ("scater-drak", "'dark' for 'drak'"),
    ("line--dark", "a '-' too many"),
])
def test_parse_hints_close_tokens(spec: str, hint: str) -> None:
    with pytest.raises(ValueError, match=re.escape(hint)):
        kx.parse(spec)


@pytest.mark.parametrize(("spec", "match"), [
    ("line-nope", "unknown tokens"),
    ("line--dark", "unknown tokens"),
    ("line-scatter", "more than one kind"),
    ("dark-light", "more than one theme"),
])
def test_parse_rejects_bad_specs(spec: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        kx.parse(spec)


# ---------- each kind, with and without z ----------
@pytest.mark.parametrize("with_z", [False, True])
def test_line(with_z: bool) -> None:
    ax = kx.plot(X, Y, Z if with_z else None, "line")
    assert len(ax.lines) == (2 if with_z else 1)


@pytest.mark.parametrize("with_z", [False, True])
def test_scatter_colorbar_only_with_z(with_z: bool) -> None:
    ax = kx.plot(X, Y, Z if with_z else None, "scatter")
    assert len(ax.collections) == 1
    assert len(ax.figure.axes) == (2 if with_z else 1)


def test_scatter_text_z_becomes_legend() -> None:
    labels = np.where(Z > 2, "high", "low")
    ax = kx.plot(X, Y, pd.Series(labels, name="level"), "scatter")
    assert len(ax.figure.axes) == 1                    # no colorbar
    assert len(ax.collections) == 2
    leg = ax.get_legend()
    assert leg.get_title().get_text() == "level"
    assert [t.get_text() for t in leg.get_texts()] == ["high", "low"]
    assert sum(len(c.get_offsets()) for c in ax.collections) == len(X)


def test_scatter_categorical_keeps_category_order_and_bool_is_category() -> None:
    cats = pd.Series(pd.Categorical(np.where(Z > 2, "b", "a"), categories=["b", "a"]))
    assert [t.get_text() for t in kx.plot(X, Y, cats, "scatter").get_legend().get_texts()] == ["b", "a"]
    assert [t.get_text() for t in kx.plot(X, Y, Z > 2, "scatter-leg").get_legend().get_texts()] == ["False", "True"]


@pytest.mark.parametrize("with_z", [False, True])
def test_step(with_z: bool) -> None:
    ax = kx.plot(X, Y, Z if with_z else None, "step")
    assert len(ax.lines) == (2 if with_z else 1)
    assert all(ln.get_drawstyle() == "steps-mid" for ln in ax.lines)


def test_area_overlaps_by_default() -> None:
    ax = kx.plot(X, Y, Z, "area")
    assert len(ax.lines) == 2
    assert len(ax.collections) == 2


def test_area_stack() -> None:
    ax = kx.plot(X, Y, Z, "area-stack-leg")
    assert len(ax.collections) == 2
    assert len(ax.lines) == 0
    top = ax.collections[1].get_paths()[0].vertices[:, 1].max()
    assert top == pytest.approx((Y + Z).max())


def test_area_norm_is_100_percent_stack() -> None:
    ax = kx.plot(X, Y, Z, "area-norm")
    top = ax.collections[1].get_paths()[0].vertices[:, 1].max()
    assert top == pytest.approx(100)
    assert ax.get_ylim() == (0, 100)
    assert ax.yaxis.get_major_formatter()(50, 0) == "50%"


def test_area_norm_legend_sits_outside() -> None:
    leg = kx.plot(X, Y, Z, "area-norm-leg").get_legend()
    assert leg.get_bbox_to_anchor().bounds[0] > leg.axes.bbox.bounds[0]    # anchored right of the axes


def test_band_is_y_plus_minus_z() -> None:
    ax = kx.plot(X, Y, np.full_like(Y, 0.5), "band-leg")
    ys = ax.collections[0].get_paths()[0].vertices[:, 1]
    assert ys.min() == pytest.approx(Y.min() - 0.5)
    assert ys.max() == pytest.approx(Y.max() + 0.5)
    assert to_hex(ax.collections[0].get_facecolor()[0]) == to_hex(ax.lines[0].get_color())
    assert len(ax.get_legend().get_texts()) == 2


@pytest.mark.parametrize(("spec", "call", "match"), [
    ("band", lambda s: kx.plot(X, Y, spec=s), "needs z"),
    ("area-norm", lambda s: kx.plot(X, Y, spec=s), "needs two series"),
    ("line-stack", lambda s: kx.plot(X, Y, Z, s), "only works with kind"),
    ("bar-norm", lambda s: kx.plot(X, Y, Z, s), "only works with kind"),
    ("line-cum", lambda s: kx.plot(X, Y, Z, s), "only works with kind"),
    ("kde-stack", lambda s: kx.plot(X, Y, Z, s), "only works with kind"),
    ("hist", lambda s: kx.plot(X, spec=s, by=X > 5), "by= only works with kind"),
    ("kde", lambda s: kx.plot(X, Y, spec=s, by=X > 5), "pass only x"),
    ("kde", lambda s: kx.plot(X, spec=s, by=[1, 2]), "has 2 values but x has 20"),
    ("kde", lambda s: kx.plot(np.ones(5), spec=s), "at least 2 distinct"),
    ("ecdf", lambda s: kx.plot(X, Y, spec=s, by=X > 5), "pass only x"),
    ("box", lambda s: kx.plot([np.nan], spec=s), "needs a finite value"),
    ("violin", lambda s: kx.plot(np.ones(5), spec=s), "at least 2 distinct"),
    ("strip", lambda s: kx.plot([np.nan], spec=s), "needs a finite value"),
    ("ridge", lambda s: kx.plot(np.ones(5), spec=s), "at least 2 distinct"),
    ("qq", lambda s: kx.plot(np.ones(5), spec=s), "at least 2 distinct"),
    ("ridge-ann", lambda s: kx.plot(X, spec=s), "only works with kind"),
    ("box-stack", lambda s: kx.plot(X, spec=s), "only works with kind"),
])
def test_new_kind_errors(spec: str, call: Callable[[str], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call(spec)


@pytest.mark.parametrize("with_z", [False, True])
def test_bar_groups_and_category_ticks(with_z: bool) -> None:
    cats = ["a", "b", "c"]
    ax = kx.plot(cats, [1, 2, 3], [3, 2, 1] if with_z else None, "bar")
    assert len(ax.patches) == (6 if with_z else 3)
    assert [t.get_text() for t in ax.get_xticklabels()] == cats


@pytest.mark.parametrize("name", ["bar", "barh"])
def test_bar_stack_puts_z_on_top_of_y(name: str) -> None:
    ax = kx.plot(["a", "b", "c"], [1, 2, 3], [3, 2, 1], f"{name}-stack-leg")
    _, z_bars = ax.containers
    start, length, thickness = ("get_x", "get_width", "get_height") if name == "barh" \
        else ("get_y", "get_height", "get_width")
    assert [getattr(p, start)() for p in z_bars] == [1, 2, 3]
    assert [getattr(p, start)() + getattr(p, length)() for p in z_bars] == [4, 4, 4]
    assert {getattr(p, thickness)() for p in ax.patches} == {0.8}          # full width, not split
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["y", "z"]


@pytest.mark.parametrize("with_z", [False, True])
def test_barh_categories_top_to_bottom(with_z: bool) -> None:
    cats = ["a", "b", "c"]
    ax = kx.plot(cats, [1, 2, 3], [3, 2, 1] if with_z else None, "barh")
    assert len(ax.patches) == (6 if with_z else 3)
    assert [p.get_width() for p in ax.containers[0]] == [1, 2, 3]
    assert [t.get_text() for t in ax.get_yticklabels()] == cats
    assert ax.yaxis_inverted()


@pytest.mark.parametrize("arrays", [1, 2, 3])
def test_hist_overlays_each_array(arrays: int) -> None:
    ax = kx.plot(*[X, Y, Z][:arrays], spec="hist")
    assert len(ax.containers) == arrays


def test_hist_shares_bins_across_arrays() -> None:
    ax = kx.plot(np.arange(10.0), np.arange(5.0, 20.0), spec="hist")
    lefts = [[p.get_x() for p in c] for c in ax.containers]
    assert lefts[0] == lefts[1]
    assert lefts[0][0] == 0
    assert lefts[0][-1] + ax.containers[0][-1].get_width() == pytest.approx(19)


def test_hist_stack_puts_y_on_top_of_x() -> None:
    ax = kx.plot(np.arange(10.0), np.arange(10.0), spec="hist-stack-leg")
    xs, ys = ax.containers
    assert [p.get_y() for p in ys] == [p.get_height() for p in xs]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["x", "y"]


@pytest.mark.parametrize(("spec", "total"), [("hist-norm", None), ("hist-cum", 20), ("hist-norm-cum", 1)])
def test_hist_norm_and_cum(spec: str, total: float | None) -> None:
    ax = kx.plot(X, spec=spec)
    if total is None:                           # density: bar areas sum to 1
        assert sum(p.get_height() * p.get_width() for p in ax.containers[0]) == pytest.approx(1)
    else:                                       # cumulative outline: ends at everything
        assert ax.patches[0].get_xy()[:, 1].max() == pytest.approx(total)


def test_hist_ignores_nan() -> None:
    ax = kx.plot(np.r_[X, np.nan, np.inf], spec="hist-cum")
    assert ax.patches[0].get_xy()[:, 1].max() == len(X)


def test_hist_stack_cum_keeps_filled_bars() -> None:
    ax = kx.plot(X, Y, spec="hist-stack-cum")
    assert ax.containers[1][-1].get_y() + ax.containers[1][-1].get_height() == 2 * len(X)


@pytest.mark.parametrize("arrays", [1, 2, 3])
def test_kde_draws_a_curve_per_array_with_area_one(arrays: int) -> None:
    ax = kx.plot(*[X, Y, Z][:arrays], spec="kde-leg")
    assert len(ax.lines) == arrays
    for ln in ax.lines:
        gx, gy = ln.get_data()
        assert (gy.sum() * (gx[1] - gx[0])) == pytest.approx(1, abs=1e-3)       # even grid: area = sum * step
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["x", "y", "z"][:arrays]


def test_kde_matches_gaussian_formula() -> None:
    from kx.kinds.dist import density
    v, grid = np.array([0.0, 1.0]), np.array([0.0, 0.5])
    expected = (np.exp(-0.5 * ((grid[:, None] - v) / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))).mean(axis=1)
    assert density(v, grid, 0.5) == pytest.approx(expected)


def test_kde_by_groups() -> None:
    df = pd.DataFrame({"v": np.r_[X, X + 20], "g": ["a"] * len(X) + ["b"] * len(X)})
    ax = kx.plot(df.v, spec="kde", by=df.g)
    assert len(ax.lines) == 2
    peaks = [ln.get_xdata()[np.argmax(ln.get_ydata())] for ln in ax.lines]
    assert peaks[0] < 12 < peaks[1]
    leg = ax.get_legend()
    assert leg.get_title().get_text() == "g"
    assert [t.get_text() for t in leg.get_texts()] == ["a", "b"]


def test_kde_by_numeric_groups_sorted() -> None:
    ax = kx.plot(np.r_[X, X], spec="kde", by=[2] * len(X) + [1] * len(X))
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["1", "2"]


def test_ridge_stacks_first_series_on_top_on_one_scale() -> None:
    ax = kx.plot(X, X + 20, np.r_[X, X], spec="ridge-leg", overlap=2)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["x", "y", "z"]
    assert list(ax.get_yticks()) == [2, 1, 0]                # x on the top row
    bases = [ln.get_ydata().min() for ln in ax.lines]
    assert bases == pytest.approx([2, 1, 0], abs=1e-3)
    heights = [ln.get_ydata().max() - b for ln, b in zip(ax.lines, bases, strict=True)]
    assert max(heights) == pytest.approx(2)     # overlap= is the tallest ridge in rows
    assert heights[0] == pytest.approx(heights[1])           # same shape, same scale
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["x", "y", "z"]


def test_ridge_by_groups_and_marks() -> None:
    df = pd.DataFrame({"v": np.r_[X, X + 20], "g": ["a"] * 20 + ["b"] * 20})
    ax = kx.plot(df, "v", spec="ridge-mean-median", by="g")
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b"]
    assert list(ax.get_yticks()) == [1, 0]
    assert ax.get_ylabel() == "g"
    lines = [c for c in ax.collections if isinstance(c, mpl.collections.LineCollection)]
    marks = [seg[0] for c in lines for seg in c.get_segments()]
    assert sorted(m[0] for m in marks) == pytest.approx(sorted([X.mean(), np.median(X), X.mean() + 20,
                                                                 np.median(X) + 20]))


@pytest.mark.parametrize("arrays", [1, 2, 3])
def test_ecdf_steps_from_zero_to_one(arrays: int) -> None:
    ax = kx.plot(*[X, Y, Z][:arrays], spec="ecdf-leg")
    assert len(ax.lines) == arrays
    gx, gy = ax.lines[0].get_data()
    assert gy[0] == 0
    assert gy[-1] == 1
    assert list(gx[1:]) == sorted(X)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["x", "y", "z"][:arrays]


def test_ecdf_ignores_nan_and_counts_ties() -> None:
    ax = kx.plot([1.0, 2.0, 2.0, np.nan, 4.0], spec="ecdf")
    gx, gy = ax.lines[0].get_data()
    assert list(gx) == [1, 1, 2, 2, 4]
    assert list(gy) == [0, 0.25, 0.5, 0.75, 1]


def test_ecdf_by_groups() -> None:
    ax = kx.plot(pd.Series(np.r_[X, X + 20]), spec="ecdf", by=pd.Series(["a"] * 20 + ["b"] * 20, name="g"))
    assert len(ax.lines) == 2
    assert ax.get_legend().get_title().get_text() == "g"


@pytest.mark.parametrize("kind", ["box", "violin", "strip"])
def test_box_and_violin_one_per_array(kind: str) -> None:
    ax = kx.plot(X, pd.Series(Y, name="sin"), Z, spec=kind)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["x", "sin", "z"]


@pytest.mark.parametrize("kind", ["box", "violin", "strip"])
def test_box_and_violin_by_groups(kind: str) -> None:
    df = pd.DataFrame({"v": np.r_[X, X + 20], "g": pd.Categorical(["b"] * 20 + ["a"] * 20, ["b", "a"])})
    ax = kx.plot(df.v, spec=kind, by=df.g)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["b", "a"]
    assert ax.get_xlabel() == "g"


def test_box_median_and_colours() -> None:
    ax = kx.plot(X, X + 5, spec="box")
    assert ax.lines[4].get_ydata()[0] == pytest.approx(np.median(X))     # whiskers, caps, then median
    faces = [to_hex(p.get_facecolor(), keep_alpha=False) for p in ax.patches]
    assert faces[0] != faces[1]


def test_strip_draws_every_value_jittered_around_its_column() -> None:
    ax = kx.plot(np.r_[X, np.nan], np.ones(5), spec="strip", jitter=0.1)
    first, second = (c.get_offsets() for c in ax.collections)
    assert len(first) == len(X)                 # NaN dropped
    assert sorted(first[:, 1]) == pytest.approx(sorted(X))
    assert np.all(np.abs(first[:, 0] - 1) <= 0.1)
    assert np.all(np.abs(second[:, 0] - 2) <= 0.1)
    assert np.ptp(second[:, 0]) > 0             # ties spread sideways
    again = kx.plot(np.r_[X, np.nan], np.ones(5), spec="strip", jitter=0.1).collections[0].get_offsets()
    assert np.array_equal(first, again)         # same data, same picture
    faces = [to_hex(c.get_facecolor()[0], keep_alpha=False) for c in ax.collections]
    assert faces[0] != faces[1]


def test_qq_against_normal_quantiles_with_quartile_line() -> None:
    v = np.array([3.0, 1.0, np.nan, 2.0, 4.0])
    ax = kx.plot(v, spec="qq")
    pts = ax.collections[0].get_offsets()
    expected = [NormalDist().inv_cdf((k - 0.5) / 4) for k in range(1, 5)]
    assert list(pts[:, 0]) == pytest.approx(expected)
    assert list(pts[:, 1]) == [1, 2, 3, 4]      # sorted, NaN dropped
    (x0, x1), (y0, y1) = ax.lines[0].get_data()
    q1, q3 = np.percentile([1, 2, 3, 4], [25, 75])
    z1, z3 = NormalDist().inv_cdf(0.25), NormalDist().inv_cdf(0.75)
    assert (y1 - y0) / (x1 - x0) == pytest.approx((q3 - q1) / (z3 - z1))
    assert ax.get_xlabel() == "normal quantiles"


def test_qq_normal_sample_lies_on_its_line_and_by_groups() -> None:
    v = np.array([NormalDist(5, 2).inv_cdf(p) for p in (np.arange(1, 201) - 0.5) / 200])
    ax = kx.plot(v, spec="qq")
    (x0, x1), (y0, y1) = ax.lines[0].get_data()
    assert (y1 - y0) / (x1 - x0) == pytest.approx(2, rel=0.01)
    df = pd.DataFrame({"v": np.r_[X, X * 3], "g": ["a"] * 20 + ["b"] * 20})
    ax = kx.plot(df, "v", spec="qq", by="g")
    assert len(ax.collections) == 2
    assert ax.get_legend().get_title().get_text() == "g"


def test_violin_marks_median() -> None:
    ax = kx.plot(X, spec="violin")
    assert ax.collections[-1].get_offsets()[0][1] == pytest.approx(np.median(X))


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


PASSENGERS = kx.load("passengers")


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


# ---------- machine-learning kinds ----------
def test_cm_counts_true_rows_and_predicted_columns() -> None:
    ax = kx.plot(["a", "b", "b", "c", "c"], ["a", "b", "a", "a", "b"], "cm-ann")
    assert ax.images[0].get_array().tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 0]]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]    # 'c' is never predicted
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "c"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("predicted", "true")
    assert [t.get_text() for t in ax.texts] == ["1", "0", "0", "1", "1", "0", "1", "1", "0"]
    assert ax.figure.axes[1].get_ylabel() == "count"


def test_cm_extra_predicted_class_gets_a_blank_norm_row() -> None:
    ax = kx.plot([0, 0, 1], [0, 2, 1], "cm-norm")
    shares = ax.images[0].get_array()
    assert shares.shape == (3, 3)
    assert np.allclose(shares[:2].sum(axis=1), 1)
    assert shares[:2].tolist() == [[0.5, 0, 0.5], [0, 1, 0]]
    assert np.ma.is_masked(shares[2, 0]) or np.isnan(shares[2, 0])     # class 2 is never true: no recall


def test_cm_norm_rows_sum_to_one_with_a_percent_colorbar() -> None:
    m = kx.load("multiclass")
    ax = kx.plot(m, "y", "pred", "cm-norm-ann")
    assert np.allclose(ax.images[0].get_array().sum(axis=1), 1)
    assert all(t.get_text().endswith("%") for t in ax.texts)
    cb = ax.figure.axes[1]
    assert isinstance(cb.yaxis.get_major_formatter(), mpl.ticker.PercentFormatter)
    assert len(kx.plot(m, "y", "pred", "cm-nocb").figure.axes) == 1


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot([0, 1, 2], np.eye(3), "cm"), r"proba\.argmax\(axis=1\)"),
    (lambda: kx.plot(np.eye(3), [0, 1, 2], "cm"), "y_true is 2-D"),
    (lambda: kx.plot([0, 1], [0, 1], [0, 1], "cm"), "takes no z"),
    (lambda: kx.plot([0, 1], [0, 1, 1], "cm"), "y_true has 2 values but y_pred has 3"),
    (lambda: kx.plot([0, np.nan], [0, 1], "cm"), "missing values"),
])
def test_cm_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


BINARY, MULTI, REGRESSION = kx.load("binary"), kx.load("multiclass"), kx.load("regression")


def _curves(ax: Axes) -> list:
    """The labelled lines: one per model or class, not the chance line."""
    return [ln for ln in ax.lines if not ln.get_label().startswith("_")]


def _legend(ax: Axes) -> list[str]:
    return [t.get_text() for t in ax.get_legend().get_texts()]


def test_roc_runs_from_corner_to_corner_with_auc_in_the_legend() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "roc")
    curve, = _curves(ax)
    assert (curve.get_xdata()[0], curve.get_ydata()[0]) == (0, 0)
    assert (curve.get_xdata()[-1], curve.get_ydata()[-1]) == (1, 1)
    assert _legend(ax) == [f"gbm (AUC {kx.roc_auc(BINARY.y, BINARY.gbm):.2f})"]   # drawn without 'leg'
    assert (ax.get_xlim(), ax.get_ylim()) == ((0, 1), (0, 1))
    assert ax.get_aspect() == 1
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("false positive rate", "true positive rate")
    chance = [ln for ln in ax.lines if ln.get_linestyle() == ":"]
    assert len(chance) == 1


def test_roc_z_is_a_second_model() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "logreg", "roc")
    assert len(_curves(ax)) == 2
    assert _legend(ax)[1].startswith("logreg (AUC ")
    assert _legend(kx.plot(BINARY.y.to_numpy(), BINARY.gbm.to_numpy(), "roc"))[0].startswith("y (AUC ")


def test_roc_multiclass_one_curve_per_column() -> None:
    ax = kx.plot(MULTI.y, MULTI[["cat", "dog", "bird"]], "roc")
    assert [t.split(" ")[0] for t in _legend(ax)] == ["cat", "dog", "bird"]
    want = kx.roc_auc(MULTI.y == "dog", MULTI.dog)
    assert _legend(ax)[1] == f"dog (AUC {want:.2f})"


def test_roc_multiclass_array_takes_the_sorted_labels() -> None:
    codes = MULTI.y.map({"bird": 0, "cat": 1, "dog": 2})
    ax = kx.plot(codes, MULTI[["bird", "cat", "dog"]].to_numpy(), "roc")
    assert [t.split(" ")[0] for t in _legend(ax)] == ["0", "1", "2"]
    renamed = MULTI[["bird", "cat", "dog"]].set_axis(["b", "c", "d"], axis=1)  # columns in label order, other names
    assert [t.split(" ")[0] for t in _legend(kx.plot(codes, renamed, "roc"))] == ["b", "c", "d"]


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot(MULTI.y, MULTI.cat, "roc"), "pass a 2-D score, one column per class"),
    (lambda: kx.plot(MULTI.y, MULTI[["cat", "dog"]].to_numpy(), "roc"), "2 columns but y_true has 3 labels"),
    (lambda: kx.plot(MULTI.y, MULTI[["cat", "dog", "bird"]], MULTI.cat, "roc"), "z adds a second binary model"),
    (lambda: kx.plot([1, 1, 1], [0.2, 0.5, 0.9], "roc"), "only positives"),
])
def test_roc_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def _dotted(ax: Axes) -> list:
    return [ln for ln in ax.lines if ln.get_linestyle() == ":"]


def test_pr_baseline_at_the_positive_rate_and_ap_in_the_legend() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "pr")
    curve, = _curves(ax)
    assert (curve.get_xdata()[0], curve.get_ydata()[0]) == (0, 1)
    assert curve.get_xdata()[-1] == 1
    assert curve.get_drawstyle() == "steps-pre"
    assert _legend(ax) == [f"gbm (AP {kx.pr_ap(BINARY.y, BINARY.gbm):.2f})"]
    base, = _dotted(ax)
    assert list(base.get_ydata()) == pytest.approx([BINARY.y.mean()] * 2)
    assert list(base.get_xdata()) == [0, 1]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("recall", "precision")
    assert (ax.get_xlim(), ax.get_ylim()) == ((0, 1), (0, 1))


def test_pr_z_shares_one_baseline() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "logreg", "pr")
    assert len(_curves(ax)) == 2
    assert len(_dotted(ax)) == 1
    assert _legend(ax)[1].startswith("logreg (AP ")


def test_pr_multiclass_one_baseline_per_class_in_its_colour() -> None:
    ax = kx.plot(MULTI.y, MULTI[["cat", "dog", "bird"]], "pr")
    curves, bases = _curves(ax), _dotted(ax)
    assert [t.split(" ")[0] for t in _legend(ax)] == ["cat", "dog", "bird"]
    assert [b.get_ydata()[0] for b in bases] == pytest.approx([(MULTI.y == c).mean() for c in ("cat", "dog", "bird")])
    assert [to_hex(b.get_color()) for b in bases] == [to_hex(c.get_color()) for c in curves]


def test_calib_perfect_model_lies_on_the_diagonal_without_empty_bins() -> None:
    score = [0.05] * 20 + [0.25] * 4 + [0.75] * 4             # bins 0, 2 and 7 of 10; the rest are empty
    truth = [1] + [0] * 19 + [1, 0, 0, 0] + [1, 1, 1, 0]       # each comes true as often as it says
    ax = kx.plot(truth, score, "calib")
    pts = ax.collections[0]
    assert np.allclose(pts.get_offsets(), [[0.05, 0.05], [0.25, 0.25], [0.75, 0.75]])
    sizes = pts.get_sizes()
    assert sizes[0] > sizes[1] == sizes[2]                    # 20 rows against 4
    want = kx._metrics.brier(truth, score)
    assert _legend(ax) == [f"y (Brier {want:.3f})"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("mean score", "observed positive rate")
    assert len(_dotted(ax)) == 1


def test_calib_bins_and_a_second_model() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "logreg", "calib", bins=5)
    assert len(_curves(ax)) == 2
    assert all(len(c.get_xdata()) <= 5 for c in _curves(ax))
    gbm, logreg = (kx._metrics.brier(BINARY.y, BINARY[c]) for c in ("gbm", "logreg"))
    assert _legend(ax) == [f"gbm (Brier {gbm:.3f})", f"logreg (Brier {logreg:.3f})"]


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot([0, 1], [-0.5, 2.0], "calib"), "between 0 and 1"),
    (lambda: kx.plot([0, 1], [0.2, 0.8], "calib", bins=0), "whole number from 1"),
    (lambda: kx.plot([0, 1], [0.2, 0.8], "calib", bins=2.5), "whole number from 1"),
])
def test_calib_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_proba_one_histogram_per_true_class_on_bins_over_0_to_1() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "proba")
    assert len(ax.containers) == 2
    for c in ax.containers:
        assert c[0].get_x() == pytest.approx(0)
        assert c[-1].get_x() + c[-1].get_width() == pytest.approx(1)
        assert len(c) == 20
    counts = [sum(p.get_height() for p in c) for c in ax.containers]
    assert counts == [(BINARY.y == 0).sum(), (BINARY.y == 1).sum()]
    assert _legend(ax) == ["0", "1"]
    assert ax.get_legend().get_title().get_text() == "y"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("gbm", "count")


def test_proba_norm_gives_areas_of_one() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "proba-norm", bins=10)
    for c in ax.containers:
        assert len(c) == 10
        assert sum(p.get_height() * p.get_width() for p in c) == pytest.approx(1)
    assert ax.get_ylabel() == "density"


def test_proba_multiclass_takes_the_score_of_the_true_class() -> None:
    ax = kx.plot(MULTI.y, MULTI[["cat", "dog", "bird"]], "proba")
    assert _legend(ax) == ["cat", "dog", "bird"]
    dog = sum(p.get_height() for p in ax.containers[1])
    assert dog == (MULTI.y == "dog").sum()
    assert ax.get_xlabel() == "score of the true class"


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot(BINARY, "y", "gbm", "logreg", "proba"), "one model's scores"),
    (lambda: kx.plot([0, 1], [-0.5, 2.0], "proba"), "between 0 and 1"),
    (lambda: kx.plot([0, 1], [0.2, np.nan], "proba"), "NaN"),
    (lambda: kx.plot(MULTI.y, MULTI.cat, "proba"), "pass a 2-D score"),
])
def test_proba_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_thresh_precision_recall_f1_by_hand() -> None:
    ax = kx.plot([0, 1, 0, 1], [0.1, 0.4, 0.35, 0.8], "thresh")
    lines = {ln.get_label(): ln for ln in ax.lines}
    assert list(lines["precision"].get_xdata()) == [0.8, 0.4, 0.35, 0.1]
    assert list(lines["precision"].get_ydata()) == pytest.approx([1, 1, 2 / 3, 0.5])
    assert list(lines["recall"].get_ydata()) == pytest.approx([0.5, 1, 1, 1])
    assert list(lines["F1"].get_ydata()) == pytest.approx([2 / 3, 1, 0.8, 2 / 3])
    assert _legend(ax) == ["precision", "recall", "F1"]
    assert ax.get_xlabel() == "threshold on score"


def test_thresh_best_marks_the_f1_maximum() -> None:
    ax = kx.plot([0, 1, 0, 1], [0.1, 0.4, 0.35, 0.8], "thresh-best")
    best, = _dashed(ax)
    assert list(best.get_xdata()) == [0.4, 0.4]
    assert _legend(ax)[-1] == "best F1 1.00 at 0.4"
    ax = kx.plot(BINARY, "y", "gbm", "thresh-best")
    _, _, _, f1 = kx._metrics.thresholds(BINARY.y, BINARY.gbm)
    assert _legend(ax)[-1].startswith(f"best F1 {f1.max():.2f} at ")
    assert ax.get_xlabel() == "threshold on gbm"


def test_thresh_f1_is_zero_where_nothing_is_found() -> None:
    _, precision, recall, f1 = kx._metrics.thresholds([1, 0], [0.2, 0.9])
    assert list(precision) == [0, 0.5]
    assert list(recall) == [0, 1]
    assert list(f1) == pytest.approx([0, 2 / 3])


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot(BINARY, "y", "gbm", "logreg", "thresh"), "one model's scores"),
    (lambda: kx.plot([0, 0], [0.2, 0.9], "thresh"), "only negatives"),
    (lambda: kx.parse("roc-best"), "flag 'best' only works with kind"),
])
def test_thresh_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_gain_ends_at_one_with_the_perfect_corner_at_the_positive_rate() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "gain")
    curves = {ln.get_label(): ln for ln in _curves(ax)}
    assert list(curves) == ["gbm", "perfect"]
    model = curves["gbm"]
    assert (model.get_xdata()[0], model.get_ydata()[0]) == (0, 0)
    assert (model.get_xdata()[-1], model.get_ydata()[-1]) == (1, 1)
    assert list(curves["perfect"].get_xdata()) == pytest.approx([0, BINARY.y.mean(), 1])
    assert list(curves["perfect"].get_ydata()) == [0, 1, 1]
    assert list(_dotted(ax)[0].get_ydata()) == [0, 1]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("rows taken, highest score first", "positives found")


def test_gain_by_hand_and_a_second_model() -> None:
    ax = kx.plot([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], "gain")
    model = _curves(ax)[0]
    assert list(model.get_xdata()) == [0, 0.25, 0.5, 0.75, 1]
    assert list(model.get_ydata()) == [0, 0.5, 0.5, 1, 1]
    ax = kx.plot(BINARY, "y", "gbm", "logreg", "gain")
    assert _legend(ax) == ["gbm", "logreg", "perfect"]


def test_lift_is_one_when_every_row_is_taken() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "logreg", "lift")
    for curve in _curves(ax):
        assert curve.get_xdata()[-1] == 1
        assert curve.get_ydata()[-1] == pytest.approx(1)
        assert curve.get_ydata()[0] > 1.5                   # the top rows hold far more positives than average
    assert list(_dotted(ax)[0].get_ydata()) == [1, 1]
    assert ax.get_ylabel() == "lift"
    lifted = _curves(kx.plot([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], "lift"))[0]
    assert list(lifted.get_ydata()) == pytest.approx([2, 1, 4 / 3, 1])


def test_gain_multiclass_one_perfect_corner_per_class() -> None:
    ax = kx.plot(MULTI.y, MULTI[["cat", "dog", "bird"]], "gain")
    assert _legend(ax) == ["cat", "dog", "bird"]
    corners = [ln for ln in _dotted(ax) if len(ln.get_xdata()) == 3]
    assert [c.get_xdata()[1] for c in corners] == pytest.approx([(MULTI.y == c).mean() for c in ("cat", "dog", "bird")])


def test_pva_metrics_by_hand_in_the_corner() -> None:
    assert kx._metrics.fit_errors([1, 2, 3, 4], [1, 2, 3, 5]) == pytest.approx((0.8, 0.5, 0.25))
    ax = kx.plot(pd.Series([1, 2, 3, 4], name="price"), pd.Series([1, 2, 3, 5], name="guess"), "pva")
    box, = ax.texts
    assert box.get_text() == "R² 0.800\nRMSE 0.5\nMAE 0.25"
    line, = _dashed(ax)
    assert (line.get_xdata().min(), line.get_xdata().max()) == (1, 5)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("price", "guess")
    assert ax.get_aspect() == 1


def test_pva_switches_to_hexbin_over_5000_points() -> None:
    rng = np.random.default_rng(0)
    small = kx.plot(REGRESSION.actual, REGRESSION.predicted, "pva")
    assert isinstance(small.collections[0], mpl.collections.PathCollection)
    assert len(small.figure.axes) == 1
    a = rng.normal(size=5001)
    big = kx.plot(a, a + rng.normal(size=5001), "pva")
    assert isinstance(big.collections[0], mpl.collections.PolyCollection)     # hexagons, with a count colorbar
    assert big.figure.axes[1].get_ylabel() == "count"
    assert len(_dashed(big)) == 1


def test_resid_sign_line_at_zero_and_z_colour() -> None:
    ax = kx.plot([3, 1], [1, 2], "resid")                   # actual 3 predicted 1: +2; actual 1 predicted 2: -1
    assert ax.collections[0].get_offsets().tolist() == [[1, 2], [2, -1]]
    zero, = _dashed(ax)
    assert list(zero.get_ydata()) == [0, 0]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("predicted", "residual (actual - predicted)")
    coloured = kx.plot(REGRESSION, "actual", "predicted", "x", "resid-sym")
    assert coloured.figure.axes[1].get_ylabel() == "x"
    assert len(kx.plot(REGRESSION, "actual", "predicted", "x", "resid-nocb").figure.axes) == 1


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot([1, 2], [1, 2], [1, 2], "pva"), "takes no z"),
    (lambda: kx.plot([1, 2], [1, np.nan], "pva"), "NaN"),
    (lambda: kx.plot([1, 2, 3], [1, 2], "resid"), "actual has 3 values but predicted has 2"),
    (lambda: kx.plot([1, 2], [1, 2], "resid-logc"), "colour a numeric z"),
])
def test_pva_resid_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


IMPORTANCE = kx.load("importance")


def test_imp_largest_on_top_with_error_bars_from_z() -> None:
    ax = kx.plot(IMPORTANCE, "feature", "importance", "std", "imp")
    want = IMPORTANCE.sort_values("importance", ascending=False)
    bars, = (c for c in ax.containers if isinstance(c, mpl.container.BarContainer))
    assert [p.get_width() for p in bars] == list(want.importance)
    assert [t.get_text() for t in ax.get_yticklabels()] == list(want.feature)
    assert ax.yaxis_inverted()                              # the first bar, the largest, is at the top
    segments = bars.errorbar.lines[2][0].get_segments()
    assert [s[1][0] - s[0][0] for s in segments] == pytest.approx(list(2 * want["std"]))
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("importance ± std", "feature")


def test_imp_top_cuts_and_says_how_many_were_left_out() -> None:
    ax = kx.plot(IMPORTANCE, "feature", "importance", "imp", top=5)
    assert len(ax.patches) == 5
    assert [p.get_width() for p in ax.patches] == list(IMPORTANCE.importance.nlargest(5))
    assert ax.get_ylabel() == "feature, top 5 (10 left out)"
    assert not any(isinstance(c, mpl.container.ErrorbarContainer) for c in ax.containers)
    assert ax.get_ylabel() != kx.plot(IMPORTANCE, "feature", "importance", "imp", top=15).get_ylabel()


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot(IMPORTANCE, "feature", "importance", "imp", top=0), "whole number from 1"),
    (lambda: kx.plot(["a", "b"], [1.0, np.nan], "imp"), "NaN"),
    (lambda: kx.plot(["a", "b"], [1.0, 2.0], [0.1], "imp"), "one value per feature"),
])
def test_imp_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


HISTORY = kx.load("history")


def test_loss_pairs_train_and_val_in_one_colour() -> None:
    ax = kx.plot(HISTORY, "loss")
    train, val = ax.lines
    assert (train.get_label(), val.get_label()) == ("loss", "val_loss")
    assert (train.get_linestyle(), val.get_linestyle()) == ("-", "--")
    assert to_hex(train.get_color()) == to_hex(val.get_color())
    assert list(train.get_xdata()) == list(HISTORY.epoch)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("epoch", "loss")
    assert _legend(ax) == ["loss", "val_loss"]


def test_loss_takes_a_dict_or_a_keras_like_history() -> None:
    class History:                              # what model.fit returns, without importing Keras
        def __init__(self) -> None:
            self.history = {"loss": [0.9, 0.5, 0.4], "val_loss": [1.0, 0.7, 0.8]}

    for data in (History(), History().history):
        ax = kx.plot(data, "loss")
        assert [ln.get_label() for ln in ax.lines] == ["loss", "val_loss"]
        assert list(ax.lines[0].get_xdata()) == [1, 2, 3]    # no epoch column: the row number from 1


def test_loss_best_at_the_lowest_val_loss_or_highest_accuracy() -> None:
    ax = kx.plot(HISTORY, "loss-best")
    i = HISTORY.val_loss.idxmin()
    best, = _dashed(ax)[1:]                     # after the val_loss curve
    assert list(best.get_xdata()) == [HISTORY.epoch[i]] * 2
    assert _legend(ax)[-1] == f"best val_loss {HISTORY.val_loss[i]:.3g} at epoch {HISTORY.epoch[i]}"
    acc = kx.plot(HISTORY[["accuracy", "val_accuracy"]], "loss-best")
    j = HISTORY.val_accuracy.idxmax()
    assert _legend(acc)[-1].endswith(f"at epoch {j + 1}")
    assert acc.get_ylabel() == "accuracy"


def test_loss_without_val_columns_still_plots() -> None:
    ax = kx.plot(HISTORY[["epoch", "loss"]], "loss-best")
    assert ax.lines[0].get_label() == "loss"
    assert _legend(ax)[-1].startswith(f"best loss {HISTORY.loss.min():.3g} at epoch ")


def test_loss_errors() -> None:
    with pytest.raises(ValueError, match="at least one metric column"):
        kx.plot(HISTORY[["epoch"]], "loss")
    with pytest.raises(TypeError, match="takes a whole table"):
        kx.plot(HISTORY, "epoch", "loss", "loss")           # the trailing 'loss' is the kind, not the column
    assert len(kx.plot(HISTORY, "epoch", "loss", "line").lines) == 1     # other kinds still read it as a column


CURVES = kx.load("curves")
TRAIN, VAL = CURVES.filter(like="train_"), CURVES.filter(like="val_")


def test_lcurve_band_edges_are_mean_plus_minus_std() -> None:
    ax = kx.plot(CURVES["size"], TRAIN, VAL, "lcurve")
    train, val = ax.lines
    assert list(train.get_ydata()) == pytest.approx(list(TRAIN.mean(axis=1)))
    assert list(val.get_ydata()) == pytest.approx(list(VAL.mean(axis=1)))
    edges = ax.collections[1].get_paths()[0].vertices[:, 1]     # the validation band
    lo, hi = VAL.mean(axis=1) - VAL.std(axis=1, ddof=0), VAL.mean(axis=1) + VAL.std(axis=1, ddof=0)
    assert edges.min() == pytest.approx(lo.min())
    assert edges.max() == pytest.approx(hi.max())
    assert to_hex(ax.collections[1].get_facecolor()[0][:3]) == to_hex(val.get_color())
    assert _legend(ax) == ["train", "validation"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("size", "score")


def test_lcurve_one_fold_is_a_line_without_a_band() -> None:
    ax = kx.plot([10, 20, 30], [0.9, 0.8, 0.8], np.array([[0.5], [0.6], [0.7]]), "lcurve")
    assert len(ax.lines) == 2
    assert len(ax.collections) == 0
    assert ax.get_xlabel() == "training set size"


def test_vcurve_only_changes_the_x_label() -> None:
    ax = kx.plot(np.logspace(-3, 1, 10), TRAIN.to_numpy(), VAL.to_numpy(), "vcurve-logx")
    assert ax.get_xlabel() == "parameter"
    assert ax.get_xscale() == "log"
    assert len(ax.collections) == 2


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot([1, 2, 3], TRAIN, VAL, "lcurve"), r"train has shape \(10, 5\) but x has 3 values"),
    (lambda: kx.plot(CURVES["size"], TRAIN, VAL.iloc[:5], "vcurve"), "validation has shape"),
])
def test_lcurve_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


KS = np.arange(1, 9)
INERTIA = pd.Series([100, 50, 20, 15, 12, 10, 9, 8.0], name="inertia")


def test_knee_is_scale_free_and_needs_three_points() -> None:
    assert kx._metrics.knee(KS, INERTIA) == 2
    assert kx._metrics.knee(KS * 10, INERTIA * 1000) == 2
    assert kx._metrics.knee(np.arange(1, 11), 1000 / np.arange(1, 11)) == 2
    with pytest.raises(ValueError, match="at least 3 points"):
        kx._metrics.knee([1, 2], [3, 1])


def test_elbow_best_marks_the_knee() -> None:
    ax = kx.plot(KS, INERTIA, "elbow-best")
    best, = _dashed(ax)
    assert list(best.get_xdata()) == [3, 3]
    assert _legend(ax) == ["inertia", "knee at k = 3"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("k", "inertia")
    assert ax.lines[0].get_marker() == "o"


def test_elbow_z_on_a_twin_axis() -> None:
    silhouette = pd.Series([0.2, 0.45, 0.6, 0.5, 0.4, 0.35, 0.3, 0.3], name="silhouette")
    ax = kx.plot(KS, INERTIA, silhouette, "elbow-best")
    twin, = (a for a in ax.figure.axes if a is not ax)
    assert twin.get_ylabel() == "silhouette"
    assert list(twin.lines[0].get_ydata()) == list(silhouette)
    assert ax.get_shared_x_axes().joined(ax, twin)
    assert [t.get_text() for t in twin.get_legend().get_texts()] == ["inertia", "silhouette", "knee at k = 3"]
    assert ax.get_legend() is None
    assert len(kx.plot(KS, INERTIA, "elbow").figure.axes) == 1


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot([1, 2], [5, 3], "elbow-best"), "at least 3 points"),
    (lambda: kx.plot(KS, INERTIA, [0.1, 0.2], "elbow"), "x has 8 values but z has 2"),
])
def test_elbow_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def _titles(axs: np.ndarray) -> list[str]:
    return [ax.get_title() for ax in axs]


def test_evaluate_binary_is_roc_pr_calib_proba_with_every_score() -> None:
    axs = kx.evaluate(BINARY.y, BINARY.gbm, BINARY.logreg)
    assert isinstance(axs, np.ndarray)
    assert axs.shape == (4,)
    assert all(isinstance(ax, Axes) for ax in axs)
    assert _titles(axs) == ["ROC", "precision-recall", "calibration", "scores by class: gbm"]
    assert all(ax.figure is axs[0].figure for ax in axs)
    assert [len(_curves(ax)) for ax in axs[:3]] == [2, 2, 2]  # both models in roc, pr and calib
    assert _legend(axs[0]) == [f"gbm (AUC {kx.roc_auc(BINARY.y, BINARY.gbm):.2f})",
                               f"logreg (AUC {kx.roc_auc(BINARY.y, BINARY.logreg):.2f})"]
    assert len(axs[3].containers) == 2


def test_evaluate_names_unnamed_scores() -> None:
    axs = kx.evaluate(BINARY.y.to_numpy(), BINARY.gbm.to_numpy(), BINARY.logreg.to_numpy())
    assert [t.split(" (")[0] for t in _legend(axs[0])] == ["score 1", "score 2"]
    assert _legend(kx.evaluate(BINARY.y.to_numpy(), BINARY.gbm.to_numpy())[0])[0].startswith("score (AUC")


def test_evaluate_multiclass_is_roc_cm_of_argmax_and_proba() -> None:
    axs = kx.evaluate(MULTI.y, MULTI[["cat", "dog", "bird"]])
    assert _titles(axs) == ["ROC, one-vs-rest", "confusion matrix", "score of the true class"]
    assert len(_curves(axs[0])) == 3
    want = pd.crosstab(MULTI.y, MULTI.pred, normalize="index")      # pred is the argmax column in the data
    assert np.allclose(axs[1].images[0].get_array(), want.to_numpy())
    codes = MULTI.y.map({"bird": 0, "cat": 1, "dog": 2})
    by_code = kx.evaluate(codes, MULTI[["bird", "cat", "dog"]].to_numpy())
    assert np.allclose(by_code[1].images[0].get_array(), want.to_numpy())


def test_evaluate_regression_is_pva_resid_and_a_residual_hist() -> None:
    axs = kx.evaluate(REGRESSION.actual, REGRESSION.predicted)
    assert _titles(axs) == ["predicted vs actual", "residuals", "residual distribution"]
    assert axs[0].texts[0].get_text().startswith("R² ")
    assert axs[2].get_xlabel() == "residual (actual - predicted)"
    heights = sum(p.get_height() for p in axs[2].patches)
    assert heights == len(REGRESSION)


def test_evaluate_spec_reaches_every_panel() -> None:
    with kx.context("light"):
        axs = kx.evaluate(BINARY.y, BINARY.gbm, spec="dark-grid")
    dark = mpl.rc_params_from_file(kx._theme.style_path("dark"), use_default_template=False)
    assert to_hex(axs[0].figure.get_facecolor()) == to_hex(dark["figure.facecolor"])   # the figure too, not only panels
    for ax in axs:
        assert to_hex(ax.get_facecolor()) == to_hex(dark["axes.facecolor"])
        assert ax.xaxis.get_gridlines()[0].get_visible()


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.evaluate(BINARY.y), "at least one score"),
    (lambda: kx.evaluate(BINARY.y, BINARY.gbm, BINARY.logreg, BINARY.gbm), "at most 2 models"),
    (lambda: kx.evaluate(REGRESSION.actual, REGRESSION.predicted, REGRESSION.x), "one model's predictions"),
    (lambda: kx.evaluate(MULTI.y, MULTI.cat), "pass a 2-D score"),
])
def test_evaluate_errors(call: Callable[[], np.ndarray], match: str) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        call()


def _heights(ax: Axes) -> list[list[float]]:
    return [[p.get_height() for p in c] for c in ax.containers]


def test_count_matches_value_counts_with_nan_as_a_bar() -> None:
    ax = kx.plot(PASSENGERS, "deck", "count-ann")
    want = PASSENGERS.deck.value_counts(dropna=False)
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    assert ticks == [*"ABCDEFG", "NaN"]
    assert _heights(ax) == [[want[np.nan if t == "NaN" else t] for t in ticks]]
    assert [t.get_text() for t in ax.texts] == [str(want[np.nan if t == "NaN" else t]) for t in ticks]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("deck", "count")


def test_count_sort_puts_the_most_common_first() -> None:
    ax = kx.plot(PASSENGERS, "port", "count-sort")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Southampton", "Cherbourg", "Queenstown", "NaN"]
    assert _heights(ax)[0] == sorted(_heights(ax)[0], reverse=True)


def test_count_numeric_categories_in_order() -> None:
    ax = kx.plot([3, 1, 10, 1, 3, 3], spec="count")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "3", "10"]
    assert _heights(ax) == [[2, 3, 1]]


def test_count_by_gives_one_container_per_group() -> None:
    ax = kx.plot(PASSENGERS, "pclass", spec="count", by="port")
    assert len(ax.containers) == 4                  # 3 ports and NaN
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Cherbourg", "Queenstown", "Southampton", "NaN"]
    assert ax.get_legend().get_title().get_text() == "port"
    want = pd.crosstab(PASSENGERS.port, PASSENGERS.pclass)
    assert _heights(ax)[:3] == want.to_numpy().tolist()
    assert [p.get_width() for p in ax.patches] == pytest.approx([0.2] * 12)      # 0.8 shared by 4 groups


def test_count_norm_shares() -> None:
    ax = kx.plot(PASSENGERS, "sex", "count-norm-ann")
    assert sum(_heights(ax)[0]) == pytest.approx(1)
    assert all(t.get_text().endswith("%") for t in ax.texts)
    assert isinstance(ax.yaxis.get_major_formatter(), mpl.ticker.PercentFormatter)
    stacked = kx.plot(PASSENGERS, "pclass", "count-stack-norm", by="survived")
    assert np.allclose(np.sum(_heights(stacked), axis=0), 1)           # each category's bar is 100%
    assert stacked.get_legend().get_bbox_to_anchor().x0 > stacked.get_position().x1 - 0.01    # outside
    grouped = kx.plot(PASSENGERS, "pclass", "count-norm", by="survived")
    assert np.allclose(np.sum(_heights(grouped), axis=1), 1)           # each group's bars are 100%


def test_count_errors() -> None:
    with pytest.raises(ValueError, match="split them with by="):
        kx.plot(PASSENGERS, "sex", "pclass", "count")


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


# ---------- flags, title, labels ----------
def test_flags() -> None:
    ax = kx.plot(X, Y, Z, "line-grid-leg-logx-logy-tight", title="t")
    assert ax.get_xscale() == ax.get_yscale() == "log"
    assert ax.get_legend() is not None
    assert all(g.get_visible() for g in ax.get_xgridlines())
    assert ax.get_title() == "t"


def test_scale_flags() -> None:
    ax = kx.plot(X, Y, "line-logxy")
    assert ax.get_xscale() == ax.get_yscale() == "log"
    assert kx.plot(X, Z, "line-symlogy").get_yscale() == "symlog"
    assert kx.plot(X, Y, "line-eq").get_aspect() == 1


@pytest.mark.parametrize(("name", "lim"), [("line", "get_ylim"), ("barh", "get_xlim")])
def test_zero_stretches_the_value_axis(name: str, lim: str) -> None:
    ax = kx.plot(X, Y + 10, f"{name}-zero")
    lo, hi = getattr(ax, lim)()
    assert min(lo, hi) == 0


@pytest.mark.parametrize(("spec", "axis", "value", "text"), [
    ("bar-pct", "yaxis", 0.25, "25%"), ("barh-pct", "xaxis", 0.25, "25%"), ("line-si", "yaxis", 1500, "1.5k"),
])
def test_number_format_on_the_value_axis(spec: str, axis: str, value: float, text: str) -> None:
    ax = kx.plot(["a", "b"], [0.1, 0.2], spec=spec) if "bar" in spec else kx.plot(X, Y, spec=spec)
    out = getattr(ax, axis).get_major_formatter()(value, 0)
    assert out.replace(".0%", "%") == text       # PercentFormatter adds decimals on a narrow range


def test_pct_keeps_area_norm_percent() -> None:
    assert kx.plot(X, Y, Z, "area-norm-pct").yaxis.get_major_formatter()(50, 0) == "50%"


def test_legout_moves_a_kinds_own_legend_and_keeps_its_title() -> None:
    ax = kx.plot(np.r_[X, X], spec="kde-legout", by=pd.Series(["a"] * 20 + ["b"] * 20, name="g"))
    leg = ax.get_legend()
    assert leg.get_title().get_text() == "g"
    assert leg.get_bbox_to_anchor().bounds[0] > leg.axes.bbox.bounds[0]


def test_rot_turns_x_tick_labels() -> None:
    ax = kx.plot(["a", "b", "c"], [1, 2, 3], "bar-rot")
    assert {t.get_rotation() for t in ax.get_xticklabels()} == {45}


def _dashed(ax: Axes) -> list:
    return [ln for ln in ax.lines if ln.get_linestyle() == "--"]


@pytest.mark.parametrize("spec", ["scatter-diag", "hexbin-diag"])
def test_diag_spans_the_shared_range(spec: str) -> None:
    ax = kx.plot(np.array([0, 2, 5.0]), np.array([2, 10, 4.0]), spec)
    line, = _dashed(ax)
    assert (line.get_xdata().min(), line.get_xdata().max()) == (0, 10)
    assert np.array_equal(line.get_xdata(), line.get_ydata())
    assert len(kx.plot(X, Y, "scatter").lines) == 0


def test_diag_stays_straight_on_log_axes() -> None:
    line, = _dashed(kx.plot(X, Y, "scatter-diag-logx"))
    assert len(line.get_xdata()) > 2                       # many points, so y = x curves right on a log x axis


def test_fit_slope_and_r2_in_the_legend() -> None:
    x = np.array([0, 1, 2, 3, np.nan])
    ax = kx.plot(x, 2 * x + 1, "scatter-fit")
    line, = ax.lines
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["slope 2, R² 1.00"]
    assert list(line.get_xdata()) == [0, 3]
    assert list(line.get_ydata()) == pytest.approx([1, 7])
    noisy = kx.plot([0, 1, 2, 3], [0, 2, 1, 3], "scatter-fit")
    assert "slope 0.8, R² 0.64" in [t.get_text() for t in noisy.get_legend().get_texts()]


def test_fit_one_line_per_group() -> None:
    ax = kx.plot(DF, "t", "speed", "group", "scatter-fit")
    assert len(ax.lines) == 2
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert len(texts) == 2                                  # one entry per group: its points and its line
    assert texts[0].startswith("early (slope ")
    assert texts[1].startswith("late (slope ")
    assert ax.get_legend().get_title().get_text() == "group"
    assert to_hex(ax.lines[0].get_color()) == to_hex(ax.collections[0].get_facecolor()[0])


@pytest.mark.parametrize(("spec", "match"), [
    ("scatter-fit-logx", "a curve on a logx axis"),
    ("scatter-fit-logxy", "a curve on a logxy axis"),
    ("scatter-fit-date", "needs numbers on x"),
])
def test_fit_raises(spec: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        kx.plot(X, Y, spec)


def test_fit_needs_two_x_values() -> None:
    with pytest.raises(ValueError, match="at least 2 distinct"):
        kx.plot([1, 1], [2, 3], "scatter-fit")


LONG = ["standard single journey", "short", "first class saloon pass"]


@pytest.mark.parametrize(("spec", "axis"), [("bar-wrap", "x"), ("barh-wrap", "y"), ("line-wrap", "x")])
def test_wrap_breaks_long_tick_labels_at_spaces(spec: str, axis: str) -> None:
    ax = kx.plot(LONG, [1, 2, 3], spec)
    ax.figure.canvas.draw()
    labels = [t.get_text() for t in getattr(ax, f"get_{axis}ticklabels")()]
    assert labels == ["standard\nsingle\njourney", "short", "first class\nsaloon pass"]
    assert all(len(line) <= 12 for t in labels for line in t.split("\n"))


def test_wrap_keeps_fixed_ticks_and_numbers() -> None:
    ax = kx.plot(LONG, [1e6, 2e6, 3e6], "bar-wrap-rot-si")
    assert isinstance(ax.xaxis.get_major_locator(), mpl.ticker.FixedLocator)
    assert list(ax.get_xticks()) == [0, 1, 2]
    assert {t.get_rotation() for t in ax.get_xticklabels()} == {45}
    ax.figure.canvas.draw()
    assert all("\n" not in t.get_text() for t in ax.get_yticklabels())
    assert "1M" in [t.get_text() for t in ax.get_yticklabels()]
    words = kx.plot(["supercalifragilistic", "b"], [1, 2], "bar-wrap")      # one long word stays whole
    assert words.get_xticklabels()[0].get_text() == "supercalifragilistic"


def test_date_reads_text_as_dates() -> None:
    days = pd.Series(pd.date_range("2026-01-01", periods=len(X)).strftime("%Y-%m-%d"))
    ax = kx.plot(days, Y, "line-date")
    assert isinstance(ax.xaxis.get_major_formatter(), mpl.dates.ConciseDateFormatter)
    assert mpl.dates.num2date(ax.lines[0].get_xdata(orig=False)[0]).year == 2026


@pytest.mark.parametrize("name", ["line", "step", "band", "area"])
def test_mk_marks_every_point(name: str) -> None:
    assert kx.plot(X, Y, Z, f"{name}-mk").lines[0].get_marker() == "o"


@pytest.mark.parametrize(("spec", "order"), [("bar-sort", ["b", "c", "a"]), ("bar-sort-stack", ["c", "b", "a"])])
def test_sort_largest_first(spec: str, order: list[str]) -> None:
    ax = kx.plot(["a", "b", "c"], pd.Series([1, 3, 2], name="v"), [0, 0, 5], spec + "-leg")
    assert [t.get_text() for t in ax.get_xticklabels()] == order
    assert ax.get_legend().get_texts()[0].get_text() == "v"


def test_ann_writes_bar_values() -> None:
    ax = kx.plot(["a", "b"], [1.5, 20], "bar-ann")
    assert [t.get_text() for t in ax.texts] == ["1.5", "20"]


@pytest.mark.parametrize("name", ["hist", "kde", "ecdf"])
def test_mean_and_median_lines(name: str) -> None:
    v = np.r_[X, 100.0]
    ax = kx.plot(v, spec=f"{name}-mean-median-ann")
    marks = [ln.get_xdata()[0] for ln in ax.lines if ln.get_linestyle() in ("--", ":")]
    assert marks == pytest.approx([v.mean(), np.median(v)])
    assert [t.get_text() for t in ax.texts] == [f"{v.mean():.3g}", f"{np.median(v):.3g}"]


@pytest.mark.parametrize("name", ["box", "violin", "strip"])
def test_box_and_violin_mean_and_ann(name: str) -> None:
    ax = kx.plot(X, X + 5, spec=f"{name}-mean-ann")
    assert list(ax.collections[-1].get_offsets()[:, 1]) == pytest.approx([X.mean(), X.mean() + 5])
    assert ax.texts[0].get_text().startswith(f"{np.median(X):.3g}")


def test_scatter_colour_flags() -> None:
    ax = kx.plot(X, Y, Z - 2, "scatter-sym-nocb")
    lo, hi = ax.collections[0].get_clim()
    assert lo == -hi
    assert len(ax.figure.axes) == 1
    assert isinstance(kx.plot(X, Y, Y, "scatter-logc").collections[0].norm, mpl.colors.LogNorm)


@pytest.mark.parametrize(("spec", "call", "match"), [
    ("line-pct-si", lambda s: kx.plot(X, Y, spec=s), "do not go together"),
    ("line-logy-logxy", lambda s: kx.plot(X, Y, spec=s), "do not go together"),
    ("line-zero-logy", lambda s: kx.plot(X, Y, spec=s), "do not go together"),
    ("scatter-sym-logc", lambda s: kx.plot(X, Y, Z, s), "do not go together"),
    ("hist-date", lambda s: kx.plot(X, spec=s), "only works with kind"),
    ("line-sort", lambda s: kx.plot(X, Y, spec=s), "only works with kind"),
    ("hist-ann", lambda s: kx.plot(X, spec=s), "add 'mean' or 'median'"),
    ("area-stack-mk", lambda s: kx.plot(X, Y, Z, s), "a stack has no lines"),
    ("scatter-nocb", lambda s: kx.plot(X, Y, spec=s), "colour a numeric z"),
    ("scatter-sym", lambda s: kx.plot(X, Y, Z > 2, s), "colour a numeric z"),
    ("scatter-logc", lambda s: kx.plot(X, Y, Z - 2, s), "needs z > 0"),
])
def test_flag_errors(spec: str, call: Callable[[str], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call(spec)


def test_leg_without_labelled_artists_is_quiet() -> None:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ax = kx.plot(X, Y, Z, "scatter-leg")    # numeric z is a colorbar: nothing for a legend
        assert ax.get_legend() is None
        assert kx.plot(X, Y, Z, "scatter-legout").get_legend() is None


def test_series_names_become_labels() -> None:
    df = pd.DataFrame({"x": X, "speed": Y, "accel": Z})
    ax = kx.plot(df.x, df.speed, df.accel, "line-leg")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["speed", "accel"]


# ---------- call shapes ----------
def test_spec_as_third_positional() -> None:
    ax = kx.plot(X, Y, "bar")
    assert len(ax.patches) == len(X)


def test_spec_given_twice() -> None:
    with pytest.raises(TypeError):
        kx.plot(X, Y, "line", spec="bar")


DF = pd.DataFrame({"t": X, "speed": Y, "accel": Z, "group": np.where(X > 5, "late", "early")})


def test_dataframe_columns_by_name() -> None:
    ax = kx.plot(DF, "t", "speed", "accel", "line-leg")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["speed", "accel"]
    assert list(ax.lines[0].get_ydata()) == list(Y)


def test_dataframe_spec_as_keyword_and_by_column() -> None:
    assert len(kx.plot(DF, "t", "speed", spec="bar").patches) == len(X)
    ax = kx.plot(DF, "speed", spec="box", by="group")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["early", "late"]
    assert ax.get_xlabel() == "group"


def test_dataframe_column_named_like_a_kind_is_a_column() -> None:
    df = pd.DataFrame({"x": X, "bar": Y})
    ax = kx.plot(df, "x", "bar")
    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_ydata()) == list(Y)


@pytest.mark.parametrize(("call", "error", "match"), [
    (lambda: kx.plot(DF, "t", "sped"), ValueError, r"no column 'sped'.*Did you mean \['speed'\]"),
    (lambda: kx.plot(DF, "t", "speed", "linee"), ValueError, "unknown tokens"),
    (lambda: kx.plot(DF, spec="line"), TypeError, "needs the column names"),
    (lambda: kx.plot(DF, "t", DF.speed), TypeError, "give column names"),
    (lambda: kx.plot(X, spec="kde", by="group"), TypeError, "needs a DataFrame"),
    (lambda: kx.plot(X, Y, Z, Z), TypeError, "up to 3 arrays, got 4"),
    (lambda: kx.plot(), TypeError, "got 0"),
])
def test_call_shape_errors(call: Callable[[], Axes], error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        call()


@pytest.fixture
def table_kind() -> Iterator[list]:
    """A dummy table kind 'tbl' that records what draw gets, removed after the test."""
    seen = []
    kx._spec.kind("tbl", table=True)(lambda ax, x, y, z, flags, **kw: seen.append((x, y, z)))
    yield seen
    del kx._spec.KINDS["tbl"]


def test_table_kind_gets_the_whole_frame(table_kind: list) -> None:
    assert not kx._spec.KINDS["tbl"].needs_y
    kx.plot(DF, "tbl")
    kx.plot(DF, spec="tbl")
    for x, y, z in table_kind:
        assert x is DF
        assert y is None
        assert z is None


def test_table_kind_takes_a_2d_array_with_numbered_labels(table_kind: list) -> None:
    m = np.arange(6).reshape(2, 3)
    kx.plot(m, "tbl")
    x = table_kind[0][0]
    assert isinstance(x, pd.DataFrame)
    assert (x.to_numpy() == m).all()
    assert list(x.columns) == [0, 1, 2]
    assert list(x.index) == [0, 1]


@pytest.mark.parametrize("call", [
    lambda: kx.plot(DF, "t", "speed", "tbl"),
    lambda: kx.plot(DF, "t", "tbl"),
    lambda: kx.plot(DF, "tbl", by="group"),
    lambda: kx.plot(X, "tbl"),
])
def test_table_kind_rejects_columns_y_z_and_by(table_kind: list, call: Callable[[], Axes]) -> None:
    with pytest.raises(TypeError, match=r"takes a whole table.*kx.plot\(df\[cols\], 'tbl'\)"):
        call()


def test_frame_without_columns_hints_table_kinds(table_kind: list) -> None:
    with pytest.raises(TypeError, match=r"needs the column names.*or a table kind: heat, loss, missing, nullmap, tbl"):
        kx.plot(DF, "line")


def test_grammar_says_a_table_kind_takes_a_frame(table_kind: list, capsys: pytest.CaptureFixture[str]) -> None:
    kx.grammar("tbl")
    assert "takes a DataFrame or 2-D array" in capsys.readouterr().out
    kx.grammar("line")
    assert "DataFrame" not in capsys.readouterr().out


def test_kwargs_reach_matplotlib_and_win_over_defaults() -> None:
    ln = kx.plot(X, Y, Z, "line-mk", linewidth=4, marker="s").lines
    assert {(x.get_linewidth(), x.get_marker()) for x in ln} == {(4, "s")}
    assert [t.get_text() for t in kx.plot(X, Y, "line-leg", label="mine").get_legend().get_texts()] == ["mine"]
    sc = kx.plot(X, Y, Z, "scatter", cmap="magma", s=50).collections[0]
    assert (sc.get_cmap().name, list(sc.get_sizes())) == ("magma", [50])


@pytest.mark.parametrize(("spec", "kw", "check"), [
    ("step", {"linestyle": "--"}, lambda ax: ax.lines[0].get_linestyle() == "--"),
    ("area", {"alpha": 0.5}, lambda ax: ax.collections[0].get_alpha() == 0.5),
    ("area-stack", {"alpha": 0.5}, lambda ax: ax.collections[0].get_alpha() == 0.5),
    ("band", {"color": "red"}, lambda ax: to_hex(ax.collections[0].get_facecolor()[0]) == "#ff0000"),
    ("kde", {"linestyle": ":"}, lambda ax: ax.lines[0].get_linestyle() == ":"),
    ("ecdf", {"color": "red"}, lambda ax: to_hex(ax.lines[0].get_color()) == "#ff0000"),
    ("box", {"showfliers": False}, lambda ax: len(ax.lines) == 5),             # whiskers, caps, median; no fliers
    ("violin", {"widths": 0.2}, lambda ax: np.ptp(ax.collections[0].get_paths()[0].vertices[:, 0]) <= 0.2 + 1e-9),
])
def test_kwargs_for_each_kind(spec: str, kw: dict, check: Callable[[Axes], bool]) -> None:
    assert check(kx.plot(X, Y, np.abs(Z), spec, **kw) if spec in ("step", "area", "area-stack", "band")
                 else kx.plot(np.r_[X, 100], spec=spec, **kw))


@pytest.mark.parametrize(("spec", "key", "size"), [("bar", "width", "get_width"), ("barh", "height", "get_height")])
def test_bar_width_is_the_space_per_category(spec: str, key: str, size: str) -> None:
    ax = kx.plot(["a", "b"], [1, 2], [2, 1], spec, **{key: 0.5})
    assert {getattr(p, size)() for p in ax.patches} == {0.25}


def test_hist_bins_keyword_replaces_shared_bins() -> None:
    assert len(kx.plot(X, spec="hist", bins=5).patches) == 5
    ax = kx.plot(X, Y, spec="hist", bins=np.linspace(0, 10, 11))
    assert [len(c) for c in ax.containers] == [10, 10]


def test_unknown_kwarg_is_matplotlibs_error() -> None:
    with pytest.raises(AttributeError, match="nonsense"):
        kx.plot(X, Y, "line", nonsense=1)


def test_non_hist_needs_y() -> None:
    with pytest.raises(ValueError, match="needs both x and y"):
        kx.plot(X, spec="line")


def test_draws_into_given_ax() -> None:
    import matplotlib.pyplot as plt
    _, (a, b) = plt.subplots(1, 2)
    assert kx.plot(X, Y, "line", ax=b) is b
    assert len(a.lines) == 0


# ---------- themes ----------
def test_theme_in_spec_is_local() -> None:
    before = dict(mpl.rcParams)
    ax = kx.plot(X, Y, "line-dark")
    assert to_hex(ax.get_facecolor()) == "#111418"
    assert {k for k in before if str(before[k]) != str(mpl.rcParams[k])} == set()


def test_use_sets_global_theme() -> None:
    kx.use("dark")
    assert to_hex(mpl.rcParams["axes.facecolor"]) == "#111418"


def test_use_resets_before_applying() -> None:
    kx.use("dark")                              # dark sets bold titles, light does not
    kx.use("light")
    assert mpl.rcParams["axes.titleweight"] == "normal"


def test_context_applies_inside_and_restores_after() -> None:
    kx.use("light", palette="okabe")
    before = dict(mpl.rcParams)
    with kx.context("dark", palette=["#264653", "#e9c46a"], cmap="magma", figure__figsize=(5, 2)):
        assert to_hex(mpl.rcParams["axes.facecolor"]) == "#111418"
        assert mpl.rcParams["axes.prop_cycle"].by_key()["color"] == ["#264653", "#e9c46a"]
        assert mpl.rcParams["image.cmap"] == "magma"
        ax = kx.plot(X, Y, "line")
    assert to_hex(ax.get_facecolor()) == "#111418"          # the plot keeps the theme it was drawn in
    assert {k for k in before if str(before[k]) != str(mpl.rcParams[k])} == set()


def test_context_restores_after_an_error() -> None:
    kx.use("light")
    with pytest.raises(RuntimeError), kx.context("dark"):
        raise RuntimeError
    assert to_hex(mpl.rcParams["axes.facecolor"]) == "#ffffff"
    with pytest.raises(ValueError, match="unknown theme"), kx.context("nope"):
        pass
    assert to_hex(mpl.rcParams["axes.facecolor"]) == "#ffffff"


def test_use_palette_and_rc_overrides() -> None:
    kx.use("light", palette=["#264653", "#e9c46a"], figure__figsize=(5, 2))
    assert mpl.rcParams["axes.prop_cycle"].by_key()["color"] == ["#264653", "#e9c46a"]
    assert tuple(mpl.rcParams["figure.figsize"]) == (5, 2)


# ---------- palettes ----------
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]


@pytest.mark.parametrize(("name", "mpl_name"), [("set1", "Set1"), ("set2", "Set2"), ("dark2", "Dark2"),
                                                ("paired", "Paired"), ("tab10", "tab10"), ("tab20", "tab20")])
def test_palettes_match_matplotlib(name: str, mpl_name: str) -> None:
    assert [c.lower() for c in kx.palette(name)] == [to_hex(c) for c in mpl.colormaps[mpl_name].colors]


def test_okabe_palette() -> None:
    assert "okabe" in kx.palettes()
    assert kx.palette("okabe") == OKABE


@pytest.mark.parametrize("name", kx.palettes())
def test_palette_file_is_valid(name: str) -> None:
    colors = kx.palette(name)
    assert len(colors) >= 3
    assert all(re.fullmatch(r"#[0-9A-F]{6}", c) for c in colors), colors
    assert len(set(colors)) == len(colors)
    assert "#FFFFFF" not in colors              # white vanishes on light themes


def test_use_named_palette() -> None:
    kx.use("light", palette="okabe")
    ax = kx.plot(X, Y, Z, "line")
    assert [to_hex(ln.get_color()).upper() for ln in ax.lines] == OKABE[:2]


def test_palette_keeps_theme_linestyles() -> None:
    kx.use("paper", palette="okabe")
    cycle = mpl.rcParams["axes.prop_cycle"].by_key()
    assert cycle["color"] == OKABE
    assert cycle["linestyle"][:6] == ["-", "--", ":", "-.", "-", "--"]
    assert len(cycle["linestyle"]) == len(OKABE)


def test_unknown_palette() -> None:
    with pytest.raises(ValueError, match="unknown palette"):
        kx.use("dark", palette="nope")


def test_palettes_discovers_new_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "styles" / "palettes").mkdir(parents=True)
    (tmp_path / "styles" / "palettes" / "mine.txt").write_text("# comment\n\n112233\n  AABBCC  \n")
    monkeypatch.setattr(kx._files, "root", str(tmp_path))
    assert kx.palettes() == ["mine"]
    assert kx.styles() == []                    # the subfolder is not a theme
    assert kx.palette("mine") == ["#112233", "#AABBCC"]


# ---------- colormaps ----------
def test_cmaps_lists_matplotlib_builtins() -> None:
    names = (*kx.BUILTIN_CMAPS, *kx.BREWER_CMAPS, *kx.VENDORED_CMAPS)
    assert kx.cmaps() == sorted(n for n in names if n in mpl.colormaps)
    assert set(kx.BUILTIN_CMAPS) - {"berlin", "managua", "vanimo"} <= set(kx.cmaps())   # only these need 3.10


def test_cmaps_skips_names_missing_from_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kx._theme, "BUILTIN_CMAPS", ("viridis", "notacmap"))
    monkeypatch.setattr(kx._theme, "VENDORED_CMAPS", ())
    monkeypatch.setattr(kx._theme, "BREWER_CMAPS", {})
    assert kx.cmaps() == ["viridis"]


@pytest.mark.parametrize("name", kx.VENDORED_CMAPS)
def test_vendored_cmaps_are_registered(name: str) -> None:
    assert mpl.colormaps[name].N == 256
    assert name in kx.cmaps()


@pytest.mark.parametrize(("name", "brewer"), kx.BREWER_CMAPS.items())
def test_brewer_cmaps_are_lowercase_copies(name: str, brewer: str) -> None:
    assert name in mpl.colormaps
    assert name in kx.cmaps()
    assert mpl.colormaps[name].name == name
    for v in (0, 0.5, 1):
        assert to_hex(mpl.colormaps[name](v)) == to_hex(mpl.colormaps[brewer](v))


@pytest.mark.parametrize("name", ["viridis", *kx.BREWER_CMAPS, *kx.VENDORED_CMAPS])
def test_use_reversed_cmap(name: str) -> None:
    kx.use("light", cmap=f"{name}_r")
    rev = kx.plot(X, Y, Z, "scatter").collections[0].get_cmap()
    assert rev.name == f"{name}_r"
    assert to_hex(rev(1.0)) == to_hex(mpl.colormaps[name](0.0))


@pytest.mark.parametrize("name", [*kx.BUILTIN_CMAPS, *kx.BREWER_CMAPS, *kx.VENDORED_CMAPS])
def test_use_cmap_colours_scatter(name: str) -> None:
    if name not in mpl.colormaps:
        pytest.skip(f"{name} needs a newer matplotlib")
    kx.use("light", cmap=name)
    assert mpl.rcParams["image.cmap"] == name
    assert kx.plot(X, Y, Z, "scatter").collections[0].get_cmap().name == name


def test_unknown_cmap_leaves_theme_untouched() -> None:
    kx.use("dark")
    with pytest.raises(ValueError, match="unknown cmap"):
        kx.use("light", cmap="nope")
    with pytest.raises(ValueError, match="unknown cmap"):
        kx.use("light", cmap="nope_r")
    assert to_hex(mpl.rcParams["axes.facecolor"]) == "#111418"


@pytest.mark.parametrize("fn", [kx.use, kx.show])
def test_unknown_theme(fn) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="unknown theme"):
        fn("nope")


def test_styles_discovers_new_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "styles").mkdir()
    (tmp_path / "styles" / "mine.mplstyle").write_text("lines.linewidth: 3\n")
    monkeypatch.setattr(kx._files, "root", str(tmp_path))
    assert kx.styles() == ["mine"]
    assert kx.plot(X, Y, "line-mine").lines[0].get_linewidth() == 3


def test_paper_cycles_linestyles_for_greyscale_print() -> None:
    ax = kx.plot(X, Y, Z, "line-paper")
    assert ax.lines[0].get_linestyle() != ax.lines[1].get_linestyle()


def test_scatter_size_follows_theme_markersize() -> None:
    small = kx.plot(X, Y, "scatter-science").collections[0].get_sizes()[0]
    large = kx.plot(X, Y, "scatter-poster").collections[0].get_sizes()[0]
    assert kx.plot(X, Y, "scatter").collections[0].get_sizes()[0] == 18
    assert small < 18 < large


def test_every_shipped_style_loads() -> None:
    for name in kx.styles():
        kx.use(name)


# ---------- transparency ----------
def test_show_grammar_source(capsys: pytest.CaptureFixture[str]) -> None:
    kx.show("dark")
    assert "111418" in capsys.readouterr().out
    kx.grammar()
    out = capsys.readouterr().out
    assert "scatter" in out
    assert "dark" in out
    assert "logy" in out
    assert "okabe" in out
    assert "viridis" in out
    kx.source()
    assert "def plot(" in capsys.readouterr().out



def test_grammar_and_source_of_one_kind(capsys: pytest.CaptureFixture[str]) -> None:
    kx.grammar("area")
    out = capsys.readouterr().out
    assert "100% stack" in out
    assert "'norm'" in out
    assert "'grid'" in out
    kx.source("band")
    out = capsys.readouterr().out
    assert out.lstrip().startswith("@kind(\"band\"")
    assert "def scatter(" not in out


def _suptitle(fig: mpl.figure.Figure) -> str:
    """The figure's suptitle text; Figure.get_suptitle is matplotlib >= 3.8 only."""
    return fig._suptitle.get_text() if fig._suptitle else ""


@pytest.fixture
def panel_kind() -> Iterator[None]:
    """A dummy kind 'quad' that splits ax into 2 x 2 panels, and 'nopanels' that forgets to return them."""
    def quad(ax: Axes, x: ArrayLike, y: ArrayLike, z: None, flags: set[str], **kw: Any) -> np.ndarray:
        axs = kx._util.split(ax, 2, 2)
        for a in axs:
            a.plot(x, y, label="y")
        return axs
    kx._spec.kind("quad", panels=True)(quad)
    kx._spec.kind("nopanels", panels=True)(lambda ax, x, y, z, flags, **kw: None)
    yield
    del kx._spec.KINDS["quad"], kx._spec.KINDS["nopanels"]


def test_panel_kind_returns_its_axes_in_place_of_ax(panel_kind: None) -> None:
    axs = kx.plot(X, Y, "quad-grid", title="four")
    assert isinstance(axs, np.ndarray)
    assert axs.shape == (4,)
    fig = axs[0].figure
    assert fig.axes == list(axs)                # the host axes is gone
    assert all(g.get_visible() for a in axs for g in a.get_xgridlines())   # common flags reach every panel
    assert _suptitle(fig) == "four"         # the panels fill the figure: one title over them


def test_panel_kind_stays_in_its_grid_cell(panel_kind: None) -> None:
    left, right = kx.grid(2)
    cell = right.get_subplotspec()
    axs = kx.plot(X, Y, "quad", title="four", ax=right)
    fig = left.figure
    assert right not in fig.axes
    assert left in fig.axes
    assert len(fig.axes) == 5
    assert all(a.get_subplotspec().get_topmost_subplotspec() == cell for a in axs)
    assert axs[0].get_title() == "four"        # a title over one cell of a bigger figure: on its first panel
    assert _suptitle(fig) == ""


def test_panel_kind_must_return_its_axes(panel_kind: None) -> None:
    with pytest.raises(TypeError, match="must return their axes"):
        kx.plot(X, Y, "nopanels")


def test_split_needs_a_grid_cell() -> None:
    import matplotlib.pyplot as plt
    ax = plt.figure().add_axes((0.1, 0.1, 0.8, 0.8))
    with pytest.raises(TypeError, match="add_axes"):
        kx._util.split(ax, 1, 2)


@pytest.fixture
def polar_kind() -> Iterator[list]:
    """A dummy kind 'circle' on polar axes that records the axes it draws on."""
    seen = []
    kx._spec.kind("circle", projection="polar")(lambda ax, x, y, z, flags, **kw: seen.append(ax))
    yield seen
    del kx._spec.KINDS["circle"]


def test_polar_kind_gets_polar_axes(polar_kind: list) -> None:
    ax = kx.plot(X, Y, "circle", title="round")
    assert ax.name == "polar"
    assert polar_kind == [ax]
    assert ax.get_title() == "round"


def test_polar_kind_swaps_a_grid_cell_in_place(polar_kind: list) -> None:
    left, right = kx.grid(2)
    cell = right.get_subplotspec()
    ax = kx.plot(X, Y, "circle", ax=right)
    assert ax.name == "polar"
    assert ax.get_subplotspec() == cell
    assert right not in ax.figure.axes
    assert left in ax.figure.axes
    assert left.name == "rectilinear"


def test_polar_kind_keeps_polar_axes(polar_kind: list) -> None:
    import matplotlib.pyplot as plt
    given = plt.subplots(subplot_kw={"projection": "polar"})[1]
    assert kx.plot(X, Y, "circle", ax=given) is given


def test_grammar_says_how_a_kind_draws(panel_kind: None, polar_kind: list,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    kx.grammar("quad")
    assert "flat array" in capsys.readouterr().out
    kx.grammar("circle")
    assert "polar" in capsys.readouterr().out


@pytest.mark.parametrize("fn", [kx.grammar, kx.source])
def test_unknown_kind(fn) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="unknown kind"):
        fn("nope")


@pytest.mark.parametrize("name", ["line", "grid", "a-b"])
def test_kind_names_stay_unique_tokens(name: str) -> None:
    with pytest.raises(ValueError, match="taken"):
        kx._spec.kind(name)(lambda *a: None)

# ---------- grid ----------
@pytest.mark.parametrize(("n", "cols", "shape"), [(5, 3, (2, 3)), (2, 3, (1, 2)), (12, 4, (3, 4)), (1, 3, (1, 1))])
def test_grid_is_flat_with_n_panels(n: int, cols: int, shape: tuple[int, int]) -> None:
    mpl.rcParams["figure.figsize"] = (4, 3)
    axs = kx.grid(n, cols=cols)
    assert isinstance(axs, np.ndarray)
    assert axs.shape == (n,)
    fig = axs[0].figure
    assert fig.axes == list(axs)                # unused cells are gone
    assert tuple(fig.get_size_inches()) == (4 * shape[1], 3 * shape[0])


def test_grid_passes_subplots_keywords() -> None:
    axs = kx.grid(4, cols=2, sharex=True, figsize=(6, 6))
    assert axs[0].get_shared_x_axes().joined(axs[0], axs[3])
    assert tuple(axs[0].figure.get_size_inches()) == (6, 6)


def test_grid_panels_take_kx_plots_with_tight_and_colorbars() -> None:
    a, b = kx.grid(2)
    kx.plot(X, Y, Z, "scatter", ax=a)           # a colorbar, then tight: no layout-engine clash
    kx.plot(X, Y, "line-tight", ax=b)
    assert len(a.figure.axes) == 3


@pytest.mark.parametrize(("n", "cols"), [(0, 3), (3, 0)])
def test_grid_rejects_empty(n: int, cols: int) -> None:
    with pytest.raises(ValueError, match="n >= 1 and cols >= 1"):
        kx.grid(n, cols=cols)


# ---------- saving ----------
@pytest.fixture
def kaggle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake /kaggle/working, and a different current folder."""
    (tmp_path / "working").mkdir()
    (tmp_path / "cwd").mkdir()
    monkeypatch.setattr(kx._save, "KAGGLE", str(tmp_path / "working"))
    monkeypatch.chdir(tmp_path / "cwd")
    return tmp_path


def test_save_goes_to_kaggle_working(kaggle: Path) -> None:
    kx.plot(X, Y, "line")
    path = kx.save("plots/fig.png")
    assert path == str(kaggle / "working" / "plots" / "fig.png")
    assert Path(path).stat().st_size > 0


def test_save_outside_kaggle_uses_current_folder(kaggle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kx._save, "KAGGLE", str(kaggle / "missing"))
    kx.plot(X, Y, "line")
    assert Path(kx.save("fig.png")).resolve() == kaggle / "cwd" / "fig.png"


def test_save_absolute_path_and_given_ax(kaggle: Path) -> None:
    import matplotlib.pyplot as plt
    ax = kx.plot(X, Y, "line")
    plt.figure()
    target = kaggle / "elsewhere" / "fig.png"
    assert kx.save(str(target), ax, dpi=50) == str(target)
    assert plt.imread(target).shape[1] == int(ax.figure.get_figwidth() * 50)


def test_save_without_extension_uses_theme_format(kaggle: Path) -> None:
    with kx.context("thesis"):
        kx.plot(X, Y, "line")
        path = kx.save("fig")
    assert path.endswith("fig.pdf")
    assert Path(path).read_bytes().startswith(b"%PDF")
    kx.plot(X, Y, "line")
    assert kx.save("fig2", format="svg").endswith("fig2.svg")


# ---------- metrics ----------
_I = np.arange(200)
SCORE = np.round(((_I * 7919) % 200) / 200, 1)            # 11 distinct values, so many tied scores
TRUTH = (((_I * 7919) % 200) / 200 + ((_I * 104729) % 97) / 97 * 0.8) > 0.9


def test_auc_and_ap_match_scikit_learn() -> None:
    # computed once with sklearn.metrics.roc_auc_score and average_precision_score on TRUTH, SCORE
    assert kx.roc_auc(TRUTH, SCORE) == pytest.approx(0.8937043339005105, abs=1e-12)
    assert kx.pr_ap(TRUTH, SCORE) == pytest.approx(0.8694172331641548, abs=1e-12)


@pytest.mark.parametrize(("score", "auc", "ap"), [
    ([0.1, 0.2, 0.8, 0.9], 1.0, 1.0),                      # perfect
    ([0.9, 0.8, 0.2, 0.1], 0.0, 5 / 12),                   # inverted: precision 1/3 then 2/4
    ([0.5, 0.5, 0.5, 0.5], 0.5, 0.5),                      # all tied: one step, the positive rate
])
def test_auc_and_ap_by_hand(score: list[float], auc: float, ap: float) -> None:
    y = [0, 0, 1, 1]
    assert kx.roc_auc(y, score) == pytest.approx(auc)
    assert kx.pr_ap(y, score) == pytest.approx(ap)


def test_roc_and_pr_curves_ends_and_ties() -> None:
    fpr, tpr, thr = kx._metrics.roc_curve([0, 1, 0, 1], [0.3, 0.3, 0.1, 0.9])
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0, 0, 1, 1)
    assert list(thr) == [np.inf, 0.9, 0.3, 0.1]             # the tied 0.3 is one step
    precision, recall, _ = kx._metrics.pr_curve([0, 1, 0, 1], [0.3, 0.3, 0.1, 0.9])
    assert list(precision) == [1, 1, 2 / 3, 0.5]
    assert list(recall) == [0, 0.5, 1, 1]


@pytest.mark.parametrize(("y", "positive"), [
    ([False, True, True], [0, 1, 1]),
    ([1.0, 0.0, 1.0], [1, 0, 1]),
    (["no", "yes", "no"], [0, 1, 0]),
    (pd.Series(["b", "a", "a"]), [1, 0, 0]),
])
def test_binary_positive_class(y: list, positive: list[int]) -> None:
    assert list(kx._metrics.binary(y)) == positive


@pytest.mark.parametrize(("y", "score", "match"), [
    ([1, 1, 1], [0.1, 0.5, 0.9], "only positives"),
    ([0, 0, 0], [0.1, 0.5, 0.9], "only negatives"),
    (["a", "a"], [0.1, 0.9], "one label only"),
    ([0, 1, 2], [0.1, 0.5, 0.9], "pass a 2-D score, one column per class"),
    ([0, 1, 1], [[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]], "a binary metric takes one score per row"),
    ([0, 1], [0.1, 0.5, 0.9], "y_true has 2 values but the score has 3"),
    ([0, 1, 1], [0.1, np.nan, 0.9], "NaN"),
    ([0, np.nan, 1], [0.1, 0.5, 0.9], "missing values"),
])
def test_metrics_raise(y: list, score: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        kx.roc_auc(y, score)


def test_confusion_counts_and_unpredicted_class() -> None:
    m = kx._metrics.confusion(["a", "b", "b", "c", "c"], ["a", "b", "a", "a", "b"])
    assert m.tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 0]]  # 'c' is never predicted but keeps its column
    assert kx._metrics.confusion([0, 1], [1, 1], labels=[1, 0]).tolist() == [[1, 0], [1, 0]]


def test_brier_calibration_and_gain() -> None:
    assert kx._metrics.brier([0, 1, 1], [0.0, 1.0, 0.5]) == pytest.approx(0.25 / 3)
    mean, rate, n = kx._metrics.calibration([0, 0, 1, 1, 1], [0.05, 0.15, 0.15, 0.95, 1.0], bins=10)
    assert mean.tolist() == pytest.approx([0.05, 0.15, 0.975])  # empty bins left out, 1.0 in the last bin
    assert rate.tolist() == [0, 0.5, 1]
    assert n.tolist() == [1, 2, 2]
    with pytest.raises(ValueError, match="between 0 and 1"):
        kx._metrics.calibration([0, 1], [-0.5, 2.0])
    taken, found = kx._metrics.gain([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
    assert taken.tolist() == [0, 0.25, 0.5, 0.75, 1]
    assert found.tolist() == [0, 0.5, 0.5, 1, 1]


# ---------- statistics ----------
# computed once with scipy.stats (kruskal, mannwhitneyu(method="asymptotic"), chi2.sf) and
# scipy.cluster.hierarchy (linkage, leaves_list) on these values
GA = [2.0, -2.6, 0.4, -0.6, -0.5, -0.2, -2.0, -0.2, -0.9, 3.3, 0.2, -0.4, -0.3, -0.7, -1.1]
GB = [0.1, 1.0, 0.3, 1.5, 0.3, 0.5, 2.0, 1.0, -0.0, 0.3, 1.0, 2.4]
GC = [0.6, 0.6, 2.5, -0.3, 0.6, 2.3, 1.9, 1.1, 2.0, -3.2]
POINTS = np.array([[1.0213, -0.9596, -1.6686], [0.2764, 0.7005, -0.4448], [-1.0764, 0.0261, -0.0527],
                   [1.4056, 0.7474, 0.1938], [1.1116, -0.2055, -0.9259], [0.5841, 0.5825, -0.2148],
                   [-0.7828, 0.2292, -2.4939], [0.6901, 0.4914, -1.6389], [0.0614, -0.9641, 0.7572],
                   [-2.0342, -0.9145, 0.7096], [1.1564, -2.158, -0.498], [0.328, -0.6092, 1.5906]])


def test_ranks_share_ties() -> None:
    r, ties = kx._stats.ranks([3, 1, 3, 2, 3])
    assert list(r) == [4, 1, 4, 2, 4]
    assert ties == 24                           # one run of 3 ties: 3³ - 3


def test_kruskal_and_mannwhitney_match_scipy() -> None:
    h, p = kx._stats.kruskal(GA, GB, GC)
    assert h == pytest.approx(9.722380612972211, rel=1e-12)
    assert p == pytest.approx(0.007741263928270015, rel=1e-12)
    u, p = kx._stats.mannwhitney(GA, GB)
    assert u == 29.5
    assert p == pytest.approx(0.003365928106937215, rel=1e-12)


def test_rank_tests_drop_nans_and_reject_ties_only() -> None:
    assert kx._stats.kruskal([*GA, np.nan], GB) == pytest.approx(kx._stats.kruskal(GA, GB))
    with pytest.raises(ValueError, match="every value is the same"):
        kx._stats.kruskal([1, 1], [1, 1, 1])
    with pytest.raises(ValueError, match="every value is the same"):
        kx._stats.mannwhitney([2, 2], [2])
    with pytest.raises(ValueError, match="2 or more groups"):
        kx._stats.kruskal(GA)
    with pytest.raises(ValueError, match="group 2 has no values"):
        kx._stats.kruskal(GA, [np.nan])


@pytest.mark.parametrize(("x", "df", "want"), [
    (0.5, 1, 0.47950012218695337),
    (3.0, 2, 0.22313016014842982),
    (10.0, 4, 0.04042768199451279),
    (40.0, 3, 1.065509033425585e-08),
    (1e-3, 5, 0.9999999983185123),
    (100.0, 10, 5.4497019829205215e-17),
    (0.0, 3, 1.0),
])
def test_chi2_sf_matches_scipy(x: float, df: int, want: float) -> None:
    assert kx._stats.chi2_sf(x, df) == pytest.approx(want, rel=1e-12)


def test_holm_keeps_order_and_rises() -> None:
    assert kx._stats.holm([0.01, 0.04, 0.03, 0.2]) == pytest.approx([0.04, 0.09, 0.09, 0.2])
    assert kx._stats.holm([0.6, 0.5]) == pytest.approx([1, 1])      # capped at 1


def test_cramers_v() -> None:
    x, y = list("aabbccabcaabbcacbcab"), list("xxyyzzxyzxyyzzxzyzxx")
    # scipy.stats.contingency.association(crosstab, method="cramer") for the uncorrected V
    assert kx._stats.cramers_v(x, y, corrected=False) == pytest.approx(0.7911070345636262, rel=1e-12)
    assert kx._stats.cramers_v(x, y) < kx._stats.cramers_v(x, y, corrected=False)
    assert kx._stats.cramers_v(x, x) == pytest.approx(1)
    assert np.isnan(kx._stats.cramers_v(["a"] * 5, list("xyxyx")))   # one category: no association to measure
    labels = np.arange(400)
    noise = (labels * 7919) % 13, (labels * 104729) % 11              # independent-ish, many categories
    assert kx._stats.cramers_v(*noise) < 0.5 * kx._stats.cramers_v(*noise, corrected=False)


def test_cramers_v_drops_missing_pairs() -> None:
    x, y = ["a", "a", "b", "b", None], ["x", "x", "y", "y", "y"]
    assert kx._stats.cramers_v(x, y, corrected=False) == pytest.approx(1)


@pytest.mark.parametrize(("method", "heights", "order"), [
    ("average", [0.4018747193, 0.9442378567, 1.062273124, 1.1151604858, 1.285533481, 1.5437650372, 1.7350597984,
                 2.1503478341, 2.5121640859, 2.724401083, 2.8300260832],
     [8, 11, 2, 9, 10, 6, 3, 1, 5, 7, 0, 4]),
    ("ward", [0.4018747193, 0.9442378567, 1.062273124, 1.2840972925, 1.3719731035, 1.5437650372, 2.554341681,
              2.7326952074, 3.213215435, 3.4674198816, 5.1363884748],
     [3, 1, 5, 6, 7, 0, 4, 2, 9, 10, 8, 11]),
])
def test_linkage_matches_scipy(method: str, heights: list[float], order: list[int]) -> None:
    z = kx._stats.linkage(POINTS, method)
    assert z.shape == (11, 4)
    assert z[:, 2] == pytest.approx(heights, abs=1e-9)
    assert z[-1, 3] == 12                       # the last join holds every leaf
    assert (z[:, 0] < z[:, 1]).all()
    assert kx._stats.leaves(z) == order


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
def test_linkage_heights_rise_and_precomputed_agrees(method: str) -> None:
    z = kx._stats.linkage(POINTS, method)
    assert (np.diff(z[:, 2]) >= -1e-12).all()
    d = np.sqrt(((POINTS[:, None] - POINTS[None]) ** 2).sum(-1))
    assert kx._stats.linkage(d, method, distances=True) == pytest.approx(z)
    assert sorted(kx._stats.leaves(z)) == list(range(12))


@pytest.mark.parametrize(("args", "kw", "match"), [
    ([[1.0]], {}, "2 or more rows"),
    ([np.array([[1.0, np.nan], [2.0, 3.0]])], {}, "finite"),
    ([np.zeros((2, 3))], {"distances": True}, "square"),
    ([np.zeros((3, 2))], {"method": "median"}, "unknown method"),
    ([np.zeros((kx._stats.MAX_LEAVES + 1, 1))], {}, "at most"),
])
def test_linkage_errors(args: list, kw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        kx._stats.linkage(*args, **kw)


# ---------- data ----------
def test_datasets_and_load() -> None:
    assert {"waves", "cloud", "monthly", "dists", "daily", "passengers"} <= set(kx.datasets())
    assert list(kx.load("waves").columns) == ["x", "y", "z"]


def test_daily_is_two_full_years_of_days() -> None:
    daily = kx.load("daily")
    assert list(daily.columns) == ["date", "temp", "sales"]
    days = pd.to_datetime(daily.date)
    assert days.diff().iloc[1:].eq(pd.Timedelta("1D")).all()
    assert (days.iloc[0], days.iloc[-1]) == (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31"))


def test_passengers_has_categories_and_nans() -> None:
    p = kx.load("passengers")
    assert p.shape == (500, 8)
    assert list(p.columns) == ["age", "fare", "sex", "pclass", "port", "deck", "ticket_type", "survived"]
    assert all(pd.api.types.is_float_dtype(p[c]) for c in ("age", "fare"))
    assert all(pd.api.types.is_integer_dtype(p[c]) for c in ("pclass", "survived"))
    for c in ("sex", "port", "deck", "ticket_type"):
        assert all(isinstance(v, str) for v in p[c].dropna())
    assert set(p.pclass) == {1, 2, 3}
    assert set(p.survived) == {0, 1}
    assert p.sex.nunique() == 2
    assert p.port.nunique() == 3
    assert set(p.deck.dropna()) == set("ABCDEFG")
    assert p.ticket_type.str.len().min() > 12
    assert p.fare.mean() > p.fare.median()
    nans = p.isna().mean()
    assert nans.age == pytest.approx(0.2, abs=0.05)
    assert nans.deck == pytest.approx(0.75, abs=0.05)
    assert 0 < nans.port < 0.03
    assert (nans.drop(["age", "deck", "port"]) == 0).all()


def test_ml_datasets_shapes_and_story() -> None:
    b, m, r = kx.load("binary"), kx.load("multiclass"), kx.load("regression")
    h, imp, c = kx.load("history"), kx.load("importance"), kx.load("curves")
    assert list(b.columns) == ["y", "gbm", "logreg"]
    assert list(m.columns) == ["y", "cat", "dog", "bird", "pred"]
    assert list(r.columns) == ["actual", "predicted", "x"]
    assert list(h.columns) == ["epoch", "loss", "val_loss", "accuracy", "val_accuracy"]
    assert list(imp.columns) == ["feature", "importance", "std"]
    assert list(c.columns) == ["size", *(f"train_{k}" for k in range(1, 6)), *(f"val_{k}" for k in range(1, 6))]
    assert (b.shape, m.shape, r.shape, h.shape, imp.shape, c.shape) == (
        (1000, 3), (600, 5), (500, 3), (60, 5), (15, 3), (10, 11))
    assert set(b.y) == {0, 1}
    assert b[["gbm", "logreg"]].stack().between(0, 1).all()
    assert kx.roc_auc(b.y, b.gbm) > kx.roc_auc(b.y, b.logreg)
    assert set(m.y) == set(m.pred) == {"cat", "dog", "bird"}
    assert np.allclose(m[["cat", "dog", "bird"]].sum(axis=1), 1)
    assert (m.pred == m[["cat", "dog", "bird"]].idxmax(axis=1)).all()
    err = (r.actual - r.predicted).abs()
    assert err[r.x > 7].mean() > 2 * err[r.x < 3].mean()
    assert 0 < h.val_loss.idxmin() < len(h) - 1             # overfits: the minimum is inside, not at the end
    assert h.loss.idxmin() > h.val_loss.idxmin()
    assert imp.importance.sum() == pytest.approx(1, abs=1e-3)
    assert c.val_1.iloc[-1] > c.val_1.iloc[0]
    assert c.train_1.iloc[-1] < c.train_1.iloc[0]


def test_load_unknown() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        kx.load("nope")


def test_version_comes_from_pyproject() -> None:
    toml = (Path(kx.__file__).parent.parent / "pyproject.toml").read_text()
    assert f'\nversion = "{kx.__version__}"\n' in toml
