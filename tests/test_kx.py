import re
from collections.abc import Callable
from pathlib import Path
from statistics import NormalDist

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_hex

import kx

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
    assert kx.cmaps() == sorted(n for n in kx.BUILTIN_CMAPS + kx.VENDORED_CMAPS if n in mpl.colormaps)
    assert set(kx.BUILTIN_CMAPS) - {"berlin", "managua", "vanimo"} <= set(kx.cmaps())   # only these need 3.10


def test_cmaps_skips_names_missing_from_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kx._theme, "BUILTIN_CMAPS", ("viridis", "notacmap"))
    monkeypatch.setattr(kx._theme, "VENDORED_CMAPS", ())
    assert kx.cmaps() == ["viridis"]


@pytest.mark.parametrize("name", kx.VENDORED_CMAPS)
def test_vendored_cmaps_are_registered(name: str) -> None:
    assert mpl.colormaps[name].N == 256
    assert name in kx.cmaps()


@pytest.mark.parametrize("name", ["viridis", *kx.VENDORED_CMAPS])
def test_use_reversed_cmap(name: str) -> None:
    kx.use("light", cmap=f"{name}_r")
    rev = kx.plot(X, Y, Z, "scatter").collections[0].get_cmap()
    assert rev.name == f"{name}_r"
    assert to_hex(rev(1.0)) == to_hex(mpl.colormaps[name](0.0))


@pytest.mark.parametrize("name", kx.BUILTIN_CMAPS + kx.VENDORED_CMAPS)
def test_use_cmap_colours_scatter(name: str) -> None:
    if name not in mpl.colormaps:
        pytest.skip(f"{name} needs a newer matplotlib")
    kx.use("light", cmap=name)
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


# ---------- data ----------
def test_datasets_and_load() -> None:
    assert {"waves", "cloud", "monthly", "dists", "daily"} <= set(kx.datasets())
    assert list(kx.load("waves").columns) == ["x", "y", "z"]


def test_daily_is_two_full_years_of_days() -> None:
    daily = kx.load("daily")
    assert list(daily.columns) == ["date", "temp", "sales"]
    days = pd.to_datetime(daily.date)
    assert days.diff().iloc[1:].eq(pd.Timedelta("1D")).all()
    assert (days.iloc[0], days.iloc[-1]) == (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31"))


def test_load_unknown() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        kx.load("nope")


def test_version_comes_from_pyproject() -> None:
    toml = (Path(kx.__file__).parent.parent / "pyproject.toml").read_text()
    assert f'\nversion = "{kx.__version__}"\n' in toml
