"""Flags, titles and labels."""
from collections.abc import Callable
from itertools import pairwise

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from helpers import DF, PASSENGERS, X, Y, Z, _dashed
from matplotlib.axes import Axes
from matplotlib.colors import to_hex

import kx
import kx._stats
import kx.kinds.dist


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


# ---------- facet ----------
def _shared(axs: np.ndarray, which: str) -> bool:
    return all(getattr(a, f"get_shared_{which}_axes")().joined(a, axs[0]) for a in axs)


def test_facet_crosstab_one_panel_per_value_on_one_colour_scale() -> None:
    axs = kx.plot(PASSENGERS, "pclass", "port", "crosstab-facet-ann", facet="sex", title="ports")
    assert isinstance(axs, np.ndarray)
    assert [a.get_title() for a in axs] == ["sex = female", "sex = male"]
    for a, sex in zip(axs, ["female", "male"], strict=True):
        part = PASSENGERS[PASSENGERS.sex == sex]
        want = pd.crosstab(part.port, part.pclass).to_numpy()
        assert np.array_equal(a.images[0].get_array(), want)
    assert axs[0].images[0].get_clim() == axs[1].images[0].get_clim()
    fig = axs[0].figure
    bars = [a for a in fig.axes if a not in axs]
    assert len(bars) == 1                                         # one colorbar for both
    assert bars[0].get_ylabel() == "count"
    assert fig._suptitle.get_text() == "ports"


def test_facet_keeps_every_category_in_every_panel() -> None:
    axs = kx.plot(PASSENGERS, "deck", spec="count-facet", facet="pclass")
    ticks = [[t.get_text() for t in a.get_xticklabels()] for a in axs]
    assert ticks[0] == ticks[1] == ticks[2] == [*"ABCDEFG", "NaN"]
    second = PASSENGERS[PASSENGERS.pclass == 2].deck.value_counts(dropna=False)
    heights = [p.get_height() for p in axs[1].patches]
    assert heights[0] == second.get("A", 0)                      # a deck missing in this class: a 0 bar
    assert _shared(axs, "x")
    assert _shared(axs, "y")
    assert len({a.get_ylim() for a in axs}) == 1


def test_facet_scatter_shares_axes_and_the_z_colour_scale() -> None:
    frame = kx.load("embedding")
    axs = kx.plot(frame, "x", "y", "size", spec="scatter-facet", facet="group")
    assert len(axs) == 2
    assert axs[0].collections[0].get_clim() == axs[1].collections[0].get_clim()
    assert axs[0].get_xlim() == axs[1].get_xlim()
    assert len(axs[0].figure.axes) == 3
    for a, g in zip(axs, ["batch_a", "batch_b"], strict=True):
        assert len(a.collections[0].get_offsets()) == (frame.group == g).sum()


def test_facet_with_by_and_a_table_kind() -> None:
    axs = kx.plot(PASSENGERS, "age", spec="kde-facet", by="sex", facet="pclass")
    assert [len(a.lines) for a in axs] == [2, 2, 2]
    assert _shared(axs, "x")
    corr = kx.plot(PASSENGERS[["age", "fare", "pclass", "sex"]], "heat-corr-facet", facet="sex")
    labels = [t.get_text() for t in corr[0].get_xticklabels()]
    assert labels == ["age", "fare", "pclass"]                   # the facet column is not a column of the table


def test_facet_does_not_share_axes_that_show_different_categories() -> None:
    frame = pd.DataFrame({"cat": ["a", "b", "c", "d"], "v": [1.0, 2, 3, 4], "g": ["x", "x", "y", "y"]})
    axs = kx.plot(frame, "cat", "v", "bar-facet", facet="g")
    assert [t.get_text() for t in axs[0].get_xticklabels()] == ["a", "b"]
    assert [t.get_text() for t in axs[1].get_xticklabels()] == ["c", "d"]
    assert not axs[1].get_shared_x_axes().joined(axs[1], axs[0])
    assert _shared(axs, "y")                                      # the values still share a scale


def test_facet_polar_kind_and_layout() -> None:
    frame = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 1, 4, 3], "c": [3.0, 3, 1, 2], "g": list("pqpq")})
    axs = kx.plot(frame, "radar-facet", facet="g")
    assert all(a.name == "polar" for a in axs)
    many = kx.plot(np.arange(12.0), np.arange(12.0), "scatter-facet", facet=np.repeat(list("abcdef"), 2))
    assert len(many) == 6
    gs = many[0].get_subplotspec().get_gridspec()
    assert (gs.nrows, gs.ncols) == (2, 3)                        # past 4 values: 3 per row
    w, h = mpl.rcParams["figure.figsize"]
    assert tuple(many[0].figure.get_size_inches()) == pytest.approx((3 * w, 2 * h))


def test_facet_in_a_grid_cell() -> None:
    left, right = kx.grid(2)
    size = tuple(left.figure.get_size_inches())
    axs = kx.plot(X, Y, "line-facet", facet=X > 5, title="split", ax=right)
    assert len(axs) == 2
    assert tuple(left.figure.get_size_inches()) == size
    assert all(a.get_subplotspec().get_topmost_subplotspec() == right.get_subplotspec() for a in axs)
    assert axs[0].get_title() == "split\nfacet = False"


@pytest.mark.parametrize(("call", "error", "match"), [
    (lambda: kx.plot(X, Y, "line-facet"), ValueError, "needs facet="),
    (lambda: kx.plot(X, Y, "line", facet=X > 5), ValueError, "add the flag: line-facet"),
    (lambda: kx.plot(PASSENGERS, "pair-facet", facet="sex"), ValueError, "draws several panels already"),
    (lambda: kx.plot(X, Y, "line-facet", facet=[1, 2]), ValueError, "facet= has 2 values but the data has 20"),
    (lambda: kx.plot(X, Y, "line-facet", facet="sex"), TypeError, "needs a DataFrame first"),
])
def test_facet_errors(call: Callable[[], Axes], error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        call()


# ---------- sig ----------
LOW = np.arange(20.0)
SAME = LOW + 0.5                                                  # overlaps LOW: no difference
HIGH = LOW + 30                                                   # above both: differs from each


def _brackets(ax: Axes) -> list[tuple[float, float, float]]:
    """(left, right, height) of each significance bracket: the 4-point lines."""
    return [(ln.get_xdata()[0], ln.get_xdata()[2], ln.get_ydata()[1]) for ln in ax.lines if len(ln.get_xdata()) == 4]


@pytest.mark.parametrize("kind", ["box", "violin"])
def test_sig_brackets_only_the_pairs_that_differ(kind: str) -> None:
    ax = kx.plot(LOW, SAME, HIGH, f"{kind}-sig")
    assert sorted((a, b) for a, b, _ in _brackets(ax)) == [(1, 3), (2, 3)]
    pairs = [(LOW, SAME), (LOW, HIGH), (SAME, HIGH)]
    adjusted = kx._stats.holm([kx._stats.mannwhitney(a, b)[1] for a, b in pairs])
    marks = next(m for limit, m in kx.kinds.dist.STARS if adjusted[1] < limit)
    assert [t.get_text() for t in ax.texts].count(marks) == 2
    _, p = kx._stats.kruskal(LOW, SAME, HIGH)
    assert f"Kruskal-Wallis {kx.kinds.dist._p(p)}" in [t.get_text() for t in ax.texts]
    assert all(h > HIGH.max() for _, _, h in _brackets(ax))         # above the data
    assert ax.get_ylim()[1] > max(h for _, _, h in _brackets(ax))   # and inside the axes


def test_sig_ann_writes_p_values_and_stacks_without_overlap() -> None:
    groups = [LOW, LOW + 25, LOW + 50, LOW + 75]
    ax = kx.plot(np.concatenate(groups), spec="box-sig-ann", by=np.repeat(list("abcd"), 20))
    brackets = _brackets(ax)
    assert len(brackets) == 6                                     # every pair differs
    assert sum(t.get_text().startswith("p ") for t in ax.texts) == 6
    for h in {h for _, _, h in brackets}:                         # one level: spans never touch
        spans = sorted((a, b) for a, b, hh in brackets if hh == h)
        assert all(b1 < a2 for (_, b1), (a2, _) in pairwise(spans))
    assert min(b - a for a, b, _ in brackets) == 1
    lowest = min(h for _, _, h in brackets)
    assert all(b - a == 1 for a, b, h in brackets if h == lowest)   # neighbours sit lowest


def test_sig_over_six_groups_is_the_corner_note_only() -> None:
    values = np.concatenate([LOW + 10 * k for k in range(7)])
    ax = kx.plot(values, spec="violin-sig", by=np.repeat(list("abcdefg"), 20))
    assert not _brackets(ax)
    assert any(t.get_text().startswith("Kruskal-Wallis") for t in ax.texts)


def test_sig_works_with_facet_and_by_columns() -> None:
    axs = kx.plot(PASSENGERS, "fare", spec="box-sig-facet", by="pclass", facet="sex")
    assert all(_brackets(a) for a in axs)


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot(LOW, spec="box-sig"), "2 or more groups"),
    (lambda: kx.plot(LOW, HIGH, spec="violin-sig-logy"), "linear value axis"),
    (lambda: kx.plot([1.0, 1.0], [1.0, 1.0], spec="box-sig"), "every value is the same"),
    (lambda: kx.plot(LOW, spec="kde-sig"), "only works with kind"),
])
def test_sig_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()
