"""Basic kinds: line, scatter, step, area, band, bar, barh, lollipop, count, pie, donut, radar, parallel."""
import datetime as dt
from collections.abc import Callable

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from helpers import PASSENGERS, X, Y, Z, _drawn, _same, _to_polars
from matplotlib.axes import Axes
from matplotlib.colors import to_hex

import kx


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


def _heights(ax: Axes) -> list[list[float]]:
    return [[p.get_height() for p in c] for c in ax.containers]


CATS = ["a", "b", "c"]


def _dots(ax: Axes) -> list[list[float]]:
    """Each lollipop dot series as (value, row) pairs, top row first."""
    return [list(zip(line.get_xdata(), line.get_ydata(), strict=True)) for line in ax.lines]


def _segments(ax: Axes) -> list[tuple[float, float, float]]:
    """The stems (or dumbbell bars) as (start, end, row)."""
    (lines,) = ax.collections
    return [(a[0], b[0], a[1]) for a, b in lines.get_segments()]


def test_lollipop_dots_on_stems_from_zero() -> None:
    ax = kx.plot(CATS, [3, -1, 2], "lollipop")
    assert _dots(ax) == [[(3, 0), (-1, 1), (2, 2)]]
    assert _segments(ax) == [(0, 3, 0), (0, -1, 1), (0, 2, 2)]
    assert [t.get_text() for t in ax.get_yticklabels()] == CATS
    assert ax.yaxis_inverted()
    assert to_hex(ax.lines[0].get_color()) == to_hex("C0")
    assert ax.get_legend() is None
    assert ax.get_xlim()[0] < -1


def test_lollipop_dumbbell_joins_y_and_z() -> None:
    y, z = pd.Series([1, 4, 2], name="y2025"), pd.Series([3, 5, 1], name="y2026")
    ax = kx.plot(CATS, y, z, "lollipop")
    assert _dots(ax) == [[(1, 0), (4, 1), (2, 2)], [(3, 0), (5, 1), (1, 2)]]
    assert _segments(ax) == [(1, 3, 0), (4, 5, 1), (2, 1, 2)]
    assert [to_hex(line.get_color()) for line in ax.lines] == [to_hex("C0"), to_hex("C1")]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["y2025", "y2026"]


@pytest.mark.parametrize(("z", "cats"), [(None, ["b", "c", "a"]), ([3, 5, 1], ["a", "b", "c"])])
def test_lollipop_sort_by_value_or_difference(z: list | None, cats: list) -> None:
    ax = kx.plot(CATS, [1, 4, 2], z, "lollipop-sort")
    assert [t.get_text() for t in ax.get_yticklabels()] == cats
    values = dict(zip(CATS, [1, 4, 2], strict=True))
    assert [v for v, _ in _dots(ax)[0]] == [values[c] for c in cats]


def test_lollipop_ann_beside_each_dot() -> None:
    ax = kx.plot(CATS, [3, -1, 2], "lollipop-ann")
    texts = {t.get_text(): (t.xy, t.get_ha()) for t in ax.texts}
    assert texts == {"3": ((3, 0), "left"), "-1": ((-1, 1), "right"), "2": ((2, 2), "left")}


def test_lollipop_ann_on_the_outside_of_a_dumbbell() -> None:
    ax = kx.plot(CATS, [1, 4, 2], [3, 5, 1], "lollipop-ann")
    sides = {(t.xy[1], t.xy[0]): t.get_ha() for t in ax.texts}
    assert sides == {(0, 1): "right", (0, 3): "left", (1, 4): "right", (1, 5): "left",
                     (2, 2): "left", (2, 1): "right"}


def test_lollipop_value_axis_is_x() -> None:
    ax = kx.plot(CATS, [0.1, 0.5, 0.25], "lollipop-pct")
    assert isinstance(ax.xaxis.get_major_formatter(), mpl.ticker.PercentFormatter)
    assert kx.plot(CATS, [2, 3, 4], [5, 6, 7], "lollipop-zero").get_xlim()[0] == 0


def test_lollipop_errors() -> None:
    with pytest.raises(ValueError, match="one value per category"):
        kx.plot(CATS, [1, 2], "lollipop")
    with pytest.raises(ValueError, match="1-D"):
        kx.plot(CATS, np.ones((3, 2)), "lollipop")


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


# ---------- pie, donut, radar, parallel ----------
def _spans(ax: Axes) -> list[float]:
    """Each wedge's angle as a share of the circle."""
    return [round((w.theta2 - w.theta1) / 360, 6) for w in ax.patches]


def test_pie_counts_categories_with_nan_from_twelve_clockwise() -> None:
    ax = kx.plot(PASSENGERS, "port", "pie")
    want = PASSENGERS.port.value_counts(dropna=False)
    names = [w.get_label() for w in ax.patches]
    assert names == ["Cherbourg", "Queenstown", "Southampton", "NaN"]
    assert _spans(ax) == pytest.approx([want[np.nan if n == "NaN" else n] / len(PASSENGERS) for n in names])
    assert ax.patches[0].theta2 == pytest.approx(90)            # starts at 12 o'clock, going clockwise
    assert [t.get_text() for t in ax.texts] == names              # labels beside the slices


def test_pie_y_sums_values_per_category() -> None:
    ax = kx.plot(["a", "b", "a", "c"], [1, 2, 3, 4], "pie-sort")
    assert [w.get_label() for w in ax.patches] == ["a", "c", "b"]   # a = 1 + 3 ties c; stable keeps a first
    assert _spans(ax) == pytest.approx([0.4, 0.4, 0.2])


def test_pie_folds_slices_past_top_into_grey_other() -> None:
    cats = list("abcdefghij")
    ax = kx.plot(cats, list(range(10, 0, -1)), "pie", top=4)
    assert [w.get_label() for w in ax.patches] == ["a", "b", "c", "other"]
    assert _spans(ax)[-1] == pytest.approx(sum(range(1, 8)) / 55)
    assert to_hex(ax.patches[-1].get_facecolor()) == to_hex("0.6")
    ax = kx.plot(["other", "x"], [1, 1], "pie")                    # a real 'other' keeps its colour
    assert to_hex(ax.patches[0].get_facecolor()) != to_hex("0.6")


def test_pie_ann_writes_shares_in_contrasting_ink() -> None:
    ax = kx.plot(["a", "a", "a", "b"], spec="pie-ann")
    shares = [t for t in ax.texts if t.get_text().endswith("%")]
    assert [t.get_text() for t in shares] == ["75%", "25%"]
    assert all(t.get_color() in ("black", "white") for t in shares)


@pytest.mark.parametrize("flag", ["leg", "legout"])
def test_pie_legend_replaces_slice_labels_and_sits_beside(flag: str) -> None:
    ax = kx.plot(PASSENGERS, "sex", f"pie-{flag}")
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["female", "male"]
    assert legend.get_title().get_text() == "sex"
    assert not [t for t in ax.texts if t.get_text() in ("female", "male")]
    assert legend.get_bbox_to_anchor().x0 >= ax.get_window_extent().x1 - 1   # outside the pie, on the right


def test_donut_ring_with_the_total_in_the_hole() -> None:
    ax = kx.plot(PASSENGERS, "pclass", "donut", hole=0.5)
    assert all(w.width == pytest.approx(0.5) for w in ax.patches)
    assert "500" in [t.get_text() for t in ax.texts]
    assert sum(_spans(ax)) == pytest.approx(1)


@pytest.mark.parametrize(("call", "error", "match"), [
    (lambda: kx.plot(["a"], [1], [2], "pie"), ValueError, "no z"),
    (lambda: kx.plot(["a", "b"], [1, -1], "pie"), ValueError, "negative"),
    (lambda: kx.plot(["a", "b"], [0, 0], "pie"), ValueError, "positive total"),
    (lambda: kx.plot(["a", "b"], [1, 2, 3], "pie"), ValueError, "2 categories but y has 3"),
    (lambda: kx.plot(["a"], spec="donut", hole=1), ValueError, "hole="),
])
def test_pie_errors(call: Callable[[], Axes], error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        call()


PROFILES = pd.DataFrame({"asym": [0.2, 0.8, 0.5, 0.4], "border": [1.0, 3.0, 2.0, 2.0],
                         "colour": [10.0, 30, 20, 20], "diam": [5.0, 5, 6, 8],
                         "dx": ["mel", "mel", "nev", "nev"]})


def test_radar_by_averages_groups_scaled_per_column() -> None:
    ax = kx.plot(PROFILES, "radar", by="dx")
    assert ax.name == "polar"
    assert [ln.get_label() for ln in ax.lines] == ["mel", "nev"]
    mel, nev = (ln.get_ydata() for ln in ax.lines)
    assert list(mel) == pytest.approx([1, 0.5, 0.5, 0, 1])       # means 0.5 2 20 5 against 0.45 2 20 7, closed
    assert list(nev) == pytest.approx([0, 0.5, 0.5, 1, 0])       # border and colour tie: a constant sits mid
    assert [t.get_text() for t in ax.get_xticklabels()][::3] == ["asym\n0.45 to 0.5", "diam\n5 to 7"]
    assert ax.get_legend().get_title().get_text() == "dx"
    assert ax.get_ylim() == (0, 1)


def test_radar_rows_raw_first_column_on_top() -> None:
    ax = kx.plot(PROFILES.drop(columns="dx").iloc[:2], "radar-raw")
    assert list(ax.lines[1].get_ydata()) == [0.8, 3, 30, 5, 0.8]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["asym", "border", "colour", "diam"]
    assert ax.get_theta_offset() == pytest.approx(np.pi / 2)
    assert ax.get_theta_direction() == -1


def test_radar_takes_an_array_by_and_swaps_a_grid_cell() -> None:
    _, right = kx.grid(2)
    ax = kx.plot(PROFILES.drop(columns="dx"), "radar", by=np.array(["a", "b", "a", "b"]), ax=right)
    assert ax.name == "polar"
    assert ax.get_subplotspec() == right.get_subplotspec()
    assert len(ax.lines) == 2


@pytest.mark.parametrize(("frame", "match"), [
    (PROFILES, r"columns \['dx'\] are not"),
    (PROFILES[["asym", "diam"]], "at least 3 numeric columns"),
    (pd.DataFrame(np.ones((11, 3))), "11 rows are too many"),
])
def test_radar_errors(frame: pd.DataFrame, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        kx.plot(frame, "radar")


def test_parallel_lines_per_group_scaled_with_real_ends() -> None:
    ax = kx.plot(PROFILES, "parallel-mean", by="dx")
    groups = ax.collections[:2]
    assert [len(c.get_segments()) for c in groups] == [2, 2]
    first = groups[0].get_segments()[0]
    assert list(first[:, 0]) == [0, 1, 2, 3]
    assert list(first[:, 1]) == pytest.approx([0, 0, 0, 0])     # the first row is the minimum everywhere
    assert {"0.2", "0.8", "1", "3", "5", "8"} <= {t.get_text() for t in ax.texts}   # each axis' min and max
    assert [t.get_text() for t in ax.get_xticklabels()] == ["asym", "border", "colour", "diam"]
    means = [ln for ln in ax.lines if ln.get_linewidth() == 3]
    assert len(means) == 2
    assert list(means[0].get_ydata()) == pytest.approx([0.5, 0.5, 0.5, 0])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["mel", "nev"]
    assert to_hex(groups[0].get_color()[0]) == to_hex(ax.get_legend().legend_handles[0].get_color())


def test_parallel_without_by_is_one_fading_set() -> None:
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(400, 3)), columns=list("abc"))
    frame["c"] = 1.0                                            # a constant column sits mid-height
    ax = kx.plot(frame, "parallel")
    lines = ax.collections[0]
    assert len(lines.get_segments()) == 400
    assert lines.get_alpha() == pytest.approx(0.1)
    assert all(seg[2, 1] == 0.5 for seg in lines.get_segments())
    assert ax.get_legend() is None


def test_table_kinds_take_by_only_when_they_group() -> None:
    with pytest.raises(TypeError, match=r"takes a whole table and no columns, y, z or by="):
        kx.plot(PROFILES.drop(columns="dx"), "heat", by=PROFILES.dx)
    with pytest.raises(ValueError, match="no column 'dz'"):
        kx.plot(PROFILES, "parallel", by="dz")


# ---------- pandas and polars draw the same ----------
_rng = np.random.default_rng(0)
MIXED = pd.DataFrame({
    "t": np.arange(40, dtype=float),
    "a": _rng.normal(size=40).round(2),
    "b": _rng.normal(size=40).round(2) + 1,
    "c": _rng.uniform(1, 5, size=40).round(2),
    "g": _rng.choice(["x", "y", "z"], size=40).astype(object),
    "h": _rng.choice(["p", "q"], size=40),
    "k": _rng.integers(0, 4, size=40),
    "flag": _rng.choice([True, False], size=40),
    "day": pd.date_range("2025-01-01", periods=40, freq="D"),
    "text day": pd.date_range("2025-01-01", periods=40, freq="D").strftime("%Y-%m-%d"),
    "level": pd.Categorical(_rng.choice(["lo", "hi"], size=40), categories=["lo", "mid", "hi"]),
})
MIXED.loc[[3, 7], "g"] = None
MIXED.loc[5, "a"] = np.nan
NUMBERS = MIXED[["a", "b", "c", "t"]]
SHORT = pd.DataFrame({"name": list("abcde"), "v": [3.0, 1, 4, 1.5, 9], "w": [2.0, 2, 5, 1, 7]})

PARITY = {
    "line": lambda f: kx.plot(f(MIXED), "t", "a", "b", "line-leg"),
    "line endlab": lambda f: kx.plot(f(MIXED), "t", "a", "b", "line-endlab"),
    "line date": lambda f: kx.plot(f(MIXED), "day", "a", "line-date"),
    "line text date": lambda f: kx.plot(f(MIXED), "text day", "a", "line-date-mk"),
    "line ci numbers": lambda f: kx.plot(f(MIXED), "k", "a", "line-ci", n_boot=50),
    "line ci text": lambda f: kx.plot(f(MIXED), "g", "a", "line-ci", n_boot=50),
    "line ci enum": lambda f: kx.plot(f(MIXED), "level", "a", "line-ci", n_boot=50),
    "line ci bool": lambda f: kx.plot(f(MIXED), "flag", "a", "line-ci", n_boot=50),
    "step ci": lambda f: kx.plot(f(MIXED), "k", "b", "step-ci", n_boot=50),
    "area": lambda f: kx.plot(f(MIXED), "t", "c", "b", "area-leg"),
    "area stack": lambda f: kx.plot(f(MIXED), "t", "c", "b", "area-stack"),
    "area norm": lambda f: kx.plot(f(MIXED), "t", "c", "b", "area-norm-leg"),
    "band": lambda f: kx.plot(f(MIXED), "t", "a", "c", "band-leg"),
    "scatter numbers z": lambda f: kx.plot(f(MIXED), "a", "b", "c", "scatter-sym"),
    "scatter text z fit": lambda f: kx.plot(f(MIXED), "a", "b", "g", "scatter-fit"),
    "scatter enum z": lambda f: kx.plot(f(MIXED), "a", "b", "level", "scatter"),
    "scatter bool z": lambda f: kx.plot(f(MIXED), "a", "b", "flag", "scatter"),
    "scatter fit diag": lambda f: kx.plot(f(MIXED), "a", "b", "scatter-fit-diag"),
    "bar": lambda f: kx.plot(f(SHORT), "name", "v", "w", "bar-ann-sort"),
    "bar stack": lambda f: kx.plot(f(SHORT), "name", "v", "w", "bar-stack-ann"),
    "bar ci": lambda f: kx.plot(f(MIXED), "g", "a", "bar-ci", n_boot=50),
    "barh": lambda f: kx.plot(f(SHORT), "name", "v", "barh-sort"),
    "barh ci enum": lambda f: kx.plot(f(MIXED), "level", "a", "barh-ci", n_boot=50),
    "count": lambda f: kx.plot(f(MIXED), "g", spec="count-ann"),
    "count by": lambda f: kx.plot(f(MIXED), "g", spec="count-stack-norm", by="h"),
    "count numbers": lambda f: kx.plot(f(MIXED), "k", spec="count-sort"),
    "count bool": lambda f: kx.plot(f(MIXED), "flag", spec="count"),
    "count by enum": lambda f: kx.plot(f(MIXED), "h", spec="count", by="level"),
    "count passengers": lambda f: kx.plot(f(PASSENGERS), "port", spec="count", by="sex"),
    "lollipop": lambda f: kx.plot(f(SHORT), "name", "v", "lollipop-ann"),
    "dumbbell": lambda f: kx.plot(f(SHORT), "name", "v", "w", "lollipop-sort-ann"),
    "pie": lambda f: kx.plot(f(MIXED), "g", spec="pie-ann"),
    "pie y": lambda f: kx.plot(f(SHORT), "name", "v", "pie-sort"),
    "pie y with gaps": lambda f: kx.plot(f(MIXED), "g", "c", "pie"),
    "donut": lambda f: kx.plot(f(PASSENGERS), "pclass", spec="donut-ann", top=3),
    "radar": lambda f: kx.plot(f(NUMBERS.head(3)), "radar"),
    "radar by": lambda f: kx.plot(f(MIXED[["a", "b", "c", "t", "g"]]), "radar", by="g"),
    "radar raw": lambda f: kx.plot(f(NUMBERS.head(4)), "radar-raw"),
    "parallel": lambda f: kx.plot(f(NUMBERS), "parallel"),
    "parallel by": lambda f: kx.plot(f(MIXED[["a", "b", "c", "g"]]), "parallel-mean", by="g"),
    "parallel constant and empty": lambda f: kx.plot(f(NUMBERS.assign(k=1.0, b=np.nan)), "parallel"),
    "bar facet": lambda f: kx.plot(f(MIXED), "g", "a", "bar-ci-facet", facet="h", n_boot=50),
    "count facet": lambda f: kx.plot(f(MIXED), "g", spec="count-facet", facet="h"),
    "scatter facet enum": lambda f: kx.plot(f(MIXED), "a", "b", "g", "scatter-facet", facet="level"),
    "line facet": lambda f: kx.plot(f(MIXED), "t", "a", "line-facet", facet="h"),
    "radar facet series": lambda f: kx.plot(f(MIXED[["a", "b", "c", "h"]]), "radar-facet", by="h",
                                            facet=f(MIXED["flag"])),
    "line series": lambda f: kx.plot(f(MIXED["t"]), f(MIXED["a"]), "line-leg"),
}


@pytest.mark.parametrize("name", PARITY)
def test_basic_kinds_draw_the_same_from_polars(name: str) -> None:
    pytest.importorskip("polars")
    assert _drawn(PARITY[name](_to_polars)) == _drawn(PARITY[name](_same))


def test_count_on_an_enum_keeps_its_order_and_unused_levels() -> None:
    pl = pytest.importorskip("polars")
    levels = ["low", "mid", "high"]
    values = ["high", "low", "high", None]
    polars = kx.plot(pl.DataFrame({"v": pl.Series(values, dtype=pl.Enum(levels))}), "v", spec="count")
    pandas = kx.plot(pd.DataFrame({"v": pd.Categorical(values, categories=levels)}), "v", spec="count")
    assert [t.get_text() for t in polars.get_xticklabels()] == ["low", "mid", "high", "NaN"]
    assert [p.get_height() for p in polars.patches] == [1, 0, 2, 1]
    assert _drawn(polars) == _drawn(pandas)


def test_count_on_a_polars_categorical_keeps_its_order() -> None:
    pl = pytest.importorskip("polars")
    ax = kx.plot(pl.DataFrame({"v": pl.Series(["z", "y", "z"], dtype=pl.Categorical)}), "v", spec="count")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["z", "y"]


def test_line_date_on_a_polars_date_column() -> None:
    pl = pytest.importorskip("polars")
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(30)]
    polars = kx.plot(pl.DataFrame({"day": days, "v": np.arange(30.0)}), "day", "v", "line-date")
    pandas = kx.plot(pd.DataFrame({"day": pd.to_datetime(days), "v": np.arange(30.0)}), "day", "v", "line-date")
    assert _drawn(polars) == _drawn(pandas)
