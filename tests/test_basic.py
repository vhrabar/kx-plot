"""Basic kinds: line, scatter, step, area, band, bar, barh, count, pie, donut, radar, parallel."""
from collections.abc import Callable

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from helpers import PASSENGERS, X, Y, Z
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
