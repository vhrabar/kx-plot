"""Basic kinds: line, scatter, step, area, band, bar, barh, count."""
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
