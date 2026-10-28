"""Distribution kinds: hist, kde, ridge, ecdf, box, violin, strip, qq."""
from statistics import NormalDist

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from helpers import X, Y, Z
from matplotlib.colors import to_hex

import kx


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
