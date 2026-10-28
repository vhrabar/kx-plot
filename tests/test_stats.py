"""Statistics in kx._stats, checked against scipy."""
import numpy as np
import pytest

import kx
import kx._stats

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
