"""
Statistics for the significance, association and clustering kinds, in numpy with no scipy needed
"""
import math

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

MAX_LEAVES = 1000
METHODS = ("single", "complete", "average", "ward")


def _values(a: ArrayLike, name: str) -> np.ndarray:
    """a as a flat float array without NaNs, checked to have a value left."""
    v = np.asarray(a, dtype=float).ravel()
    v = v[~np.isnan(v)]
    if not len(v):
        raise ValueError(f"{name} has no values (or only NaNs)")
    return v


def ranks(a: ArrayLike) -> tuple[np.ndarray, float]:
    """Ranks from 1, tied values sharing the mean of their ranks, and Σ(t³ - t) over the runs of t ties."""
    v = np.asarray(a, dtype=float).ravel()
    order = np.argsort(v, kind="mergesort")
    starts = np.r_[0, np.flatnonzero(np.diff(v[order])) + 1]   # the first row of each run of tied values
    sizes = np.diff(np.r_[starts, len(v)])
    out = np.empty(len(v))
    out[order] = np.repeat(starts + (sizes + 1) / 2, sizes)    # mean of ranks starts+1 .. starts+size
    return out, float((sizes ** 3 - sizes).sum())


def _gamma_q(a: float, x: float) -> float:
    """The regularized upper incomplete gamma Q(a, x): a series below a + 1, a continued fraction above."""
    if x <= 0:
        return 1.0
    log_front = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1:
        term = total = 1 / a
        for n in range(1, 500):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1 - total * math.exp(log_front))
    tiny = 1e-300                               # modified Lentz, as in Numerical Recipes' gcf
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for n in range(1, 500):
        an = -n * (n - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return math.exp(log_front) * h


def chi2_sf(x: float, df: int) -> float:
    """P(X > x) for X chi-squared with df degrees of freedom."""
    return _gamma_q(df / 2, x / 2)


def kruskal(*groups: ArrayLike) -> tuple[float, float]:
    """Kruskal-Wallis H and its p-value (chi² with k - 1 degrees of freedom), corrected for ties.
    NaNs are dropped; all values tied has no test and raises."""
    if len(groups) < 2:
        raise ValueError(f"Kruskal-Wallis compares 2 or more groups, got {len(groups)}")
    values = [_values(g, f"group {i + 1}") for i, g in enumerate(groups)]
    sizes = np.array([len(v) for v in values])
    n = sizes.sum()
    r, ties = ranks(np.concatenate(values))
    sums = np.add.reduceat(r, np.r_[0, np.cumsum(sizes)[:-1]])
    correction = 1 - ties / (n ** 3 - n)
    if correction == 0:
        raise ValueError("every value is the same; there is nothing to rank")
    h = (12 / (n * (n + 1)) * (sums ** 2 / sizes).sum() - 3 * (n + 1)) / correction
    return float(h), chi2_sf(h, len(groups) - 1)


def mannwhitney(a: ArrayLike, b: ArrayLike) -> tuple[float, float]:
    """Mann-Whitney U of a and the two-sided p-value, by the normal approximation with tie and continuity
    corrections (scipy's method='asymptotic'). NaNs are dropped."""
    x, y = _values(a, "a"), _values(b, "b")
    n1, n2 = len(x), len(y)
    n = n1 + n2
    r, ties = ranks(np.r_[x, y])
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))) if n > 1 else 0.0
    if sd == 0:
        raise ValueError("every value is the same; there is nothing to rank")
    z = max(abs(u - n1 * n2 / 2) - 0.5, 0) / sd
    return float(u), min(1.0, math.erfc(z / math.sqrt(2)))


def holm(p: ArrayLike) -> np.ndarray:
    """Holm-adjusted p-values, in the order given: the i-th smallest times (m - i), kept rising and at most 1."""
    v = np.asarray(p, dtype=float).ravel()
    order = np.argsort(v, kind="mergesort")
    scaled = np.minimum(1, np.maximum.accumulate(v[order] * (len(v) - np.arange(len(v)))))
    out = np.empty(len(v))
    out[order] = scaled
    return out


def cramers_v(x: ArrayLike, y: ArrayLike, corrected: bool = True) -> float:
    """Cramér's V of two categoricals, 0 (independent) .. 1 (one fixes the other). corrected = the bias correction
    of Bergsma (2013), which keeps V near 0 for independent columns with many categories. Rows with a NaN in
    either are dropped; NaN when either has one category only."""
    pairs = pd.DataFrame({"x": np.asarray(x, dtype=object).ravel(), "y": np.asarray(y, dtype=object).ravel()})
    table = pd.crosstab(pairs.x, pairs.y).to_numpy(dtype=float)
    n = table.sum()
    rows, cols = table.shape
    if rows < 2 or cols < 2:
        return math.nan
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / n
    phi2 = ((table - expected) ** 2 / expected).sum() / n
    if not corrected:
        return math.sqrt(phi2 / (min(rows, cols) - 1))
    phi2 = max(0.0, phi2 - (rows - 1) * (cols - 1) / (n - 1))
    rows, cols = rows - (rows - 1) ** 2 / (n - 1), cols - (cols - 1) ** 2 / (n - 1)
    return math.sqrt(phi2 / (min(rows, cols) - 1))


def linkage(x: ArrayLike, method: str = "average", distances: bool = False) -> np.ndarray:
    """Agglomerative clustering in scipy's format: row k joins clusters a < b at height h into cluster n + k,
    as [a, b, h, size]. x is n points (rows) with euclidean distances, or with distances=True an n x n distance
    matrix (1 - |r| for features). method: single, complete, average or ward."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {list(METHODS)}")
    v = np.asarray(x, dtype=float)
    v = v.reshape(-1, 1) if v.ndim == 1 else v
    if distances and (v.ndim != 2 or v.shape[0] != v.shape[1]):
        raise ValueError(f"distances=True takes a square matrix, got shape {v.shape}")
    if not np.isfinite(v).all():
        raise ValueError("linkage needs finite values; fill or drop NaNs first")
    n = len(v)
    if n < 2:
        raise ValueError(f"linkage needs 2 or more rows, got {n}")
    if n > MAX_LEAVES:
        raise ValueError(f"linkage of {n} rows is slow (O(n³)); cluster at most {MAX_LEAVES}, or a sample")
    d = v.copy() if distances else np.sqrt(((v[:, None, :] - v[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    ids, sizes = np.arange(n), np.ones(n)
    alive = np.ones(n, dtype=bool)
    out = np.empty((n - 1, 4))
    for k in range(n - 1):
        i, j = sorted(np.unravel_index(np.argmin(d), d.shape))
        h, ni, nj = d[i, j], sizes[i], sizes[j]
        out[k] = [*sorted((ids[i], ids[j])), h, ni + nj]
        if method == "single":
            new = np.minimum(d[i], d[j])
        elif method == "complete":
            new = np.maximum(d[i], d[j])
        elif method == "average":
            new = (ni * d[i] + nj * d[j]) / (ni + nj)
        else:
            nk = sizes
            new = np.sqrt(np.maximum(0, (nk + ni) * d[i] ** 2 + (nk + nj) * d[j] ** 2 - nk * h ** 2)
                          / (nk + ni + nj))
        new[~alive] = np.inf
        d[i], d[:, i] = new, new
        d[i, i] = np.inf
        d[j], d[:, j] = np.inf, np.inf
        alive[j] = False
        ids[i], sizes[i] = n + k, ni + nj
    return out


def leaves(z: ArrayLike) -> list[int]:
    """The leaf order of a linkage, left to right, as scipy's leaves_list and the dendrogram draw it."""
    z = np.asarray(z)
    n = len(z) + 1
    order, stack = [], [2 * n - 2]
    while stack:
        node = stack.pop()
        if node < n:
            order.append(int(node))
        else:
            a, b = z[node - n, :2].astype(int)
            stack += [b, a]
    return order
