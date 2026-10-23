"""
Metrics for the machine-learning kinds
"""
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def binary(y: ArrayLike) -> np.ndarray:
    """y as a 0/1 int array: 1/True is positive, else the later of two labels in sorted order."""
    v = np.asarray(y).ravel()
    if pd.isna(v).any():
        raise ValueError("y_true has missing values; drop those rows first")
    if v.dtype == bool:
        return v.astype(int)
    found = sorted(pd.unique(v), key=lambda c: (str(type(c)), c))
    if set(found) <= {0, 1}:
        return (v == 1).astype(int)
    if len(found) == 2:
        return (v == found[1]).astype(int)
    if len(found) == 1:
        raise ValueError(f"y_true has one label only ({found[0]!r}); a binary metric needs two classes")
    raise ValueError(f"y_true has {len(found)} labels but the score is 1-D; "
                     "for multiclass pass a 2-D score, one column per class")


def _pair(y: ArrayLike, s: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """0/1 labels and float scores of the same length, checked."""
    sc = np.asarray(s, dtype=float)
    if sc.ndim > 1 and sc.shape[1] > 1:
        raise ValueError(f"score has {sc.shape[1]} columns; a binary metric takes one score per row")
    sc = sc.ravel()
    t = binary(y)
    if len(t) != len(sc):
        raise ValueError(f"y_true has {len(t)} values but the score has {len(sc)}")
    if not np.isfinite(sc).all():
        raise ValueError("score has NaN or inf values; drop those rows first")
    return t, sc


def _counts(y: ArrayLike, s: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """True and false positives at each distinct threshold, highest score first, so tied scores are one step.
    Returns tps, fps, thresholds and the number of positives and negatives."""
    t, sc = _pair(y, s)
    order = np.argsort(-sc, kind="mergesort")
    t, sc = t[order], sc[order]
    last = np.r_[np.flatnonzero(np.diff(sc)), len(sc) - 1]   # the last row of each run of tied scores
    tps = np.cumsum(t)[last]
    fps = (last + 1) - tps
    return tps, fps, sc[last], int(t.sum()), len(t) - int(t.sum())


def _both(pos: int, neg: int, what: str) -> None:
    if not pos or not neg:
        raise ValueError(f"{what} needs both classes in y_true; it has only {'negatives' if neg else 'positives'}")


def roc_curve(y: ArrayLike, s: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """False and true positive rates at each distinct threshold, from (0, 0) at threshold inf to (1, 1)."""
    tps, fps, thr, pos, neg = _counts(y, s)
    _both(pos, neg, "a ROC curve")
    return np.r_[0, fps / neg], np.r_[0, tps / pos], np.r_[np.inf, thr]


def pr_curve(y: ArrayLike, s: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision and recall at each distinct threshold, highest first, from (1, 0) at threshold inf."""
    tps, fps, thr, pos, _ = _counts(y, s)
    if not pos:
        raise ValueError("a PR curve needs positives in y_true; it has only negatives")
    return np.r_[1, tps / (tps + fps)], np.r_[0, tps / pos], np.r_[np.inf, thr]


def roc_auc(y_true: ArrayLike, score: ArrayLike) -> float:
    """Area under the ROC curve, by trapezoids: 1 is a perfect ranking, 0.5 a random one."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def pr_ap(y_true: ArrayLike, score: ArrayLike) -> float:
    """Average precision: the step sum Σ (Rₙ - Rₙ₋₁) Pₙ over thresholds, as scikit-learn computes it.
    Trapezoids would overstate it."""
    precision, recall, _ = pr_curve(y_true, score)
    return float(np.sum(np.diff(recall) * precision[1:]))


def confusion(y: ArrayLike, p: ArrayLike, labels: list[Any] | None = None) -> np.ndarray:
    """Counts with the true class down the rows and the predicted class along the columns.
    labels default to the union of both arrays in sorted order, so a class never predicted still gets a column."""
    t, q = np.asarray(y, dtype=object).ravel(), np.asarray(p, dtype=object).ravel()
    if len(t) != len(q):
        raise ValueError(f"y_true has {len(t)} values but y_pred has {len(q)}")
    if labels is None:
        labels = sorted(set(pd.unique(t)) | set(pd.unique(q)), key=lambda c: (str(type(c)), c))
    index = {c: i for i, c in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for a, b in zip(t, q, strict=True):
        if a in index and b in index:
            m[index[a], index[b]] += 1
    return m


def brier(y: ArrayLike, s: ArrayLike) -> float:
    """Mean squared gap between the score and the 0/1 outcome: 0 is perfect."""
    t, sc = _pair(y, s)
    return float(np.mean((sc - t) ** 2))


def calibration(y: ArrayLike, s: ArrayLike, bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean score, observed positive rate and row count in each of bins equal-width bins over 0..1.
    Empty bins are left out."""
    t, sc = _pair(y, s)
    if sc.min() < 0 or sc.max() > 1:
        raise ValueError("calibration needs scores between 0 and 1, such as predicted probabilities")
    which = np.clip(np.floor(sc * bins).astype(int), 0, bins - 1)     # a score of exactly 1 goes in the last bin
    n = np.bincount(which, minlength=bins)
    full = n > 0
    mean = np.bincount(which, weights=sc, minlength=bins)[full] / n[full]
    rate = np.bincount(which, weights=t, minlength=bins)[full] / n[full]
    return mean, rate, n[full]


def gain(y: ArrayLike, s: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Share of rows taken, best scores first, and the share of positives found, from (0, 0) to (1, 1)."""
    tps, fps, _, pos, neg = _counts(y, s)
    if not pos:
        raise ValueError("a gain curve needs positives in y_true; it has only negatives")
    return np.r_[0, (tps + fps) / (pos + neg)], np.r_[0, tps / pos]
