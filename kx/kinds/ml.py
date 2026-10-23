"""
Machine-learning kinds: truth first, then predictions or scores.
"""
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from .._metrics import binary, brier, calibration, classes, confusion, pr_ap, pr_curve, roc_auc, roc_curve
from .._spec import kind
from .._util import label
from .table import _matrix


def _flat(a: ArrayLike, name: str) -> np.ndarray:
    """a as a 1-D object array of class labels, or an error saying how to get one from scores."""
    if np.ndim(a) == 2 and np.shape(a)[1] > 1:
        raise ValueError(f"{name} is 2-D, one column per class; kind 'cm' takes one label per row: "
                         "pass proba.argmax(axis=1), or df.idxmax(axis=1) for class names")
    v = np.asarray(a, dtype=object).ravel()
    if pd.isna(v).any():
        raise ValueError(f"{name} has missing values; drop those rows first")
    return v


def _check_cm(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if z is not None:
        raise ValueError("kind 'cm' compares y_true (x) with y_pred (y); it takes no z")
    _flat(y, "y_pred")


@kind("cm", flags=("norm", "ann", "nocb"), check=_check_cm)
def cm(ax: Axes, x: ArrayLike, y: ArrayLike, z: None, flags: set[str], **kw: Any) -> None:
    """Confusion matrix: kx.plot(y_true, y_pred, 'cm'). True classes down the rows, predicted along the columns,
    as in scikit-learn; every label of either array gets a row and a column.
    norm = each row sums to 1 (the recall of each class), ann = the value in each cell, nocb = no colorbar.
    kw: ax.imshow."""
    t, p = _flat(x, "y_true"), _flat(y, "y_pred")
    if len(t) != len(p):
        raise ValueError(f"y_true has {len(t)} values but y_pred has {len(p)}")
    labels = classes(t, p)
    frame = pd.DataFrame(confusion(t, p, labels), index=pd.Index(labels, name="true"),
                         columns=pd.Index(labels, name="predicted"))
    if "norm" in flags:                         # a class never true has no row total: its cells stay blank
        frame = frame.div(frame.sum(axis=1).replace(0, np.nan), axis=0)
        _matrix(ax, frame, flags, "share of true class", "{:.0%}", **{"vmin": 0, "vmax": 1, **kw})
    else:
        _matrix(ax, frame, flags, "count", "{:g}", **{"vmin": 0, **kw})


def _check_scores(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if z is not None and np.ndim(y) == 2 and np.shape(y)[1] > 1:
        raise ValueError("a 2-D score already draws one curve per class; z adds a second binary model only")
    if z is not None and np.ndim(z) == 2 and np.shape(z)[1] > 1:
        raise ValueError("z is a second model's 1-D score; for multiclass pass one 2-D score as y")


def _scored(x: ArrayLike, y: ArrayLike, z: ArrayLike | None) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """(0/1 truth, score, name) per curve: y and z for a binary task, one-vs-rest per column of a 2-D y.
    The class of each column is its DataFrame name when y_true has it, else the sorted labels of y_true."""
    if not (np.ndim(y) == 2 and np.shape(y)[1] > 1):
        return [(binary(x), np.asarray(s, dtype=float).ravel(), label(s, n)) for s, n in ((y, "y"), (z, "z"))
                if s is not None]
    truth = np.asarray(x, dtype=object).ravel()
    found, scores = classes(truth), np.asarray(y, dtype=float)
    cols = list(y.columns) if isinstance(y, pd.DataFrame) else None
    if cols is not None and set(cols) <= set(found):
        pairs = list(zip(cols, cols, strict=True))
    elif len(found) == scores.shape[1]:
        pairs = list(zip(found, cols or found, strict=True))
    else:
        raise ValueError(f"the score has {scores.shape[1]} columns but y_true has {len(found)} labels "
                         f"{found}; pass one column per class, named after the labels or in their sorted order")
    return [((truth == c).astype(int), scores[:, j], str(name)) for j, (c, name) in enumerate(pairs)]


def _chance(ax: Axes, xs: list[float], ys: list[float], color: Any = None) -> None:
    """A dotted line for what a random model scores, in the text colour unless it belongs to one curve."""
    ax.plot(xs, ys, color=color or plt.rcParams["text.color"], linestyle=":", linewidth=1, marker="",
            alpha=0.6, zorder=1)


def _unit(ax: Axes, xlabel: str, ylabel: str, loc: str) -> None:
    """Both axes from 0 to 1, labelled, with the legend drawn: the metric is the point of the plot."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=loc)


@kind("roc", check=_check_scores)
def roc(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """ROC curve: kx.plot(y_true, score, 'roc'), the true positive rate against the false positive rate as the
    threshold falls, with the AUC in the legend. z = a second model's score; a 2-D score (one column per class)
    draws one curve per class, one-vs-rest. kw: the curves (ax.plot)."""
    for t, s, name in _scored(x, y, z):
        fpr, tpr, _ = roc_curve(t, s)
        ax.plot(fpr, tpr, **{"label": f"{name} (AUC {roc_auc(t, s):.2f})", **kw})
    _chance(ax, [0, 1], [0, 1])
    ax.set_aspect("equal")
    _unit(ax, "false positive rate", "true positive rate", "lower right")


@kind("pr", check=_check_scores)
def pr(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """Precision-recall curve: kx.plot(y_true, score, 'pr'), precision against recall as the threshold falls,
    with the average precision in the legend and a dotted baseline at the positive rate, what a random model gets.
    z = a second model's score; a 2-D score (one column per class) draws one curve and baseline per class,
    one-vs-rest. kw: the curves (ax.plot)."""
    curves = _scored(x, y, z)
    shared = len(curves) == 1 or z is not None              # y and z score the same truth: one baseline
    for t, s, name in curves:
        precision, recall, _ = pr_curve(t, s)
        ln, = ax.plot(recall, precision, **{"drawstyle": "steps-pre",      # the steps AP sums
                                            "label": f"{name} (AP {pr_ap(t, s):.2f})", **kw})
        if not shared:
            _chance(ax, [0, 1], [t.mean()] * 2, ln.get_color())
    if shared:
        _chance(ax, [0, 1], [curves[0][0].mean()] * 2)
    _unit(ax, "recall", "precision", "lower left")


@kind("calib", check=_check_scores)
def calib(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], bins: int = 10,
          **kw: Any) -> None:
    """Reliability diagram: kx.plot(y_true, score, 'calib'), the observed positive rate against the mean score in
    bins= equal-width bins over 0..1 (10), with markers sized by the rows in each bin and the Brier score in the
    legend. On the dotted diagonal, a score of 0.7 comes true 70% of the time; empty bins are left out.
    z = a second model's score; a 2-D score draws one line per class, one-vs-rest. kw: the lines (ax.plot)."""
    curves = [(name, *calibration(t, s, bins), brier(t, s)) for t, s, name in _scored(x, y, z)]
    most = max(n.max() for _, _, _, n, _ in curves)
    for name, mean, rate, n, score in curves:
        ln, = ax.plot(mean, rate, **{"marker": "", "clip_on": False, "label": f"{name} (Brier {score:.3f})", **kw})
        ax.scatter(mean, rate, s=10 + 90 * n / most, color=ln.get_color(), zorder=3, clip_on=False)   # 0 and 1 too
    _chance(ax, [0, 1], [0, 1])
    ax.set_aspect("equal")
    _unit(ax, "mean score", "observed positive rate", "upper left")
