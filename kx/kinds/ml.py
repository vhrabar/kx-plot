"""
Machine-learning kinds: truth first, then predictions or scores.
"""
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from .._metrics import binary, classes, confusion, roc_auc, roc_curve
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


def _chance(ax: Axes, xs: list[float], ys: list[float]) -> None:
    """A dotted line for what a random model scores."""
    ax.plot(xs, ys, color=plt.rcParams["text.color"], linestyle=":", linewidth=1, marker="", alpha=0.6,
            zorder=1)


def _unit(ax: Axes, xlabel: str, ylabel: str) -> None:
    """Both axes from 0 to 1, labelled, with the legend in the lower right: the metric is the point of the plot."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right")


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
    _unit(ax, "false positive rate", "true positive rate")
