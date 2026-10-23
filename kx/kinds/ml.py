"""
Machine-learning kinds: truth first, then predictions or scores.
"""
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from .._metrics import classes, confusion
from .._spec import kind
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
