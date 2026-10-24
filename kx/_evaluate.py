"""kx.evaluate: one call, every panel a model needs, picked from the data."""
import contextlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ._grid import grid
from ._metrics import classes
from ._plot import plot
from ._spec import parse
from ._theme import style_path

LABELS = 10                                     # a numeric y with more distinct values than this is a regression


def evaluate(y: ArrayLike, *scores: ArrayLike, spec: str = "", **kw: Any) -> np.ndarray:
    """A model report in one figure: kx.evaluate(y_true, score) or kx.evaluate(y_true, gbm, logreg).
    The task comes from the data:
      binary y, 1-D scores    roc, pr, calib, proba (2 x 2), every score in each panel but proba (the first)
      2-D score (multiclass)  roc per class, cm of the most likely class, proba of the true class
      numeric y, many values  pva, resid, and a hist of the residuals
    spec= adds a theme and flags to every panel: kx.evaluate(y, score, spec='dark-grid').
    Other keywords go to kx.grid (figsize=...). Returns the flat array of axes; the figure is axs[0].figure."""
    if not scores:
        raise TypeError("kx.evaluate needs y_true and at least one score: kx.evaluate(y_true, score)")
    task = _task(y, scores)
    if task != "binary" and len(scores) > 1:
        raise ValueError(f"a {task} report takes one model's {'scores' if task == 'multiclass' else 'predictions'}; "
                         f"got {len(scores)}")
    if task == "binary" and len(scores) > 2:
        raise ValueError(f"the panels compare at most 2 models; got {len(scores)}. Call kx.evaluate once per pair")
    named = [s if getattr(s, "name", None) is not None or np.ndim(s) == 2
             else pd.Series(np.asarray(s).ravel(), name="score" if len(scores) == 1 else f"score {i + 1}")
             for i, s in enumerate(scores)]
    theme = parse(spec).theme if spec else None
    style = plt.style.context(["default", style_path(theme)]) if theme else contextlib.nullcontext()
    with style:
        panels = PANELS[task](y, named)
        axs = grid(len(panels), cols=2 if len(panels) == 4 else 3, **kw)
        for ax, (title, args, kind) in zip(axs, panels, strict=True):
            plot(*args, spec="-".join([kind, *([spec] if spec else [])]), ax=ax, title=title)
        if task == "regression":
            axs[2].set_xlabel("residual (actual - predicted)")
    return axs


def _task(y: ArrayLike, scores: tuple) -> str:
    if any(np.ndim(s) == 2 and np.shape(s)[1] > 1 for s in scores):
        return "multiclass"
    v = pd.Series(np.asarray(y).ravel())
    if pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v) and v.nunique() > LABELS:
        return "regression"
    return "binary"


def _binary(y: ArrayLike, scores: list) -> list[tuple[str, tuple, str]]:
    return [("ROC", (y, *scores), "roc"), ("precision-recall", (y, *scores), "pr"),
            ("calibration", (y, *scores), "calib"), (f"scores by class: {scores[0].name}", (y, scores[0]), "proba")]


def _multiclass(y: ArrayLike, scores: list) -> list[tuple[str, tuple, str]]:
    score = scores[0]
    found = classes(y)
    cols = list(score.columns) if isinstance(score, pd.DataFrame) else None
    names = cols if cols is not None and set(cols) <= set(found) else found   # as _scored pairs columns
    pred = np.asarray(names, dtype=object)[np.asarray(score, dtype=float).argmax(axis=1)]
    return [("ROC, one-vs-rest", (y, score), "roc"), ("confusion matrix", (y, pred), "cm-norm-ann"),
            ("score of the true class", (y, score), "proba")]


def _regression(y: ArrayLike, scores: list) -> list[tuple[str, tuple, str]]:
    actual, predicted = np.asarray(y, dtype=float).ravel(), np.asarray(scores[0], dtype=float).ravel()
    return [("predicted vs actual", (y, scores[0]), "pva"), ("residuals", (y, scores[0]), "resid"),
            ("residual distribution", (pd.Series(actual - predicted, name="residual"),), "hist")]


PANELS = {"binary": _binary, "multiclass": _multiclass, "regression": _regression}
