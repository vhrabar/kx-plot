"""
Machine-learning kinds: truth first, then predictions or scores.
"""
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from .._metrics import (
    binary,
    brier,
    calibration,
    classes,
    confusion,
    fit_errors,
    pr_ap,
    pr_curve,
    roc_auc,
    roc_curve,
    thresholds,
)
from .._metrics import gain as gains
from .._spec import kind
from .._util import COLOUR, label
from .basic import _check_scatter, scatter
from .dist import _hist
from .table import _matrix, hexbin

HEXBIN = 5000                                   # pva: more points than this draw as hexagons


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


def _check_proba(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if z is not None:
        raise ValueError("kind 'proba' shows one model's scores; compare two models with 'roc', 'pr' or 'calib'")


@kind("proba", flags=("norm",), check=_check_proba)
def proba(ax: Axes, x: ArrayLike, y: ArrayLike, z: None, flags: set[str], bins: int = 20, **kw: Any) -> None:
    """Score distribution per true class: kx.plot(y_true, score, 'proba'), one histogram per class on the same
    bins= equal-width bins over 0..1 (20). Where they overlap, the model confuses the classes.
    A 2-D score (one column per class) shows each row's score for its own true class.
    norm = densities (area 1 each), for imbalanced classes. kw: ax.hist."""
    if np.ndim(y) == 2 and np.shape(y)[1] > 1:
        series = [(s[t == 1], name) for t, s, name in _scored(x, y, None)]
        xlabel = "score of the true class"
    else:
        s, t = np.asarray(y, dtype=float).ravel(), np.asarray(x, dtype=object).ravel()
        binary(t)                               # two classes, or a hint for multiclass
        if len(s) != len(t):
            raise ValueError(f"y_true has {len(t)} values but the score has {len(s)}")
        series = [(s[t == c], str(c)) for c in classes(t)]
        xlabel = label(y, "score")
    every = np.concatenate([v for v, _ in series])
    if not np.isfinite(every).all():
        raise ValueError("score has NaN or inf values; drop those rows first")
    if every.min() < 0 or every.max() > 1:
        raise ValueError("kind 'proba' needs scores between 0 and 1, such as predicted probabilities")
    _hist(ax, series, np.linspace(0, 1, bins + 1), flags & {"norm"}, **kw)
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density" if "norm" in flags else "count")
    ax.legend(title=label(x, "true class"))


def _check_thresh(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if z is not None:
        raise ValueError("kind 'thresh' shows one model's scores; compare two models with 'roc', 'pr' or 'calib'")


def _best(ax: Axes, at: float, text: str) -> None:
    """best: a dashed vertical line at the best x, with what makes it best in the legend."""
    ax.axvline(at, color=plt.rcParams["text.color"], linestyle="--", linewidth=1, alpha=0.8, label=text)


@kind("thresh", flags=("best",), check=_check_thresh)
def thresh(ax: Axes, x: ArrayLike, y: ArrayLike, z: None, flags: set[str], **kw: Any) -> None:
    """Precision, recall and F1 against the decision threshold: kx.plot(y_true, score, 'thresh').
    A row is positive when its score is at or above the threshold. best = a dashed line at the threshold with the
    highest F1. kw: the three lines (ax.plot)."""
    thr, precision, recall, f1 = thresholds(x, y)
    for values, name in ((precision, "precision"), (recall, "recall"), (f1, "F1")):
        ax.plot(thr, values, **{"label": name, **kw})
    if "best" in flags:
        i = int(np.argmax(f1))                  # ties: the highest threshold, the fewest positives
        _best(ax, thr[i], f"best F1 {f1[i]:.2f} at {thr[i]:.3g}")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"threshold on {label(y, 'score')}")
    ax.legend(loc="lower left")


def _ranked(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, lift: bool, **kw: Any) -> None:
    """gain or lift of each model against the share of rows taken, best scores first."""
    curves = _scored(x, y, z)
    shared = len(curves) == 1 or z is not None              # y and z score the same truth: one perfect model
    for t, s, name in curves:
        taken, found = gains(t, s)
        if lift:                                # found / taken, from the first rows taken on
            ln, = ax.plot(taken[1:], found[1:] / taken[1:], **{"label": name, **kw})
        else:
            ln, = ax.plot(taken, found, **{"label": name, **kw})
            if not shared:
                _chance(ax, [0, t.mean(), 1], [0, 1, 1], ln.get_color())
    if lift:
        _chance(ax, [0, 1], [1, 1])
    else:
        if shared:
            rate = curves[0][0].mean()          # the perfect model takes every positive first
            ax.plot([0, rate, 1], [0, 1, 1], color=plt.rcParams["text.color"], linestyle="--", linewidth=1,
                    marker="", alpha=0.6, label="perfect")
        _chance(ax, [0, 1], [0, 1])
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    ax.set_xlim(0, 1)
    ax.set_xlabel("rows taken, highest score first")
    if lift:
        ax.set_ylim(bottom=0)
        ax.set_ylabel("lift")
        ax.legend(loc="upper right")
    else:
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
        ax.set_ylim(0, 1)
        ax.set_ylabel("positives found")
        ax.legend(loc="lower right")


@kind("gain", check=_check_scores)
def gain(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """Cumulative gain: kx.plot(y_true, score, 'gain'), the share of positives found against the share of rows
    taken, highest scores first. The dotted diagonal is a random order, the dashed corner a perfect one.
    z = a second model's score; a 2-D score draws one curve per class, one-vs-rest. kw: the curves (ax.plot)."""
    _ranked(ax, x, y, z, False, **kw)


@kind("lift", check=_check_scores)
def lift(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """Lift: kx.plot(y_true, score, 'lift'), the gain divided by a random order's, so how many times more positives
    the top rows hold than the same share picked at random; the dotted line at 1 is random.
    z = a second model's score; a 2-D score draws one curve per class, one-vs-rest. kw: the curves (ax.plot)."""
    _ranked(ax, x, y, z, True, **kw)


def _check_pva(y: ArrayLike, z: ArrayLike | None, flags: set[str]) -> None:
    if z is not None:
        raise ValueError("kind 'pva' compares actual (x) with predicted (y); it takes no z. "
                         "To colour by a third column, use 'resid'")


@kind("pva", check=_check_pva)
def pva(ax: Axes, x: ArrayLike, y: ArrayLike, z: None, flags: set[str], **kw: Any) -> None:
    """Predicted against actual: kx.plot(actual, predicted, 'pva'), with the dashed y = x line a perfect model lies
    on and R², RMSE and MAE in the corner. Over 5,000 points it draws hexagons (hexbin), so big test sets stay
    readable. kw: ax.scatter, or ax.hexbin over 5,000 points."""
    r2, rmse, mae = fit_errors(x, y)
    if np.size(x) > HEXBIN:
        hexbin(ax, x, y, None, {"diag"}, **kw)
    else:
        scatter(ax, x, y, None, {"diag"}, **kw)
    ax.text(0.03, 0.97, f"R² {r2:.3f}\nRMSE {rmse:.3g}\nMAE {mae:.3g}", transform=ax.transAxes, ha="left",
            va="top", bbox={"boxstyle": "round", "facecolor": ax.get_facecolor(), "edgecolor": "none",
                            "alpha": 0.85})
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(label(x, "actual"))
    ax.set_ylabel(label(y, "predicted"))


@kind("resid", flags=COLOUR, check=_check_scatter)
def resid(ax: Axes, x: ArrayLike, y: ArrayLike, z: ArrayLike | None, flags: set[str], **kw: Any) -> None:
    """Residuals: kx.plot(actual, predicted, 'resid'), actual - predicted against predicted, with a line at 0.
    Above it the model guessed too low, below too high; a funnel means the error grows with the prediction.
    z colours the points as in scatter (a colorbar for numbers, a legend for categories), with nocb, sym and logc.
    kw: ax.scatter."""
    fit_errors(x, y)                            # same lengths, no NaN
    a, p = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    scatter(ax, p, a - p, z, flags & set(COLOUR), **kw)
    ax.axhline(0, color=plt.rcParams["text.color"], linestyle="--", linewidth=1, alpha=0.6, zorder=1)
    ax.set_xlabel(label(y, "predicted"))
    ax.set_ylabel("residual (actual - predicted)")
