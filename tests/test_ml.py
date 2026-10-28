"""Machine-learning kinds, kx.evaluate and the metrics behind them."""
from collections.abc import Callable

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from helpers import _dashed
from matplotlib.axes import Axes
from matplotlib.colors import to_hex

import kx


# ---------- machine-learning kinds ----------
def test_cm_counts_true_rows_and_predicted_columns() -> None:
    ax = kx.plot(["a", "b", "b", "c", "c"], ["a", "b", "a", "a", "b"], "cm-ann")
    assert ax.images[0].get_array().tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 0]]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]    # 'c' is never predicted
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "c"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("predicted", "true")
    assert [t.get_text() for t in ax.texts] == ["1", "0", "0", "1", "1", "0", "1", "1", "0"]
    assert ax.figure.axes[1].get_ylabel() == "count"


def test_cm_extra_predicted_class_gets_a_blank_norm_row() -> None:
    ax = kx.plot([0, 0, 1], [0, 2, 1], "cm-norm")
    shares = ax.images[0].get_array()
    assert shares.shape == (3, 3)
    assert np.allclose(shares[:2].sum(axis=1), 1)
    assert shares[:2].tolist() == [[0.5, 0, 0.5], [0, 1, 0]]
    assert np.ma.is_masked(shares[2, 0]) or np.isnan(shares[2, 0])     # class 2 is never true: no recall


def test_cm_norm_rows_sum_to_one_with_a_percent_colorbar() -> None:
    m = kx.load("multiclass")
    ax = kx.plot(m, "y", "pred", "cm-norm-ann")
    assert np.allclose(ax.images[0].get_array().sum(axis=1), 1)
    assert all(t.get_text().endswith("%") for t in ax.texts)
    cb = ax.figure.axes[1]
    assert isinstance(cb.yaxis.get_major_formatter(), mpl.ticker.PercentFormatter)
    assert len(kx.plot(m, "y", "pred", "cm-nocb").figure.axes) == 1


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot([0, 1, 2], np.eye(3), "cm"), r"proba\.argmax\(axis=1\)"),
    (lambda: kx.plot(np.eye(3), [0, 1, 2], "cm"), "y_true is 2-D"),
    (lambda: kx.plot([0, 1], [0, 1], [0, 1], "cm"), "takes no z"),
    (lambda: kx.plot([0, 1], [0, 1, 1], "cm"), "y_true has 2 values but y_pred has 3"),
    (lambda: kx.plot([0, np.nan], [0, 1], "cm"), "missing values"),
])
def test_cm_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


BINARY, MULTI, REGRESSION = kx.load("binary"), kx.load("multiclass"), kx.load("regression")


def _curves(ax: Axes) -> list:
    """The labelled lines: one per model or class, not the chance line."""
    return [ln for ln in ax.lines if not ln.get_label().startswith("_")]


def _legend(ax: Axes) -> list[str]:
    return [t.get_text() for t in ax.get_legend().get_texts()]


def test_roc_runs_from_corner_to_corner_with_auc_in_the_legend() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "roc")
    curve, = _curves(ax)
    assert (curve.get_xdata()[0], curve.get_ydata()[0]) == (0, 0)
    assert (curve.get_xdata()[-1], curve.get_ydata()[-1]) == (1, 1)
    assert _legend(ax) == [f"gbm (AUC {kx.roc_auc(BINARY.y, BINARY.gbm):.2f})"]   # drawn without 'leg'
    assert (ax.get_xlim(), ax.get_ylim()) == ((0, 1), (0, 1))
    assert ax.get_aspect() == 1
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("false positive rate", "true positive rate")
    chance = [ln for ln in ax.lines if ln.get_linestyle() == ":"]
    assert len(chance) == 1


def test_roc_z_is_a_second_model() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "logreg", "roc")
    assert len(_curves(ax)) == 2
    assert _legend(ax)[1].startswith("logreg (AUC ")
    assert _legend(kx.plot(BINARY.y.to_numpy(), BINARY.gbm.to_numpy(), "roc"))[0].startswith("y (AUC ")


def test_roc_multiclass_one_curve_per_column() -> None:
    ax = kx.plot(MULTI.y, MULTI[["cat", "dog", "bird"]], "roc")
    assert [t.split(" ")[0] for t in _legend(ax)] == ["cat", "dog", "bird"]
    want = kx.roc_auc(MULTI.y == "dog", MULTI.dog)
    assert _legend(ax)[1] == f"dog (AUC {want:.2f})"


def test_roc_multiclass_array_takes_the_sorted_labels() -> None:
    codes = MULTI.y.map({"bird": 0, "cat": 1, "dog": 2})
    ax = kx.plot(codes, MULTI[["bird", "cat", "dog"]].to_numpy(), "roc")
    assert [t.split(" ")[0] for t in _legend(ax)] == ["0", "1", "2"]
    renamed = MULTI[["bird", "cat", "dog"]].set_axis(["b", "c", "d"], axis=1)  # columns in label order, other names
    assert [t.split(" ")[0] for t in _legend(kx.plot(codes, renamed, "roc"))] == ["b", "c", "d"]


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot(MULTI.y, MULTI.cat, "roc"), "pass a 2-D score, one column per class"),
    (lambda: kx.plot(MULTI.y, MULTI[["cat", "dog"]].to_numpy(), "roc"), "2 columns but y_true has 3 labels"),
    (lambda: kx.plot(MULTI.y, MULTI[["cat", "dog", "bird"]], MULTI.cat, "roc"), "z adds a second binary model"),
    (lambda: kx.plot([1, 1, 1], [0.2, 0.5, 0.9], "roc"), "only positives"),
])
def test_roc_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def _dotted(ax: Axes) -> list:
    return [ln for ln in ax.lines if ln.get_linestyle() == ":"]


def test_pr_baseline_at_the_positive_rate_and_ap_in_the_legend() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "pr")
    curve, = _curves(ax)
    assert (curve.get_xdata()[0], curve.get_ydata()[0]) == (0, 1)
    assert curve.get_xdata()[-1] == 1
    assert curve.get_drawstyle() == "steps-pre"
    assert _legend(ax) == [f"gbm (AP {kx.pr_ap(BINARY.y, BINARY.gbm):.2f})"]
    base, = _dotted(ax)
    assert list(base.get_ydata()) == pytest.approx([BINARY.y.mean()] * 2)
    assert list(base.get_xdata()) == [0, 1]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("recall", "precision")
    assert (ax.get_xlim(), ax.get_ylim()) == ((0, 1), (0, 1))


def test_pr_z_shares_one_baseline() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "logreg", "pr")
    assert len(_curves(ax)) == 2
    assert len(_dotted(ax)) == 1
    assert _legend(ax)[1].startswith("logreg (AP ")


def test_pr_multiclass_one_baseline_per_class_in_its_colour() -> None:
    ax = kx.plot(MULTI.y, MULTI[["cat", "dog", "bird"]], "pr")
    curves, bases = _curves(ax), _dotted(ax)
    assert [t.split(" ")[0] for t in _legend(ax)] == ["cat", "dog", "bird"]
    assert [b.get_ydata()[0] for b in bases] == pytest.approx([(MULTI.y == c).mean() for c in ("cat", "dog", "bird")])
    assert [to_hex(b.get_color()) for b in bases] == [to_hex(c.get_color()) for c in curves]


def test_calib_perfect_model_lies_on_the_diagonal_without_empty_bins() -> None:
    score = [0.05] * 20 + [0.25] * 4 + [0.75] * 4             # bins 0, 2 and 7 of 10; the rest are empty
    truth = [1] + [0] * 19 + [1, 0, 0, 0] + [1, 1, 1, 0]       # each comes true as often as it says
    ax = kx.plot(truth, score, "calib")
    pts = ax.collections[0]
    assert np.allclose(pts.get_offsets(), [[0.05, 0.05], [0.25, 0.25], [0.75, 0.75]])
    sizes = pts.get_sizes()
    assert sizes[0] > sizes[1] == sizes[2]                    # 20 rows against 4
    want = kx._metrics.brier(truth, score)
    assert _legend(ax) == [f"y (Brier {want:.3f})"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("mean score", "observed positive rate")
    assert len(_dotted(ax)) == 1


def test_calib_bins_and_a_second_model() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "logreg", "calib", bins=5)
    assert len(_curves(ax)) == 2
    assert all(len(c.get_xdata()) <= 5 for c in _curves(ax))
    gbm, logreg = (kx._metrics.brier(BINARY.y, BINARY[c]) for c in ("gbm", "logreg"))
    assert _legend(ax) == [f"gbm (Brier {gbm:.3f})", f"logreg (Brier {logreg:.3f})"]


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot([0, 1], [-0.5, 2.0], "calib"), "between 0 and 1"),
    (lambda: kx.plot([0, 1], [0.2, 0.8], "calib", bins=0), "whole number from 1"),
    (lambda: kx.plot([0, 1], [0.2, 0.8], "calib", bins=2.5), "whole number from 1"),
])
def test_calib_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_proba_one_histogram_per_true_class_on_bins_over_0_to_1() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "proba")
    assert len(ax.containers) == 2
    for c in ax.containers:
        assert c[0].get_x() == pytest.approx(0)
        assert c[-1].get_x() + c[-1].get_width() == pytest.approx(1)
        assert len(c) == 20
    counts = [sum(p.get_height() for p in c) for c in ax.containers]
    assert counts == [(BINARY.y == 0).sum(), (BINARY.y == 1).sum()]
    assert _legend(ax) == ["0", "1"]
    assert ax.get_legend().get_title().get_text() == "y"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("gbm", "count")


def test_proba_norm_gives_areas_of_one() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "proba-norm", bins=10)
    for c in ax.containers:
        assert len(c) == 10
        assert sum(p.get_height() * p.get_width() for p in c) == pytest.approx(1)
    assert ax.get_ylabel() == "density"


def test_proba_multiclass_takes_the_score_of_the_true_class() -> None:
    ax = kx.plot(MULTI.y, MULTI[["cat", "dog", "bird"]], "proba")
    assert _legend(ax) == ["cat", "dog", "bird"]
    dog = sum(p.get_height() for p in ax.containers[1])
    assert dog == (MULTI.y == "dog").sum()
    assert ax.get_xlabel() == "score of the true class"


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot(BINARY, "y", "gbm", "logreg", "proba"), "one model's scores"),
    (lambda: kx.plot([0, 1], [-0.5, 2.0], "proba"), "between 0 and 1"),
    (lambda: kx.plot([0, 1], [0.2, np.nan], "proba"), "NaN"),
    (lambda: kx.plot(MULTI.y, MULTI.cat, "proba"), "pass a 2-D score"),
])
def test_proba_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_thresh_precision_recall_f1_by_hand() -> None:
    ax = kx.plot([0, 1, 0, 1], [0.1, 0.4, 0.35, 0.8], "thresh")
    lines = {ln.get_label(): ln for ln in ax.lines}
    assert list(lines["precision"].get_xdata()) == [0.8, 0.4, 0.35, 0.1]
    assert list(lines["precision"].get_ydata()) == pytest.approx([1, 1, 2 / 3, 0.5])
    assert list(lines["recall"].get_ydata()) == pytest.approx([0.5, 1, 1, 1])
    assert list(lines["F1"].get_ydata()) == pytest.approx([2 / 3, 1, 0.8, 2 / 3])
    assert _legend(ax) == ["precision", "recall", "F1"]
    assert ax.get_xlabel() == "threshold on score"


def test_thresh_best_marks_the_f1_maximum() -> None:
    ax = kx.plot([0, 1, 0, 1], [0.1, 0.4, 0.35, 0.8], "thresh-best")
    best, = _dashed(ax)
    assert list(best.get_xdata()) == [0.4, 0.4]
    assert _legend(ax)[-1] == "best F1 1.00 at 0.4"
    ax = kx.plot(BINARY, "y", "gbm", "thresh-best")
    _, _, _, f1 = kx._metrics.thresholds(BINARY.y, BINARY.gbm)
    assert _legend(ax)[-1].startswith(f"best F1 {f1.max():.2f} at ")
    assert ax.get_xlabel() == "threshold on gbm"


def test_thresh_f1_is_zero_where_nothing_is_found() -> None:
    _, precision, recall, f1 = kx._metrics.thresholds([1, 0], [0.2, 0.9])
    assert list(precision) == [0, 0.5]
    assert list(recall) == [0, 1]
    assert list(f1) == pytest.approx([0, 2 / 3])


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot(BINARY, "y", "gbm", "logreg", "thresh"), "one model's scores"),
    (lambda: kx.plot([0, 0], [0.2, 0.9], "thresh"), "only negatives"),
    (lambda: kx.parse("roc-best"), "flag 'best' only works with kind"),
])
def test_thresh_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_gain_ends_at_one_with_the_perfect_corner_at_the_positive_rate() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "gain")
    curves = {ln.get_label(): ln for ln in _curves(ax)}
    assert list(curves) == ["gbm", "perfect"]
    model = curves["gbm"]
    assert (model.get_xdata()[0], model.get_ydata()[0]) == (0, 0)
    assert (model.get_xdata()[-1], model.get_ydata()[-1]) == (1, 1)
    assert list(curves["perfect"].get_xdata()) == pytest.approx([0, BINARY.y.mean(), 1])
    assert list(curves["perfect"].get_ydata()) == [0, 1, 1]
    assert list(_dotted(ax)[0].get_ydata()) == [0, 1]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("rows taken, highest score first", "positives found")


def test_gain_by_hand_and_a_second_model() -> None:
    ax = kx.plot([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], "gain")
    model = _curves(ax)[0]
    assert list(model.get_xdata()) == [0, 0.25, 0.5, 0.75, 1]
    assert list(model.get_ydata()) == [0, 0.5, 0.5, 1, 1]
    ax = kx.plot(BINARY, "y", "gbm", "logreg", "gain")
    assert _legend(ax) == ["gbm", "logreg", "perfect"]


def test_lift_is_one_when_every_row_is_taken() -> None:
    ax = kx.plot(BINARY, "y", "gbm", "logreg", "lift")
    for curve in _curves(ax):
        assert curve.get_xdata()[-1] == 1
        assert curve.get_ydata()[-1] == pytest.approx(1)
        assert curve.get_ydata()[0] > 1.5                   # the top rows hold far more positives than average
    assert list(_dotted(ax)[0].get_ydata()) == [1, 1]
    assert ax.get_ylabel() == "lift"
    lifted = _curves(kx.plot([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], "lift"))[0]
    assert list(lifted.get_ydata()) == pytest.approx([2, 1, 4 / 3, 1])


def test_gain_multiclass_one_perfect_corner_per_class() -> None:
    ax = kx.plot(MULTI.y, MULTI[["cat", "dog", "bird"]], "gain")
    assert _legend(ax) == ["cat", "dog", "bird"]
    corners = [ln for ln in _dotted(ax) if len(ln.get_xdata()) == 3]
    assert [c.get_xdata()[1] for c in corners] == pytest.approx([(MULTI.y == c).mean() for c in ("cat", "dog", "bird")])


def test_pva_metrics_by_hand_in_the_corner() -> None:
    assert kx._metrics.fit_errors([1, 2, 3, 4], [1, 2, 3, 5]) == pytest.approx((0.8, 0.5, 0.25))
    ax = kx.plot(pd.Series([1, 2, 3, 4], name="price"), pd.Series([1, 2, 3, 5], name="guess"), "pva")
    box, = ax.texts
    assert box.get_text() == "R² 0.800\nRMSE 0.5\nMAE 0.25"
    line, = _dashed(ax)
    assert (line.get_xdata().min(), line.get_xdata().max()) == (1, 5)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("price", "guess")
    assert ax.get_aspect() == 1


def test_pva_switches_to_hexbin_over_5000_points() -> None:
    rng = np.random.default_rng(0)
    small = kx.plot(REGRESSION.actual, REGRESSION.predicted, "pva")
    assert isinstance(small.collections[0], mpl.collections.PathCollection)
    assert len(small.figure.axes) == 1
    a = rng.normal(size=5001)
    big = kx.plot(a, a + rng.normal(size=5001), "pva")
    assert isinstance(big.collections[0], mpl.collections.PolyCollection)     # hexagons, with a count colorbar
    assert big.figure.axes[1].get_ylabel() == "count"
    assert len(_dashed(big)) == 1


def test_resid_sign_line_at_zero_and_z_colour() -> None:
    ax = kx.plot([3, 1], [1, 2], "resid")                   # actual 3 predicted 1: +2; actual 1 predicted 2: -1
    assert ax.collections[0].get_offsets().tolist() == [[1, 2], [2, -1]]
    zero, = _dashed(ax)
    assert list(zero.get_ydata()) == [0, 0]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("predicted", "residual (actual - predicted)")
    coloured = kx.plot(REGRESSION, "actual", "predicted", "x", "resid-sym")
    assert coloured.figure.axes[1].get_ylabel() == "x"
    assert len(kx.plot(REGRESSION, "actual", "predicted", "x", "resid-nocb").figure.axes) == 1


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot([1, 2], [1, 2], [1, 2], "pva"), "takes no z"),
    (lambda: kx.plot([1, 2], [1, np.nan], "pva"), "NaN"),
    (lambda: kx.plot([1, 2, 3], [1, 2], "resid"), "actual has 3 values but predicted has 2"),
    (lambda: kx.plot([1, 2], [1, 2], "resid-logc"), "colour a numeric z"),
])
def test_pva_resid_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


IMPORTANCE = kx.load("importance")


def test_imp_largest_on_top_with_error_bars_from_z() -> None:
    ax = kx.plot(IMPORTANCE, "feature", "importance", "std", "imp")
    want = IMPORTANCE.sort_values("importance", ascending=False)
    bars, = (c for c in ax.containers if isinstance(c, mpl.container.BarContainer))
    assert [p.get_width() for p in bars] == list(want.importance)
    assert [t.get_text() for t in ax.get_yticklabels()] == list(want.feature)
    assert ax.yaxis_inverted()                              # the first bar, the largest, is at the top
    segments = bars.errorbar.lines[2][0].get_segments()
    assert [s[1][0] - s[0][0] for s in segments] == pytest.approx(list(2 * want["std"]))
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("importance ± std", "feature")


def test_imp_top_cuts_and_says_how_many_were_left_out() -> None:
    ax = kx.plot(IMPORTANCE, "feature", "importance", "imp", top=5)
    assert len(ax.patches) == 5
    assert [p.get_width() for p in ax.patches] == list(IMPORTANCE.importance.nlargest(5))
    assert ax.get_ylabel() == "feature, top 5 (10 left out)"
    assert not any(isinstance(c, mpl.container.ErrorbarContainer) for c in ax.containers)
    assert ax.get_ylabel() != kx.plot(IMPORTANCE, "feature", "importance", "imp", top=15).get_ylabel()


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot(IMPORTANCE, "feature", "importance", "imp", top=0), "whole number from 1"),
    (lambda: kx.plot(["a", "b"], [1.0, np.nan], "imp"), "NaN"),
    (lambda: kx.plot(["a", "b"], [1.0, 2.0], [0.1], "imp"), "one value per feature"),
])
def test_imp_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


HISTORY = kx.load("history")


def test_loss_pairs_train_and_val_in_one_colour() -> None:
    ax = kx.plot(HISTORY, "loss")
    train, val = ax.lines
    assert (train.get_label(), val.get_label()) == ("loss", "val_loss")
    assert (train.get_linestyle(), val.get_linestyle()) == ("-", "--")
    assert to_hex(train.get_color()) == to_hex(val.get_color())
    assert list(train.get_xdata()) == list(HISTORY.epoch)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("epoch", "loss")
    assert _legend(ax) == ["loss", "val_loss"]


def test_loss_takes_a_dict_or_a_keras_like_history() -> None:
    class History:                              # what model.fit returns, without importing Keras
        def __init__(self) -> None:
            self.history = {"loss": [0.9, 0.5, 0.4], "val_loss": [1.0, 0.7, 0.8]}

    for data in (History(), History().history):
        ax = kx.plot(data, "loss")
        assert [ln.get_label() for ln in ax.lines] == ["loss", "val_loss"]
        assert list(ax.lines[0].get_xdata()) == [1, 2, 3]    # no epoch column: the row number from 1


def test_loss_best_at_the_lowest_val_loss_or_highest_accuracy() -> None:
    ax = kx.plot(HISTORY, "loss-best")
    i = HISTORY.val_loss.idxmin()
    best, = _dashed(ax)[1:]                     # after the val_loss curve
    assert list(best.get_xdata()) == [HISTORY.epoch[i]] * 2
    assert _legend(ax)[-1] == f"best val_loss {HISTORY.val_loss[i]:.3g} at epoch {HISTORY.epoch[i]}"
    acc = kx.plot(HISTORY[["accuracy", "val_accuracy"]], "loss-best")
    j = HISTORY.val_accuracy.idxmax()
    assert _legend(acc)[-1].endswith(f"at epoch {j + 1}")
    assert acc.get_ylabel() == "accuracy"


def test_loss_without_val_columns_still_plots() -> None:
    ax = kx.plot(HISTORY[["epoch", "loss"]], "loss-best")
    assert ax.lines[0].get_label() == "loss"
    assert _legend(ax)[-1].startswith(f"best loss {HISTORY.loss.min():.3g} at epoch ")


def test_loss_errors() -> None:
    with pytest.raises(ValueError, match="at least one metric column"):
        kx.plot(HISTORY[["epoch"]], "loss")
    with pytest.raises(TypeError, match="takes a whole table"):
        kx.plot(HISTORY, "epoch", "loss", "loss")           # the trailing 'loss' is the kind, not the column
    assert len(kx.plot(HISTORY, "epoch", "loss", "line").lines) == 1     # other kinds still read it as a column


CURVES = kx.load("curves")
TRAIN, VAL = CURVES.filter(like="train_"), CURVES.filter(like="val_")


def test_lcurve_band_edges_are_mean_plus_minus_std() -> None:
    ax = kx.plot(CURVES["size"], TRAIN, VAL, "lcurve")
    train, val = ax.lines
    assert list(train.get_ydata()) == pytest.approx(list(TRAIN.mean(axis=1)))
    assert list(val.get_ydata()) == pytest.approx(list(VAL.mean(axis=1)))
    edges = ax.collections[1].get_paths()[0].vertices[:, 1]     # the validation band
    lo, hi = VAL.mean(axis=1) - VAL.std(axis=1, ddof=0), VAL.mean(axis=1) + VAL.std(axis=1, ddof=0)
    assert edges.min() == pytest.approx(lo.min())
    assert edges.max() == pytest.approx(hi.max())
    assert to_hex(ax.collections[1].get_facecolor()[0][:3]) == to_hex(val.get_color())
    assert _legend(ax) == ["train", "validation"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("size", "score")


def test_lcurve_one_fold_is_a_line_without_a_band() -> None:
    ax = kx.plot([10, 20, 30], [0.9, 0.8, 0.8], np.array([[0.5], [0.6], [0.7]]), "lcurve")
    assert len(ax.lines) == 2
    assert len(ax.collections) == 0
    assert ax.get_xlabel() == "training set size"


def test_vcurve_only_changes_the_x_label() -> None:
    ax = kx.plot(np.logspace(-3, 1, 10), TRAIN.to_numpy(), VAL.to_numpy(), "vcurve-logx")
    assert ax.get_xlabel() == "parameter"
    assert ax.get_xscale() == "log"
    assert len(ax.collections) == 2


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot([1, 2, 3], TRAIN, VAL, "lcurve"), r"train has shape \(10, 5\) but x has 3 values"),
    (lambda: kx.plot(CURVES["size"], TRAIN, VAL.iloc[:5], "vcurve"), "validation has shape"),
])
def test_lcurve_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


KS = np.arange(1, 9)
INERTIA = pd.Series([100, 50, 20, 15, 12, 10, 9, 8.0], name="inertia")


def test_knee_is_scale_free_and_needs_three_points() -> None:
    assert kx._metrics.knee(KS, INERTIA) == 2
    assert kx._metrics.knee(KS * 10, INERTIA * 1000) == 2
    assert kx._metrics.knee(np.arange(1, 11), 1000 / np.arange(1, 11)) == 2
    with pytest.raises(ValueError, match="at least 3 points"):
        kx._metrics.knee([1, 2], [3, 1])


def test_elbow_best_marks_the_knee() -> None:
    ax = kx.plot(KS, INERTIA, "elbow-best")
    best, = _dashed(ax)
    assert list(best.get_xdata()) == [3, 3]
    assert _legend(ax) == ["inertia", "knee at k = 3"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("k", "inertia")
    assert ax.lines[0].get_marker() == "o"


def test_elbow_z_on_a_twin_axis() -> None:
    silhouette = pd.Series([0.2, 0.45, 0.6, 0.5, 0.4, 0.35, 0.3, 0.3], name="silhouette")
    ax = kx.plot(KS, INERTIA, silhouette, "elbow-best")
    twin, = (a for a in ax.figure.axes if a is not ax)
    assert twin.get_ylabel() == "silhouette"
    assert list(twin.lines[0].get_ydata()) == list(silhouette)
    assert ax.get_shared_x_axes().joined(ax, twin)
    assert [t.get_text() for t in twin.get_legend().get_texts()] == ["inertia", "silhouette", "knee at k = 3"]
    assert ax.get_legend() is None
    assert len(kx.plot(KS, INERTIA, "elbow").figure.axes) == 1


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot([1, 2], [5, 3], "elbow-best"), "at least 3 points"),
    (lambda: kx.plot(KS, INERTIA, [0.1, 0.2], "elbow"), "x has 8 values but z has 2"),
])
def test_elbow_errors(call: Callable[[], Axes], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def _titles(axs: np.ndarray) -> list[str]:
    return [ax.get_title() for ax in axs]


def test_evaluate_binary_is_roc_pr_calib_proba_with_every_score() -> None:
    axs = kx.evaluate(BINARY.y, BINARY.gbm, BINARY.logreg)
    assert isinstance(axs, np.ndarray)
    assert axs.shape == (4,)
    assert all(isinstance(ax, Axes) for ax in axs)
    assert _titles(axs) == ["ROC", "precision-recall", "calibration", "scores by class: gbm"]
    assert all(ax.figure is axs[0].figure for ax in axs)
    assert [len(_curves(ax)) for ax in axs[:3]] == [2, 2, 2]  # both models in roc, pr and calib
    assert _legend(axs[0]) == [f"gbm (AUC {kx.roc_auc(BINARY.y, BINARY.gbm):.2f})",
                               f"logreg (AUC {kx.roc_auc(BINARY.y, BINARY.logreg):.2f})"]
    assert len(axs[3].containers) == 2


def test_evaluate_names_unnamed_scores() -> None:
    axs = kx.evaluate(BINARY.y.to_numpy(), BINARY.gbm.to_numpy(), BINARY.logreg.to_numpy())
    assert [t.split(" (")[0] for t in _legend(axs[0])] == ["score 1", "score 2"]
    assert _legend(kx.evaluate(BINARY.y.to_numpy(), BINARY.gbm.to_numpy())[0])[0].startswith("score (AUC")


def test_evaluate_multiclass_is_roc_cm_of_argmax_and_proba() -> None:
    axs = kx.evaluate(MULTI.y, MULTI[["cat", "dog", "bird"]])
    assert _titles(axs) == ["ROC, one-vs-rest", "confusion matrix", "score of the true class"]
    assert len(_curves(axs[0])) == 3
    want = pd.crosstab(MULTI.y, MULTI.pred, normalize="index")      # pred is the argmax column in the data
    assert np.allclose(axs[1].images[0].get_array(), want.to_numpy())
    codes = MULTI.y.map({"bird": 0, "cat": 1, "dog": 2})
    by_code = kx.evaluate(codes, MULTI[["bird", "cat", "dog"]].to_numpy())
    assert np.allclose(by_code[1].images[0].get_array(), want.to_numpy())


def test_evaluate_regression_is_pva_resid_and_a_residual_hist() -> None:
    axs = kx.evaluate(REGRESSION.actual, REGRESSION.predicted)
    assert _titles(axs) == ["predicted vs actual", "residuals", "residual distribution"]
    assert axs[0].texts[0].get_text().startswith("R² ")
    assert axs[2].get_xlabel() == "residual (actual - predicted)"
    heights = sum(p.get_height() for p in axs[2].patches)
    assert heights == len(REGRESSION)


def test_evaluate_spec_reaches_every_panel() -> None:
    with kx.context("light"):
        axs = kx.evaluate(BINARY.y, BINARY.gbm, spec="dark-grid")
    dark = mpl.rc_params_from_file(kx._theme.style_path("dark"), use_default_template=False)
    assert to_hex(axs[0].figure.get_facecolor()) == to_hex(dark["figure.facecolor"])   # the figure too, not only panels
    for ax in axs:
        assert to_hex(ax.get_facecolor()) == to_hex(dark["axes.facecolor"])
        assert ax.xaxis.get_gridlines()[0].get_visible()


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.evaluate(BINARY.y), "at least one score"),
    (lambda: kx.evaluate(BINARY.y, BINARY.gbm, BINARY.logreg, BINARY.gbm), "at most 2 models"),
    (lambda: kx.evaluate(REGRESSION.actual, REGRESSION.predicted, REGRESSION.x), "one model's predictions"),
    (lambda: kx.evaluate(MULTI.y, MULTI.cat), "pass a 2-D score"),
])
def test_evaluate_errors(call: Callable[[], np.ndarray], match: str) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        call()


# ---------- metrics ----------
_I = np.arange(200)
SCORE = np.round(((_I * 7919) % 200) / 200, 1)            # 11 distinct values, so many tied scores
TRUTH = (((_I * 7919) % 200) / 200 + ((_I * 104729) % 97) / 97 * 0.8) > 0.9


def test_auc_and_ap_match_scikit_learn() -> None:
    # computed once with sklearn.metrics.roc_auc_score and average_precision_score on TRUTH, SCORE
    assert kx.roc_auc(TRUTH, SCORE) == pytest.approx(0.8937043339005105, abs=1e-12)
    assert kx.pr_ap(TRUTH, SCORE) == pytest.approx(0.8694172331641548, abs=1e-12)


@pytest.mark.parametrize(("score", "auc", "ap"), [
    ([0.1, 0.2, 0.8, 0.9], 1.0, 1.0),                      # perfect
    ([0.9, 0.8, 0.2, 0.1], 0.0, 5 / 12),                   # inverted: precision 1/3 then 2/4
    ([0.5, 0.5, 0.5, 0.5], 0.5, 0.5),                      # all tied: one step, the positive rate
])
def test_auc_and_ap_by_hand(score: list[float], auc: float, ap: float) -> None:
    y = [0, 0, 1, 1]
    assert kx.roc_auc(y, score) == pytest.approx(auc)
    assert kx.pr_ap(y, score) == pytest.approx(ap)


def test_roc_and_pr_curves_ends_and_ties() -> None:
    fpr, tpr, thr = kx._metrics.roc_curve([0, 1, 0, 1], [0.3, 0.3, 0.1, 0.9])
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0, 0, 1, 1)
    assert list(thr) == [np.inf, 0.9, 0.3, 0.1]             # the tied 0.3 is one step
    precision, recall, _ = kx._metrics.pr_curve([0, 1, 0, 1], [0.3, 0.3, 0.1, 0.9])
    assert list(precision) == [1, 1, 2 / 3, 0.5]
    assert list(recall) == [0, 0.5, 1, 1]


@pytest.mark.parametrize(("y", "positive"), [
    ([False, True, True], [0, 1, 1]),
    ([1.0, 0.0, 1.0], [1, 0, 1]),
    (["no", "yes", "no"], [0, 1, 0]),
    (pd.Series(["b", "a", "a"]), [1, 0, 0]),
])
def test_binary_positive_class(y: list, positive: list[int]) -> None:
    assert list(kx._metrics.binary(y)) == positive


@pytest.mark.parametrize(("y", "score", "match"), [
    ([1, 1, 1], [0.1, 0.5, 0.9], "only positives"),
    ([0, 0, 0], [0.1, 0.5, 0.9], "only negatives"),
    (["a", "a"], [0.1, 0.9], "one label only"),
    ([0, 1, 2], [0.1, 0.5, 0.9], "pass a 2-D score, one column per class"),
    ([0, 1, 1], [[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]], "a binary metric takes one score per row"),
    ([0, 1], [0.1, 0.5, 0.9], "y_true has 2 values but the score has 3"),
    ([0, 1, 1], [0.1, np.nan, 0.9], "NaN"),
    ([0, np.nan, 1], [0.1, 0.5, 0.9], "missing values"),
])
def test_metrics_raise(y: list, score: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        kx.roc_auc(y, score)


def test_confusion_counts_and_unpredicted_class() -> None:
    m = kx._metrics.confusion(["a", "b", "b", "c", "c"], ["a", "b", "a", "a", "b"])
    assert m.tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 0]]  # 'c' is never predicted but keeps its column
    assert kx._metrics.confusion([0, 1], [1, 1], labels=[1, 0]).tolist() == [[1, 0], [1, 0]]


def test_brier_calibration_and_gain() -> None:
    assert kx._metrics.brier([0, 1, 1], [0.0, 1.0, 0.5]) == pytest.approx(0.25 / 3)
    mean, rate, n = kx._metrics.calibration([0, 0, 1, 1, 1], [0.05, 0.15, 0.15, 0.95, 1.0], bins=10)
    assert mean.tolist() == pytest.approx([0.05, 0.15, 0.975])  # empty bins left out, 1.0 in the last bin
    assert rate.tolist() == [0, 0.5, 1]
    assert n.tolist() == [1, 2, 2]
    with pytest.raises(ValueError, match="between 0 and 1"):
        kx._metrics.calibration([0, 1], [-0.5, 2.0])
    taken, found = kx._metrics.gain([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
    assert taken.tolist() == [0, 0.25, 0.5, 0.75, 1]
    assert found.tolist() == [0, 0.5, 0.5, 1, 1]
