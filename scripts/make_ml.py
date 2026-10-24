"""
Write the machine-learning demo tables to kx/data/: binary, multiclass, regression, history, importance and curves.
"""
import os

import numpy as np
import pandas as pd

SEED = 5
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kx", "data")
CLASSES = ("cat", "dog", "bird")
FEATURES = ("age", "income", "tenure", "visits", "basket_size", "discount", "region", "device", "weekday", "hour",
            "returns", "reviews", "email_opens", "referrals", "plan")


def _sigmoid(v: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-v))


def binary(rng: np.random.Generator, n: int = 1000) -> pd.DataFrame:
    """Two models scoring the same 0/1 outcome: gbm ranks better but is overconfident, logreg is calibrated."""
    logit = rng.normal(-0.4, 1.6, n)
    y = (rng.random(n) < _sigmoid(logit)).astype(int)
    gbm = _sigmoid(2.2 * (logit + rng.normal(0, 0.6, n)))
    seen = logit + rng.normal(0, 1.3, n)
    logreg = _sigmoid(0.85 * (-0.4 + 0.6 * (seen + 0.4)))       # E[logit | seen], shrunk for the noise: calibrated
    return pd.DataFrame({"y": y, "gbm": gbm.round(4), "logreg": logreg.round(4)})


def multiclass(rng: np.random.Generator, n: int = 600) -> pd.DataFrame:
    """Three classes with one probability column each, summing to 1, and the predicted class."""
    y = rng.choice(len(CLASSES), n, p=[0.45, 0.35, 0.2])
    logits = 1.8 * np.eye(len(CLASSES))[y] + rng.normal(0, 1, (n, len(CLASSES)))
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    p = p.round(4)
    p[:, -1] = (1 - p[:, :-1].sum(axis=1)).round(4)             # rounding must not break the sum of 1
    df = pd.DataFrame(p, columns=list(CLASSES))
    df.insert(0, "y", np.array(CLASSES)[y])
    df["pred"] = np.array(CLASSES)[p.argmax(axis=1)]
    return df


def regression(rng: np.random.Generator, n: int = 500) -> pd.DataFrame:
    """Actual against predicted values, with the error growing with x."""
    x = rng.uniform(0, 10, n)
    actual = 5 + 3 * x + rng.normal(0, 2, n)
    predicted = actual + rng.normal(0, 0.5 + 0.6 * x)
    return pd.DataFrame({"actual": actual.round(3), "predicted": predicted.round(3), "x": x.round(3)})


def history(rng: np.random.Generator, epochs: int = 60) -> pd.DataFrame:
    """Training curves that overfit after about 30 epochs: val_loss turns up while loss keeps falling."""
    e = np.arange(1, epochs + 1)
    loss = 0.25 + 0.9 * np.exp(-e / 10) + rng.normal(0, 0.01, epochs)
    val_loss = 0.38 + 0.85 * np.exp(-e / 9) + 0.00025 * np.maximum(e - 30, 0) ** 2 + rng.normal(0, 0.015, epochs)
    accuracy = 0.95 - 0.45 * np.exp(-e / 10) + rng.normal(0, 0.005, epochs)
    val_accuracy = 0.88 - 0.4 * np.exp(-e / 9) - 0.00008 * np.maximum(e - 30, 0) ** 2 + rng.normal(0, 0.008, epochs)
    return pd.DataFrame({"epoch": e, "loss": loss.round(4), "val_loss": val_loss.round(4),
                         "accuracy": accuracy.round(4), "val_accuracy": val_accuracy.round(4)})


def importance(rng: np.random.Generator) -> pd.DataFrame:
    """Feature importances summing to 1 with their spread across folds, in no particular order."""
    raw = np.exp(-np.arange(len(FEATURES)) / 3) * rng.uniform(0.8, 1.2, len(FEATURES))
    imp = raw / raw.sum()
    std = imp * rng.uniform(0.1, 0.3, len(imp))
    df = pd.DataFrame({"feature": FEATURES, "importance": imp.round(4), "std": std.round(4)})
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def curves(rng: np.random.Generator, folds: int = 5) -> pd.DataFrame:
    """Learning-curve scores per fold: train falls and validation rises as the training set grows."""
    size = np.array([50, 100, 200, 300, 400, 500, 600, 700, 800, 1000])
    df = pd.DataFrame({"size": size})
    train = 0.82 + 0.16 * np.exp(-size / 250)
    val = 0.80 - 0.3 * np.exp(-size / 220)
    for k in range(1, folds + 1):
        df[f"train_{k}"] = (train + rng.normal(0, 0.01, len(size))).round(4)
    for k in range(1, folds + 1):
        df[f"val_{k}"] = (val + rng.normal(0, 0.025, len(size))).round(4)
    return df


def make(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    return {f.__name__: f(rng) for f in (binary, multiclass, regression, history, importance, curves)}


if __name__ == "__main__":
    for name, df in make().items():
        out = os.path.join(DATA, f"{name}.csv")
        df.to_csv(out, index=False)
        print(f"wrote {out}")
