"""Demo datasets, kx.load and the version."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import kx


# ---------- data ----------
def test_datasets_and_load() -> None:
    assert {"waves", "cloud", "monthly", "dists", "daily", "passengers"} <= set(kx.datasets())
    assert list(kx.load("waves").columns) == ["x", "y", "z"]


def test_daily_is_two_full_years_of_days() -> None:
    daily = kx.load("daily")
    assert list(daily.columns) == ["date", "temp", "sales"]
    days = pd.to_datetime(daily.date)
    assert days.diff().iloc[1:].eq(pd.Timedelta("1D")).all()
    assert (days.iloc[0], days.iloc[-1]) == (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31"))


def test_passengers_has_categories_and_nans() -> None:
    p = kx.load("passengers")
    assert p.shape == (500, 8)
    assert list(p.columns) == ["age", "fare", "sex", "pclass", "port", "deck", "ticket_type", "survived"]
    assert all(pd.api.types.is_float_dtype(p[c]) for c in ("age", "fare"))
    assert all(pd.api.types.is_integer_dtype(p[c]) for c in ("pclass", "survived"))
    for c in ("sex", "port", "deck", "ticket_type"):
        assert all(isinstance(v, str) for v in p[c].dropna())
    assert set(p.pclass) == {1, 2, 3}
    assert set(p.survived) == {0, 1}
    assert p.sex.nunique() == 2
    assert p.port.nunique() == 3
    assert set(p.deck.dropna()) == set("ABCDEFG")
    assert p.ticket_type.str.len().min() > 12
    assert p.fare.mean() > p.fare.median()
    nans = p.isna().mean()
    assert nans.age == pytest.approx(0.2, abs=0.05)
    assert nans.deck == pytest.approx(0.75, abs=0.05)
    assert 0 < nans.port < 0.03
    assert (nans.drop(["age", "deck", "port"]) == 0).all()


def test_ml_datasets_shapes_and_story() -> None:
    b, m, r = kx.load("binary"), kx.load("multiclass"), kx.load("regression")
    h, imp, c = kx.load("history"), kx.load("importance"), kx.load("curves")
    assert list(b.columns) == ["y", "gbm", "logreg"]
    assert list(m.columns) == ["y", "cat", "dog", "bird", "pred"]
    assert list(r.columns) == ["actual", "predicted", "x"]
    assert list(h.columns) == ["epoch", "loss", "val_loss", "accuracy", "val_accuracy"]
    assert list(imp.columns) == ["feature", "importance", "std"]
    assert list(c.columns) == ["size", *(f"train_{k}" for k in range(1, 6)), *(f"val_{k}" for k in range(1, 6))]
    assert (b.shape, m.shape, r.shape, h.shape, imp.shape, c.shape) == (
        (1000, 3), (600, 5), (500, 3), (60, 5), (15, 3), (10, 11))
    assert set(b.y) == {0, 1}
    assert b[["gbm", "logreg"]].stack().between(0, 1).all()
    assert kx.roc_auc(b.y, b.gbm) > kx.roc_auc(b.y, b.logreg)
    assert set(m.y) == set(m.pred) == {"cat", "dog", "bird"}
    assert np.allclose(m[["cat", "dog", "bird"]].sum(axis=1), 1)
    assert (m.pred == m[["cat", "dog", "bird"]].idxmax(axis=1)).all()
    err = (r.actual - r.predicted).abs()
    assert err[r.x > 7].mean() > 2 * err[r.x < 3].mean()
    assert 0 < h.val_loss.idxmin() < len(h) - 1             # overfits: the minimum is inside, not at the end
    assert h.loss.idxmin() > h.val_loss.idxmin()
    assert imp.importance.sum() == pytest.approx(1, abs=1e-3)
    assert c.val_1.iloc[-1] > c.val_1.iloc[0]
    assert c.train_1.iloc[-1] < c.train_1.iloc[0]


def test_load_unknown() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        kx.load("nope")


def test_embedding_clusters_by_label() -> None:
    e = kx.load("embedding")
    assert e.shape == (600, 15)
    assert list(e.columns) == ["x", "y", "label", "group", "size", *(f"f{k}" for k in range(10))]
    assert set(e.label) == {"alpha", "beta", "gamma", "delta", "epsilon"}
    assert e.label.value_counts().is_monotonic_decreasing                # imbalanced, largest first
    assert set(e.group) == {"batch_a", "batch_b"}
    centres = e.groupby("label")[["x", "y"]].mean()
    spread = e.groupby("label")[["x", "y"]].std().mean(axis=1)
    assert np.hypot(*(centres.loc["alpha"] - centres.loc["beta"])) > 4 * spread.max()
    assert e["size"].corr(e.x) > 0.8
    f = e.filter(regex=r"^f\d$").to_numpy()
    s = np.linalg.svd(f - f.mean(0), compute_uv=False) ** 2
    assert s[:2].sum() / s.sum() > 0.8                                    # PCA finds the 2-D layout again


def test_shapes_is_a_dict_of_image_arrays() -> None:
    d = kx.load("shapes")
    assert isinstance(d, dict)
    assert set(d) == {"images", "masks", "labels", "pred", "proba", "saliency", "artifact", "classes"}
    n = 210
    assert (d["images"].shape, d["images"].dtype) == ((n, 40, 40, 3), np.uint8)
    assert (d["masks"].shape, d["masks"].dtype) == ((n, 40, 40), np.bool_)
    assert d["proba"].shape == (n, 7)
    assert d["saliency"].shape == (n, 10, 10)
    assert d["saliency"].min() == 0
    assert d["saliency"].max() == 1
    assert list(d["classes"]) == ["blob", "ring", "square", "cross", "stripes", "dots", "crescent"]
    counts = pd.Series(d["labels"]).value_counts()
    assert list(counts[list(d["classes"])]) == [60, 40, 30, 25, 20, 20, 15]
    assert np.allclose(d["proba"].sum(axis=1), 1, atol=1e-5)
    assert (d["pred"] == d["classes"][d["proba"].argmax(axis=1)]).all()
    assert 0.6 < (d["pred"] == d["labels"]).mean() < 0.9                 # some mistakes for miscls
    assert set(d["artifact"]) == {"", "border", "line"}
    assert all(d["masks"][i].any() for i in range(n))
    ink = d["images"].astype(float).mean(axis=3)
    assert all(ink[i][d["masks"][i]].mean() < ink[i][~d["masks"][i]].mean() for i in range(n))  # dark shapes


def test_datasets_lists_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data").mkdir()
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "data" / "tab.csv", index=False)
    np.savez(tmp_path / "data" / "arr.npz", v=np.arange(3))
    monkeypatch.setattr(kx._files, "root", str(tmp_path))
    assert kx.datasets() == ["arr", "tab"]
    assert list(kx.load("arr")["v"]) == [0, 1, 2]
    assert isinstance(kx.load("tab"), pd.DataFrame)
    with pytest.raises(ValueError, match="unknown dataset"):
        kx.load("nope")


def test_version_comes_from_pyproject() -> None:
    toml = (Path(kx.__file__).parent.parent / "pyproject.toml").read_text()
    assert f'\nversion = "{kx.__version__}"\n' in toml
