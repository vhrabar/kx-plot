"""Transparency (show, grammar, source), kx.grid and kx.save."""
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pytest
from helpers import X, Y, Z

import kx
import kx._stats


# ---------- transparency ----------
def test_show_grammar_source(capsys: pytest.CaptureFixture[str]) -> None:
    kx.show("dark")
    assert "111418" in capsys.readouterr().out
    kx.grammar()
    out = capsys.readouterr().out
    assert "scatter" in out
    assert "dark" in out
    assert "logy" in out
    assert "okabe" in out
    assert "viridis" in out
    kx.source()
    assert "def plot(" in capsys.readouterr().out


def test_grammar_and_source_of_one_kind(capsys: pytest.CaptureFixture[str]) -> None:
    kx.grammar("area")
    out = capsys.readouterr().out
    assert "100% stack" in out
    assert "'norm'" in out
    assert "'grid'" in out
    kx.source("band")
    out = capsys.readouterr().out
    assert out.lstrip().startswith("@kind(\"band\"")
    assert "def scatter(" not in out

# ---------- grid ----------
@pytest.mark.parametrize(("n", "cols", "shape"), [(5, 3, (2, 3)), (2, 3, (1, 2)), (12, 4, (3, 4)), (1, 3, (1, 1))])
def test_grid_is_flat_with_n_panels(n: int, cols: int, shape: tuple[int, int]) -> None:
    mpl.rcParams["figure.figsize"] = (4, 3)
    axs = kx.grid(n, cols=cols)
    assert isinstance(axs, np.ndarray)
    assert axs.shape == (n,)
    fig = axs[0].figure
    assert fig.axes == list(axs)                # unused cells are gone
    assert tuple(fig.get_size_inches()) == (4 * shape[1], 3 * shape[0])


def test_grid_passes_subplots_keywords() -> None:
    axs = kx.grid(4, cols=2, sharex=True, figsize=(6, 6))
    assert axs[0].get_shared_x_axes().joined(axs[0], axs[3])
    assert tuple(axs[0].figure.get_size_inches()) == (6, 6)


def test_grid_panels_take_kx_plots_with_tight_and_colorbars() -> None:
    a, b = kx.grid(2)
    kx.plot(X, Y, Z, "scatter", ax=a)           # a colorbar, then tight: no layout-engine clash
    kx.plot(X, Y, "line-tight", ax=b)
    assert len(a.figure.axes) == 3


@pytest.mark.parametrize(("n", "cols"), [(0, 3), (3, 0)])
def test_grid_rejects_empty(n: int, cols: int) -> None:
    with pytest.raises(ValueError, match="n >= 1 and cols >= 1"):
        kx.grid(n, cols=cols)


# ---------- saving ----------
@pytest.fixture
def kaggle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake /kaggle/working, and a different current folder."""
    (tmp_path / "working").mkdir()
    (tmp_path / "cwd").mkdir()
    monkeypatch.setattr(kx._save, "KAGGLE", str(tmp_path / "working"))
    monkeypatch.chdir(tmp_path / "cwd")
    return tmp_path


def test_save_goes_to_kaggle_working(kaggle: Path) -> None:
    kx.plot(X, Y, "line")
    path = kx.save("plots/fig.png")
    assert path == str(kaggle / "working" / "plots" / "fig.png")
    assert Path(path).stat().st_size > 0


def test_save_outside_kaggle_uses_current_folder(kaggle: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kx._save, "KAGGLE", str(kaggle / "missing"))
    kx.plot(X, Y, "line")
    assert Path(kx.save("fig.png")).resolve() == kaggle / "cwd" / "fig.png"


def test_save_absolute_path_and_given_ax(kaggle: Path) -> None:
    import matplotlib.pyplot as plt
    ax = kx.plot(X, Y, "line")
    plt.figure()
    target = kaggle / "elsewhere" / "fig.png"
    assert kx.save(str(target), ax, dpi=50) == str(target)
    assert plt.imread(target).shape[1] == int(ax.figure.get_figwidth() * 50)


def test_save_without_extension_uses_theme_format(kaggle: Path) -> None:
    with kx.context("thesis"):
        kx.plot(X, Y, "line")
        path = kx.save("fig")
    assert path.endswith("fig.pdf")
    assert Path(path).read_bytes().startswith(b"%PDF")
    kx.plot(X, Y, "line")
    assert kx.save("fig2", format="svg").endswith("fig2.svg")


# ---------- kx.embed ----------
EMBEDDING = kx.load("embedding")
FEATURES = EMBEDDING[[f"f{k}" for k in range(10)]]


def test_embed_pca_matches_svd_with_variance_in_the_labels() -> None:
    ax = kx.embed(FEATURES, by=EMBEDDING.label)
    z = (FEATURES - FEATURES.mean()) / FEATURES.std(ddof=0)
    u, s, _ = np.linalg.svd(z.to_numpy(), full_matrices=False)
    for k in range(2):                                            # the same axes up to sign
        assert abs(np.corrcoef(ax.embedding[:, k], u[:, k])[0, 1]) == pytest.approx(1)
    share = s ** 2 / (s ** 2).sum()
    assert ax.get_xlabel() == f"PC1 ({share[0]:.0%})"
    assert len(ax.collections) == EMBEDDING.label.nunique()
    again = kx.embed(FEATURES, by=EMBEDDING.label)
    assert np.array_equal(again.embedding, ax.embedding)         # signs fixed: the same picture each run


def test_embed_takes_a_column_name_spec_and_images() -> None:
    ax = kx.embed(EMBEDDING.drop(columns=["x", "y", "group"]), by="label", spec="ann", title="pca")
    assert ax.get_title() == "pca"
    assert {t.get_text() for t in ax.texts} == set(EMBEDDING.label)
    assert ax.embedding.shape == (600, 2)                         # label left out of the features
    shapes = kx.load("shapes")
    pics = kx.embed(shapes["images"], by=shapes["labels"])
    assert pics.embedding.shape == (210, 2)


def test_embed_tsne_and_umap_when_installed() -> None:
    pytest.importorskip("sklearn")
    ax = kx.embed(FEATURES.iloc[:100], method="tsne", by=EMBEDDING.label.iloc[:100], perplexity=10)
    assert ax.get_xlabel() == "t-SNE 1"
    assert ax.embedding.shape == (100, 2)
    pytest.importorskip("umap")
    ax = kx.embed(FEATURES.iloc[:100], method="umap", n_neighbors=10)
    assert ax.get_xlabel() == "UMAP 1"


def test_embed_says_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    real = builtins.__import__

    def missing(name: str, *args: object, **kw: object) -> object:
        if name.startswith("sklearn"):
            raise ImportError(name)
        return real(name, *args, **kw)
    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ImportError, match=r"needs scikit-learn: pip install 'kx-plot\[embed\]'"):
        kx.embed(FEATURES, method="tsne")


@pytest.mark.parametrize(("call", "error", "match"), [
    (lambda: kx.embed(FEATURES, method="mds"), ValueError, "unknown method"),
    (lambda: kx.embed(FEATURES, perplexity=5), TypeError, "takes no options"),
    (lambda: kx.embed(FEATURES.assign(f0=np.nan)), ValueError, "fill or drop the NaNs"),
    (lambda: kx.embed(EMBEDDING[["label"]]), ValueError, "needs numeric columns"),
    (lambda: kx.embed(FEATURES, by="nope"), ValueError, "no column 'nope'"),
    (lambda: kx.embed(FEATURES.to_numpy(), by="label"), TypeError, "needs a DataFrame"),
    (lambda: kx.embed(FEATURES.iloc[:2]), ValueError, "at least 3 rows"),
    (lambda: kx.embed([np.zeros((4, 4)), np.zeros((5, 5)), np.zeros((4, 4))]), ValueError, "one size"),
])
def test_embed_errors(call: Callable[[], object], error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        call()


# ---------- kx.cramers ----------
PASSENGERS = kx.load("passengers")


def test_cramers_matrix_of_the_categorical_columns() -> None:
    v = kx.cramers(PASSENGERS, plot=False)
    assert list(v.columns) == ["sex", "pclass", "port", "deck", "ticket_type", "survived"]   # no age, no fare
    assert np.allclose(v.to_numpy(), v.to_numpy().T)
    assert np.allclose(np.diag(v), 1)
    assert v.loc["sex", "survived"] == pytest.approx(kx._stats.cramers_v(PASSENGERS.sex, PASSENGERS.survived))
    assert v.loc["sex", "survived"] > 0.3 > v.loc["sex", "port"]


def test_cramers_heat_map_on_0_1_with_two_decimals() -> None:
    ax = kx.cramers(PASSENGERS, cols=["sex", "survived", "pclass"], title="V")
    image = ax.images[0]
    assert image.norm.vmin == 0
    assert image.norm.vmax == 1
    assert image.colorbar.ax.get_ylabel() == "Cramér's V"
    assert "1.00" in [t.get_text() for t in ax.texts]
    assert ax.get_title() == "V"


def test_cramers_cluster_puts_associated_columns_together() -> None:
    order = list(kx.cramers(PASSENGERS, cluster=True, plot=False).columns)
    assert abs(order.index("pclass") - order.index("ticket_type")) == 1
    assert abs(order.index("sex") - order.index("survived")) == 1


def test_cramers_errors() -> None:
    with pytest.raises(ValueError, match=r"cols= names \['nope'\]"):
        kx.cramers(PASSENGERS, cols=["sex", "nope"])
    with pytest.raises(ValueError, match="compares pairs"):
        kx.cramers(PASSENGERS[["age", "fare", "sex"]])
