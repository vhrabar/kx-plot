"""Transparency (show, grammar, source), kx.grid and kx.save."""
import datetime as dt
import subprocess
import sys
import typing
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from helpers import DF, X, Y, Z

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


# ---------- dataframe libraries (narwhals) ----------
def _drawn(axs: object) -> list:
    """What a plot draws, comparable across runs: line and patch geometry, points, images and tick labels."""
    out = []
    for ax in np.atleast_1d(axs):
        out.append([ln.get_xydata().tolist() for ln in ax.lines])
        out.append([p.get_path().vertices.tolist() for p in ax.patches])
        out.append([np.asarray(c.get_offsets()).tolist() for c in ax.collections])
        out.append([np.asarray(im.get_array()).tolist() for im in ax.images])
        out.append([t.get_text() for t in [*ax.get_xticklabels(), *ax.get_yticklabels()]])
        out.append([ax.get_xlabel(), ax.get_ylabel(), ax.get_title()])
    return out


def _to_polars(obj: object) -> object:
    """A pandas frame or Series as polars, the gaps as nulls."""
    pl = pytest.importorskip("polars")
    if isinstance(obj, pd.Series):
        return pl.Series(obj.name, obj.astype(object).where(obj.notna(), None).tolist())
    return pl.DataFrame({c: obj[c].astype(object).where(obj[c].notna(), None).tolist() for c in obj.columns})


def _same(f: Callable[[object], object]) -> object:
    return f


BINARY = kx.load("binary")
SHAPES = kx.load("shapes")
PLOTS = {
    "line": lambda f: kx.plot(f(DF), "t", "speed", "accel", "line-leg"),
    "box by=": lambda f: kx.plot(f(DF), "speed", spec="box", by="group"),
    "heat-corr": lambda f: kx.plot(f(DF.drop(columns="group")), "heat-corr-ann"),
    "roc": lambda f: kx.plot(f(BINARY), "y", "gbm", "logreg", "roc"),
    "imgrid": lambda f: kx.plot(SHAPES["images"][:30], f(pd.Series(SHAPES["labels"][:30], name="label")), "imgrid",
                                n=2),
    "count": lambda f: kx.plot(f(kx.load("passengers")), "port", spec="count"),
}


@pytest.mark.parametrize("name", PLOTS)
def test_polars_input_draws_the_same_as_pandas(name: str) -> None:
    pytest.importorskip("polars")
    assert _drawn(PLOTS[name](_to_polars)) == _drawn(PLOTS[name](_same))


def test_polars_dtypes_become_their_pandas_counterparts() -> None:
    pl = pytest.importorskip("polars")
    from kx._frame import to_pandas

    when = pl.Series("t", [dt.datetime(2025, 1, 1, 12), None]).dt.replace_time_zone("Europe/Zagreb")
    frame = pl.DataFrame({
        "i": [1, None], "s": ["a", None], "b": [True, None], "d": [dt.date(2025, 1, 1), None], "t": when,
        "dur": [dt.timedelta(days=1), None],
        "e": pl.Series(["hi", None], dtype=pl.Enum(["lo", "mid", "hi"])),
        "c": pl.Series(["z", "y"], dtype=pl.Categorical),
    })
    got = to_pandas(frame)
    assert list(got.columns) == frame.columns
    assert got["i"].dtype == float
    assert np.isnan(got["i"][1])
    assert got["s"][0] == "a"
    assert pd.isna(got["s"][1])
    assert got["b"][0] is True
    assert pd.isna(got["b"][1])
    assert got["d"].dtype == "datetime64[ns]"
    assert pd.isna(got["d"][1])
    assert str(got["t"].dt.tz) == "Europe/Zagreb"
    assert got["t"][0].hour == 12
    assert got["dur"][0] == pd.Timedelta(days=1)
    assert list(got["e"].cat.categories) == ["lo", "mid", "hi"]
    assert pd.isna(got["e"][1])
    assert list(got["c"].cat.categories) == ["z", "y"]
    assert to_pandas(pl.Series([1, 2])).name is None
    assert to_pandas(pl.Series("v", [1, 2])).name == "v"


def test_pandas_input_passes_through_untouched() -> None:
    from kx._frame import to_pandas, wrap

    column = DF["t"]
    assert to_pandas(DF) is DF
    assert to_pandas(column) is column
    assert wrap(X) is X
    assert wrap("line") == "line"


def test_lazy_frames_are_refused() -> None:
    pl = pytest.importorskip("polars")
    with pytest.raises(TypeError, match=r"\.collect\(\)"):
        kx.plot(pl.DataFrame({"x": [1, 2], "y": [3, 4]}).lazy(), "x", "y")


def test_polars_columns_by_name_with_typo_hint() -> None:
    pytest.importorskip("polars")
    with pytest.raises(ValueError, match="Did you mean"):
        kx.plot(_to_polars(DF), "t", "sped", "line")


def test_polars_reaches_evaluate_embed_and_cramers() -> None:
    pytest.importorskip("polars")
    pb = _to_polars(BINARY)
    assert _drawn(kx.evaluate(pb["y"], pb["gbm"])) == _drawn(kx.evaluate(BINARY["y"], BINARY["gbm"]))
    e = kx.load("embedding")
    assert np.allclose(kx.embed(_to_polars(e), by="label").embedding, kx.embed(e, by="label").embedding)
    p = kx.load("passengers")
    assert kx.cramers(_to_polars(p), plot=False).equals(kx.cramers(p, plot=False))


def test_load_into_polars() -> None:
    pl = pytest.importorskip("polars")
    p, pandas = kx.load("passengers", backend="polars"), kx.load("passengers")
    assert isinstance(p, pl.DataFrame)
    assert p.columns == list(pandas.columns)
    assert p.null_count().row(0) == tuple(pandas.isna().sum())
    assert isinstance(kx.load("shapes", backend="polars"), dict)
    with pytest.raises(ValueError, match="bogus"):
        kx.load("daily", backend="bogus")


def test_import_kx_does_not_import_polars() -> None:
    code = "import sys, kx; print('polars' in sys.modules, 'pyarrow' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


def test_typing_names_the_input_types() -> None:
    assert {"DataFrame", "Series", "Column"} <= set(kx.typing.__all__)
    hints = typing.get_type_hints(kx.plot)
    assert hints["by"] == kx.typing.Column | str | None
    assert typing.get_type_hints(kx.cramers)["df"] is kx.typing.DataFrame
    assert typing.get_type_hints(kx.evaluate)["y"] == kx.typing.Column
    for f in (kx.missing, kx.embed, kx.load):
        typing.get_type_hints(f)                # every public annotation resolves
