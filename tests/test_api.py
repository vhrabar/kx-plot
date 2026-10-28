"""Transparency (show, grammar, source), kx.grid and kx.save."""
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pytest
from helpers import X, Y, Z

import kx


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
