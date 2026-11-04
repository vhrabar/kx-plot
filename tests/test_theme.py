"""Themes, palettes and colormaps."""
import re
from pathlib import Path

import matplotlib as mpl
import pytest
from helpers import X, Y, Z
from matplotlib.colors import to_hex

import kx


# ---------- themes ----------
def test_theme_in_spec_is_local() -> None:
    before = dict(mpl.rcParams)
    ax = kx.plot(X, Y, "line-dark")
    assert to_hex(ax.get_facecolor()) == "#111418"
    assert {k for k in before if str(before[k]) != str(mpl.rcParams[k])} == set()


def test_use_sets_global_theme() -> None:
    kx.use("dark")
    assert to_hex(mpl.rcParams["axes.facecolor"]) == "#111418"


def test_use_resets_before_applying() -> None:
    kx.use("dark")                              # dark sets bold titles, light does not
    kx.use("light")
    assert mpl.rcParams["axes.titleweight"] == "normal"


def test_context_applies_inside_and_restores_after() -> None:
    kx.use("light", palette="okabe")
    before = dict(mpl.rcParams)
    with kx.context("dark", palette=["#264653", "#e9c46a"], cmap="magma", figure__figsize=(5, 2)):
        assert to_hex(mpl.rcParams["axes.facecolor"]) == "#111418"
        assert mpl.rcParams["axes.prop_cycle"].by_key()["color"] == ["#264653", "#e9c46a"]
        assert mpl.rcParams["image.cmap"] == "magma"
        ax = kx.plot(X, Y, "line")
    assert to_hex(ax.get_facecolor()) == "#111418"          # the plot keeps the theme it was drawn in
    assert {k for k in before if str(before[k]) != str(mpl.rcParams[k])} == set()


def test_context_restores_after_an_error() -> None:
    kx.use("light")
    with pytest.raises(RuntimeError), kx.context("dark"):
        raise RuntimeError
    assert to_hex(mpl.rcParams["axes.facecolor"]) == "#ffffff"
    with pytest.raises(ValueError, match="unknown theme"), kx.context("nope"):
        pass
    assert to_hex(mpl.rcParams["axes.facecolor"]) == "#ffffff"


def test_use_palette_and_rc_overrides() -> None:
    kx.use("light", palette=["#264653", "#e9c46a"], figure__figsize=(5, 2))
    assert mpl.rcParams["axes.prop_cycle"].by_key()["color"] == ["#264653", "#e9c46a"]
    assert tuple(mpl.rcParams["figure.figsize"]) == (5, 2)


# ---------- palettes ----------
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]


@pytest.mark.parametrize(("name", "mpl_name"), [("set1", "Set1"), ("set2", "Set2"), ("dark2", "Dark2"),
                                                ("paired", "Paired"), ("tab10", "tab10"), ("tab20", "tab20")])
def test_palettes_match_matplotlib(name: str, mpl_name: str) -> None:
    assert [c.lower() for c in kx.palette(name)] == [to_hex(c) for c in mpl.colormaps[mpl_name].colors]


def test_okabe_palette() -> None:
    assert "okabe" in kx.palettes()
    assert kx.palette("okabe") == OKABE


@pytest.mark.parametrize("name", kx.palettes())
def test_palette_file_is_valid(name: str) -> None:
    colors = kx.palette(name)
    assert len(colors) >= 3
    assert all(re.fullmatch(r"#[0-9A-F]{6}", c) for c in colors), colors
    assert len(set(colors)) == len(colors)
    assert "#FFFFFF" not in colors              # white vanishes on light themes


def test_use_named_palette() -> None:
    kx.use("light", palette="okabe")
    ax = kx.plot(X, Y, Z, "line")
    assert [to_hex(ln.get_color()).upper() for ln in ax.lines] == OKABE[:2]


def test_palette_keeps_theme_linestyles() -> None:
    kx.use("paper", palette="okabe")
    cycle = mpl.rcParams["axes.prop_cycle"].by_key()
    assert cycle["color"] == OKABE
    assert cycle["linestyle"][:6] == ["-", "--", ":", "-.", "-", "--"]
    assert len(cycle["linestyle"]) == len(OKABE)


def test_unknown_palette() -> None:
    with pytest.raises(ValueError, match="unknown palette"):
        kx.use("dark", palette="nope")


def test_palettes_discovers_new_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "styles" / "palettes").mkdir(parents=True)
    (tmp_path / "styles" / "palettes" / "mine.txt").write_text("# comment\n\n112233\n  AABBCC  \n")
    monkeypatch.setattr(kx._files, "root", str(tmp_path))
    assert kx.palettes() == ["mine"]
    assert kx.styles() == []                    # the subfolder is not a theme
    assert kx.palette("mine") == ["#112233", "#AABBCC"]


# ---------- colormaps ----------
def test_cmaps_lists_matplotlib_builtins() -> None:
    names = (*kx.BUILTIN_CMAPS, *kx.BREWER_CMAPS, *kx.VENDORED_CMAPS)
    assert kx.cmaps() == sorted(n for n in names if n in mpl.colormaps)
    assert set(kx.BUILTIN_CMAPS) - {"berlin", "managua", "vanimo"} <= set(kx.cmaps())   # only these need 3.10


def test_cmaps_skips_names_missing_from_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kx._theme, "BUILTIN_CMAPS", ("viridis", "notacmap"))
    monkeypatch.setattr(kx._theme, "VENDORED_CMAPS", ())
    monkeypatch.setattr(kx._theme, "BREWER_CMAPS", {})
    assert kx.cmaps() == ["viridis"]


@pytest.mark.parametrize("name", kx.VENDORED_CMAPS)
def test_vendored_cmaps_are_registered(name: str) -> None:
    assert mpl.colormaps[name].N == 256
    assert name in kx.cmaps()


@pytest.mark.parametrize(("name", "brewer"), kx.BREWER_CMAPS.items())
def test_brewer_cmaps_are_lowercase_copies(name: str, brewer: str) -> None:
    assert name in mpl.colormaps
    assert name in kx.cmaps()
    assert mpl.colormaps[name].name == name
    for v in (0, 0.5, 1):
        assert to_hex(mpl.colormaps[name](v)) == to_hex(mpl.colormaps[brewer](v))


@pytest.mark.parametrize("name", ["viridis", *kx.BREWER_CMAPS, *kx.VENDORED_CMAPS])
def test_use_reversed_cmap(name: str) -> None:
    kx.use("light", cmap=f"{name}_r")
    rev = kx.plot(X, Y, Z, "scatter").collections[0].get_cmap()
    assert rev.name == f"{name}_r"
    assert to_hex(rev(1.0)) == to_hex(mpl.colormaps[name](0.0))


@pytest.mark.parametrize("name", [*kx.BUILTIN_CMAPS, *kx.BREWER_CMAPS, *kx.VENDORED_CMAPS])
def test_use_cmap_colours_scatter(name: str) -> None:
    if name not in mpl.colormaps:
        pytest.skip(f"{name} needs a newer matplotlib")
    kx.use("light", cmap=name)
    assert mpl.rcParams["image.cmap"] == name
    assert kx.plot(X, Y, Z, "scatter").collections[0].get_cmap().name == name


def test_unknown_cmap_leaves_theme_untouched() -> None:
    kx.use("dark")
    with pytest.raises(ValueError, match="unknown cmap"):
        kx.use("light", cmap="nope")
    with pytest.raises(ValueError, match="unknown cmap"):
        kx.use("light", cmap="nope_r")
    assert to_hex(mpl.rcParams["axes.facecolor"]) == "#111418"


@pytest.mark.parametrize("fn", [kx.use, kx.show])
def test_unknown_theme(fn) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="unknown theme"):
        fn("nope")


def test_styles_discovers_new_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "styles").mkdir()
    (tmp_path / "styles" / "mine.mplstyle").write_text("lines.linewidth: 3\n")
    monkeypatch.setattr(kx._files, "root", str(tmp_path))
    assert kx.styles() == ["mine"]
    assert kx.plot(X, Y, "line-mine").lines[0].get_linewidth() == 3


def test_paper_cycles_linestyles_for_greyscale_print() -> None:
    ax = kx.plot(X, Y, Z, "line-paper")
    assert ax.lines[0].get_linestyle() != ax.lines[1].get_linestyle()


def test_scatter_size_follows_theme_markersize() -> None:
    small = kx.plot(X, Y, "scatter-science").collections[0].get_sizes()[0]
    large = kx.plot(X, Y, "scatter-poster").collections[0].get_sizes()[0]
    assert kx.plot(X, Y, "scatter").collections[0].get_sizes()[0] == 18
    assert small < 18 < large


def test_every_shipped_style_loads() -> None:
    for name in kx.styles():
        kx.use(name)
