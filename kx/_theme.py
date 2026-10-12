"""Themes, palettes and colormaps: discovery, kx.use and kx.context."""
import contextlib
from collections.abc import Iterator
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from . import _files

BUILTIN_CMAPS = ("viridis", "magma", "plasma", "inferno", "cividis", "turbo",
                 "twilight", "coolwarm", "berlin", "managua", "vanimo")
# styles/cmaps/<name>.txt: cmocean, then Crameri's Scientific colour maps 8
VENDORED_CMAPS = ("thermal", "haline", "deep", "balance",
                  "batlow", "lipari", "hawaii", "oslo", "davos", "vik", "roma", "bam", "cork", "romaO")


def styles() -> list[str]:
    """List available theme names."""
    return _files.names("styles", ".mplstyle")


def style_path(name: str) -> str:
    """Path to styles/<name>.mplstyle; unknown names raise ValueError."""
    if name not in styles():
        raise ValueError(f"unknown theme {name!r}. Available: {styles()}")
    return _files.path("styles", f"{name}.mplstyle")


def palettes() -> list[str]:
    """List available palette names."""
    return _files.names("styles/palettes", ".txt")


def palette(name: str) -> list[str]:
    """Colors of styles/palettes/<name>.txt as '#RRGGBB' strings (one hex per line, '#' lines are comments)."""
    return _read_palette(name)


def _read_palette(name: str) -> list[str]:
    """palette() under a name that kx.use's `palette` argument does not shadow."""
    if name not in palettes():
        raise ValueError(f"unknown palette {name!r}. Available: {palettes()}")
    return _read_hex(_files.path("styles", "palettes", f"{name}.txt"))


def _read_hex(path: str) -> list[str]:
    """One hex per line as '#RRGGBB'; '#' lines are comments."""
    with open(path) as f:
        lines = (line.strip() for line in f)
        return [f"#{line}" for line in lines if line and not line.startswith("#")]


def _register_cmaps() -> None:
    """Add the vendored colormaps to matplotlib, once."""
    for name in VENDORED_CMAPS:
        if name not in mpl.colormaps:
            cmap = ListedColormap(_read_hex(_files.path("styles", "cmaps", f"{name}.txt")), name=name)
            mpl.colormaps.register(cmap)


_register_cmaps()


def cmaps() -> list[str]:
    """List supported colormap names that this matplotlib version has, plus the vendored ones."""
    return sorted(n for n in BUILTIN_CMAPS + VENDORED_CMAPS if n in mpl.colormaps)


def show(name: str) -> None:
    """Print the contents of a style file."""
    with open(style_path(name)) as f:
        print(f.read())


def use(name: str = "dark", palette: str | list[str] | None = None, cmap: str | None = None,
        **rc: Any) -> None:
    """Apply a theme, optional palette (name from styles/palettes/ or list of colors),
    colormap (one of kx.cmaps(), '_r' reverses it), and rcParam overrides.
    Use double underscore for dots:  figure__figsize=(8, 4)  ->  figure.figsize
    """
    if cmap is not None and cmap.removesuffix("_r") not in cmaps():
        raise ValueError(f"unknown cmap {cmap!r}. Available: {cmaps()} (add '_r' to reverse)")
    plt.style.use(["default", style_path(name)])       # reset first so themes never mix
    if cmap:
        if cmap not in mpl.colormaps:                   # '<name>_r' that matplotlib does not ship
            mpl.colormaps.register(mpl.colormaps[cmap.removesuffix("_r")].reversed(), name=cmap)
        plt.rcParams["image.cmap"] = cmap
    if palette:
        colors = _read_palette(palette) if isinstance(palette, str) else list(palette)
        # swap only the colors; other cycled props (paper's linestyles) repeat to the new length
        rest = {k: v for k, v in plt.rcParams["axes.prop_cycle"].by_key().items() if k != "color"}
        plt.rcParams["axes.prop_cycle"] = plt.cycler(
            color=colors, **{k: [v[i % len(v)] for i in range(len(colors))] for k, v in rest.items()})
    plt.rcParams.update({k.replace("__", "."): v for k, v in rc.items()})


@contextlib.contextmanager
def context(name: str = "dark", palette: str | list[str] | None = None, cmap: str | None = None,
            **rc: Any) -> Iterator[None]:
    """kx.use for one with block:  with kx.context('paper', palette='okabe'): ...
    Same arguments as kx.use; the previous rcParams come back on exit, also after an error."""
    with mpl.rc_context():
        use(name, palette, cmap, **rc)
        yield
