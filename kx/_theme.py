"""Themes, palettes and colormaps: discovery, kx.use and kx.context."""
import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap, to_hex

from . import _files

BUILTIN_CMAPS = ("viridis", "magma", "plasma", "inferno", "cividis", "turbo",
                 "twilight", "coolwarm", "berlin", "managua", "vanimo")
# lowercase names for matplotlib's ColorBrewer maps, registered as copies
BREWER_CMAPS = {"blues": "Blues", "greys": "Greys", "rdbu": "RdBu"}
# styles/cmaps/<name>.txt: cmocean, then Crameri's Scientific colour maps 8
VENDORED_CMAPS = ("thermal", "haline", "deep", "balance",
                  "batlow", "lipari", "hawaii", "oslo", "davos", "vik", "roma", "bam", "cork", "romaO")

# kx.register: name -> '#rrggbb' colours, for this session
_RUNTIME_PALETTES: dict[str, list[str]] = {}
# kx.register(cmap=True): names added to mpl.colormaps
_RUNTIME_CMAPS: set[str] = set()


def styles() -> list[str]:
    """List available theme names."""
    return _files.names("styles", ".mplstyle")


def style_path(name: str) -> str:
    """Path to styles/<name>.mplstyle; unknown names raise ValueError."""
    if name not in styles():
        raise ValueError(f"unknown theme {name!r}. Available: {styles()}")
    return _files.path("styles", f"{name}.mplstyle")


def palettes() -> list[str]:
    """List available palette names: the files in styles/palettes/ and those added with kx.register."""
    return sorted({*_files.names("styles/palettes", ".txt"), *_RUNTIME_PALETTES})


def palette(name: str) -> list[str]:
    """Colors of styles/palettes/<name>.txt as '#RRGGBB' strings (one hex per line, '#' lines are comments)."""
    return _read_palette(name)


def _read_palette(name: str) -> list[str]:
    """palette() under a name that kx.use's `palette` argument does not shadow."""
    if name in _RUNTIME_PALETTES:
        return list(_RUNTIME_PALETTES[name])
    if name not in palettes():
        raise ValueError(f"unknown palette {name!r}. Available: {palettes()}")
    return _read_hex(_files.path("styles", "palettes", f"{name}.txt"))


def _read_hex(path: str) -> list[str]:
    """One hex per line as '#RRGGBB'; '#' lines are comments."""
    with open(path) as f:
        lines = (line.strip() for line in f)
        return [f"#{line}" for line in lines if line and not line.startswith("#")]


def _register_cmaps() -> None:
    """Add the vendored colormaps and the lowercase ColorBrewer names to matplotlib, once."""
    for name in VENDORED_CMAPS:
        if name not in mpl.colormaps:
            cmap = ListedColormap(_read_hex(_files.path("styles", "cmaps", f"{name}.txt")), name=name)
            mpl.colormaps.register(cmap)
    for name, brewer in BREWER_CMAPS.items():
        if name not in mpl.colormaps:
            # mpl.colormaps.register(mpl.colormaps[brewer], name=name)
            # older mpl syntax
            cmap = mpl.colormaps[brewer]
            cmap.name = name
            mpl.colormaps.register(cmap)


_register_cmaps()


def cmaps() -> list[str]:
    """List supported colormap names that this matplotlib version has, plus the vendored ones and those added
    with kx.register."""
    names = (*BUILTIN_CMAPS, *BREWER_CMAPS, *VENDORED_CMAPS, *_RUNTIME_CMAPS)
    return sorted({n for n in names if n in mpl.colormaps})


def register(name: str, colors: Sequence[Any] | Colormap, cmap: bool = False) -> None:
    """Add a palette, or with cmap=True a colormap, for this session:  kx.register('brand', ['#0b3d91', 'tab:red'])
    colors: anything matplotlib reads as a colour ('#hex', names, RGB tuples); a Colormap is registered as a
    colormap. A palette then works as kx.use(palette='brand'), a colormap as kx.use(cmap='brand') and 'brand_r'.
    A name a shipped palette or a matplotlib colormap already has raises; registering the same name again
    replaces it, so a notebook cell can run twice. To keep one, add a file to styles/palettes/ or styles/cmaps/."""
    if not isinstance(name, str) or not name.strip() or name.endswith("_r"):
        raise ValueError(f"register needs a name that does not end in '_r' (kept for reversed maps), got {name!r}")
    if isinstance(colors, Colormap) or cmap:
        _register_cmap(name, colors)
        return
    if name in _files.names("styles/palettes", ".txt"):
        raise ValueError(f"palette {name!r} ships with kx; pick another name")
    _RUNTIME_PALETTES[name] = _hexes(colors, least=1)


def _register_cmap(name: str, colors: Sequence[Any] | Colormap) -> None:
    """kx.register with cmap=True: a copy of a Colormap, or a smooth map through the colours, in mpl.colormaps."""
    if name in mpl.colormaps and name not in _RUNTIME_CMAPS:
        raise ValueError(f"colormap {name!r} exists in matplotlib or kx; pick another name")
    if isinstance(colors, Colormap):
        new = colors.copy()
        new.name = name
    else:
        new = LinearSegmentedColormap.from_list(name, _hexes(colors, least=2))
    for old in (name, f"{name}_r"):             # the old map, and a reversed copy kx.use made of it
        if old in mpl.colormaps:
            mpl.colormaps.unregister(old)
    mpl.colormaps.register(new, name=name)
    _RUNTIME_CMAPS.add(name)


def _hexes(colors: Sequence[Any], least: int) -> list[str]:
    """colors as '#rrggbb', at least `least` of them; a bad colour raises naming it."""
    if isinstance(colors, str) or len(colors) < least:
        raise ValueError(f"register needs a list of at least {least} colours, got {colors!r}")
    out = []
    for c in colors:
        try:
            out.append(to_hex(c))
        except (ValueError, TypeError):
            raise ValueError(f"register: {c!r} is not a colour") from None
    return out


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
