"""
kx: tiny matplotlib helper.

Token grammar for kx.plot(spec):   kind-theme-flag-flag-...
  kind  : line | step | area | band | scatter | bar | barh | hist | kde | ridge | ecdf | box | violin | strip
          | qq | heat | crosstab | count
  theme : any file name in styles/
  flags : grid | leg | legout | tight | rot | logx | logy | logxy | symlogy | zero | eq | pct | si
          and per kind: stack norm cum (bars, area, hist), date mk (lines), sort ann (bars, count),
          stack norm (count),
          mean median ann (distributions), nocb sym logc (scatter, heat), corr ann (heat),
          norm ann nocb logc (crosstab).
          kx.grammar(kind) lists them.
by=    : group labels for x, one series per group (kde, ridge, ecdf, box, violin, strip, qq, count)
table  : heat takes a whole DataFrame or 2-D array, with no column names: kx.plot(df, 'heat-corr')

Palettes (kx.use(theme, palette="okabe")): any file name in styles/palettes/
Colormaps (kx.use(theme, cmap="magma")):  kx.cmaps()
with kx.context(theme, ...): the same as kx.use, for one block only.
kx.save("fig.png"): saves to /kaggle/working on Kaggle, the current folder elsewhere.
kx.grid(5): a figure of 5 panels as a flat array of axes, for kx.plot(..., ax=).
kx.grammar("area") and kx.source("area") explain one kind; kx.parse(spec) shows how a spec is read.
"""
import glob
import importlib.metadata
import inspect
import os
import re

from ._data import datasets, load
from ._grid import grid
from ._plot import plot
from ._save import save
from ._spec import COMMON_FLAGS, KINDS, Spec, all_flags, parse
from ._theme import BREWER_CMAPS, BUILTIN_CMAPS, VENDORED_CMAPS, cmaps, context, palette, palettes, show, styles, use

__all__ = ["BREWER_CMAPS", "BUILTIN_CMAPS", "VENDORED_CMAPS", "Spec", "__version__", "cmaps", "context", "datasets",
           "grammar", "grid", "load", "palette", "palettes", "parse", "plot", "save", "show", "source", "styles", "use"]


def _version() -> str:
    """project.version from pyproject.toml, the only place it is written.

    """
    toml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyproject.toml")
    if os.path.isfile(toml):
        with open(toml) as f:
            m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.MULTILINE)
        if m:
            return m.group(1)
    try:
        return importlib.metadata.version("kx-plot")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


__version__ = _version()


def _kind(name: str):  # noqa: ANN202
    if name not in KINDS:
        raise ValueError(f"unknown kind {name!r}. Available: {sorted(KINDS)}")
    return KINDS[name]


def grammar(kind: str | None = None) -> None:
    """Print the valid tokens for kx.plot specs, or what one kind does and which flags it takes."""
    if kind:
        k = _kind(kind)
        print(f"{k.name}: {inspect.getdoc(k.draw)}")
        print("flags:", sorted(COMMON_FLAGS), "+", sorted(k.flags) or "none of its own")
        if k.takes_table:
            print("data : takes a DataFrame or 2-D array")
        if k.takes_by:
            print("by=  : group labels for x, one per group")
        return
    print("kind :", sorted(KINDS))
    print("theme:", styles())
    print("flags:", sorted(all_flags()))
    print("palettes (kx.use):", palettes())
    print("cmaps    (kx.use):", cmaps())


def source(kind: str | None = None) -> None:
    """Print the code that draws one kind, or every file in this package."""
    if kind:
        print(inspect.getsource(_kind(kind).draw))
        return
    here = os.path.dirname(os.path.abspath(__file__))
    for p in sorted(glob.glob(os.path.join(here, "**", "*.py"), recursive=True)):
        with open(p) as f:
            print(f"# ---------- kx/{os.path.relpath(p, here)} ----------\n{f.read()}")
