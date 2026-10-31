"""
kx: tiny matplotlib helper.

Token grammar for kx.plot(spec):   kind-theme-flag-flag-...
  kind  : line | step | area | band | scatter | bar | barh | hist | kde | ridge | ecdf | box | violin | strip
          | qq | heat | crosstab | count | missing | nullmap | hexbin | pie | donut | radar | parallel | pair
          | cluster | mosaic | treemap | sunburst
          | cm | roc | pr | calib | proba | thresh | gain | lift | pva | resid | imp | loss
          | lcurve | vcurve | elbow
          | imgrid
  theme : any file name in styles/
  flags : grid | leg | legout | tight | rot | wrap | logx | logy | logxy | symlogy | zero | eq | pct | si | facet
          and per kind: stack norm cum (bars, area, hist), date mk (lines), sort ann (bars, count),
          stack norm (count),
          mean median ann (distributions), sig (box, violin),
          nocb sym logc (scatter, heat, hexbin, cluster), corr ann (heat, cluster),
          norm ann nocb logc (crosstab), ann all (missing), sort (nullmap), diag (scatter, hexbin),
          fit (scatter), norm ann nocb (cm), norm (proba), best (thresh, loss, elbow),
          nocb sym logc (resid), ann sort (pie, donut), ann (mosaic, treemap, sunburst), raw (radar), mean (parallel).
          kx.grammar(kind) lists them.
by=    : group labels for x, one series per group (kde, ridge, ecdf, box, violin, strip, qq, count);
         for radar, parallel and pair a column of the table: kx.plot(df, 'radar', by='class')
images : image kinds take an (N, H, W[, C]) array or a list of arrays or file paths, labels as y:
         kx.plot(images, labels, 'imgrid')
facet= : with the flag facet, one panel per value of a column, on shared axes and one colour scale:
         kx.plot(df, 'pclass', 'port', 'crosstab-facet', facet='sex')
table  : heat, cluster, missing, nullmap, loss, radar, parallel, pair take a whole DataFrame or 2-D array:
         kx.plot(df, 'heat-corr'); a dict or a Keras History works too: kx.plot(model.fit(...), 'loss')

Palettes (kx.use(theme, palette="okabe")): any file name in styles/palettes/
Colormaps (kx.use(theme, cmap="magma")):  kx.cmaps()
with kx.context(theme, ...): the same as kx.use, for one block only.
kx.save("fig.png"): saves to /kaggle/working on Kaggle, the current folder elsewhere.
kx.load("passengers"): a demo table; kx.load("shapes"): a dict of image arrays. kx.datasets() lists them.
kx.missing(df): the share of missing values per column, the same as kx.plot(df, 'missing-ann').
kx.grid(5): a figure of 5 panels as a flat array of axes, for kx.plot(..., ax=).
kx.roc_auc(y_true, score) and kx.pr_ap(y_true, score): area under the ROC curve and average precision, in numpy.
kx.evaluate(y_true, score): a model report, the panels picked from the data (binary, multiclass or regression).
kx.grammar("area") and kx.source("area") explain one kind; kx.parse(spec) shows how a spec is read.
"""
import glob
import importlib.metadata
import inspect
import os
import re
from typing import Any

import pandas as pd
from matplotlib.axes import Axes

from ._data import datasets, load
from ._evaluate import evaluate
from ._grid import grid
from ._metrics import pr_ap, roc_auc
from ._plot import plot
from ._save import save
from ._spec import COMMON_FLAGS, KINDS, Spec, all_flags, parse
from ._theme import BREWER_CMAPS, BUILTIN_CMAPS, VENDORED_CMAPS, cmaps, context, palette, palettes, show, styles, use

__all__ = ["BREWER_CMAPS", "BUILTIN_CMAPS", "VENDORED_CMAPS", "Spec", "__version__", "cmaps", "context", "datasets",
           "evaluate", "grammar", "grid", "load", "missing", "palette", "palettes", "parse", "plot", "pr_ap", "roc_auc",
           "save", "show", "source", "styles", "use"]


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
        if k.panels:
            print("axes : draws several panels; kx.plot returns them as a flat array")
        if k.projection:
            print(f"axes : {k.projection}; kx.plot swaps a given ax for {k.projection} axes in its place")
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


def missing(df: pd.DataFrame, spec: str = "", **kw: Any) -> Axes:
    """The share of missing values in each column, labelled: kx.plot(df, 'missing-ann-' + spec).
    kx.missing(df, 'dark-wrap') adds a theme and flags; other keywords (title=, ax=, ...) go to kx.plot."""
    return plot(df, "-".join(["missing", "ann", *([spec] if spec else [])]), **kw)
