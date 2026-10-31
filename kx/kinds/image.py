"""Image kinds: a set of images (an array, or a list of arrays or file paths) as the data. See kx/_image.py."""
import math
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from .._image import bare, count, fit_figure, show, take, tile
from .._spec import kind
from .._util import split
from .basic import _levels

MAX_CLASSES = 20                                # imgrid rows: one per class


def _check_imgrid(y: ArrayLike | None, z: ArrayLike | None, flags: set[str]) -> None:
    if z is not None:
        raise ValueError("kind 'imgrid' takes the images (x) and maybe their labels (y), no z")


@kind("imgrid", needs_y=False, panels=True, check=_check_imgrid)
def imgrid(ax: Axes, x: Any, y: ArrayLike | None, z: None, flags: set[str], n: int | None = None, seed: int = 0,
           idx: list[int] | None = None, cols: int | None = None, **kw: Any) -> np.ndarray:
    """Sample images: with labels (y), one row per class of n= random images (6), the class at the left;
    without, a grid of n= random images (16). idx= shows those images instead, titled with their labels.
    seed= picks the same images each time; only the images shown are read. Returns their panels.
    kw: ax.imshow."""
    total = count(x)
    if y is not None and np.size(y) != total:
        raise ValueError(f"there are {total} images but {np.size(y)} labels")
    names = np.asarray(y, dtype=object).ravel() if y is not None else None
    rng = np.random.default_rng(seed)
    if idx is not None or y is None:
        picks = list(idx) if idx is not None else \
            [int(i) for i in rng.choice(total, min(n or 16, total), replace=False)]
        cols = min(len(picks), cols or math.ceil(math.sqrt(len(picks))))
        fit_figure(ax, -(-len(picks) // cols), cols)
        axs = tile(ax, len(picks), cols)
        for a, img, i in zip(axs, take(x, picks), picks, strict=True):
            show(a, img, **kw)
            if names is not None:
                a.set_title(str(names[i]), fontsize="small")
        return axs
    classes, masks = _levels(y)
    if len(classes) > MAX_CLASSES:
        raise ValueError(f"{len(classes)} classes are too many rows; pass idx= or a subset of the classes")
    per = n or 6
    rows = [[int(i) for i in rng.permutation(np.flatnonzero(m))[:per]] for m in masks]
    width = max(len(r) for r in rows)
    fit_figure(ax, len(classes), width)
    grid = split(ax, len(classes), width, wspace=0.05, hspace=0.08).reshape(len(classes), width)
    shown = []
    for row, cls, picks in zip(grid, classes, rows, strict=True):
        for a, img in zip(row, take(x, picks), strict=False):
            bare(a)
            show(a, img, **kw)
            shown.append(a)
        for a in row[len(picks):]:              # a class with fewer images than n leaves its row short
            a.remove()
        row[0].set_ylabel(str(cls), rotation=0, ha="right", va="center")
    return np.array(shown)
