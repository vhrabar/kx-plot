"""Image kinds: a set of images (an array, or a list of arrays or file paths) as the data. See kx/_image.py."""
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from .._image import bare, count, fit_figure, histograms, moments, show, take, tile
from .._spec import kind
from .._util import split
from .basic import _levels

MAX_CLASSES = 20                                # imgrid rows: one per class


def _check_labelled(name: str) -> Any:
    """A check that an image kind got no z."""
    def check(y: ArrayLike | None, z: ArrayLike | None, flags: set[str]) -> None:
        if z is not None:
            raise ValueError(f"kind {name!r} takes the images (x) and maybe their labels (y), no z")
    return check


@kind("imgrid", needs_y=False, panels=True, check=_check_labelled("imgrid"))
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


def _classes(x: Any, y: ArrayLike | None) -> tuple[list[str], list[np.ndarray]]:
    """Class names and the image positions in each; without labels one group of every image."""
    total = count(x)
    if y is None:
        return ["all"], [np.arange(total)]
    if np.size(y) != total:
        raise ValueError(f"there are {total} images but {np.size(y)} labels")
    names, masks = _levels(y)
    return names, [np.flatnonzero(m) for m in masks]


@kind("meanimg", flags=("std",), needs_y=False, panels=True, check=_check_labelled("meanimg"))
def meanimg(ax: Axes, x: Any, y: ArrayLike | None, z: None, flags: set[str], cols: int | None = None,
            **kw: Any) -> np.ndarray:
    """The mean image of each class (y), or of all images without labels, titled with the class and its count.
    Sizes that differ are brought to the most common one (nearest neighbour). std = how much each pixel varies
    instead, the channels averaged, on one colour scale for every class with one colorbar. Up to 8 a row (cols=).
    kw: ax.imshow."""
    names, groups = _classes(x, y)
    stats, _ = moments(x, groups)
    cols = min(len(names), cols or 8)
    fit_figure(ax, -(-len(names) // cols), cols, side=1.8)
    axs = tile(ax, len(names), cols)
    if "std" in flags:
        maps = [np.sqrt(np.clip(sq - mean ** 2, 0, None)) for mean, sq, _ in stats]
        maps = [m.mean(axis=2) if m.ndim == 3 else m for m in maps]
        top = max(float(m.max()) for m in maps) or 1.0
        shown = [a.imshow(m, **{"cmap": "magma", "vmin": 0, "vmax": top, "interpolation": "nearest", **kw})
                 for a, m in zip(axs, maps, strict=True)]
        axs[0].figure.colorbar(shown[0], ax=list(axs), label="std", shrink=0.8)
    else:
        for a, (mean, _, _) in zip(axs, stats, strict=True):
            show(a, np.clip(mean, 0, 1), **kw)
    for a, name, (_, _, n) in zip(axs, names, stats, strict=True):
        a.set_title(f"{name} (n={n})", fontsize="small")
    return axs


@kind("colorhist", flags=("hsv", "map"), needs_y=False, panels=True, check=_check_labelled("colorhist"))
def colorhist(ax: Axes, x: Any, y: ArrayLike | None, z: None, flags: set[str], bins: int = 32,
              **kw: Any) -> np.ndarray:
    """How the pixel values are spread in each class (y): one panel per channel (red, green, blue; grey images
    one), one line per class, each the class's average share of pixels in bins= bins (32) over 0..1.
    hsv = hue, saturation and value instead; map = a class x value heat map per channel, on one colour scale.
    kw: the lines (ax.plot) or the heat maps (ax.imshow)."""
    if bins < 2:
        raise ValueError(f"bins= needs at least 2 bins, got {bins}")
    names, groups = _classes(x, y)
    hists, channels = histograms(x, groups, bins, "hsv" in flags)
    fig = ax.figure
    if len(fig.axes) == 1:                      # a figure of its own: each channel the size of a small plot
        w, h = plt.rcParams["figure.figsize"]
        fig.set_size_inches(len(channels) * w * 0.7 + 1, h * (0.4 + 0.08 * len(names) if "map" in flags else 0.8))
        if fig.get_layout_engine() is None:
            fig.set_layout_engine("constrained")
    axs = split(ax, 1, len(channels), wspace=0.08)
    centres = (np.arange(bins) + 0.5) / bins
    if "map" in flags:
        top = float(hists.max()) or 1.0
        for c, (a, channel) in enumerate(zip(axs, channels, strict=True)):
            im = a.imshow(hists[:, c], **{"cmap": "magma", "vmin": 0, "vmax": top, "aspect": "auto",
                                          "extent": (0, 1, len(names) - 0.5, -0.5), "interpolation": "nearest",
                                          **kw})
            a.set_yticks(range(len(names)), names if c == 0 else [])
            a.set_title(channel)
            a.set_xlabel("value")
        fig.colorbar(im, ax=list(axs), label="share of pixels", shrink=0.8)
        return axs
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for c, (a, channel) in enumerate(zip(axs, channels, strict=True)):
        for g, name in enumerate(names):          # past the palette's colours, the lines repeat them dashed
            a.plot(centres, hists[g, c], **{"label": name, "color": colours[g % len(colours)],
                                            "linestyle": ("-", "--", ":")[g // len(colours) % 3], **kw})
        a.set_title(channel)
        a.set_xlabel("value")
        a.set_xlim(0, 1)
        if c:
            a.sharey(axs[0])
            a.tick_params(labelleft=False)
    axs[0].set_ylabel("share of pixels")
    axs[0].set_ylim(0, float(hists.max()) * 1.05 or 1)   # the highest line of any channel, not just the first
    if y is not None:
        axs[-1].legend(title=getattr(y, "name", None), loc="upper left", bbox_to_anchor=(1.02, 1))
    return axs
