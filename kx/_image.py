"""Images as input: arrays, lists of arrays or file paths, turned into a list of 0..1 float images."""
import math
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from ._util import split

CHANNELS = (1, 3, 4)


def images(x: Any) -> list[np.ndarray]:
    """x as a list of float images in 0..1, each (H, W) grey or (H, W, 3) RGB. Sizes may differ.
    x: one 2-D image, an (N, H, W) or (N, H, W, C) array (or (N, C, H, W) as PyTorch has it), or a list of
    arrays, PIL images or file paths (anything plt.imread reads). A single colour image is [img] or img[None].
    Integers are divided by 255 (or 65535 above 255), floats outside 0..1 are scaled per image to fill it,
    alpha is dropped and (H, W, 1) is grey."""
    if isinstance(x, (str, os.PathLike)):
        x = [x]
    if isinstance(x, np.ndarray) and x.dtype != object:
        if x.ndim == 2:
            x = x[None]
        elif x.ndim == 4 and x.shape[1] in CHANNELS and x.shape[-1] not in CHANNELS:
            x = x.transpose(0, 2, 3, 1)         # channels first: (N, C, H, W)
        elif x.ndim not in (3, 4):
            raise ValueError(f"images as one array are (N, H, W) or (N, H, W, C); got shape {x.shape}")
    out = [_image(v, i) for i, v in enumerate(x)]
    if not out:
        raise ValueError("no images given")
    return out


def _image(v: Any, i: int) -> np.ndarray:
    """One image as a 0..1 float array, (H, W) or (H, W, 3)."""
    a = plt.imread(os.fspath(v)) if isinstance(v, (str, os.PathLike)) else np.asarray(v)
    if a.ndim == 3 and a.shape[-1] not in CHANNELS:
        raise ValueError(f"image {i} has shape {a.shape}: the last axis must be 1, 3 or 4 channels")
    if a.ndim not in (2, 3):
        raise ValueError(f"image {i} has shape {a.shape}; an image is (H, W) or (H, W, C)")
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] == 1 else a[..., :3]
    if a.dtype == bool:
        return a.astype(float)
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(float) / (255 if a.max(initial=0) <= 255 else 65535)
    a = a.astype(float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if lo >= 0 and hi <= 1:
        return a
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def tile(ax: Axes, n: int, cols: int | None = None, **kw: Any) -> np.ndarray:
    """ax's area as n image panels, `cols` per row (about square by default), with no ticks or frames but room
    for a title and a y label. Cells past n in the last row are removed. kw: subgridspec (wspace=, hspace=)."""
    if n < 1:
        raise ValueError(f"no images to show (n={n})")
    cols = min(n, cols or math.ceil(math.sqrt(n)))
    axs = split(ax, math.ceil(n / cols), cols, **{"wspace": 0.05, "hspace": 0.15, **kw})
    for a in axs[n:]:
        a.remove()
    for a in axs[:n]:
        a.set_xticks([])
        a.set_yticks([])
        a.spines[:].set_visible(False)
    return axs[:n]


def show(ax: Axes, image: np.ndarray, **kw: Any) -> AxesImage:
    """image (from images()) on ax: grey in the grey colormap, on the full 0..1 scale, pixels kept square."""
    grey = {"cmap": "gray"} if image.ndim == 2 else {}
    return ax.imshow(image, **{"vmin": 0, "vmax": 1, "interpolation": "nearest", **grey, **kw})
