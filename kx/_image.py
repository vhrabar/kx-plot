"""Images as input: arrays, lists of arrays or file paths, turned into a list of 0..1 float images."""
import math
import os
from collections import Counter
from typing import Any

import matplotlib as mpl
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


def count(x: Any) -> int:
    """How many images x holds, without reading any: one 2-D array or path is one."""
    if isinstance(x, (str, os.PathLike)):
        return 1
    if isinstance(x, np.ndarray) and x.dtype != object:
        return 1 if x.ndim == 2 else len(x)
    return len(x)


def take(x: Any, idx: list[int]) -> list[np.ndarray]:
    """The images at positions idx, read and scaled like images(); the rest are never read."""
    n = count(x)
    bad = [i for i in idx if not -n <= i < n]
    if bad:
        raise ValueError(f"idx= {bad} is out of range for {n} images")
    if isinstance(x, (str, os.PathLike)) or (isinstance(x, np.ndarray) and x.dtype != object and x.ndim == 2):
        return images(x)
    if isinstance(x, np.ndarray) and x.dtype != object:
        return images(x[np.asarray(idx, dtype=int)])
    return images([x[i] for i in idx])


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


def resize(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """image at shape (H, W) by nearest neighbour: each new pixel takes the old one it falls in. A selection, so
    the resize of a sum is the sum of the resizes."""
    h, w = image.shape[:2]
    rows = (np.arange(shape[0]) * h // shape[0]).astype(int)
    cols = (np.arange(shape[1]) * w // shape[1]).astype(int)
    return image[rows][:, cols]


def as_rgb(image: np.ndarray) -> np.ndarray:
    """A grey (H, W) image as (H, W, 3); colour stays as it is."""
    return np.repeat(image[..., None], 3, axis=2) if image.ndim == 2 else image


CHUNK = 256


def moments(x: Any, groups: list[np.ndarray]) -> tuple[list[tuple[np.ndarray, np.ndarray, int]], tuple[int, int]]:
    """Per group of image positions: the mean image, the mean of squares, and the count, at the size most images
    have (nearest neighbour), grey made colour if any image is colour. Reads each image once, CHUNK at a time:
    sums are kept per size and resized at the end, which equals resizing every image first."""
    sums: list[dict[tuple[int, ...], list]] = [{} for _ in groups]
    sizes: Counter[tuple[int, int]] = Counter()
    for g, idx in enumerate(groups):
        for start in range(0, len(idx), CHUNK):
            chunk = [int(i) for i in idx[start:start + CHUNK]]
            for img in take(x, chunk):
                sizes[img.shape[:2]] += 1
                acc = sums[g].setdefault(img.shape, [np.zeros(img.shape), np.zeros(img.shape), 0])
                acc[0] += img
                acc[1] += img ** 2
                acc[2] += 1
    if not sizes:
        raise ValueError("no images to average")
    shape = sizes.most_common(1)[0][0]
    rgb = any(len(k) == 3 for group in sums for k in group)
    out = []
    for group in sums:
        total, squares, n = 0.0, 0.0, 0
        for s, q, k in group.values():
            s, q = (as_rgb(v) if rgb else v for v in (s, q))
            total, squares, n = total + resize(s, shape), squares + resize(q, shape), n + k
        out.append((total / n, squares / n, n))
    return out, shape


RGB_NAMES, HSV_NAMES = ("red", "green", "blue"), ("hue", "saturation", "value")


def histograms(x: Any, groups: list[np.ndarray], bins: int, hsv: bool) -> tuple[np.ndarray, list[str]]:
    """Per group, each channel's histogram over 0..1 as shares of the pixels, averaged over the group's images
    (so a big image weighs as much as a small one): an array (groups, channels, bins) and the channel names.
    All grey: one 'grey' channel; grey among colour counts as R = G = B. hsv: hue, saturation, value."""
    colour = np.zeros((len(groups), 3, bins))
    grey = np.zeros((len(groups), 1, bins))
    seen = np.zeros(len(groups), dtype=int)
    any_colour = False

    def hist(channel: np.ndarray) -> np.ndarray:
        return np.histogram(channel, bins=bins, range=(0, 1))[0] / channel.size
    for g, idx in enumerate(groups):
        for start in range(0, len(idx), CHUNK):
            for img in take(x, [int(i) for i in idx[start:start + CHUNK]]):
                seen[g] += 1
                if img.ndim == 2 and not hsv:
                    grey[g, 0] += hist(img)
                    continue
                any_colour = True
                img = mpl.colors.rgb_to_hsv(np.clip(as_rgb(img), 0, 1)) if hsv else img
                colour[g] += [hist(img[..., c]) for c in range(3)]
    if not seen.all():
        raise ValueError("a class without images has no histogram")
    if not any_colour:
        return grey / seen[:, None, None], ["grey"]
    return (colour + grey) / seen[:, None, None], list(HSV_NAMES if hsv else RGB_NAMES)


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
        bare(a)
    return axs[:n]


def bare(ax: Axes) -> None:
    """No ticks or frame, but the y label stays for a class name."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


def fit_figure(ax: Axes, rows: int, cols: int, side: float = 1.4) -> None:
    """A figure of its own sized to the image grid (`side` inches an image) and laid out constrained."""
    fig = ax.figure
    if len(fig.axes) != 1:
        return                                  # a cell of a bigger figure keeps that figure
    fig.set_size_inches(max(2.5, cols * side + 0.6), max(2.0, rows * side + 0.4))
    if fig.get_layout_engine() is None:
        fig.set_layout_engine("constrained")


def show(ax: Axes, image: np.ndarray, **kw: Any) -> AxesImage:
    """image (from images()) on ax: grey in the grey colormap, on the full 0..1 scale, pixels kept square."""
    grey = {"cmap": "gray"} if image.ndim == 2 else {}
    return ax.imshow(image, **{"vmin": 0, "vmax": 1, "interpolation": "nearest", **grey, **kw})
