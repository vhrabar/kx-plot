"""
Write kx/data/shapes.npz: a small synthetic image-classification set for the image kinds, with masks,
a model's predictions and saliency maps.
"""
import os

import numpy as np

SEED = 11
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kx", "data")
SIZE = 40                                       # pixels per side
CAM = 10                                        # saliency map pixels per side, coarse as a Grad-CAM is
CLASSES = ("blob", "ring", "square", "cross", "stripes", "dots", "crescent")
COUNTS = (60, 40, 30, 25, 20, 20, 15)           # imbalanced, as real classes are
HUES = ((0.45, 0.25, 0.15), (0.30, 0.20, 0.35), (0.20, 0.30, 0.25), (0.50, 0.35, 0.10),
        (0.25, 0.15, 0.10), (0.35, 0.30, 0.30), (0.40, 0.15, 0.25))   # each class's mean shape colour
CONFUSED = {"blob": "crescent", "crescent": "blob", "ring": "dots", "dots": "ring", "square": "cross",
            "cross": "square", "stripes": "square"}                   # the class a model mistakes it for


def _shape(name: str, rng: np.random.Generator) -> tuple[np.ndarray, tuple[float, float]]:
    """A 0..1 coverage map of one shape of class `name`, and its centre (row, column)."""
    cy, cx = rng.uniform(13, SIZE - 13, 2)
    r = rng.uniform(7, 11)
    yy, xx = np.mgrid[:SIZE, :SIZE] + 0.5
    dy, dx = yy - cy, xx - cx
    a = rng.uniform(0, np.pi)
    u, v = dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a)    # rotated coordinates
    dist = np.hypot(dx, dy)
    if name == "blob":
        cover = dist < r * (1 + 0.15 * np.sin(3 * np.arctan2(dy, dx) + a))
    elif name == "ring":
        cover = (dist < r) & (dist > 0.6 * r)
    elif name == "square":
        cover = (np.abs(u) < 0.8 * r) & (np.abs(v) < 0.8 * r)
    elif name == "cross":
        cover = ((np.abs(u) < 0.3 * r) | (np.abs(v) < 0.3 * r)) & (np.abs(u) < r) & (np.abs(v) < r)
    elif name == "stripes":
        cover = (dist < r) & (np.sin(u * 1.4) > 0)
    elif name == "dots":
        cover = np.zeros((SIZE, SIZE), dtype=bool)
        for t in np.linspace(0, 2 * np.pi, 6, endpoint=False) + a:
            cover |= np.hypot(dx - 0.65 * r * np.cos(t), dy - 0.65 * r * np.sin(t)) < 0.25 * r
    else:                                       # crescent
        cover = (dist < r) & (np.hypot(dx - 0.5 * r * np.cos(a), dy - 0.5 * r * np.sin(a)) > 0.85 * r)
    return cover.astype(float), (cy, cx)


def _artifact(img: np.ndarray, kind: str, rng: np.random.Generator) -> None:
    """Draw a dark border or a thin dark line (a hair) on img in place."""
    if kind == "border":
        yy, xx = np.mgrid[:SIZE, :SIZE] + 0.5
        edge = np.hypot(yy - SIZE / 2, xx - SIZE / 2) / (SIZE / 2)
        img *= np.clip(1.6 - edge, 0.05, 1)[..., None]
    elif kind == "line":
        t = np.linspace(0, 1, 200)
        y0, y1 = rng.uniform(0, SIZE, 2)
        ys = y0 + (y1 - y0) * t + 3 * np.sin(t * rng.uniform(2, 6))
        for y, x in zip(ys, t * SIZE, strict=True):
            iy, ix = int(y), int(x)
            if 0 <= iy < SIZE and 0 <= ix < SIZE:
                img[iy, ix] *= 0.25


def _saliency(centre: tuple[float, float], rng: np.random.Generator, off: bool) -> np.ndarray:
    """A coarse CAM x CAM map peaking at the shape, or (off) somewhere else, scaled to 0..1."""
    cy, cx = rng.uniform(0, SIZE, 2) if off else centre
    yy, xx = (np.mgrid[:CAM, :CAM] + 0.5) * SIZE / CAM
    m = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rng.uniform(5, 9) ** 2)) + rng.uniform(0, 0.15, (CAM, CAM))
    return ((m - m.min()) / (m.max() - m.min())).astype(np.float32)


def shapes(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """images (n, 40, 40, 3) uint8, masks (n, 40, 40) bool, labels, pred, proba (n, 7), saliency (n, 10, 10),
    artifact ('', 'border' or 'line') and the class names, in a shuffled order."""
    names = np.repeat(CLASSES, COUNTS)
    rng.shuffle(names)
    n = len(names)
    images = np.empty((n, SIZE, SIZE, 3), dtype=np.uint8)
    masks = np.empty((n, SIZE, SIZE), dtype=bool)
    artifact = rng.choice(["", "border", "line"], n, p=[0.7, 0.15, 0.15])
    centres = []
    for i, name in enumerate(names):
        cover, centre = _shape(name, rng)
        skin = rng.uniform([0.80, 0.62, 0.52], [0.95, 0.78, 0.68])      # a light background, varying
        ink = np.clip(np.array(HUES[CLASSES.index(name)]) + rng.normal(0, 0.05, 3), 0, 1)
        img = skin * (1 - cover[..., None]) + ink * cover[..., None] + rng.normal(0, 0.02, (SIZE, SIZE, 3))
        _artifact(img, artifact[i], rng)
        images[i] = np.round(np.clip(img, 0, 1) * 255)
        masks[i] = cover > 0.5
        centres.append(centre)
    true = np.array([CLASSES.index(c) for c in names])
    logits = rng.normal(0, 1, (n, len(CLASSES)))
    logits[np.arange(n), true] += 2.6
    near = np.array([CLASSES.index(CONFUSED[c]) for c in names])
    logits[np.arange(n), near] += 1.2 + 0.8 * (artifact != "")       # artifacts make the mistake likelier
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    pred = proba.argmax(axis=1)
    saliency = np.stack([_saliency(c, rng, off=p != t) for c, p, t in zip(centres, pred, true, strict=True)])
    return {"images": images, "masks": masks, "labels": names.astype("U8"), "pred": np.array(CLASSES)[pred],
            "proba": proba.astype(np.float32), "saliency": saliency, "artifact": artifact.astype("U6"),
            "classes": np.array(CLASSES)}


if __name__ == "__main__":
    out = os.path.join(DATA, "shapes.npz")
    np.savez_compressed(out, **shapes(np.random.default_rng(SEED)))
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")
