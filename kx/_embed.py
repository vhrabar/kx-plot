"""kx.embed: project a table or images to 2-D (PCA, t-SNE or UMAP) and plot it with the embed kind."""
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from ._image import images
from ._plot import plot

METHODS = ("pca", "tsne", "umap")
NEEDS = {"tsne": ("sklearn.manifold", "TSNE", "scikit-learn"), "umap": ("umap", "UMAP", "umap-learn")}


def embed(data: Any, method: str = "pca", by: ArrayLike | str | None = None, spec: str = "", seed: int = 0,
          ax: Axes | None = None, title: str | None = None, **kw: Any) -> Axes:
    """kx.embed(df, by='label') or kx.embed(images, labels): the rows in 2-D, coloured by label.
    data: a DataFrame (its numeric columns, standardised; by= may name a column, left out of them), a 2-D
    array of features, or images (an (N, H, W[, C]) array or a list, flattened; all one size).
    method: 'pca' (numpy, the explained variance in the axis labels), 'tsne' (scikit-learn) or 'umap'
    (umap-learn): pip install 'kx-plot[embed]'. seed= fixes t-SNE and UMAP; other keywords go to them.
    spec= adds a theme and flags (kx.embed(df, by='label', spec='dark-ann')). Returns the axes; the coordinates
    are ax.embedding."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {list(METHODS)}")
    if kw and method == "pca":
        raise TypeError(f"method='pca' takes no options, got {sorted(kw)}")
    features, by = _features(data, by)
    coords, names = _project(features, method, seed, kw)
    axes = plot(pd.Series(coords[:, 0], name=names[0]), pd.Series(coords[:, 1], name=names[1]),
                spec="-".join(["embed", *([spec] if spec else [])]), by=by, ax=ax, title=title)
    axes.embedding = coords
    return axes


def _features(data: Any, by: ArrayLike | str | None) -> tuple[np.ndarray, ArrayLike | None]:
    """The rows as a float matrix to project, and by= as labels (a column name taken out of a table)."""
    if isinstance(data, pd.DataFrame):
        if isinstance(by, str):
            if by not in data.columns:
                raise ValueError(f"no column {by!r} in the DataFrame. Columns: {list(data.columns)}")
            data, by = data.drop(columns=by), data[by]
        cols = [c for c in data.columns
                if pd.api.types.is_numeric_dtype(data[c]) and not pd.api.types.is_bool_dtype(data[c])]
        if not cols:
            raise ValueError("kx.embed needs numeric columns to project; this table has none")
        values = data[cols].to_numpy(dtype=float)
        spread = values.std(axis=0)
        values = (values - values.mean(axis=0)) / np.where(spread > 0, spread, 1)   # one unit per column
    elif isinstance(by, str):
        raise TypeError(f"by={by!r} names a column, which needs a DataFrame: kx.embed(df, by={by!r})")
    elif isinstance(data, np.ndarray) and data.dtype != object and data.ndim == 2:
        values = data.astype(float)
    else:
        found = images(data)
        if len({im.shape for im in found}) > 1:
            raise ValueError("kx.embed compares images pixel by pixel; they need one size and one channel count")
        values = np.stack([im.ravel() for im in found])
    if not np.isfinite(values).all():
        raise ValueError("kx.embed needs every value; fill or drop the NaNs first (df.dropna())")
    if len(values) < 3:
        raise ValueError(f"kx.embed needs at least 3 rows, got {len(values)}")
    if by is not None and np.size(by) != len(values):
        raise ValueError(f"there are {len(values)} rows but {np.size(by)} labels")
    return values, by


def _project(values: np.ndarray, method: str, seed: int, kw: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """2-D coordinates and the axis names."""
    if method == "pca":
        if values.shape[1] < 2:
            raise ValueError("PCA to 2-D needs at least 2 features")
        centred = values - values.mean(axis=0)
        u, s, vt = np.linalg.svd(centred, full_matrices=False)
        big = np.argmax(np.abs(vt[:2]), axis=1)
        signs = np.sign(vt[[0, 1], big])
        coords = u[:, :2] * s[:2] * signs
        share = s ** 2 / (s ** 2).sum()
        return coords, [f"PC{k + 1} ({share[k]:.0%})" for k in range(2)]
    module, cls, package = NEEDS[method]
    try:
        model = getattr(__import__(module, fromlist=[cls]), cls)
    except ImportError as err:
        raise ImportError(f"method={method!r} needs {package}: pip install 'kx-plot[embed]'") from err
    found = model(n_components=2, random_state=seed, **kw).fit_transform(values)
    name = "t-SNE" if method == "tsne" else "UMAP"
    return np.asarray(found, dtype=float), [f"{name} 1", f"{name} 2"]
