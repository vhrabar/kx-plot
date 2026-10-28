"""Demo data that ships in data/."""
import numpy as np
import pandas as pd

from . import _files


def datasets() -> list[str]:
    """List the names available to kx.load(): tables (CSV) and array sets (NPZ, such as images)."""
    return sorted(_files.names("data", ".csv") + _files.names("data", ".npz"))


def load(name: str) -> pd.DataFrame | dict[str, np.ndarray]:
    """Read data/<name>.csv as a DataFrame, or data/<name>.npz as a dict of arrays:
    images = kx.load('shapes')['images']."""
    if name not in datasets():
        raise ValueError(f"unknown dataset {name!r}. Available: {datasets()}")
    if name in _files.names("data", ".npz"):
        with np.load(_files.path("data", f"{name}.npz"), allow_pickle=False) as arrays:
            return dict(arrays)
    return pd.read_csv(_files.path("data", f"{name}.csv"))
