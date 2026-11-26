"""Demo data that ships in data/."""
from __future__ import annotations

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd

from . import _files
from .typing import DataFrame


def datasets() -> list[str]:
    """List the names available to kx.load(): tables (CSV) and array sets (NPZ, such as images)."""
    return sorted(_files.names("data", ".csv") + _files.names("data", ".npz"))


def load(name: str, backend: str = "pandas") -> pd.DataFrame | DataFrame | dict[str, np.ndarray]:
    """Read data/<name>.csv as a DataFrame, or data/<name>.npz as a dict of arrays:
    images = kx.load('shapes')['images']. backend= reads a table into another dataframe library that kx takes,
    such as 'polars' or 'pyarrow' (installed separately), with the gaps as nulls; a dict of arrays stays one."""
    if name not in datasets():
        raise ValueError(f"unknown dataset {name!r}. Available: {datasets()}")
    if name in _files.names("data", ".npz"):
        with np.load(_files.path("data", f"{name}.npz"), allow_pickle=False) as arrays:
            return dict(arrays)
    df = pd.read_csv(_files.path("data", f"{name}.csv"))
    if backend == "pandas":
        return df
    columns = {c: df[c].astype(object).where(df[c].notna(), None).tolist() if df[c].isna().any()
               else df[c].to_numpy() for c in df.columns}
    try:
        return nw.to_native(nw.from_dict(columns, backend=backend))
    except ModuleNotFoundError as err:
        raise ImportError(f"backend={backend!r} needs {backend}: pip install {backend}") from err
