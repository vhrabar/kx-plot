"""
The input types of kx's functions, as the signatures and the docs name them.
"""
from narwhals.typing import IntoDataFrame, IntoSeries
from numpy.typing import ArrayLike

__all__ = ["Column", "DataFrame", "Series"]

DataFrame = IntoDataFrame
"""A dataframe of any library narwhals supports: pandas, polars, pyarrow, modin, cuDF, ..."""

Series = IntoSeries
"""One column of any of those libraries: a pandas or polars ``Series``, a pyarrow ``ChunkedArray``, ..."""

Column = Series | ArrayLike
"""One column of data: a ``Series``, a numpy array or a list."""
