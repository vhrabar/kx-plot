"""
Dataframe input through narwhals: kx reads pandas, polars, pyarrow and the other narwhals backends the same way.
"""
from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd


def wrap(obj: Any) -> Any:
    """obj as a narwhals DataFrame or Series if it is a dataframe or series of any backend, else obj unchanged.
    A lazy frame raises: kx draws data, it does not run queries."""
    found = nw.from_native(obj, pass_through=True, allow_series=True)
    if isinstance(found, nw.LazyFrame):
        raise TypeError("kx draws data, not queries: call .collect() on the lazy frame first")
    return found


def column(v: Any) -> nw.Series:
    """v as a narwhals Series: a Series of any backend as it is, anything else 1-D (an array, a list) as a pandas
    Series that kx builds for itself, so a helper reads every input the same way."""
    found = wrap(v)
    if isinstance(found, nw.Series):
        return found
    return nw.from_native(pd.Series(np.asarray(v).ravel()), series_only=True)


def name(v: Any) -> Any:
    """The name of a Series (narwhals or pandas), None for an array or an unnamed Series. polars names an
    unnamed Series ''."""
    found = getattr(v, "name", None)
    if found == "" and isinstance(v, nw.Series) and not v.implementation.is_pandas_like():
        return None
    return found
