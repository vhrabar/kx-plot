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


def to_pandas(obj: Any) -> Any:
    """obj (native or narwhals) as pandas if it is a dataframe or series, else obj unchanged."""
    found = wrap(obj)
    if not isinstance(found, (nw.DataFrame, nw.Series)):
        return obj
    native = nw.to_native(found)
    if isinstance(native, (pd.DataFrame, pd.Series)):
        return native
    if found.implementation.is_polars():
        if isinstance(found, nw.Series):
            return _polars_series(native)
        return pd.DataFrame({c: _polars_series(native[c]) for c in native.columns}, columns=list(native.columns))
    return found.to_pandas()


def _polars_series(s: Any) -> pd.Series:
    """One polars Series as pandas with to_numpy: categories in polars' order, nulls as NaN / None, dates as
    datetime64[ns] with the time zone kept. An unnamed Series stays unnamed."""
    dtype = type(s.dtype).__name__
    name = s.name if s.name != "" else None
    if dtype in ("Enum", "Categorical"):
        values = s.to_list()
        if dtype == "Enum":
            cats = s.dtype.categories.to_list()
        else:                                   # the order polars gave the categories: by their codes
            codes = s.to_physical().to_list()
            first = {c: v for c, v in zip(codes, values, strict=True) if c is not None}
            cats = [first[c] for c in sorted(first)]
        return pd.Series(pd.Categorical(values, categories=cats), name=name)
    values = s.to_numpy()
    if dtype in ("Date", "Datetime"):
        out = pd.Series(values.astype("datetime64[ns]"), name=name)
        tz = getattr(s.dtype, "time_zone", None)
        return out.dt.tz_localize("UTC").dt.tz_convert(tz) if tz else out
    if dtype == "Duration":
        return pd.Series(values.astype("timedelta64[ns]"), name=name)
    return pd.Series(np.asarray(values), name=name)
