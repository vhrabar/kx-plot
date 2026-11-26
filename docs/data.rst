Data input
==========

kx reads dataframes through `narwhals <https://narwhals-dev.github.io/narwhals/>`_, so a frame from any library
narwhals supports works where the docs say ``DataFrame``: pandas, polars, pyarrow, modin, cuDF and others. pandas
and polars are tested; the others should work and are worth an issue if they do not.

.. code-block:: python

   import polars as pl
   import kx

   df = pl.read_csv("passengers.csv")          # or kx.load("passengers", backend="polars")
   kx.plot(df, "age", "fare", "sex", spec="scatter")
   kx.plot(df, "fare", spec="box", by="pclass")
   kx.plot(df.select(pl.selectors.numeric()), "heat-corr")

Columns are named as for pandas, and ``by=`` and ``facet=`` can name a column or take a Series of their own.
polars ``Enum`` and ``Categorical`` columns keep their category order, as a pandas categorical does, and nulls
count as missing values.

What is not accepted
--------------------

- **Lazy frames** (a polars ``LazyFrame``, a DuckDB relation): kx draws data, it does not run queries. Call
  ``.collect()`` first, after selecting the columns you need.
- **Duplicate column names**: a plot names its columns, so each name must be unique.

Demo data in another library
----------------------------

``kx.load(name, backend="polars")`` (or ``"pyarrow"``) returns a demo table in that library, with the gaps as
nulls. The library is installed separately; kx itself never imports polars or pyarrow.

Types in the signatures
-----------------------

.. automodule:: kx.typing
   :members: DataFrame, Series, Column
