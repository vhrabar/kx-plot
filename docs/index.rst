kx-plot
=======

**One-line matplotlib plots and themes for notebooks and scripts.**

``kx`` is a small, plain-Python package with ``.mplstyle`` themes, palettes and colourmaps. It needs nothing beyond
matplotlib, numpy and pandas and works offline, so it also runs in Kaggle competition notebooks.

.. code-block:: python

   import kx; kx.use("light")

   kx.plot(x, y, z, "scatter-grid-leg")         # spec: kind-theme-flag-...
   kx.plot(df, "month", "y2025", "bar-paper")   # DataFrame columns by name, a theme for this plot only
   kx.plot(df, "value", spec="kde", by="group") # one density curve per group
   kx.save("fig.png")

.. image:: _static/kinds/kde.png
   :width: 32%
.. image:: _static/kinds/roc.png
   :width: 32%
.. image:: _static/kinds/heat.png
   :width: 32%

.. toctree::
   :maxdepth: 2
   :caption: User guide

   install
   spec

.. toctree::
   :maxdepth: 2
   :caption: Kinds

   kinds/basic
   kinds/dist
   kinds/table
   kinds/ml
   kinds/image

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   changelog

.. toctree::
   :caption: Links

   Gallery <https://vhrabar.github.io/kx-plot>
   GitHub <https://github.com/vhrabar/kx-plot>
   PyPI <https://pypi.org/project/kx-plot/>
   Example notebook (Kaggle) <https://www.kaggle.com/code/vhrabar/kx-one-line-matplotlib-plots-and-themes>
