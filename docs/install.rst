Installation
============

.. code-block:: bash

   pip install kx-plot

   # adds t-SNE and UMAP to kx.embed (PCA needs neither)
   pip install 'kx-plot[embed]'

Python 3.10 or newer. kx needs only matplotlib, numpy and pandas; the optional ``[embed]`` extra adds
scikit-learn and umap-learn for ``method='tsne'`` and ``method='umap'`` in ``kx.embed``.

uv
--

In a project managed by `uv <https://docs.astral.sh/uv/>`_, add it to ``pyproject.toml``:

.. code-block:: bash

   uv add kx-plot
   uv add 'kx-plot[embed]'

For a one-off script, without adding it to a project:

.. code-block:: bash

   uv run --with kx-plot script.py

Kaggle
------

In a competition notebook without internet, attach the dataset instead
(**Add Input → Datasets → vhrabar/kx-plot**), then:

.. code-block:: python

   import sys; sys.path.append("/kaggle/input/datasets/vhrabar/kx-plot")
   import kx; kx.use("dark")

``kx.save`` writes to ``/kaggle/working`` on Kaggle and to the current folder elsewhere.
