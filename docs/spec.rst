The spec
========

Every plot is one call, ``kx.plot(data..., spec)``. The spec is a string of tokens joined by ``-``:

.. code-block:: text

   kind-theme-flag-flag-...

* exactly one kind: what to draw (see the *Kinds* pages),
* at most one theme: applies to this plot only; ``kx.use`` sets the default,
* any number of flags: the common ones below work with every kind, the rest belong to a kind.

Token order does not matter: ``kx.parse("leg-scatter-dark")`` and ``kx.parse("scatter-dark-leg")`` are the same
plot, and ``str(kx.parse(spec))`` gives the canonical form.

.. code-block:: python

   kx.plot(x, y, z, "scatter-grid-leg")
   kx.plot(x, y, "bar-paper")
   kx.plot(x, y, "line", linewidth=3)     # other keywords go to matplotlib
   kx.grammar("hist")                     # what one kind does and which flags it takes

Themes
------

.. include:: kinds/_themes.rst

``kx.use`` sets the theme for the session, ``kx.context`` for one block, and both take a palette and a colormap:

.. code-block:: python

   kx.use("light", palette="okabe", cmap="cividis")
   with kx.context("paper"):
       ...

**Palettes:**

.. include:: kinds/_palettes.rst

**Colormaps** (add ``_r`` to reverse one):

.. include:: kinds/_cmaps.rst

``kx.register(name, colors)`` adds a palette (``cmap=True``: a colormap) of your own for the session.

.. _common-flags:

Common flags
------------

These work with every kind; each kind's page lists the flags of its own.

.. list-table::
   :header-rows: 1
   :widths: 1 4

   * - Flag
     - Effect
   * - ``grid``
     - grid lines
   * - ``leg`` / ``legout``
     - legend inside / outside the axes
   * - ``tight``
     - tight layout
   * - ``rot`` / ``wrap``
     - rotate / wrap long tick labels
   * - ``logx`` ``logy`` ``logxy`` ``symlogy``
     - log scales
   * - ``zero``
     - stretch the value axis to include 0
   * - ``eq``
     - equal aspect
   * - ``pct`` / ``si``
     - percent / SI-prefix tick labels
   * - ``facet``
     - one panel per value of ``facet=``

Some flags ask for opposite things. Use at most one from each group, or kx.plot raises an error:

.. include:: kinds/_exclusive.rst
