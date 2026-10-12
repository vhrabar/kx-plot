# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Plot kind** `ridge`: density curves stacked top to bottom, one per array or `by=` group, on one shared height
  scale; `overlap=` sets the tallest ridge's height in rows. `mean` and `median` mark each ridge.
- **Plot kind** `strip`: every value as a point, one column per array or `by=` group, jittered sideways so ties
  stay visible (the same way every run; `jitter=` sets the spread). Takes `mean` and `ann` like `box`.
- `kx.grid(n, cols=3)` makes a figure of `n` panels and returns them as a flat array of axes. Each panel gets
  the theme's figure size, empty cells are removed, and constrained layout leaves room for colorbars and outside
  legends. Other keywords go to `plt.subplots`.
- `kx.save('fig.png')` saves the current figure (or the one of a given `ax` or figure) and returns the path.
  A relative name goes to `/kaggle/working` on Kaggle and to the current folder elsewhere; missing folders are
  created. Without an extension the theme's format is used, so `thesis` saves a PDF.
- `with kx.context(theme, palette=..., cmap=..., **rc):` takes the same arguments as `kx.use` and puts the
  previous settings back when the block ends, also after an error.
- `kx.parse(spec)` returns a `Spec` with `kind`, `theme` and `flags`. Printed, it is the canonical spec string
  (kind, theme, then flags A–Z), and `kx.plot` takes a `Spec` wherever it takes a spec string. Mistyped tokens get
  the closest valid ones as a hint.
- `kx.plot(df, 'x', 'y', 'z', spec)` names DataFrame columns, and `by=` can name a column. A mistyped column
  gets the closest names as a hint.
- `kx.plot(..., **kwargs)` passes other keywords to the matplotlib call that draws the kind, over kx's defaults:
  `kx.plot(x, y, 'line', linewidth=3)`. `bins=` sets the shared `hist` bins and `width=` (`height=` for
  `barh`) the space each category's bars fill.
- **Flags for every kind**
  - Layout: `legout` puts the legend outside on the right, keeping a kind's legend title. `rot` rotates the x tick
    labels 45°.
  - Axes: `logxy` (log on both), `symlogy` (symmetric log y, so negatives work), `zero` (the value axis includes 0)
    and `eq` (equal aspect).
  - Number format: `pct` (0.25 → 25%) and `si` (1500 → 1.5k) on the value axis, which is x for `barh` and y
    otherwise.
- **Flags for some kinds**
  - `date` for `line`, `step`, `area`, `band` and `scatter`: `x` is read as dates (text works too) and gets compact
    date ticks.
  - `mk` for `line`, `step`, `area` and `band`: a marker on every point.
  - `sort` for `bar` and `barh`: largest first, by `y`, or by the total with `stack`.
  - `ann` for `bar` and `barh`: the value on each bar, inside each segment with `stack`.
  - `mean` and `median` for `hist`, `kde` and `ecdf`: a dashed / dotted line at each series' mean / median.
    `mean` for `box` and `violin`: a diamond at each mean. `ann` writes the values.
  - `nocb`, `sym` and `logc` for `scatter` with a numeric `z`: no colorbar, colour limits symmetric around 0,
    log colour scale.
- Flags that ask for opposite things (`pct` and `si`, `logy` and `logxy`, `zero` and a log y, `sym` and `logc`)
  raise an error.

### Changed

- The `tight` flag leaves a figure with constrained layout (such as `kx.grid`'s) alone instead of failing
  when it has a colorbar.

## [0.2.0] - 2026-10-09

### Added

- **Plot kinds**
  - `barh`: horizontal bars, first category on top. `y` and `z` are grouped like `bar`.
  - `kde`: smooth density curves (Gaussian KDE in numpy, no scipy needed), one per array.
  - `ecdf`: empirical CDF, the share of values at or below each x, one step curve per array.
  - `box`: box plots side by side (median, quartiles, 1.5 IQR whiskers, outliers), one per array.
  - `violin`: violin plots side by side with the median and quartiles marked, one per array.
- **Flags**
  - `stack` now works with `bar` and `barh`: `z` is drawn on top of `y`, so the bar's end is the total.
  - `stack`, `norm` (density) and the new `cum` (cumulative) work with `hist`.
- `kx.plot(..., by=groups)` splits `x` into one curve, box or violin per group (for `kde`, `ecdf`, `box`, `violin`).
- cmocean colourmaps `thermal`, `haline`, `deep` and `balance` for `kx.use(cmap=...)`.
- Crameri colourmaps (Scientific colour maps 8) `batlow`, `lipari`, `hawaii`, `oslo`, `davos`, `vik`, `roma`,
  `bam`, `cork` and cyclic `romaO`, vendored the same way.
- `kx.use(cmap="<name>_r")` reverses any colourmap in `kx.cmaps()`.
- Palettes `tollight` (Paul Tol), `set1` and `paired` (ColorBrewer) and `tab20` (Tableau, 20 categories).

### Changed

- `hist` puts every series on the same bins, so overlaid bars line up.

## [0.1.0] - 2026-10-08

### Added

- **PyPI package:** `pip install kx-plot`, then `import kx`. The wheel ships the themes, palettes and demo CSVs.
- **Palettes:** Paul Tol (`tolbright`, `tolvibrant`, `tolhc`, `tolmc`), Petroff (`petroff6`, `petroff8`, `petroff10`),
  Tableau (`tab10`, `tableaucb`), ColorBrewer (`set2`, `dark2`) and `ibm`. Their licences are in
  `kx/styles/palettes/LICENCES.md`.

### Changed

- **`kx.py` is now the `kx/` package**, with the themes in `kx/styles/` and the demo CSVs in `kx/data/`.
  `kx.source()` prints every file in the package.

## [0.0.2] - 2026-10-08

### Added

- **Plot kinds**
  - `step`: like `line`, drawn as steps centred on each x.
  - `area`: filled areas, overlapping and semi-transparent by default.
    - `area-stack` stacks `z` on top of `y`, so the top edge is the total.
    - `area-norm` draws a 100% stack with a percent y axis. With `leg`, the legend goes outside on the right.
  - `band`: a `y` line with a shaded `y ± z` band in the same colour. `z` is required.
  - `scatter` with a text, bool or pandas categorical `z` draws one colour per category with a legend titled after `z`. A numeric `z` still gets a colorbar.
- **Flags**
  - `stack` and `norm`. Both work with `area` only for now.
- **Themes**
  - `science`: one journal column (3.5 in), 8 pt serif, inward ticks on all four sides, minor ticks, Paul Tol bright colours, saves at 600 dpi.
  - `talk`: slides (16:9), 18 pt sans, thick lines, Okabe-Ito colours.
  - `poster`: 12 × 9 in, 26 pt sans, very thick lines, saves at 300 dpi.
  - `thesis`: A4 text width (160 mm), Computer Modern 10/9 pt without needing LaTeX, saves as PDF when the file name has no extension.
- **Palettes**
  - Named palettes live in `styles/palettes/<name>.txt`, one hex colour per line. Any file added there is found automatically.
  - `okabe`: the Okabe-Ito palette, 8 colours that stay distinguishable for every common type of colour blindness.
  - `kx.palettes()` lists the palettes and `kx.palette(name)` returns a palette's colours.
  - `kx.use(theme, palette="okabe")` accepts a palette name as well as a list of colours.
- **Colormaps**
  - `kx.use(theme, cmap=...)` sets the colormap for continuous colour.
  - The supported matplotlib colormaps are `viridis`, `magma`, `plasma`, `inferno`, `cividis`, `turbo`, `twilight`, `coolwarm`, `berlin`, `managua` and `vanimo`.
  - `kx.cmaps()` lists only the ones the installed matplotlib has. `berlin`, `managua` and `vanimo` need matplotlib 3.10 or newer.
- **Docs**
  - `Example Usage.ipynb`: a detailed walkthrough for the public Kaggle notebook. It loads `kx` from the attached dataset.
  - `kx.grammar()` lists the palettes and colormaps as well as the kinds, themes and flags.

### Changed

- **`paper` theme, reworked for greyscale print:**
  - black, dark grey and mid grey colours, each paired with a linestyle (solid, dashed, dotted, dash-dot)
  - greyscale colormap
  - 300 dpi export with a white background
  - serif font for maths text as well
  - 110 dpi in notebooks instead of 150
- **Palettes keep the theme's other cycled properties:** they replace only the colours, so `paper` + `okabe` keeps `paper`'s linestyles.
- **`scatter` point size follows the theme's `lines.markersize`.** Points are bigger on `poster` and smaller on `science`. `dark` and `light` look the same as before.
- **Preview images:** each theme's preview is drawn at twice that theme's own figure size instead of a fixed 12×6 in, so `talk` and `poster` previews no longer overlap.
- **`previews/demo.ipynb`:** cut down to a short demo of each kind and theme. The full walkthrough is now `Example Usage.ipynb`.
- **Strict errors** also cover:
  - `stack` or `norm` on a kind that doesn't support them
  - `band` without `z`
  - `area-norm` with only one series

## [0.0.1] - 2026-10-07

### Added

- `kx.plot` with the kinds `line`, `scatter`, `bar` and `hist`
- the flags `grid`, `leg`, `logx`, `logy` and `tight`
- the themes `dark`, `light` and `paper`
- `kx.use`, `kx.show`, `kx.source`, `kx.grammar`, `kx.load` and the demo CSVs
- a workflow that publishes the dataset to Kaggle

[Unreleased]: https://github.com/vhrabar/kx-plot/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/vhrabar/kx-plot/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/vhrabar/kx-plot/compare/v0.0.2...v0.1.0
[0.0.2]: https://github.com/vhrabar/kx-plot/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/vhrabar/kx-plot/releases/tag/v0.0.1
