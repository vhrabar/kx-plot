# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `kx.roc_auc(y_true, score)` and `kx.pr_ap(y_true, score)`: the area under the ROC curve (by trapezoids) and
  the average precision (the step sum `Σ (Rₙ − Rₙ₋₁) Pₙ`), in numpy with no scikit-learn needed, and tied scores
  counted as one step as scikit-learn does. Labels are 0/1 or bool with `1`/`True` positive; with two other labels
  the later one in sorted order is positive. More than two labels, a 2-D score or only one class raise.
- **Datasets** for the machine-learning kinds, for `kx.load`: `binary` (a 0/1 `y` scored by two models, `gbm`
  ranking better and `logreg` better calibrated), `multiclass` (a 3-class `y`, one probability column per class
  `cat`, `dog`, `bird` summing to 1, and `pred`), `regression` (`actual`, `predicted` and an `x` the error grows
  with), `history` (`loss`, `val_loss`, `accuracy`, `val_accuracy` per `epoch`, overfitting after about 30 epochs),
  `importance` (`feature`, `importance`, `std`) and `curves` (learning-curve scores per training `size`,
  `train_1..5` and `val_1..5`). `scripts/make_ml.py` writes them with a fixed seed.
- **Flag** `diag` (`scatter`, `hexbin`): a dashed `y = x` line across the range `x` and `y` share, to compare
  predicted with actual values or two measurements of the same thing.
- **Flag** `fit` (`scatter`): a least-squares line with its slope and R² in the legend, which it draws. With a
  categorical `z`, one line per group in the group's colour. With a log axis it raises, because a straight fit
  is a curve there; fit the logs instead.
- **Plot kind** `cm`: a confusion matrix, `kx.plot(y_true, y_pred, 'cm')`, with the true classes down the rows
  and the predicted ones along the columns, as in scikit-learn. Every label of either array gets a row and a
  column, so a class that is never predicted still shows. `norm` divides each row by its total (the recall of
  each class) with a percent colorbar, `ann` writes each cell's value and `nocb` drops the colorbar. A 2-D
  `y_pred` raises with a hint to pass `proba.argmax(axis=1)`.
- **Plot kind** `roc`: the ROC curve, `kx.plot(y_true, score, 'roc')`, with the AUC in the legend
  (`gbm (AUC 0.80)`), which it draws without `leg`. A dotted diagonal marks a random model, and both axes run
  from 0 to 1 at equal scale. `z` adds a second model's score. A 2-D score, one column per class, draws one
  curve per class, one-vs-rest, named after the DataFrame's columns or else the sorted labels of `y_true`.
- **Plot kind** `pr`: the precision-recall curve, `kx.plot(y_true, score, 'pr')`, with the average precision in
  the legend (`gbm (AP 0.80)`) and a dotted baseline at the positive rate, what a random model gets. The curve
  is drawn in the steps that the average precision sums. `z` adds a second model on the same baseline; a 2-D
  score draws one curve per class, each with its own baseline in its colour.
- **Plot kind** `calib`: a reliability diagram, `kx.plot(y_true, score, 'calib')`: the observed positive rate
  against the mean score in `bins=` equal-width bins over 0..1 (10 by default), with markers sized by the rows
  in each bin, the diagonal a calibrated model lies on, and the Brier score in the legend (`gbm (Brier 0.201)`).
  Empty bins are left out. `z` adds a second model; scores outside 0..1 raise.
- **Plot kind** `proba`: the score distribution per true class, `kx.plot(y_true, score, 'proba')`: one
  histogram per class on the same `bins=` equal-width bins over 0..1 (20 by default), so their overlap shows
  where the model confuses them. `norm` draws densities, for imbalanced classes. A 2-D score shows each row's
  score for its own true class.
- **Plot kind** `thresh`: precision, recall and F1 against the decision threshold,
  `kx.plot(y_true, score, 'thresh')`, where a row is positive when its score is at or above the threshold.
- **Flag** `best` (`thresh`, `loss`, `elbow`): a dashed vertical line at the threshold with the highest F1, labelled in the legend
  (`best F1 0.74 at 0.38`).
- **Plot kinds** `gain` and `lift`: `kx.plot(y_true, score, 'gain')` draws the share of positives found against
  the share of rows taken, highest scores first, between a dotted diagonal (a random order) and a dashed corner
  at the positive rate (a perfect one). `lift` divides the gain by the random order's, so how many times more
  positives the top rows hold, with a dotted line at 1. `z` adds a second model to both.
- **Plot kind** `pva`: predicted against actual for a regression, `kx.plot(actual, predicted, 'pva')`, with the
  dashed `y = x` line, equal scales, and R², RMSE and MAE in the corner. Above 5,000 points it draws hexagons
  (`hexbin`) instead of dots, so big test sets stay readable.
- **Plot kind** `resid`: residuals (`actual - predicted`) against predicted, with a dashed line at 0. `z` colours
  the points as in `scatter`, such as by a feature, and takes `nocb`, `sym` and `logc`.
- **Plot kind** `imp`: feature importance, `kx.plot(df, 'feature', 'importance', 'std', 'imp')`, as bars lying
  down with the largest on top and `z` as error bars. `top=` keeps the largest (20 by default) and the y label
  says how many were left out.
- **Plot kind** `loss`: training curves, `kx.plot(history, 'loss')`. The first metric is a solid line and its
  `val_` pair a dashed one in the same colour, against an `epoch` column or the row number from 1. Other metrics
  have other scales, so `kx.plot(h[['accuracy', 'val_accuracy']], 'loss')` draws them. `best` marks the best
  validation epoch: the lowest, or the highest for metrics named with `acc`, `auc`, `f1`, `precision` or `recall`.
- Table kinds also take a dict of columns, or anything with a `.history` dict such as a Keras `History`, so
  `kx.plot(model.fit(...), 'loss')` works without importing Keras.
- **Plot kinds** `lcurve` and `vcurve`: cross-validated scores, `kx.plot(sizes, train_scores, val_scores,
  'lcurve')`, in the shapes scikit-learn's `learning_curve` and `validation_curve` return (one row per x, one
  column per fold). The mean is a line and ± one standard deviation a band in its colour; 1-D scores draw a line
  without a band. `vcurve` is the same with a hyperparameter on x, so `logx` is the usual flag.
- **Plot kind** `elbow`: choosing the number of clusters, `kx.plot(ks, inertia, 'elbow')`, with `z` (such as the
  silhouette score) on a second y axis on the right. `best` marks the knee: the point farthest from the line
  between the first and last points, with both axes scaled to 0..1.

## [0.4.0] - 2026-10-18

### Added

- **Plot kind** `heat`: a DataFrame or 2-D array as coloured cells, columns along x and index labels down y.
  `corr` draws the correlations of the numeric columns on `rdbu` from -1 to 1, and `ann` writes each cell's value
  in black or white, whichever reads on that cell. Takes `nocb`, `sym` and `logc` like `scatter`.
- **Plot kind** `crosstab`: how often each pair of categories occurs, as a heat map with `x` along the x axis and
  `y` down the y axis: `kx.plot(df, 'sex', 'pclass', 'crosstab-ann')`. `norm` shows shares within each `x`
  category (each column sums to 100%), and `logc` leaves pairs that never occur blank.
- **Plot kind** `count`: how often each category of `x` occurs, with missing values as their own `NaN` bar.
  `by=` splits each bar into one per group (`kx.plot(df, 'pclass', spec='count', by='survived')`). `stack` puts the
  groups on top of each other, `sort` puts the most common first and `ann` writes the counts. `norm` shows shares:
  within each category with `by=` and `stack`, and within each group otherwise.
- **Plot kind** `missing`: the share of missing values in each column of a DataFrame, as bars lying down with
  the most missing on top: `kx.plot(df, 'missing-ann')`. Columns without missing values show only with `all`,
  and a frame with none shows "no missing values" instead of raising.
- **Plot kind** `nullmap`: where the missing values are, one row of the image per row of the table and the
  columns along x. Missing cells are in the palette's first colour on the theme's background. `sort` puts the rows
  with the most missing values on top and groups rows with the same gaps, so patterns show as blocks.
- **Plot kind** `hexbin`: how many points fall in each hexagon, or with `z` the mean of `z` there, with a
  colorbar labelled `count` or with `z`'s name. Rows with a NaN are left out. Takes `nocb` and `logc` (`sym` with
  `z` only), and `gridsize=` sets the number of hexagons across.
- `kx.missing(df)`: the same as `kx.plot(df, 'missing-ann')`. A spec adds a theme and flags,
  `kx.missing(df, 'dark-wrap')`, and other keywords such as `title=` and `ax=` go to `kx.plot`.
- **Flag for every kind** `wrap`: long text tick labels break into lines of at most 12 characters at spaces,
  on both axes. Numbers, dates and single long words are left as they are. Works together with `rot`.
- `kx.plot(df, spec)` and `kx.plot(matrix, spec)` hand the whole table to a table kind, with no column names. A
  2-D array becomes a DataFrame labelled `0..n`. A table kind given columns, `y`, `z` or `by=` raises, and
  `kx.grammar(kind)` says it takes a DataFrame or 2-D array.
- **Colourmaps** `blues`, `greys` and `rdbu`: lowercase names for matplotlib's ColorBrewer `Blues`, `Greys` and
  `RdBu`, for `kx.use(cmap=...)` and in `kx.cmaps()`. `_r` reverses them like the others.
- **Demo dataset** `passengers`: 500 synthetic passengers (`age`, `fare`, `sex`, `pclass`, `port`, `deck`,
  `ticket_type`, `survived`) with missing ages, decks and ports, for the table kinds: `kx.load('passengers')`.
  It is generated by `scripts/make_passengers.py`, not taken from the Titanic data.

## [0.3.0] - 2026-10-12

### Added

- **Plot kind** `qq`: sorted values against normal quantiles, one series per array or `by=` group, with a dashed
  line through each one's quartiles. The quantiles come from Python's `statistics`, so no scipy is needed.
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

### Fixed

- `leg` and `legout` on a plot with nothing to label, such as `scatter` with a numeric `z`, no longer warn
  "No artists with labels found"; the plot just has no legend.

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

[Unreleased]: https://github.com/vhrabar/kx-plot/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/vhrabar/kx-plot/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/vhrabar/kx-plot/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/vhrabar/kx-plot/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/vhrabar/kx-plot/compare/v0.0.2...v0.1.0
[0.0.2]: https://github.com/vhrabar/kx-plot/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/vhrabar/kx-plot/releases/tag/v0.0.1
