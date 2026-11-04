"""
Render previews/<theme>.png, <theme>-ml.png, <theme>-panels.png and <theme>-img.png for every theme: every kind
on the bundled demo data.
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import kx  # noqa: E402


def render(theme: str, out_dir: str) -> str:
    """Draw one panel per kind in `theme`, 7 per row, and save it as PNG."""
    waves, cloud, monthly, dists, passengers = (kx.load(n) for n in ("waves", "cloud", "monthly", "dists",
                                                                     "passengers"))
    spread = (0.1 + waves.x / 40).rename("std")
    samples = (dists.normal, dists.skewed, dists.bimodal)
    kx.use(theme)
    a = iter(axs := kx.grid(29, cols=7))
    fig = axs[0].figure
    kx.plot(waves.x, waves.y, waves.z, "line-leg", title="line", ax=next(a))
    kx.plot(monthly.month, monthly.y2025, monthly.y2026, "step-leg", title="step", ax=next(a))
    kx.plot(monthly.month, monthly.y2025, monthly.y2026, "area-stack-leg", title="area-stack", ax=next(a))
    kx.plot(waves.x, waves.y, spread, "band-leg", title="band", ax=next(a))
    kx.plot(cloud.x, cloud.y, cloud.z, "scatter", title="scatter", ax=next(a))
    kx.plot(monthly.month, monthly.y2025, monthly.y2026, "bar-leg", title="bar", ax=next(a))
    kx.plot(monthly.month, monthly.y2025, monthly.y2026, "barh-stack-leg", title="barh-stack", ax=next(a))
    kx.plot(*samples, spec="hist-leg", title="hist", ax=next(a))
    kx.plot(*samples, spec="kde-leg", title="kde", ax=next(a))
    kx.plot(*samples, spec="ridge", title="ridge", ax=next(a))
    kx.plot(*samples, spec="ecdf-leg", title="ecdf", ax=next(a))
    kx.plot(*samples, spec="box-sig", title="box-sig", ax=next(a))
    kx.plot(*samples, spec="violin", title="violin", ax=next(a))
    kx.plot(*samples, spec="strip", title="strip", ax=next(a))
    kx.plot(*samples, spec="qq", title="qq", ax=next(a))
    kx.plot(cloud, "heat-corr-ann", title="heat-corr", ax=next(a))
    kx.plot(passengers, "pclass", "port", "crosstab-ann", title="crosstab", ax=next(a))
    kx.plot(passengers, "pclass", spec="count-leg", by="survived", title="count", ax=next(a))
    kx.plot(passengers, "missing-ann-wrap", title="missing", ax=next(a))
    kx.plot(passengers, "nullmap-sort-rot", title="nullmap", ax=next(a))
    kx.plot(cloud.x, cloud.y, "hexbin", title="hexbin", gridsize=12, ax=next(a))
    kx.plot(passengers, "port", spec="pie-ann", title="pie", ax=next(a))
    kx.plot(passengers, "pclass", spec="donut-ann", title="donut", ax=next(a))
    profile = passengers[["age", "fare", "pclass", "survived", "sex"]]
    kx.plot(profile, "radar", by="sex", title="radar", ax=next(a))
    kx.plot(profile, "parallel-mean", by="sex", title="parallel", ax=next(a))
    kx.plot(passengers, "pclass", "sex", "survived", spec="mosaic-ann", title="mosaic", ax=next(a))
    kx.plot(passengers, "pclass", "sex", spec="treemap-ann", title="treemap", ax=next(a))
    kx.plot(passengers, "pclass", "sex", spec="sunburst", title="sunburst", ax=next(a))
    kx.cramers(passengers, cluster=True, title="kx.cramers", ax=next(a))
    fig.suptitle(theme)
    path = os.path.join(out_dir, f"{theme}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_ml(theme: str, out_dir: str) -> str:
    """Draw one panel per machine-learning kind in `theme`, 5 per row, and save it as PNG."""
    binary, multi, reg, history, imp, curves = (kx.load(n) for n in ("binary", "multiclass", "regression", "history",
                                                                     "importance", "curves"))
    rng = np.random.default_rng(0)
    gamma = np.logspace(-3, 2, 10)              # vcurve: a hyperparameter where validation peaks in the middle
    t = np.log10(gamma)[:, None]
    noise = rng.normal(0, 0.01, (10, 5))
    ks = np.arange(1, 11)                       # elbow: inertia and silhouette for k clusters
    kx.use(theme)
    a = iter(axs := kx.grid(16, cols=4))
    fig = axs[0].figure
    kx.plot(multi, "y", "pred", "cm-norm-ann", title="cm", ax=next(a))
    kx.plot(binary, "y", "gbm", "logreg", "roc", title="roc", ax=next(a))
    kx.plot(binary, "y", "gbm", "logreg", "pr", title="pr", ax=next(a))
    kx.plot(binary, "y", "gbm", "logreg", "calib", title="calib", ax=next(a))
    kx.plot(binary, "y", "gbm", "proba", title="proba", ax=next(a))
    kx.plot(binary, "y", "gbm", "thresh-best", title="thresh", ax=next(a))
    kx.plot(binary, "y", "gbm", "logreg", "gain", title="gain", ax=next(a))
    kx.plot(binary, "y", "gbm", "logreg", "lift", title="lift", ax=next(a))
    kx.plot(reg, "actual", "predicted", "pva", title="pva", ax=next(a))
    kx.plot(reg, "actual", "predicted", "x", "resid", title="resid", ax=next(a))
    kx.plot(imp, "feature", "importance", "std", "imp", title="imp", ax=next(a))
    kx.plot(history, "loss-best", title="loss", ax=next(a))
    kx.plot(curves["size"], curves.filter(like="train_"), curves.filter(like="val_"), "lcurve", title="lcurve",
            ax=next(a))
    kx.plot(gamma, 0.98 - 0.35 / (1 + np.exp(2 * (t + 1))) + noise, 0.83 - 0.03 * (t - 0.5) ** 2 + 2 * noise,
            "vcurve-logx", title="vcurve", ax=next(a))
    kx.plot(ks, 1000 / ks ** 1.2 + 40, 0.6 - 0.04 * np.abs(ks - 3), "elbow-best", title="elbow", ax=next(a))
    embedding = kx.load("embedding")
    kx.embed(embedding.filter(like="f"), by=embedding.label, title="kx.embed", ax=next(a))
    fig.suptitle(f"{theme}: machine learning")
    path = os.path.join(out_dir, f"{theme}-ml.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _cells(cols: int, rows: int, scale: float = 1.5) -> tuple[float, float]:
    """A figure size for cells that each hold several panels: the theme's own figure size, scaled up, per cell, so
    text keeps its room in big-type themes (talk, poster) as in small ones."""
    w, h = plt.rcParams["figure.figsize"]
    return cols * w * scale, rows * h * scale


def render_panels(theme: str, out_dir: str) -> str:
    """The kinds that draw several panels, one per cell, in `theme`, and save it as PNG."""
    passengers, embedding = kx.load("passengers"), kx.load("embedding")
    kx.use(theme)
    a = iter(axs := kx.grid(4, cols=2, figsize=_cells(2, 2)))
    fig = axs[0].figure
    kx.plot(embedding, "pair", by="label", cols=["x", "y", "size"], title="pair", ax=next(a))
    kx.plot(embedding.select_dtypes("number"), "cluster-corr", title="cluster-corr", ax=next(a))
    kx.plot(passengers, "pclass", "port", "crosstab-facet-ann", facet="sex", title="crosstab-facet", ax=next(a))
    kx.plot(passengers, "fare", spec="violin-sig-facet", by="pclass", facet="sex", title="violin-sig-facet",
            ax=next(a))
    fig.suptitle(f"{theme}: panels")
    path = os.path.join(out_dir, f"{theme}-panels.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_img(theme: str, out_dir: str) -> str:
    """The image kinds on the shapes dataset, one per cell, in `theme`, and save it as PNG."""
    d = kx.load("shapes")
    proba = d["proba"]
    kx.use(theme)
    a = iter(axs := kx.grid(7, cols=4, figsize=_cells(4, 2)))
    fig = axs[0].figure
    kx.plot(d["images"], d["labels"], "imgrid", n=4, title="imgrid", ax=next(a))
    kx.plot(d["images"], d["labels"], "meanimg", cols=4, title="meanimg", ax=next(a))
    kx.plot(d["images"], d["labels"], "colorhist", title="colorhist", ax=next(a))
    kx.plot(d["images"], d["labels"], "imhist", idx=[0, 1], title="imhist", ax=next(a))
    kx.plot(d["images"], d["masks"], np.roll(d["masks"], 2, axis=2), "segmask", idx=[0, 1], title="segmask",
            ax=next(a))
    kx.plot(d["images"], d["saliency"], "saliency", idx=[0, 2], title="saliency", ax=next(a))
    kx.plot(d["images"], d["labels"], proba, "miscls", n=6, title="miscls", ax=next(a))
    fig.suptitle(f"{theme}: images")
    path = os.path.join(out_dir, f"{theme}-img.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    out_dir = os.path.join(_root, "previews")
    os.makedirs(out_dir, exist_ok=True)
    for theme in kx.styles():
        print(render(theme, out_dir))
        print(render_ml(theme, out_dir))
        print(render_panels(theme, out_dir))
        print(render_img(theme, out_dir))


if __name__ == "__main__":
    main()
