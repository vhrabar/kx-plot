"""
Render previews/<theme>.png for every theme: every kind on the bundled demo CSVs.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import kx  # noqa: E402


def render(theme: str, out_dir: str) -> str:
    """Draw one panel per kind in `theme`, 5 per row, and save it as PNG."""
    waves, cloud, monthly, dists = (kx.load(n) for n in ("waves", "cloud", "monthly", "dists"))
    spread = (0.1 + waves.x / 40).rename("std")
    samples = (dists.normal, dists.skewed, dists.bimodal)
    kx.use(theme)
    a = iter(axs := kx.grid(14, cols=5))
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
    kx.plot(*samples, spec="box", title="box", ax=next(a))
    kx.plot(*samples, spec="violin", title="violin", ax=next(a))
    kx.plot(*samples, spec="strip", title="strip", ax=next(a))
    fig.suptitle(theme)
    path = os.path.join(out_dir, f"{theme}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    out_dir = os.path.join(_root, "previews")
    os.makedirs(out_dir, exist_ok=True)
    for theme in kx.styles():
        print(render(theme, out_dir))


if __name__ == "__main__":
    main()
