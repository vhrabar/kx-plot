"""kx.save: savefig that knows where Kaggle keeps files."""
import os
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

KAGGLE = "/kaggle/working"

def save(name: str, fig: Figure | Axes | None = None, **kw: Any) -> str:
    """Save fig (or ax.figure, or the current figure) as `name` and return the path written.

    A relative name goes to /kaggle/working on Kaggle and to the current folder elsewhere; missing folders
    are created. Without an extension the theme's savefig.format is used (thesis: PDF). Other keywords
    go to savefig. Call it before plt.show(), which leaves a new empty current figure, or pass ax / fig.
    """
    fig = plt.gcf() if fig is None else fig.figure if isinstance(fig, Axes) else fig
    path = os.path.expanduser(name)
    if not os.path.isabs(path) and os.path.isdir(KAGGLE):
        path = os.path.join(KAGGLE, path)
    if not os.path.splitext(path)[1]:
        path += "." + (kw.get("format") or plt.rcParams["savefig.format"])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, **kw)
    return path
