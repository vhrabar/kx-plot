"""Sphinx configuration for kx-plot (Read the Docs)."""
import os
import subprocess
import sys

DOCS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DOCS)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(DOCS, "_ext"))

from sphinx.util.inspect import TypeAliasForwardRef  # noqa: E402

import kx  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

with open(os.path.join(ROOT, "pyproject.toml"), "rb") as _f:
    _pyproject = tomllib.load(_f)["project"]

# the kind pages come from the @kind registry: regenerate them on every build
subprocess.run([sys.executable, os.path.join(DOCS, "gen_kinds.py")], check=True, cwd=ROOT)

project = "kx-plot"
author = "Vedran Hrabar"
copyright = "2026, Vedran Hrabar"
release = kx.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
    "kx_seo",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "kinds/_*.rst"]
default_role = "literal"

autodoc_member_order = "bysource"
autodoc_docstring_signature = False
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_type_aliases = {
    "DataFrame": "~kx.typing.DataFrame",
    "Series": "~kx.typing.Series",
    "Column": "~kx.typing.Column",
}

TypeAliasForwardRef.__repr__ = lambda self: self.name
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

html_theme = "sphinx_rtd_theme"
html_title = f"kx-plot {release}"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "style_external_links": True,
}
html_context = {
    "display_github": True,
    "github_user": "vhrabar",
    "github_repo": "kx-plot",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Read the Docs sets the canonical URL of the version being built; sitemap, llms.txt and JSON-LD use it
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
kx_seo_fallback_url = "https://kx-plot.readthedocs.io/en/latest/"
kx_seo_description = _pyproject["description"]
kx_seo_keywords = _pyproject["keywords"]
kx_seo_author_url = "https://github.com/vhrabar"
kx_seo_summary = """
kx-plot is a small, plain-Python package (import name `kx`) of one-line matplotlib plots, `.mplstyle` themes,
palettes and colourmaps. Install with `pip install kx-plot`. A plot is `kx.plot(*data, spec)`, where the spec string
is `kind-theme-flag-...`, e.g. `kx.plot(df, "month", "sales", "bar-paper-ann")`. Data can be arrays, lists, or
columns of any narwhals-supported dataframe (pandas, polars, pyarrow, ...) by name. It works offline, so it also
runs in Kaggle competition notebooks.
"""
kx_seo_links = {
    "GitHub": "https://github.com/vhrabar/kx-plot",
    "PyPI": "https://pypi.org/project/kx-plot/",
    "Gallery": "https://vhrabar.github.io/kx-plot",
    "Kaggle dataset": "https://www.kaggle.com/datasets/vhrabar/kx-plot",
    "Example notebook": "https://www.kaggle.com/code/vhrabar/kx-one-line-matplotlib-plots-and-themes",
}
