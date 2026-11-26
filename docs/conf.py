"""Sphinx configuration for kx-plot (Read the Docs)."""
import os
import subprocess
import sys

DOCS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DOCS)
sys.path.insert(0, ROOT)

from sphinx.util.inspect import TypeAliasForwardRef  # noqa: E402

import kx  # noqa: E402

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
