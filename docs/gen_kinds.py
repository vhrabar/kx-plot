"""
Write the kind reference from the @kind registry (kx._spec.KINDS): one page per kinds module, one section per kind
with its docstring, flags and an example image.
"""
import ast
import inspect
import os
import re
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

DOCS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DOCS)
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]

import make_previews  # noqa: E402

import kx  # noqa: E402
from kx._spec import COMMON_FLAGS, EXCLUSIVE, KINDS  # noqa: E402

THEME = "light"
PAGES = {
    "kx.kinds.basic": ("basic", "Basic"),
    "kx.kinds.dist": ("dist", "Distributions"),
    "kx.kinds.table": ("table", "Tables"),
    "kx.kinds.ml": ("ml", "Machine learning"),
    "kx.kinds.image": ("image", "Images"),
}


def record() -> dict[str, tuple[tuple, dict, str]]:
    """Run every make_previews sheet with kx.plot swapped for a recorder: kind -> (args, kwargs, source line)."""
    source = inspect.getsource(make_previews)
    tree = ast.parse(source)
    calls = {n.lineno: n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "kx.plot"}
    sheets = [f for f in tree.body if isinstance(f, ast.FunctionDef)]
    seen: dict[str, tuple[tuple, dict, str]] = {}
    real = kx.plot

    def recorder(*args, **kw):                  # noqa: ANN002, ANN003, ANN202
        spec = kw.get("spec") or next(a for a in args if isinstance(a, str) and _is_spec(a))
        kind = kx.parse(spec).kind
        call = calls.get(sys._getframe(1).f_lineno)
        if kind not in seen and call is not None:
            sheet = next(f for f in sheets if f.lineno <= call.lineno <= f.end_lineno)
            seen[kind] = (args, kw, example(source, sheet, call))

    kx.plot = recorder
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for sheet in (make_previews.render, make_previews.render_ml, make_previews.render_panels,
                          make_previews.render_img):
                sheet(THEME, tmp)
    finally:
        kx.plot = real
        plt.close("all")
    emb = kx.load("embedding")                  # make_previews draws embed through kx.embed, not kx.plot
    seen.setdefault("embed", ((emb, "x", "y", "embed"), {"by": "label"},
                              'import kx\n\nembedding = kx.load("embedding")\n'
                              'kx.plot(embedding, "x", "y", "embed", by="label")'))
    return seen


def _assignments(source: str, sheet: ast.FunctionDef) -> dict[str, tuple[int, str, set[str]]]:
    """Each name a sheet assigns: name -> (line, a statement that sets it alone, the names that statement reads).
    `a, b = (kx.load(n) for n in ("x", "y"))` and `a, b = kx.load("x"), kx.load("y")` split into one line each."""
    out: dict[str, tuple[int, str, set[str]]] = {}
    for st in ast.walk(sheet):
        if isinstance(st, ast.Assign) and len(st.targets) == 1:
            target, value = st.targets[0], st.value
        elif isinstance(st, ast.NamedExpr):
            target, value = st.target, st.value
        else:
            continue
        if isinstance(target, ast.Name):
            pairs = [(target.id, value)]
        elif isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
            pairs = [(t.id, v) for t, v in zip(target.elts, value.elts, strict=True)]
        elif isinstance(target, ast.Tuple) and isinstance(value, ast.GeneratorExp):   # (kx.load(n) for n in ...)
            names = [ast.literal_eval(e) for e in value.generators[0].iter.elts]
            for t, n in zip(target.elts, names, strict=True):
                out[t.id] = (st.lineno, f'{t.id} = kx.load("{n}")', {"kx"})
            continue
        else:
            continue
        for name, v in pairs:
            text = ast.get_source_segment(source, v)
            out[name] = (st.lineno, f"{name} = {text}", {n.id for n in ast.walk(v) if isinstance(n, ast.Name)})
    return out


def example(source: str, sheet: ast.FunctionDef, call: ast.Call) -> str:
    """The kx.plot call as a runnable snippet: imports, then every line of the sheet it needs, in order."""
    known = _assignments(source, sheet)
    reads = {n.id for n in ast.walk(call) if isinstance(n, ast.Name)}
    need, todo = set(), set(reads)
    while todo:
        name = todo.pop()
        if name in known and name not in need and name not in ("a", "axs", "fig"):
            need.add(name)
            todo |= known[name][2]
    lines = [known[n][1] for n in sorted(need, key=lambda n: known[n][0])]
    used = reads.union(*(known[n][2] for n in need))
    imports = ["import kx"] + [f"import {m} as {a}" for m, a in (("numpy", "np"), ("pandas", "pd")) if a in used]
    return "\n".join([*imports, "", *lines, tidy(ast.get_source_segment(source, call))])


def _is_spec(s: str) -> bool:
    try:
        kx.parse(s)
        return True
    except ValueError:
        return False


def tidy(code: str) -> str:
    """The recorded call without the preview sheet's ax= and title=, on one line where it fits."""
    code = re.sub(r",\s*ax=next\(a\)", "", code)
    code = re.sub(r",\s*title=\"[^\"]*\"", "", code)
    return re.sub(r"\s*\n\s*", " ", code)


def draw(kind: str, args: tuple, kw: dict, out: str) -> None:
    """Replay one recorded call on a figure of its own."""
    k = KINDS[kind]
    kx.use(THEME)
    w, h = plt.rcParams["figure.figsize"]
    big = k.panels or k.draw.__module__ == "kx.kinds.image"
    fig, ax = plt.subplots(figsize=(w * 1.6, h * 1.6) if big else (w, h),
                           subplot_kw={"projection": k.projection} if k.projection else {})
    kx.plot(*args, **{**kw, "ax": ax})
    fig.savefig(out, bbox_inches="tight", dpi=110)
    plt.close(fig)


def rst_text(doc: str) -> str:
    """A plain-text docstring as reST: `x` and kw= stay literal, * does not start emphasis."""
    doc = doc.replace("*", r"\*").replace("|", r"\|")
    doc = re.sub(r"(\w)_(?=[^\w`]|$)", r"\1\\_", doc, flags=re.MULTILINE)   # val_ would be a link target
    return re.sub(r"(?<![\w`])([A-Za-z_]+=)(?=[\s,.:;)]|$)", r"``\1``", doc, flags=re.MULTILINE)


def section(name: str, example: tuple[tuple, dict, str] | None, img_dir: str) -> str:
    """One kind: heading, docstring, flags, data notes, example code and image."""
    k = KINDS[name]
    lines = [f".. _kind-{name}:", "", f"``{name}``", "-" * (len(name) + 4), "", rst_text(inspect.getdoc(k.draw)), ""]
    own = ", ".join(f"``{f}``" for f in sorted(k.flags)) or "none of its own"
    lines += [".. list-table::", "   :stub-columns: 1", "   :widths: 1 4", "",
              "   * - Flags", f"     - {own}, plus the :ref:`common flags <common-flags>`"]
    for on, what in ((k.takes_table, "a whole DataFrame or 2-D array"),
                     (k.takes_by, "``by=``: one series per group"),
                     (k.panels, "several panels; ``kx.plot`` returns them as a flat array"),
                     (k.projection, f"``{k.projection}`` axes")):
        if on:
            lines += ["   * - Data", f"     - {what}"]
    lines.append("")
    if example:
        args, kw, code = example
        draw(name, args, kw, os.path.join(img_dir, f"{name}.png"))
        if code:
            lines += [".. code-block:: python", "", *(f"   {ln}" if ln else "" for ln in code.split("\n")), ""]
        lines += [f".. image:: /_static/kinds/{name}.png", f"   :alt: {name}", "   :width: 100%" if (
            k.panels or k.draw.__module__ == "kx.kinds.image") else "   :width: 70%", ""]
    return "\n".join(lines)


def main() -> None:
    img_dir = os.path.join(DOCS, "_static", "kinds")
    out_dir = os.path.join(DOCS, "kinds")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    examples = record()
    for module, (page, title) in PAGES.items():
        names = [k.name for k in KINDS.values() if k.draw.__module__ == module]
        intro = inspect.getdoc(sys.modules[module]) or ""
        body = [title, "=" * len(title), "", rst_text(intro), ""]
        body += [section(n, examples.get(n), img_dir) for n in names]
        with open(os.path.join(out_dir, f"{page}.rst"), "w") as f:
            f.write("\n".join(body))
    lists = {                                   # included by spec.rst, so the lists never go stale
        "_themes.rst": kx.styles(),
        "_palettes.rst": kx.palettes(),
        "_cmaps.rst": kx.cmaps(),
    }
    for name, items in lists.items():
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(", ".join(f"``{i}``" for i in items) + "\n")
    with open(os.path.join(out_dir, "_exclusive.rst"), "w") as f:
        f.writelines(f"* {' / '.join(f'``{flag}``' for flag in sorted(group))}\n" for group in EXCLUSIVE)
    with open(os.path.join(DOCS, "spec.rst")) as f:     # the common-flag table is prose: fail if it misses one
        undocumented = COMMON_FLAGS - set(re.findall(r"``(\w+)``", f.read()))
    if undocumented:
        raise SystemExit(f"docs/spec.rst: add {sorted(undocumented)} to the common flags table")
    missing = sorted(set(KINDS) - set(examples))
    print(f"{len(KINDS)} kinds, {len(examples)} with an example image; without: {missing}")


if __name__ == "__main__":
    main()
