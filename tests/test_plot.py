"""kx.plot itself: spec parsing, call shapes, keywords, and kinds that are table, panel or polar kinds."""
import re
from collections.abc import Callable, Iterator
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from helpers import DF, X, Y, Z
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from numpy.typing import ArrayLike

import kx


# ---------- spec parsing ----------
def test_parse_order_does_not_matter() -> None:
    assert kx.parse("scatter-dark-grid-leg") == kx.parse("leg-grid-dark-scatter")


def test_parse_defaults_to_line_and_no_theme() -> None:
    assert kx.parse("grid") == ("line", None, {"grid"})


def test_spec_fields_and_canonical_slug() -> None:
    spec = kx.parse("leg-grid-dark-scatter")
    assert (spec.kind, spec.theme, spec.flags) == ("scatter", "dark", frozenset({"grid", "leg"}))
    kind, theme, flags = spec
    assert (kind, theme, flags) == spec
    assert str(spec) == spec.slug == "scatter-dark-grid-leg"
    assert str(kx.parse("grid")) == "line-grid"
    assert kx.parse(str(spec)) == spec
    assert kx.parse(spec) is spec


def test_plot_takes_a_spec() -> None:
    assert len(kx.plot(X, Y, kx.parse("bar")).patches) == len(X)
    assert len(kx.plot(X, Y, spec=kx.parse("bar")).patches) == len(X)
    assert len(kx.plot(DF, "t", "speed", kx.parse("bar")).patches) == len(X)


@pytest.mark.parametrize(("spec", "hint"), [
    ("line-lgo", "Did you mean 'leg' or 'legout' for 'lgo'?"),
    ("scater-drak", "'dark' for 'drak'"),
    ("line--dark", "a '-' too many"),
])
def test_parse_hints_close_tokens(spec: str, hint: str) -> None:
    with pytest.raises(ValueError, match=re.escape(hint)):
        kx.parse(spec)


@pytest.mark.parametrize(("spec", "match"), [
    ("line-nope", "unknown tokens"),
    ("line--dark", "unknown tokens"),
    ("line-scatter", "more than one kind"),
    ("dark-light", "more than one theme"),
])
def test_parse_rejects_bad_specs(spec: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        kx.parse(spec)


# ---------- call shapes ----------
def test_spec_as_third_positional() -> None:
    ax = kx.plot(X, Y, "bar")
    assert len(ax.patches) == len(X)


def test_spec_given_twice() -> None:
    with pytest.raises(TypeError):
        kx.plot(X, Y, "line", spec="bar")


def test_dataframe_columns_by_name() -> None:
    ax = kx.plot(DF, "t", "speed", "accel", "line-leg")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["speed", "accel"]
    assert list(ax.lines[0].get_ydata()) == list(Y)


def test_dataframe_spec_as_keyword_and_by_column() -> None:
    assert len(kx.plot(DF, "t", "speed", spec="bar").patches) == len(X)
    ax = kx.plot(DF, "speed", spec="box", by="group")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["early", "late"]
    assert ax.get_xlabel() == "group"


def test_dataframe_column_named_like_a_kind_is_a_column() -> None:
    df = pd.DataFrame({"x": X, "bar": Y})
    ax = kx.plot(df, "x", "bar")
    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_ydata()) == list(Y)


@pytest.mark.parametrize(("call", "error", "match"), [
    (lambda: kx.plot(DF, "t", "sped"), ValueError, r"no column 'sped'.*Did you mean \['speed'\]"),
    (lambda: kx.plot(DF, "t", "speed", "linee"), ValueError, "unknown tokens"),
    (lambda: kx.plot(DF, spec="line"), TypeError, "needs the column names"),
    (lambda: kx.plot(DF, "t", DF.speed), TypeError, "give column names"),
    (lambda: kx.plot(X, spec="kde", by="group"), TypeError, "needs a DataFrame"),
    (lambda: kx.plot(X, Y, Z, Z), TypeError, "up to 3 arrays, got 4"),
    (lambda: kx.plot(), TypeError, "got 0"),
])
def test_call_shape_errors(call: Callable[[], Axes], error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        call()


@pytest.fixture
def table_kind() -> Iterator[list]:
    """A dummy table kind 'tbl' that records what draw gets, removed after the test."""
    seen = []
    kx._spec.kind("tbl", table=True)(lambda ax, x, y, z, flags, **kw: seen.append((x, y, z)))
    yield seen
    del kx._spec.KINDS["tbl"]


def test_table_kind_gets_the_whole_frame(table_kind: list) -> None:
    assert not kx._spec.KINDS["tbl"].needs_y
    kx.plot(DF, "tbl")
    kx.plot(DF, spec="tbl")
    for x, y, z in table_kind:
        assert x is DF
        assert y is None
        assert z is None


def test_table_kind_takes_a_2d_array_with_numbered_labels(table_kind: list) -> None:
    m = np.arange(6).reshape(2, 3)
    kx.plot(m, "tbl")
    x = table_kind[0][0]
    assert isinstance(x, pd.DataFrame)
    assert (x.to_numpy() == m).all()
    assert list(x.columns) == [0, 1, 2]
    assert list(x.index) == [0, 1]


@pytest.mark.parametrize("call", [
    lambda: kx.plot(DF, "t", "speed", "tbl"),
    lambda: kx.plot(DF, "t", "tbl"),
    lambda: kx.plot(DF, "tbl", by="group"),
    lambda: kx.plot(X, "tbl"),
])
def test_table_kind_rejects_columns_y_z_and_by(table_kind: list, call: Callable[[], Axes]) -> None:
    with pytest.raises(TypeError, match=r"takes a whole table.*kx.plot\(df\[cols\], 'tbl'\)"):
        call()


def test_frame_without_columns_hints_table_kinds(table_kind: list) -> None:
    tables = ", ".join(sorted(k.name for k in kx.KINDS.values() if k.takes_table))
    assert {"heat", "loss", "tbl"} <= set(tables.split(", "))
    with pytest.raises(TypeError, match=r"needs the column names.*or a table kind: " + re.escape(tables)):
        kx.plot(DF, "line")


def test_grammar_says_a_table_kind_takes_a_frame(table_kind: list, capsys: pytest.CaptureFixture[str]) -> None:
    kx.grammar("tbl")
    assert "takes a DataFrame or 2-D array" in capsys.readouterr().out
    kx.grammar("line")
    assert "DataFrame" not in capsys.readouterr().out


def test_kwargs_reach_matplotlib_and_win_over_defaults() -> None:
    ln = kx.plot(X, Y, Z, "line-mk", linewidth=4, marker="s").lines
    assert {(x.get_linewidth(), x.get_marker()) for x in ln} == {(4, "s")}
    assert [t.get_text() for t in kx.plot(X, Y, "line-leg", label="mine").get_legend().get_texts()] == ["mine"]
    sc = kx.plot(X, Y, Z, "scatter", cmap="magma", s=50).collections[0]
    assert (sc.get_cmap().name, list(sc.get_sizes())) == ("magma", [50])


@pytest.mark.parametrize(("spec", "kw", "check"), [
    ("step", {"linestyle": "--"}, lambda ax: ax.lines[0].get_linestyle() == "--"),
    ("area", {"alpha": 0.5}, lambda ax: ax.collections[0].get_alpha() == 0.5),
    ("area-stack", {"alpha": 0.5}, lambda ax: ax.collections[0].get_alpha() == 0.5),
    ("band", {"color": "red"}, lambda ax: to_hex(ax.collections[0].get_facecolor()[0]) == "#ff0000"),
    ("kde", {"linestyle": ":"}, lambda ax: ax.lines[0].get_linestyle() == ":"),
    ("ecdf", {"color": "red"}, lambda ax: to_hex(ax.lines[0].get_color()) == "#ff0000"),
    ("box", {"showfliers": False}, lambda ax: len(ax.lines) == 5),             # whiskers, caps, median; no fliers
    ("violin", {"widths": 0.2}, lambda ax: np.ptp(ax.collections[0].get_paths()[0].vertices[:, 0]) <= 0.2 + 1e-9),
])
def test_kwargs_for_each_kind(spec: str, kw: dict, check: Callable[[Axes], bool]) -> None:
    assert check(kx.plot(X, Y, np.abs(Z), spec, **kw) if spec in ("step", "area", "area-stack", "band")
                 else kx.plot(np.r_[X, 100], spec=spec, **kw))


@pytest.mark.parametrize(("spec", "key", "size"), [("bar", "width", "get_width"), ("barh", "height", "get_height")])
def test_bar_width_is_the_space_per_category(spec: str, key: str, size: str) -> None:
    ax = kx.plot(["a", "b"], [1, 2], [2, 1], spec, **{key: 0.5})
    assert {getattr(p, size)() for p in ax.patches} == {0.25}


def test_hist_bins_keyword_replaces_shared_bins() -> None:
    assert len(kx.plot(X, spec="hist", bins=5).patches) == 5
    ax = kx.plot(X, Y, spec="hist", bins=np.linspace(0, 10, 11))
    assert [len(c) for c in ax.containers] == [10, 10]


def test_unknown_kwarg_is_matplotlibs_error() -> None:
    with pytest.raises(AttributeError, match="nonsense"):
        kx.plot(X, Y, "line", nonsense=1)


def test_non_hist_needs_y() -> None:
    with pytest.raises(ValueError, match="needs both x and y"):
        kx.plot(X, spec="line")


def test_draws_into_given_ax() -> None:
    import matplotlib.pyplot as plt
    _, (a, b) = plt.subplots(1, 2)
    assert kx.plot(X, Y, "line", ax=b) is b
    assert len(a.lines) == 0


def _suptitle(fig: mpl.figure.Figure) -> str:
    """The figure's suptitle text; Figure.get_suptitle is matplotlib >= 3.8 only."""
    return fig._suptitle.get_text() if fig._suptitle else ""


@pytest.fixture
def panel_kind() -> Iterator[None]:
    """A dummy kind 'quad' that splits ax into 2 x 2 panels, and 'nopanels' that forgets to return them."""
    def quad(ax: Axes, x: ArrayLike, y: ArrayLike, z: None, flags: set[str], **kw: Any) -> np.ndarray:
        axs = kx._util.split(ax, 2, 2)
        for a in axs:
            a.plot(x, y, label="y")
        return axs
    kx._spec.kind("quad", panels=True)(quad)
    kx._spec.kind("nopanels", panels=True)(lambda ax, x, y, z, flags, **kw: None)
    yield
    del kx._spec.KINDS["quad"], kx._spec.KINDS["nopanels"]


def test_panel_kind_returns_its_axes_in_place_of_ax(panel_kind: None) -> None:
    axs = kx.plot(X, Y, "quad-grid", title="four")
    assert isinstance(axs, np.ndarray)
    assert axs.shape == (4,)
    fig = axs[0].figure
    assert fig.axes == list(axs)                # the host axes is gone
    assert all(g.get_visible() for a in axs for g in a.get_xgridlines())   # common flags reach every panel
    assert _suptitle(fig) == "four"         # the panels fill the figure: one title over them


def test_panel_kind_stays_in_its_grid_cell(panel_kind: None) -> None:
    left, right = kx.grid(2)
    cell = right.get_subplotspec()
    axs = kx.plot(X, Y, "quad", title="four", ax=right)
    fig = left.figure
    assert right not in fig.axes
    assert left in fig.axes
    assert len(fig.axes) == 5
    assert all(a.get_subplotspec().get_topmost_subplotspec() == cell for a in axs)
    assert axs[0].get_title() == "four"        # a title over one cell of a bigger figure: on its first panel
    assert _suptitle(fig) == ""


def test_panel_kind_must_return_its_axes(panel_kind: None) -> None:
    with pytest.raises(TypeError, match="must return their axes"):
        kx.plot(X, Y, "nopanels")


def test_split_needs_a_grid_cell() -> None:
    import matplotlib.pyplot as plt
    ax = plt.figure().add_axes((0.1, 0.1, 0.8, 0.8))
    with pytest.raises(TypeError, match="add_axes"):
        kx._util.split(ax, 1, 2)


@pytest.fixture
def polar_kind() -> Iterator[list]:
    """A dummy kind 'circle' on polar axes that records the axes it draws on."""
    seen = []
    kx._spec.kind("circle", projection="polar")(lambda ax, x, y, z, flags, **kw: seen.append(ax))
    yield seen
    del kx._spec.KINDS["circle"]


def test_polar_kind_gets_polar_axes(polar_kind: list) -> None:
    ax = kx.plot(X, Y, "circle", title="round")
    assert ax.name == "polar"
    assert polar_kind == [ax]
    assert ax.get_title() == "round"


def test_polar_kind_swaps_a_grid_cell_in_place(polar_kind: list) -> None:
    left, right = kx.grid(2)
    cell = right.get_subplotspec()
    ax = kx.plot(X, Y, "circle", ax=right)
    assert ax.name == "polar"
    assert ax.get_subplotspec() == cell
    assert right not in ax.figure.axes
    assert left in ax.figure.axes
    assert left.name == "rectilinear"


def test_polar_kind_keeps_polar_axes(polar_kind: list) -> None:
    import matplotlib.pyplot as plt
    given = plt.subplots(subplot_kw={"projection": "polar"})[1]
    assert kx.plot(X, Y, "circle", ax=given) is given


def test_grammar_says_how_a_kind_draws(panel_kind: None, polar_kind: list,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    kx.grammar("quad")
    assert "flat array" in capsys.readouterr().out
    kx.grammar("circle")
    assert "polar" in capsys.readouterr().out


@pytest.mark.parametrize("fn", [kx.grammar, kx.source])
def test_unknown_kind(fn) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="unknown kind"):
        fn("nope")


@pytest.mark.parametrize("name", ["line", "grid", "a-b"])
def test_kind_names_stay_unique_tokens(name: str) -> None:
    with pytest.raises(ValueError, match="taken"):
        kx._spec.kind(name)(lambda *a: None)
