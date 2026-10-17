"""Spec grammar (kind-theme-flag-flag-...) and the registry of kinds."""
import difflib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

from ._theme import styles

COMMON_FLAGS = {"grid", "leg", "legout", "tight", "rot",                   # handled by kx.plot for every kind
                "logx", "logy", "logxy", "symlogy", "zero", "eq", "pct", "si"}
EXCLUSIVE = ({"logy", "logxy", "symlogy"}, {"logx", "logxy"}, {"zero", "logy"}, {"zero", "logxy"},
             {"pct", "si"}, {"sym", "logc"})                                # pairs that ask for opposite things


@dataclass(frozen=True)
class Kind:
    """A plot kind: draw(ax, x, y, z, flags) plus what kx.plot checks before drawing."""
    name: str
    draw: Callable[..., None]
    flags: frozenset[str]                       # kind-specific flags, on top of COMMON_FLAGS
    needs_y: bool
    check: Callable[[Any, Any, set[str]], None] | None     # check(y, z, flags) raises on bad input
    takes_by: bool = False                      # draw also gets by= (group labels for x)
    value_axis: str = "y"                       # where pct, si and zero apply: "x" for barh
    takes_table: bool = False                   # x is a whole DataFrame (or 2-D array), with no y or z


KINDS: dict[str, Kind] = {}


def kind(name: str, flags: tuple[str, ...] = (), needs_y: bool = True,
         check: Callable[[Any, Any, set[str]], None] | None = None, takes_by: bool = False,
         value_axis: str = "y", table: bool = False) -> Callable:
    """Decorator that registers a draw function as kind `name`. table=True: x is the whole DataFrame."""
    def register(draw: Callable[..., None]) -> Callable[..., None]:
        if name in KINDS or name in COMMON_FLAGS or "-" in name:
            raise ValueError(f"kind name {name!r} is taken or has a '-'")
        KINDS[name] = Kind(name, draw, frozenset(flags), needs_y and not table, check, takes_by, value_axis, table)
        return draw
    return register


def all_flags() -> set[str]:
    """Common flags and every kind-specific flag."""
    return COMMON_FLAGS.union(*(k.flags for k in KINDS.values()))


class Spec(NamedTuple):
    """A parsed spec. Unpacks as kind, theme, flags; str() gives the canonical slug."""
    kind: str
    theme: str | None
    flags: frozenset[str]

    @property
    def slug(self) -> str:
        """kind, theme, then flags in alphabetical order: the same plot always gets the same string."""
        return "-".join([self.kind, *([self.theme] if self.theme else []), *sorted(self.flags)])

    def __str__(self) -> str:
        return self.slug


def parse(spec: str | Spec) -> Spec:
    """Split 'kind-theme-flag-...' into Spec(kind, theme, flags). Strict: no guessing, but hints."""
    if isinstance(spec, Spec):
        return spec
    toks = spec.split("-")
    themes, flags = set(styles()), all_flags()
    valid = KINDS.keys() | flags | themes
    unknown = [t for t in toks if t not in valid]
    if unknown:
        hints = [" or ".join(map(repr, sorted(close))) + f" for {t!r}"
                 for t in unknown if (close := difflib.get_close_matches(t, valid, n=2))]
        hint = f" Did you mean {', '.join(hints)}?" if hints else ""
        if "" in unknown:
            hint += " An empty token means a '-' too many."
        raise ValueError(f"unknown tokens: {unknown}.{hint} Run kx.grammar() to see valid ones.")
    kinds = [t for t in toks if t in KINDS]
    picked = [t for t in toks if t in themes]
    if len(kinds) > 1:
        raise ValueError(f"spec {spec!r} has more than one kind: {kinds}")
    if len(picked) > 1:
        raise ValueError(f"spec {spec!r} has more than one theme: {picked}")
    kind = (kinds or ["line"])[0]
    for flag in sorted(set(toks) & (flags - COMMON_FLAGS - KINDS[kind].flags)):
        owners = sorted(k.name for k in KINDS.values() if flag in k.flags)
        raise ValueError(f"flag {flag!r} only works with kind {owners}, not {kind!r}")
    for group in EXCLUSIVE:
        if len(both := sorted(group & set(toks))) > 1:
            raise ValueError(f"flags {both} do not go together; pick one")
    return Spec(kind, (picked or [None])[0], frozenset(toks) & flags)
