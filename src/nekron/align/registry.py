"""Registry mapping an alignment-semantic name to a :class:`PanelAligner` factory.

Adding a join semantic means writing a class that satisfies
:class:`~nekron.align.base.PanelAligner` and registering it here under a name — no
calling code changes. This mirrors the ingestion filter and preprocessing
transform registries, so a configuration entry looks the same everywhere in the
project: a short ``type`` plus a ``params`` mapping.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .aligners import ALIGNERS
from .base import AlignmentError, PanelAligner

AlignerBuilder = Callable[[Mapping[str, Any]], PanelAligner]

_REGISTRY: dict[str, AlignerBuilder] = {}


def register_aligner(name: str, builder: AlignerBuilder) -> None:
    """Register an aligner builder under a type ``name`` (overwrites in place)."""
    _REGISTRY[name] = builder


def registered_aligners() -> tuple[str, ...]:
    """Return the sorted names of all registered aligner types."""
    return tuple(sorted(_REGISTRY))


def build_aligner(name: str, params: Mapping[str, Any]) -> PanelAligner:
    """Construct the aligner registered under ``name`` from ``params``."""
    try:
        builder = _REGISTRY[name]
    except KeyError:
        raise AlignmentError(
            f"unknown aligner type {name!r}; registered types: {registered_aligners()}."
        ) from None
    try:
        return builder(params)
    except (TypeError, ValueError, KeyError) as exc:
        raise AlignmentError(f"cannot build aligner {name!r}: {exc}.") from exc


def _as_tuples(params: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce list-valued params to tuples to match frozen-dataclass field types."""
    return {
        key: tuple(value) if isinstance(value, list) else value for key, value in params.items()
    }


def _direct(cls: Callable[..., PanelAligner]) -> AlignerBuilder:
    """Builder for an aligner whose params map directly onto its constructor."""

    def build(params: Mapping[str, Any]) -> PanelAligner:
        return cls(**_as_tuples(params))

    return build


for _name, _cls in ALIGNERS.items():
    register_aligner(_name, _direct(_cls))
