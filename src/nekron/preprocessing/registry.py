"""Registry mapping a preprocessing step type name to a transform builder.

Preprocessing configurations name a transform by a short string and pass a
parameter mapping; :func:`build_transform` looks up the builder and instantiates
the transform. Adding a transform means writing a :class:`~.base.PanelTransform`
and registering a builder here under a name — no calling code changes.

Most transforms map their params directly onto a frozen dataclass (list values
are coerced to the tuples the dataclasses expect). The two composite transforms
(:class:`ConsistencyFilter`, :class:`CorporateAdjustment`) build their nested
rule / adjustment objects from lists of mappings.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .adjustments import Adjustment, CorporateAdjustment, CumulativeFactor
from .base import PanelTransform, PreprocessingError
from .calendar import TradingCalendarFilter
from .consistency import ConsistencyFilter, ConsistencyRule
from .deduplicate import DuplicateMerger
from .imputation import ConstantImputer, ForwardFillImputer
from .nan_filter import NaNEntityFilter

TransformBuilder = Callable[[Mapping[str, Any]], PanelTransform]

_REGISTRY: dict[str, TransformBuilder] = {}


def register_transform(name: str, builder: TransformBuilder) -> None:
    """Register a transform builder under a type ``name`` (overwrites in place)."""
    _REGISTRY[name] = builder


def registered_transforms() -> tuple[str, ...]:
    """Return the sorted names of all registered transform types."""
    return tuple(sorted(_REGISTRY))


def build_transform(name: str, params: Mapping[str, Any]) -> PanelTransform:
    """Construct the transform registered under ``name`` from ``params``."""
    try:
        builder = _REGISTRY[name]
    except KeyError:
        raise PreprocessingError(
            f"unknown transform type {name!r}; registered types: {registered_transforms()}."
        ) from None
    try:
        return builder(params)
    except (TypeError, KeyError, ValueError) as exc:
        raise PreprocessingError(f"cannot build transform {name!r}: {exc}.") from exc


def _as_tuples(params: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce list-valued params to tuples to match frozen-dataclass field types."""
    return {
        key: tuple(value) if isinstance(value, list) else value for key, value in params.items()
    }


def _direct(cls: Callable[..., PanelTransform]) -> TransformBuilder:
    """Builder for a transform whose params map directly onto its constructor."""

    def build(params: Mapping[str, Any]) -> PanelTransform:
        return cls(**_as_tuples(params))

    return build


def _build_consistency(params: Mapping[str, Any]) -> PanelTransform:
    rules = tuple(ConsistencyRule(**rule) for rule in params["rules"])
    return ConsistencyFilter(rules=rules)


def _build_corporate_adjustment(params: Mapping[str, Any]) -> PanelTransform:
    adjustments = tuple(
        Adjustment(
            columns=tuple(adjustment["columns"]),
            factor_column=adjustment["factor_column"],
            operation=adjustment["operation"],
        )
        for adjustment in params["adjustments"]
    )
    return CorporateAdjustment(adjustments=adjustments)


_DEFAULTS: dict[str, TransformBuilder] = {
    "trading_calendar": _direct(TradingCalendarFilter),
    "deduplicate": _direct(DuplicateMerger),
    "nan_entity_filter": _direct(NaNEntityFilter),
    "cumulative_factor": _direct(CumulativeFactor),
    "corporate_adjustment": _build_corporate_adjustment,
    "consistency": _build_consistency,
    "forward_fill": _direct(ForwardFillImputer),
    "constant_fill": _direct(ConstantImputer),
}

for _name, _builder in _DEFAULTS.items():
    register_transform(_name, _builder)
