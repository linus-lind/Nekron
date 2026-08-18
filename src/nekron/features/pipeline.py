"""Sequential composition of featurizers into a feature panel."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from nekron.constants import DATE_LEVEL, ENTITY_LEVEL
from nekron.frames import assemble

from .base import (
    FeatureError,
    Featurizer,
    FloatArray,
    PanelContext,
    declared_keys,
    resolve_level,
)
from .selection import ColumnSelection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeaturePipeline:
    """Apply an ordered sequence of :class:`Featurizer` steps to a panel.

    The input is a preprocessed panel with a two-level ``(date, entity)``
    MultiIndex; the output is a feature panel sharing that index and row order.
    Featurizers run in order and each sees the columns produced by the ones before
    it, so a feature may depend on an earlier feature (for example a rolling
    volatility of a returns column, or a cross-sectional z-score of a raw signal).

    Because the steps run top to bottom, the working panel accumulates every
    produced column, some of which are only scaffolding for a later step. What the
    pipeline *returns* is a projection of that working panel, described by two
    independent selections: ``keep_inputs`` over the original panel columns and
    ``keep_features`` over the produced ones.

    Parameters
    ----------
    featurizers:
        The steps to apply, in dependency order. The order is validated on
        construction: no column may be produced twice, and no step may consume a
        column that a later step produces.
    entity_level, date_level:
        Position or name of the entity and date levels in the panel index.
    keep_inputs:
        Which original panel columns to retain alongside the features. The default
        keeps none; ``ColumnSelection()`` keeps them all, and an explicit
        ``include`` keeps a subset — a market-cap column a downstream model needs
        but no featurizer consumes, say. Retained columns pass through untouched:
        they are not cast to ``output_dtype`` and do not affect the warm-up trim,
        so a retained non-numeric column leaves the panel only partly numeric.
        Consumers treat every column of the result as a feature, so retain only
        what a model should actually see.
    keep_features:
        Which produced feature columns to output. The default keeps them all;
        naming a scaffolding feature in ``exclude`` still computes it, so later
        featurizers can consume it, but drops it from the result. The dtype cast
        and the warm-up trim below apply only to the retained features, so
        discarding a long-horizon intermediate does not cost extra burn-in dates.
        Because the featurizers declare their outputs, a stale name here is caught
        on construction; a stale ``keep_inputs`` name is caught against the panel,
        before any feature is computed.
    output_dtype:
        Optional dtype (e.g. ``"float32"``) to cast the feature columns to. Values
        are computed in ``float64`` and cast once at the end; ``None`` keeps
        ``float64``.
    drop_warmup:
        When ``True``, discard the leading burn-in dates: the earliest dates on
        which some retained feature is undefined across the whole cross-section
        (because it has not yet seen enough history — e.g. a 252-day feature is
        all-NaN for the first 252 dates). The number of dates dropped is the
        longest such leading run over all retained feature columns, so every kept
        date has every retained feature defined for at least one entity. Dropping
        is by date (all rows of a dropped date), never by row position.
    """

    featurizers: tuple[Featurizer, ...]
    entity_level: int | str = ENTITY_LEVEL
    date_level: int | str = DATE_LEVEL
    keep_inputs: ColumnSelection = ColumnSelection(include=())
    keep_features: ColumnSelection = ColumnSelection()
    output_dtype: str | None = None
    drop_warmup: bool = False

    def __post_init__(self) -> None:
        _validate_order(self.featurizers)
        declared = [name for step in self.featurizers for name in step.outputs]
        self.keep_features.validate(declared, what="feature columns")

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels < 2:
            raise FeatureError("feature creation requires a two-level (date, entity) MultiIndex.")
        entity_pos = resolve_level(panel.index, self.entity_level)
        date_pos = resolve_level(panel.index, self.date_level)
        retained = self.keep_inputs.select(list(panel.columns), what="panel columns")

        order = _entity_major_order(panel.index, entity_pos, date_pos)
        work = panel if order is None else panel.iloc[order]
        ctx = PanelContext.from_panel(work, entity_pos, date_pos)

        produced = self._run(work, ctx)
        kept = self.keep_features.select(list(produced), what="feature columns")
        n_produced = len(produced)
        # Drop what the selection discarded before assembling, so the discarded
        # arrays are released rather than held alongside the output.
        for name in [name for name in produced if name not in kept]:
            del produced[name]
        frame = self._assemble(produced, order, panel.index)
        if retained:
            frame = assemble(panel.index, [panel.loc[:, list(retained)], frame])
        if self.drop_warmup and kept:
            frame = self._drop_warmup(frame, list(kept), date_pos)
        logger.debug(
            "selection: kept %d of %d features and %d input columns",
            len(kept),
            n_produced,
            len(retained),
        )
        return frame

    def _drop_warmup(
        self, frame: pd.DataFrame, feature_names: list[str], date_pos: int
    ) -> pd.DataFrame:
        """Drop the leading dates on which some feature has no defined value yet.

        For each retained feature column, the first date on which it is defined for
        any entity marks the end of its warm-up; the panel is cut at the latest such
        date over all of them, discarding every row on the earlier dates. Columns
        that are never defined are ignored (they are a broken feature, not a
        warm-up). The MultiIndex is respected: the cut is by date value, not by
        row position.
        """
        date_values = frame.index.get_level_values(date_pos)
        defined_by_date = frame[feature_names].notna().groupby(date_values, sort=True).any()
        if defined_by_date.empty:
            return frame
        defined = defined_by_date.to_numpy()  # [n_dates, n_features], date-ascending
        ever_defined = defined.any(axis=0)
        if not ever_defined.any():
            return frame
        warmup = int(np.argmax(defined, axis=0)[ever_defined].max())
        if warmup <= 0:
            return frame
        ordered_dates = defined_by_date.index
        if warmup >= len(ordered_dates):
            return frame.iloc[:0]
        return cast("pd.DataFrame", frame.loc[date_values >= ordered_dates[warmup]])

    def _run(self, work: pd.DataFrame, ctx: PanelContext) -> dict[str, FloatArray]:
        """Run every featurizer, returning all produced columns before selection.

        Each step reads the working panel, which carries the original columns plus
        everything produced so far; scaffolding features are therefore available to
        later steps whether or not the caller keeps them in the result.
        """
        reserved = frozenset(work.columns)
        columns: dict[str, pd.Series] = {name: work[name] for name in work.columns}
        produced: dict[str, FloatArray] = {}
        for step in self.featurizers:
            self._require_inputs(columns, step.inputs, type(step).__name__)
            outputs = step.transform(self._view(columns, step, work.index), ctx)
            if set(outputs) != set(step.outputs):
                raise FeatureError(
                    f"featurizer {type(step).__name__} declares outputs {step.outputs} but "
                    f"produced {tuple(outputs)}."
                )
            for name, values in outputs.items():
                if name in reserved:
                    raise FeatureError(
                        f"feature {name!r} collides with an input panel column; rename the "
                        f"output of {type(step).__name__}."
                    )
                array = np.asarray(values, dtype=np.float64)
                if array.shape[0] != ctx.n_rows:
                    raise FeatureError(
                        f"featurizer {type(step).__name__} returned {array.shape[0]} rows for "
                        f"{name!r}; expected {ctx.n_rows}."
                    )
                produced[name] = array
                # One reference to the array, not two. Assigning into a growing
                # frame instead would copy it — doubling the peak for the whole
                # feature set — and fragment the block manager into one block per
                # feature, which slows every later step that reads it.
                columns[name] = pd.Series(array, index=work.index, copy=False)
            logger.debug("%s: produced %s", type(step).__name__, list(outputs))
        return produced

    @staticmethod
    def _require_inputs(columns: dict[str, pd.Series], names: tuple[str, ...], step: str) -> None:
        """Raise if a step reads a column the working set does not carry."""
        missing = [name for name in names if name not in columns]
        if missing:
            raise KeyError(
                f"featurizer {step} reads columns the panel does not carry: {missing}; "
                f"available: {sorted(columns)}."
            )

    @staticmethod
    def _view(columns: dict[str, pd.Series], step: Featurizer, index: pd.Index) -> pd.DataFrame:
        """The columns ``step`` declared it reads, as a frame over the shared arrays.

        Handing each step only its declared inputs keeps this constructor cheap —
        it wires up existing Series rather than copying anything — where handing it
        an ever-growing working frame would mean rebuilding that frame as features
        accumulate. A key a step declares may name an index level rather than a
        column, which needs nothing here: the level travels on ``index``.
        """
        wanted = dict.fromkeys((*step.inputs, *declared_keys(step)))
        return pd.DataFrame(
            {name: columns[name] for name in wanted if name in columns},
            index=index,
            copy=False,
        )

    def _assemble(
        self, features: dict[str, FloatArray], order: np.ndarray | None, index: pd.Index
    ) -> pd.DataFrame:
        """Build the output frame, consuming ``features`` one column at a time.

        Both the row-order restoration and the dtype cast allocate a new array per
        column, so doing them column by column and dropping each source as it is
        consumed keeps only one extra column alive at a time. Done frame-wide
        instead — a dict comprehension, then ``astype`` — each step holds a second
        copy of *every* feature at once, which is what made a narrower
        ``output_dtype`` raise peak memory rather than lower it.
        """
        if not features:
            return pd.DataFrame(index=index)
        inverse = None if order is None else np.argsort(order, kind="stable")
        data: dict[str, pd.Series] = {}
        for name in list(features):
            values = features.pop(name)
            if inverse is not None:
                values = values[inverse]
            if self.output_dtype is not None:
                values = values.astype(cast(Any, self.output_dtype), copy=False)
            data[name] = pd.Series(values, index=index, copy=False)
        return pd.DataFrame(data, index=index, copy=False)


def _validate_order(featurizers: tuple[Featurizer, ...]) -> None:
    """Check that the declared featurizer sequence is a valid dependency order.

    Featurizers run top to bottom, so a step may only read columns the panel
    already carries or an *earlier* step produced. Both structural failures — the
    same column produced twice, and a column consumed before the step producing it
    — are caught here, at construction, rather than after a panel has been read and
    the first features computed. Grouping keys count as reads: a key some later step
    produces is the same ordering mistake, while one the panel carries on its index
    is produced by nobody and constrains nothing.
    """
    producer: dict[str, int] = {}
    for position, step in enumerate(featurizers):
        for name in step.outputs:
            if name in producer:
                raise FeatureError(
                    f"feature {name!r} is produced by more than one featurizer "
                    f"(positions {producer[name]} and {position})."
                )
            producer[name] = position
    for position, step in enumerate(featurizers):
        for name in (*step.inputs, *declared_keys(step)):
            produced_at = producer.get(name)
            if produced_at is not None and produced_at > position:
                raise FeatureError(
                    f"featurizer {type(step).__name__} at position {position} reads {name!r}, "
                    f"which is produced later at position {produced_at}; featurizers run top to "
                    f"bottom, so move that step earlier — or rename its output if {name!r} is "
                    "meant to be a panel column."
                )


def _entity_major_order(index: pd.MultiIndex, entity_pos: int, date_pos: int) -> np.ndarray | None:
    """Row order that makes the panel entity-major (``(entity, date)`` ascending).

    Returns ``None`` when the panel is already in that order, so no reordering
    work is done for the common already-sorted case.
    """
    entity_key = pd.factorize(index.get_level_values(entity_pos), sort=True)[0]
    date_key = pd.factorize(index.get_level_values(date_pos), sort=True)[0]
    if _is_entity_major(entity_key, date_key):
        return None
    return np.lexsort((date_key, entity_key))


def _is_entity_major(entity_key: np.ndarray, date_key: np.ndarray) -> bool:
    """True if entities are already contiguous with dates ascending within each."""
    if entity_key.shape[0] <= 1:
        return True
    if not bool(np.all(entity_key[1:] >= entity_key[:-1])):
        return False
    same_entity = entity_key[1:] == entity_key[:-1]
    return bool(np.all(date_key[1:][same_entity] >= date_key[:-1][same_entity]))
