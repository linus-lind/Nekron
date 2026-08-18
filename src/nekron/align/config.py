"""Typed Hydra configuration for the alignment stage.

The stage is declared as one spine plus an ordered list of joins. Each join names
the source panel, how its rows reach the spine's rows (an aligner ``{type,
params}`` resolved through :mod:`~nekron.align.registry`), which of its columns to
take, and — when the source is keyed by a different identifier — how to translate
that identifier into the spine's.

Column naming is explicit on purpose. Merging several vendor panels makes name
collisions a certainty (``ret`` exists in most of them), and silently suffixing
one of them to ``ret_y`` is how a model ends up trained on the wrong column. A
join therefore carries a ``prefix`` and a ``rename`` map, and an unresolved
collision is an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class AlignerSpec:
    """One aligner: its registered ``type`` and constructor ``params``."""

    type: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyNormalizationConfig:
    """How one join-key column is normalized before matching.

    Only :attr:`strip` and :attr:`empty_as_missing` are safe to apply blindly.
    :attr:`upper` merges identifiers whose case is significant (a Compustat issue
    id, a share class), and :attr:`zfill` must be set per column — zero-padding
    every key to a common width destroys short identifiers outright.
    """

    strip: bool = True
    upper: bool = False
    zfill: int | None = None
    empty_as_missing: bool = True


@dataclass
class LinkSpec:
    """How to translate a source panel's entity identifier into the spine's.

    Parameters
    ----------
    mode:
        ``"identity"`` — the panel already carries the spine's identifier in the
        column named by :attr:`target`, so this is a rename and a membership
        check. ``"table"`` — resolve through the named link-table panel.
        ``"spine"`` — resolve against identifier columns the *spine itself*
        carries, by projecting them into validity windows.
    table:
        Name of the link-table panel, required when ``mode`` is ``"table"``.
    source_keys:
        Columns (or index levels) of the source panel that form the join key. More
        than one makes it a composite key, e.g. ``[gvkey, iid]`` or
        ``[ticker, cusip]``.
    table_keys:
        The matching columns on the link table or spine; defaults to
        :attr:`source_keys` when the names agree.
    target:
        Column holding the spine's entity identifier — on the link table for
        ``"table"``, on the source panel itself for ``"identity"``.
    valid_from, valid_to:
        Validity-window columns on the link table. Leaving both ``null`` makes the
        link time-invariant; a null ``valid_to`` value within the column means the
        row is still valid.
    normalize:
        Per-key-column normalization.
    on_ambiguous:
        What to do when one key resolves to several targets at the same date:
        ``"error"`` (default), or ``"first"``/``"last"`` to canonicalize the
        overlapping windows by which starts earlier.
    on_unmatched:
        What to do with source rows whose key does not resolve: ``"drop"``,
        ``"keep"`` (leaving the row unlinked), or ``"error"``.
    on_collision:
        What to do when two source rows land on the same ``(date, entity)`` after
        linking — a merger, typically: ``"error"`` (default), ``"first"``,
        ``"last"``, ``"sum"`` or ``"mean"``.
    """

    mode: str = "table"
    table: str | None = None
    source_keys: list[str] = field(default_factory=list)
    table_keys: list[str] | None = None
    target: str = MISSING
    valid_from: str | None = None
    valid_to: str | None = None
    normalize: dict[str, KeyNormalizationConfig] = field(default_factory=dict)
    on_ambiguous: str = "error"
    on_unmatched: str = "drop"
    on_collision: str = "error"


@dataclass
class JoinSpec:
    """One source panel attached to the spine.

    Parameters
    ----------
    panel:
        Name of the source panel in the ingested :class:`~nekron.panel.PanelSet`.
    aligner:
        How its rows reach the spine's rows.
    link:
        Identifier translation applied before alignment; ``null`` when the source
        is already keyed by the spine's identifier.
    columns:
        Columns to take from the source; ``null`` takes all of them.
    exclude:
        Columns to drop from that set.
    prefix:
        Prepended to every taken column name, which is the cheap way to keep a
        vendor's namespace separate.
    rename:
        Explicit per-column renames, applied after :attr:`prefix`.
    required:
        When true, a source key that matches no spine row at all is an error
        rather than a column of nulls.
    min_coverage:
        Minimum fraction of spine rows that must receive a non-null value from
        this join. A botched link table shows up here immediately instead of forty
        epochs into training. ``0.0`` disables the check.
    """

    panel: str = MISSING
    aligner: AlignerSpec = field(default_factory=AlignerSpec)
    link: LinkSpec | None = None
    columns: list[str] | None = None
    exclude: list[str] = field(default_factory=list)
    prefix: str = ""
    rename: dict[str, str] = field(default_factory=dict)
    required: bool = False
    min_coverage: float = 0.0


@dataclass
class AlignmentConfig:
    """Root configuration for the alignment stage.

    Parameters
    ----------
    spine:
        Name of the panel whose ``(date, entity)`` index the merged panel takes.
        Every other panel is reindexed onto it, so the merge can only add columns —
        never rows, and never a reordering.
    joins:
        The sources attached to the spine, in order. Order affects only the column
        order of the result.
    spine_columns, spine_exclude:
        Which of the spine's own columns to carry into the merged panel; ``null``
        keeps them all.
    """

    spine: str = MISSING
    joins: list[JoinSpec] = field(default_factory=list)
    spine_columns: list[str] | None = None
    spine_exclude: list[str] = field(default_factory=list)


CONFIG_SCHEMA_NAME = "base_alignment"


def register_configs() -> None:
    """Register the alignment schema with Hydra's ConfigStore (call before @hydra.main)."""
    ConfigStore.instance().store(name=CONFIG_SCHEMA_NAME, node=AlignmentConfig)


def to_config(cfg: DictConfig) -> AlignmentConfig:
    """Materialize a composed Hydra/OmegaConf config into a typed ``AlignmentConfig``."""
    return cast(AlignmentConfig, OmegaConf.to_object(cfg))
