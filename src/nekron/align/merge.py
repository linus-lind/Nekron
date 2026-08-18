"""Merging a set of named panels onto one spine."""

from __future__ import annotations

import logging

import pandas as pd

from nekron.frames import assemble
from nekron.panel import Panel, PanelGrain, PanelSet
from nekron.selection import ColumnSelection

from .base import AlignmentError, PanelAligner, Spine, required_columns
from .config import AlignmentConfig, JoinSpec, LinkSpec
from .linking import KeyNormalization, LinkTable, relink
from .registry import build_aligner

logger = logging.getLogger(__name__)


def merge_panels(panels: PanelSet, cfg: AlignmentConfig) -> Panel:
    """Reindex every configured source panel onto the spine and return one panel.

    The result carries exactly the spine's rows, in the spine's order. Each join
    contributes columns and nothing else, so a source that resolves badly shows up
    as nulls in its own columns rather than as missing or duplicated rows.
    """
    spine_panel = panels[cfg.spine]
    spine = Spine.from_panel(spine_panel)

    blocks: list[pd.DataFrame] = [
        _project(
            spine_panel.frame,
            include=cfg.spine_columns,
            exclude=cfg.spine_exclude,
            what=cfg.spine,
        )
    ]
    taken: set[str] = set(blocks[0].columns)

    for join in cfg.joins:
        block = _align_one(panels, spine, cfg.spine, join)
        collisions = sorted(taken.intersection(block.columns))
        if collisions:
            raise AlignmentError(
                f"join on panel {join.panel!r} would overwrite existing columns {collisions}; "
                "set a prefix or an explicit rename on the join."
            )
        taken.update(str(column) for column in block.columns)
        blocks.append(block)

    merged = assemble(spine.index, blocks)
    logger.info(
        "merged %d panel(s) onto spine %r: %d rows x %d columns",
        len(cfg.joins) + 1,
        cfg.spine,
        len(merged),
        merged.shape[1],
    )
    return spine_panel.with_frame(merged)


def _align_one(panels: PanelSet, spine: Spine, spine_name: str, join: JoinSpec) -> pd.DataFrame:
    """Resolve, project, align and rename one source panel."""
    source = panels[join.panel]
    if join.link is not None:
        source = _relink(panels, spine, spine_name, source, join)

    aligner = build_aligner(join.aligner.type, join.aligner.params)
    carried = _select(source.frame, include=join.columns, exclude=join.exclude, what=join.panel)
    # The aligner may read a column the caller did not ask to carry — a point-in-time
    # join reads the availability date. Keep it for the gather, then drop it, so the
    # expensive full-length take still touches only the columns that matter.
    helpers = tuple(name for name in required_columns(aligner) if name not in carried)
    _require_present(source.frame, helpers, join)
    projected = source.with_frame(source.frame.loc[:, [*carried, *helpers]])

    _require_supported(aligner, projected, join)
    block = aligner.align(spine, projected)
    if helpers:
        block = block.loc[:, list(carried)]
    block = _rename(block, join)
    _check_coverage(block, join)
    return block


def _relink(
    panels: PanelSet, spine: Spine, spine_name: str, source: Panel, join: JoinSpec
) -> Panel:
    """Rewrite ``source``'s entity key into the spine's identifier space."""
    spec = join.link
    assert spec is not None
    link = _build_link(panels, spine, spine_name, spec)
    relinked, report = relink(
        source,
        link,
        entity_name=spine.entity_name,
        on_unmatched=spec.on_unmatched,  # type: ignore[arg-type]
        on_collision=spec.on_collision,  # type: ignore[arg-type]
    )
    logger.info("linked panel %r: %s", join.panel, report)
    return relinked


def _build_link(panels: PanelSet, spine: Spine, spine_name: str, spec: LinkSpec) -> LinkTable:
    """Construct the :class:`LinkTable` a join's link spec describes."""
    normalize = {
        column: KeyNormalization(
            strip=item.strip,
            upper=item.upper,
            zfill=item.zfill,
            empty_as_missing=item.empty_as_missing,
        )
        for column, item in spec.normalize.items()
    }
    keys = tuple(spec.source_keys)
    table_keys = tuple(spec.table_keys) if spec.table_keys is not None else keys
    if len(table_keys) != len(keys):
        raise AlignmentError(
            f"link source_keys {list(keys)} and table_keys {list(table_keys)} must name the "
            "same number of columns; they are matched pairwise, in order."
        )

    if spec.mode == "identity":
        return LinkTable.identity(column=spec.target, target=spine.entity_name)
    if spec.mode == "spine":
        return LinkTable.from_spine(
            _as_source_keys(panels[spine_name].frame, table_keys, keys),
            keys,
            date_name=spine.date_name,
            entity_name=spine.entity_name,
            normalize=normalize,
            on_ambiguous=spec.on_ambiguous,
        )
    if spec.mode != "table":
        raise AlignmentError(
            f"unknown link mode {spec.mode!r}; expected one of 'table', 'spine', 'identity'."
        )
    if spec.table is None:
        raise AlignmentError("link mode 'table' requires the name of a link-table panel.")
    return LinkTable(
        table=_as_source_keys(panels[spec.table].frame, table_keys, keys),
        source_keys=keys,
        target=spec.target,
        valid_from=spec.valid_from,
        valid_to=spec.valid_to,
        normalize=normalize,
        on_ambiguous=spec.on_ambiguous,  # type: ignore[arg-type]
    )


def _as_source_keys(
    frame: pd.DataFrame, table_keys: tuple[str, ...], source_keys: tuple[str, ...]
) -> pd.DataFrame:
    """Rename a link table's key columns to the names the source panel uses.

    A link table names its columns the way its vendor does — ``gvkey``, ``lpermno``
    — while the panel being relinked knows them by whatever its own schema called
    them. The resolver matches by name, so the two vocabularies are reconciled
    once, here, rather than forcing every source to be renamed to match its link
    file.
    """
    renames = {
        table: source
        for table, source in zip(table_keys, source_keys, strict=True)
        if table != source
    }
    if not renames:
        return frame
    missing = [column for column in renames if column not in frame.columns]
    if missing:
        raise AlignmentError(
            f"the link table is missing its key column(s) {missing}; it has {list(frame.columns)}."
        )
    collisions = [name for name in renames.values() if name in frame.columns]
    if collisions:
        raise AlignmentError(
            f"renaming the link table's keys to {collisions} would collide with columns it "
            "already has; rename them in the link table's own selection instead."
        )
    return frame.rename(columns=renames)


def _select(
    frame: pd.DataFrame, *, include: list[str] | None, exclude: list[str], what: str
) -> tuple[str, ...]:
    """Resolve the configured column selection, failing loudly on a stale name."""
    selection = ColumnSelection(
        include=tuple(include) if include is not None else None,
        exclude=tuple(exclude),
        strict=True,
    )
    return selection.select(
        [str(column) for column in frame.columns], what=f"columns of panel {what!r}"
    )


def _project(
    frame: pd.DataFrame, *, include: list[str] | None, exclude: list[str], what: str
) -> pd.DataFrame:
    """Select the configured columns, failing loudly on a stale name."""
    return frame.loc[:, list(_select(frame, include=include, exclude=exclude, what=what))]


def _require_present(frame: pd.DataFrame, columns: tuple[str, ...], join: JoinSpec) -> None:
    """Reject an aligner whose own inputs are absent from the source panel."""
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise AlignmentError(
            f"aligner {join.aligner.type!r} reads {missing} from panel {join.panel!r}, but that "
            f"panel has {list(frame.columns)}; add the column to the panel's "
            "selection.columns so it is loaded."
        )


def _require_supported(aligner: PanelAligner, panel: Panel, join: JoinSpec) -> None:
    """Reject an aligner/grain pairing before any work is done."""
    if panel.grain in aligner.accepts:
        return
    accepted = ", ".join(sorted(grain.value for grain in aligner.accepts))
    raise AlignmentError(
        f"aligner {join.aligner.type!r} accepts a {accepted} panel, but {join.panel!r} is "
        f"{panel.grain.value}. {_grain_hint(panel.grain)}"
    )


def _grain_hint(grain: PanelGrain) -> str:
    """Point at the aligner that does fit a given grain."""
    hints = {
        PanelGrain.PANEL: "use 'exact' or 'asof_entity' for a (date, entity) panel.",
        PanelGrain.TIME_SERIES: "use 'broadcast_time' for a date-keyed series.",
        PanelGrain.CROSS_SECTION: "use 'broadcast_entity' for an entity-keyed frame.",
        PanelGrain.TABLE: "a keyless table cannot be aligned directly; reference it as a link table.",
    }
    return hints[grain]


def _rename(block: pd.DataFrame, join: JoinSpec) -> pd.DataFrame:
    """Apply the join's prefix and explicit renames, rejecting a collision."""
    if not join.prefix and not join.rename:
        return block
    names = [f"{join.prefix}{column}" for column in block.columns]
    names = [join.rename.get(name, name) for name in names]
    if len(set(names)) != len(names):
        duplicated = sorted({name for name in names if names.count(name) > 1})
        raise AlignmentError(
            f"renaming the columns of panel {join.panel!r} produced duplicates {duplicated}."
        )
    block.columns = pd.Index(names)
    return block


def _check_coverage(block: pd.DataFrame, join: JoinSpec) -> None:
    """Verify the join actually reached the spine, when the config asks.

    Skipped entirely unless a check is configured: measuring coverage is a full
    pass over the block, which is not worth paying for by default.
    """
    if not join.required and join.min_coverage <= 0.0:
        return
    if block.empty:
        covered = 0.0
    else:
        covered = float(block.notna().any(axis=1).mean())
    if join.required and covered == 0.0:
        raise AlignmentError(
            f"join on panel {join.panel!r} is marked required but matched no spine row; "
            "check the identifiers and the date ranges on both sides."
        )
    if covered < join.min_coverage:
        raise AlignmentError(
            f"join on panel {join.panel!r} covered {covered:.1%} of spine rows, below the "
            f"configured minimum of {join.min_coverage:.1%}."
        )
    logger.info("join on panel %r covered %.1f%% of spine rows", join.panel, covered * 100)
