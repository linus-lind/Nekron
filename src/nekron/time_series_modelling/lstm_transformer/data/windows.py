"""Residual series to fixed-length windows, and windows to train/validation/test.

One training example is ``seq_len`` consecutive periods of one entity's residual
series. A window exists only where that entity has a residual on *every* one of
those periods: a missing date ends the run and the count starts again after it.
An entity that leaves the investable universe for a week therefore contributes no
window spanning that week, rather than one with a hole interpolated into it.

Windows are an index, not a copy
--------------------------------
Materializing the windows would be catastrophic and is unnecessary. A daily panel
of a thousand names over fifteen years carries a few million valid windows; at
252 steps each, storing them costs hundreds of gigabytes, while the residual
matrix they are all cut from is a few dozen megabytes. So the matrix is stored
once, on the training device, and a window is two integers — the entity's column
and the period the window ends on. A batch is one gather of those integers
against the shared matrix, which never leaves the device and never crosses the
host boundary during training.

Which split a window belongs to
-------------------------------
The split containing its **last** date. The preceding ``seq_len - 1`` periods may
reach back across a boundary, and that is deliberate: a window's lookback lies
entirely in the past of the date it is assigned to, so nothing later than that
date is ever read, while demanding full containment would cost ``seq_len - 1``
windows at every boundary — at the default 252 that is most of a one-year
validation window.

A target, when one is configured, is held to the stricter rule: its date must
fall inside the same split. A window at the very end of the training period would
otherwise be labelled from the first day of validation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from nekron.cv import plan_schedule
from nekron.data_adapter.base import AdapterError
from nekron.data_adapter.splitting import FoldBounds

from ..config import DataConfig
from .residuals import ResidualPanel

logger = logging.getLogger(__name__)

BoolMatrix = npt.NDArray[np.bool_]

# Cells per chunk of the validity scan. The scan needs one int32 scratch matrix of
# the shape it is working on, so doing it in column blocks bounds that scratch at
# a few megabytes however wide the panel becomes.
_SCAN_CHUNK_CELLS = 1 << 22


class WindowError(Exception):
    """Raised when a residual panel admits no usable windows."""


@dataclass(frozen=True)
class WindowBatch:
    """One batch of residual windows, already on the training device.

    Parameters
    ----------
    sequences:
        ``[batch, seq_len, channels]`` residuals, oldest step first.
    date_indices:
        ``[batch]`` positions of each window's **last** period in
        :attr:`ResidualWindows.dates`.
    entity_indices:
        ``[batch]`` positions in :attr:`ResidualWindows.entity_names`.
    targets:
        ``[batch]`` residual ``target_horizon`` periods after the window, or
        ``None`` when no target is configured.
    """

    sequences: torch.Tensor
    date_indices: torch.Tensor
    entity_indices: torch.Tensor
    targets: torch.Tensor | None

    def __len__(self) -> int:
        return int(self.sequences.shape[0])


@dataclass(frozen=True)
class ResidualWindows:
    """One split's windows as an index into a residual matrix shared by all splits.

    Nothing here is a :class:`torch.utils.data.Dataset`, deliberately. A ``Dataset``
    exists to hand single examples to a loader that batches and collates them on
    the host, and every step of that is wasted work when the whole population is
    already one resident tensor: batching *is* the gather, and the gather is one
    indexing operation on the device.

    Parameters
    ----------
    values:
        The shared ``[periods, entities]`` residual matrix, on the training
        device. Absent cells are ``NaN`` and are never indexed — a ``NaN``
        reaching a batch means a window index is wrong, which is exactly the
        failure that should be loud.
    ends, entity_indices:
        The window index: each pair is one window, ending at period ``ends[m]``
        for entity ``entity_indices[m]``.
    """

    values: torch.Tensor
    ends: torch.Tensor
    entity_indices: torch.Tensor
    seq_len: int
    target_horizon: int
    dates: pd.DatetimeIndex
    entity_names: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.ends.shape[0])

    @property
    def num_channels(self) -> int:
        """Channels per step. One: the residual series is univariate."""
        return 1

    @property
    def device(self) -> torch.device:
        return self.values.device

    def span(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """First and last period any window in this split ends on."""
        if len(self) == 0:
            return None
        return (
            pd.Timestamp(self.dates[int(self.ends.min())]),
            pd.Timestamp(self.dates[int(self.ends.max())]),
        )

    def gather(self, selection: torch.Tensor) -> WindowBatch:
        """Materialize the windows at positions ``selection`` as one batch."""
        ends = self.ends[selection]
        entities = self.entity_indices[selection]
        offsets = torch.arange(self.seq_len, device=self.values.device)
        rows = ends.unsqueeze(1) - (self.seq_len - 1) + offsets  # [B, L]
        sequences = self.values[rows, entities.unsqueeze(1)]  # [B, L]
        targets = (
            self.values[ends + self.target_horizon, entities] if self.target_horizon > 0 else None
        )
        return WindowBatch(
            sequences=sequences.unsqueeze(-1),
            date_indices=ends,
            entity_indices=entities,
            targets=targets,
        )

    def iter_batches(
        self, batch_size: int, *, shuffle: bool, generator: torch.Generator | None = None
    ) -> Iterator[WindowBatch]:
        """Yield every window once, in batches of at most ``batch_size``.

        The permutation is drawn once per epoch and moved to the device in one
        transfer; the per-example work a host-side loader would do — one gather,
        one collate and one copy per window — does not happen at all.

        It is drawn on the CPU rather than on the training device because a
        :class:`torch.Generator` belongs to one device, and seeding a run must not
        depend on which accelerator it lands on: the same seed then gives the same
        epoch order on CPU, CUDA and MPS alike.
        """
        total = len(self)
        if total == 0:
            return
        order = (
            torch.randperm(total, generator=generator).to(self.values.device)
            if shuffle
            else torch.arange(total, device=self.values.device)
        )
        for start in range(0, total, batch_size):
            yield self.gather(order[start : start + batch_size])


@dataclass(frozen=True)
class WindowSplits:
    """The three splits, plus the panel and schedule they were cut from."""

    train: ResidualWindows
    val: ResidualWindows
    test: ResidualWindows
    bounds: FoldBounds
    panel: ResidualPanel

    @property
    def num_channels(self) -> int:
        return self.train.num_channels

    @property
    def seq_len(self) -> int:
        return self.train.seq_len

    def describe(self) -> str:
        """Two lines: the window counts, and the dates each split scores."""
        counts = " ".join(
            f"{name}={len(split):,}"
            for name, split in (("train", self.train), ("val", self.val), ("test", self.test))
        )
        spans = " | ".join(
            f"{name} {_span_text(split)}"
            for name, split in (("train", self.train), ("val", self.val), ("test", self.test))
        )
        return f"windows -> {counts} (seq_len={self.seq_len})\n           {spans}"


def _span_text(split: ResidualWindows) -> str:
    span = split.span()
    return "empty" if span is None else f"{span[0].date()}..{span[1].date()}"


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WindowPanel:
    """Every valid window over a residual panel, built once, sliced per fold.

    The counterpart to
    :class:`~nekron.asset_pricing.conditional_autoencoder.data.dataset.CrossSectionPanel`,
    and it exists for the same reason: cross-validation folds overlap, so the
    expensive parts — the residual matrix on the device and the validity scan over
    it — are computed once and every fold takes a constant-time
    :meth:`splits` view onto them. Rebuilding them per fold would re-scan the whole
    matrix once per fold that covers it and hold a private copy of the tensors for
    each.

    :attr:`dates` is the periods that carry residuals, which is the sequence a
    schedule must be cut over.
    """

    values: torch.Tensor
    finite: BoolMatrix
    valid: BoolMatrix
    panel: ResidualPanel
    cfg: DataConfig

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.panel.dates

    @property
    def num_channels(self) -> int:
        return 1

    def splits(self, bounds: FoldBounds) -> WindowSplits:
        """One fold's three window sets. Constant time — nothing is re-scanned."""
        splits = {
            name: _windows_in(self.values, self.finite, self.valid, span, self.panel, self.cfg)
            for name, span in (
                ("train", bounds.train),
                ("val", bounds.val),
                ("test", bounds.test),
            )
        }
        if len(splits["train"]) == 0:
            raise WindowError(
                f"fold {bounds.index}: the train segment ({len(bounds.train)} periods) yields "
                f"no complete window of {self.cfg.seq_len} periods. Shorten data.seq_len, lower "
                "data.stride, or check that the residual panel is dense enough over that span."
            )
        for name in ("val", "test"):
            if len(splits[name]) == 0:
                logger.warning(
                    "fold %d: the %s split has no windows; its segment spans %d periods "
                    "against a seq_len of %d.",
                    bounds.index,
                    name,
                    len(getattr(bounds, name)),
                    self.cfg.seq_len,
                )
        return WindowSplits(
            train=splits["train"],
            val=splits["val"],
            test=splits["test"],
            bounds=bounds,
            panel=self.panel,
        )


def build_window_panel(
    panel: ResidualPanel, cfg: DataConfig, *, device: torch.device
) -> WindowPanel:
    """Move the residual matrix to ``device`` and find every window end on it."""
    if cfg.seq_len > panel.num_periods:
        raise WindowError(
            f"data.seq_len ({cfg.seq_len}) exceeds the {panel.num_periods} periods that carry "
            "residuals; no window can be formed."
        )
    finite = np.isfinite(panel.values)
    valid = _window_ends(finite, cfg.seq_len)
    if cfg.stride > 1:
        valid &= (np.arange(panel.num_periods) % cfg.stride == 0)[:, None]
    return WindowPanel(
        values=torch.from_numpy(np.ascontiguousarray(panel.values)).to(device),
        finite=finite,
        valid=valid,
        panel=panel,
        cfg=cfg,
    )


def window_schedule(panel: ResidualPanel, cfg: DataConfig) -> tuple[FoldBounds, ...]:
    """The fold schedule ``cfg`` describes over the periods that carry residuals.

    Cut over the residual panel's own dates rather than the raw panel's, because
    that is the sequence the window positions index; cutting over the panel's would
    shift every boundary after the first period the autoencoder could not price.

    The same sequence is passed twice, and that is correct rather than lazy: an
    inherited split was already re-expressed in the residual sequence's units when
    the panel was built (see :func:`~..data.residuals.build_residual_panel`), and a
    configured one is written against those dates to begin with. Rebasing a second
    time would convert a burn-in that has already been converted.
    """
    split_cfg = panel.split if cfg.inherit_cae_split else cfg.split
    try:
        return plan_schedule(panel.dates, panel.dates, split_cfg, what="residual")
    except AdapterError as exc:
        raise WindowError(f"could not cut a schedule over the residual periods: {exc}") from exc


def build_window_splits(
    panel: ResidualPanel, cfg: DataConfig, *, device: torch.device
) -> WindowSplits:
    """Cut ``panel`` into one fold's train/validation/test windows on ``device``.

    The single-split entry point. A schedule with more than one fold has no single
    answer to return, so this raises and points at the sweep.
    """
    schedule = window_schedule(panel, cfg)
    if len(schedule) != 1:
        raise WindowError(
            f"the configured schedule produced {len(schedule)} folds, and a single "
            "train/val/test split is only defined for one. Use the cross-validated "
            "entry point (engine.run_folds) and iterate the schedule instead."
        )
    return build_window_panel(panel, cfg, device=device).splits(schedule[0])


def _window_ends(finite: BoolMatrix, seq_len: int) -> BoolMatrix:
    """Mask of ``(period, entity)`` cells that end a gap-free run of ``seq_len``.

    The run length ending at each cell is ``t - g``, where ``g`` is the most recent
    period at or before ``t`` at which that entity had no residual — a running
    maximum, so the whole matrix is one accumulate rather than a scan per entity.
    Worked in column blocks: the accumulate needs an integer scratch matrix the
    shape of its input, and bounding that is what keeps the pass flat in memory as
    the entity axis grows.
    """
    periods, entities = finite.shape
    positions = np.arange(periods, dtype=np.int32)[:, None]
    valid = np.empty((periods, entities), dtype=np.bool_)
    width = max(1, _SCAN_CHUNK_CELLS // max(periods, 1))
    for start in range(0, entities, width):
        block = finite[:, start : start + width]
        # -1 at an observed cell, the period index at a gap: the running maximum is
        # then the most recent gap, and -1 leaves it correct before the first one.
        scratch = np.where(block, np.int32(-1), positions)
        np.maximum.accumulate(scratch, axis=0, out=scratch)
        np.subtract(positions, scratch, out=scratch)
        valid[:, start : start + width] = scratch >= seq_len
    return valid


def _windows_in(
    values: torch.Tensor,
    finite: BoolMatrix,
    valid: BoolMatrix,
    span: range,
    panel: ResidualPanel,
    cfg: DataConfig,
) -> ResidualWindows:
    """The windows whose last period falls inside ``span``."""
    horizon = cfg.target_horizon
    low = span.start
    # A target must land inside this same segment, so the last usable end position
    # moves back by the horizon rather than merely staying inside the panel.
    high = max(low, span.stop - horizon)
    selected = valid[low:high]
    if horizon > 0 and high > low:
        # ...and the target cell must itself carry a residual.
        selected = selected & finite[low + horizon : high + horizon]
    rows, columns = np.nonzero(selected)
    return ResidualWindows(
        values=values,
        # Built on the device the matrix already lives on, so a fold's index costs
        # one transfer rather than one per batch.
        ends=torch.from_numpy((rows + low).astype(np.int64)).to(values.device),
        entity_indices=torch.from_numpy(columns.astype(np.int64)).to(values.device),
        seq_len=cfg.seq_len,
        target_horizon=horizon,
        dates=panel.dates,
        entity_names=panel.entities,
    )
