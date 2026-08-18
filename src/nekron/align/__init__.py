"""Alignment of several named panels onto one spine.

One panel is the spine and defines the ``(date, entity)`` index of the result;
every other panel is reindexed onto it by a named aligner, optionally after its
entity identifier has been translated through a link table. The merge therefore
only ever adds columns — never rows, and never a reordering.
"""

from __future__ import annotations

from .aligners import (
    ALIGNERS,
    AsOfEntityAligner,
    BroadcastEntityAligner,
    BroadcastTimeAligner,
    ExactAligner,
)
from .base import AlignmentError, PanelAligner, Spine
from .config import (
    AlignerSpec,
    AlignmentConfig,
    JoinSpec,
    KeyNormalizationConfig,
    LinkSpec,
    register_configs,
    to_config,
)
from .keys import date_ranks, joint_codes, latest_at_or_before, pack
from .linking import KeyNormalization, LinkError, LinkReport, LinkTable, relink
from .merge import merge_panels
from .registry import build_aligner, register_aligner, registered_aligners

__all__ = [
    "AlignmentError",
    "PanelAligner",
    "Spine",
    "ExactAligner",
    "AsOfEntityAligner",
    "BroadcastTimeAligner",
    "BroadcastEntityAligner",
    "ALIGNERS",
    "register_aligner",
    "build_aligner",
    "registered_aligners",
    "LinkTable",
    "LinkError",
    "LinkReport",
    "KeyNormalization",
    "relink",
    "merge_panels",
    "AlignmentConfig",
    "JoinSpec",
    "AlignerSpec",
    "LinkSpec",
    "KeyNormalizationConfig",
    "register_configs",
    "to_config",
    "joint_codes",
    "date_ranks",
    "pack",
    "latest_at_or_before",
]
