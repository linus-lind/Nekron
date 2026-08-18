"""Project-wide constants shared across the data, preprocessing, and feature packages.

The ingestion schema materializes a two-level ``(date, entity)`` panel index; these
names are the single source of truth for how that index is labeled, so downstream
packages reference the levels by name instead of re-declaring them in every config.
"""

from __future__ import annotations

DATE_LEVEL = "date"
ENTITY_LEVEL = "entity"
