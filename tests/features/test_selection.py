"""Tests for :class:`ColumnSelection`."""

from __future__ import annotations

import pytest

from nekron.features import ColumnSelection

_AVAILABLE = ("close", "volume", "r1", "r5")


def test_default_keeps_every_column() -> None:
    assert ColumnSelection().select(_AVAILABLE) == _AVAILABLE  # include=None -> no restriction


def test_empty_include_keeps_nothing() -> None:
    assert ColumnSelection(include=()).select(_AVAILABLE) == ()


def test_include_preserves_selection_order() -> None:
    selection = ColumnSelection(include=("r5", "close"))
    assert selection.select(_AVAILABLE) == ("r5", "close")  # selection order, not panel order


def test_duplicate_include_names_are_collapsed() -> None:
    assert ColumnSelection(include=("r1", "r1")).select(_AVAILABLE) == ("r1",)


def test_exclude_alone_keeps_the_rest() -> None:
    assert ColumnSelection(exclude=("volume",)).select(_AVAILABLE) == ("close", "r1", "r5")


def test_exclude_wins_over_include() -> None:
    selection = ColumnSelection(include=("close", "r1"), exclude=("r1",))
    assert selection.select(_AVAILABLE) == ("close",)


def test_unknown_include_raises() -> None:
    with pytest.raises(KeyError, match="unknown"):
        ColumnSelection(include=("nope",)).select(_AVAILABLE)


def test_unknown_exclude_raises() -> None:
    # a misspelled exclusion would otherwise leave the unwanted column in the panel
    with pytest.raises(KeyError, match="unknown"):
        ColumnSelection(exclude=("clse",)).select(_AVAILABLE)


def test_error_names_the_selected_set() -> None:
    with pytest.raises(KeyError, match="feature columns"):
        ColumnSelection(include=("nope",)).select(_AVAILABLE, what="feature columns")


def test_lenient_selection_skips_absent_names() -> None:
    selection = ColumnSelection(include=("close", "absent"), exclude=("gone",), strict=False)
    assert selection.select(_AVAILABLE) == ("close",)


def test_nothing_available_selects_nothing() -> None:
    assert ColumnSelection().select(()) == ()  # an unrestricted selection of no columns
