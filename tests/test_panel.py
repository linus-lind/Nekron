"""Tests for :mod:`nekron.panel` — grains, the panel wrapper, and the panel set.

The grain check is what stands between a mislabelled source and a transform that
returns a plausible wrong answer, so these tests care less about the happy path
than about which shapes are rejected and what the rejection says.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from nekron.panel import Panel, PanelError, PanelGrain, PanelSet, UnknownPanelError


def panel_frame() -> pd.DataFrame:
    """A ``(date, entity)`` frame."""
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(["2021-01-04", "2021-01-04"]), ["A", "B"]], names=["date", "entity"]
    )
    return pd.DataFrame({"price": [1.0, 2.0]}, index=index)


def time_series_frame() -> pd.DataFrame:
    """A frame with one row per date."""
    index = pd.DatetimeIndex(["2021-01-04", "2021-01-05"], name="date")
    return pd.DataFrame({"mktrf": [0.01, -0.02]}, index=index)


def cross_section_frame() -> pd.DataFrame:
    """A frame with one row per entity."""
    index = pd.Index(["A", "B"], name="entity")
    return pd.DataFrame({"sector": ["tech", "energy"]}, index=index)


def table_frame() -> pd.DataFrame:
    """A keyless lookup relation, as a link table arrives."""
    return pd.DataFrame({"permno": [1, 2], "gvkey": ["a", "b"]})


# --------------------------------------------------------------------------- #
# PanelGrain
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("date", "entity", "expected"),
    [
        (True, True, PanelGrain.PANEL),
        (True, False, PanelGrain.TIME_SERIES),
        (False, True, PanelGrain.CROSS_SECTION),
        (False, False, PanelGrain.TABLE),
    ],
)
def test_grain_of_covers_every_combination(date: bool, entity: bool, expected: PanelGrain) -> None:
    assert PanelGrain.of(date=date, entity=entity) is expected


@pytest.mark.parametrize(
    ("grain", "has_date", "has_entity"),
    [
        (PanelGrain.PANEL, True, True),
        (PanelGrain.TIME_SERIES, True, False),
        (PanelGrain.CROSS_SECTION, False, True),
        (PanelGrain.TABLE, False, False),
    ],
)
def test_grain_reports_which_keys_it_carries(
    grain: PanelGrain, has_date: bool, has_entity: bool
) -> None:
    assert grain.has_date is has_date
    assert grain.has_entity is has_entity


def test_grain_of_round_trips_through_has_date_and_has_entity() -> None:
    for grain in PanelGrain:
        assert PanelGrain.of(date=grain.has_date, entity=grain.has_entity) is grain


# --------------------------------------------------------------------------- #
# Panel: grain and key names
# --------------------------------------------------------------------------- #


def test_panel_defaults_to_the_two_level_panel_grain() -> None:
    panel = Panel(frame=panel_frame())
    assert panel.date_name == "date"
    assert panel.entity_name == "entity"
    assert panel.grain is PanelGrain.PANEL
    assert panel.key_names == ("date", "entity")


@pytest.mark.parametrize(
    ("date_name", "entity_name", "grain", "keys"),
    [
        ("date", "entity", PanelGrain.PANEL, ("date", "entity")),
        ("date", None, PanelGrain.TIME_SERIES, ("date",)),
        (None, "entity", PanelGrain.CROSS_SECTION, ("entity",)),
        (None, None, PanelGrain.TABLE, ()),
    ],
)
def test_panel_derives_grain_and_key_names_from_its_key_names(
    date_name: str | None, entity_name: str | None, grain: PanelGrain, keys: tuple[str, ...]
) -> None:
    panel = Panel(frame=table_frame(), date_name=date_name, entity_name=entity_name)
    assert panel.grain is grain
    assert panel.key_names == keys


def test_key_names_are_in_index_order_even_when_renamed() -> None:
    panel = Panel(frame=panel_frame(), date_name="caldt", entity_name="permno")
    assert panel.key_names == ("caldt", "permno")  # date first, whatever it is called


# --------------------------------------------------------------------------- #
# Panel.with_frame
# --------------------------------------------------------------------------- #


def test_with_frame_keeps_the_key_names_and_leaves_the_original_alone() -> None:
    original = Panel(frame=panel_frame(), date_name="caldt", entity_name="permno")
    replacement = original.frame.assign(volume=[10.0, 20.0])
    derived = original.with_frame(replacement)

    assert derived is not original
    assert derived.date_name == "caldt"
    assert derived.entity_name == "permno"
    assert derived.grain is PanelGrain.PANEL
    assert list(derived.frame.columns) == ["price", "volume"]
    assert list(original.frame.columns) == ["price"]
    assert derived.frame is replacement  # the frame is carried, not copied


def test_with_frame_keeps_a_keyless_table_keyless() -> None:
    original = Panel(frame=table_frame(), date_name=None, entity_name=None)
    derived = original.with_frame(table_frame().head(1))
    assert derived.grain is PanelGrain.TABLE
    assert derived.key_names == ()


def test_panel_is_frozen() -> None:
    panel = Panel(frame=panel_frame())
    with pytest.raises(dataclasses.FrozenInstanceError):
        panel.date_name = "caldt"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Panel.require_grain
# --------------------------------------------------------------------------- #


def test_require_grain_accepts_an_allowed_grain() -> None:
    Panel(frame=panel_frame()).require_grain(PanelGrain.PANEL, what="the winsorizer")
    Panel(frame=panel_frame()).require_grain(
        PanelGrain.PANEL, PanelGrain.TIME_SERIES, what="the winsorizer"
    )


def test_require_grain_names_the_stage_and_both_grains() -> None:
    panel = Panel(frame=cross_section_frame(), date_name=None, entity_name="entity")
    with pytest.raises(PanelError) as excinfo:
        panel.require_grain(PanelGrain.PANEL, what="the cumulative factor")

    message = str(excinfo.value)
    assert "the cumulative factor" in message
    assert "grain panel" in message  # what was required
    assert "is cross_section" in message  # what it got
    assert "'entity'" in message  # and the index it actually has


def test_require_grain_lists_several_allowed_grains_in_order() -> None:
    panel = Panel(frame=table_frame(), date_name=None, entity_name=None)
    with pytest.raises(PanelError, match="grain cross_section, panel"):
        panel.require_grain(PanelGrain.PANEL, PanelGrain.CROSS_SECTION, what="the join")


def test_require_grain_with_no_allowed_grain_always_raises() -> None:
    with pytest.raises(PanelError):
        Panel(frame=panel_frame()).require_grain(what="a stage that accepts nothing")


# --------------------------------------------------------------------------- #
# Panel.validate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("frame", "date_name", "entity_name"),
    [
        (panel_frame(), "date", "entity"),
        (time_series_frame(), "date", None),
        (cross_section_frame(), None, "entity"),
        (table_frame(), None, None),
    ],
)
def test_validate_accepts_an_index_that_matches_the_declared_keys(
    frame: pd.DataFrame, date_name: str | None, entity_name: str | None
) -> None:
    Panel(frame=frame, date_name=date_name, entity_name=entity_name).validate()


def test_validate_rejects_an_index_whose_levels_are_in_the_wrong_order() -> None:
    swapped = panel_frame().reorder_levels(["entity", "date"])
    with pytest.raises(PanelError) as excinfo:
        Panel(frame=swapped).validate()
    assert "declares key levels ['date', 'entity']" in str(excinfo.value)
    assert "['entity', 'date']" in str(excinfo.value)


def test_validate_rejects_a_renamed_level() -> None:
    frame = panel_frame()
    frame.index = frame.index.set_names(["caldt", "entity"])
    with pytest.raises(PanelError, match="declares key levels"):
        Panel(frame=frame).validate()


def test_validate_rejects_a_missing_key_level() -> None:
    # A frame that lost its entity level still has a plausible index; only the
    # declared key names reveal that a level went missing.
    frame = panel_frame().droplevel("entity")
    with pytest.raises(PanelError, match=r"\['date', 'entity'\]"):
        Panel(frame=frame).validate()


def test_validate_rejects_an_unnamed_level() -> None:
    frame = panel_frame()
    frame.index = frame.index.set_names(["date", None])
    with pytest.raises(PanelError, match="declares key levels"):
        Panel(frame=frame).validate()


def test_validate_rejects_a_table_that_carries_a_named_index() -> None:
    # The mistake this catches: a link table read with an index set, which would
    # make every downstream key lookup align against the wrong thing.
    frame = table_frame().set_index("permno")
    panel = Panel(frame=frame, date_name=None, entity_name=None)
    with pytest.raises(PanelError) as excinfo:
        panel.validate()
    assert "table panel must carry a single unnamed index level" in str(excinfo.value)
    assert "['permno']" in str(excinfo.value)


def test_validate_accepts_a_table_with_an_unnamed_index() -> None:
    frame = table_frame()
    frame.index = pd.Index(np.array([7, 9], dtype=np.int64))
    Panel(frame=frame, date_name=None, entity_name=None).validate()


# --------------------------------------------------------------------------- #
# PanelSet
# --------------------------------------------------------------------------- #


def make_panel_set() -> PanelSet:
    """One panel of each grain, keyed the way a config would name them."""
    return PanelSet(
        {
            "crsp": Panel(frame=panel_frame()),
            "ff": Panel(frame=time_series_frame(), entity_name=None),
            "sectors": Panel(frame=cross_section_frame(), date_name=None),
            "link": Panel(frame=table_frame(), date_name=None, entity_name=None),
        }
    )


def test_panel_set_is_a_mapping() -> None:
    panels = make_panel_set()
    assert len(panels) == 4
    assert list(panels) == ["crsp", "ff", "sectors", "link"]  # insertion order kept
    assert list(panels.keys()) == ["crsp", "ff", "sectors", "link"]
    assert panels["crsp"].grain is PanelGrain.PANEL
    assert [panel.grain for panel in panels.values()] == [
        PanelGrain.PANEL,
        PanelGrain.TIME_SERIES,
        PanelGrain.CROSS_SECTION,
        PanelGrain.TABLE,
    ]
    assert [name for name, _ in panels.items()] == ["crsp", "ff", "sectors", "link"]


def test_panel_set_copies_the_mapping_it_was_given() -> None:
    source = {"crsp": Panel(frame=panel_frame())}
    panels = PanelSet(source)
    source["extra"] = Panel(frame=panel_frame())
    assert list(panels) == ["crsp"]


def test_unknown_panel_names_the_panels_that_were_loaded() -> None:
    panels = make_panel_set()
    with pytest.raises(PanelError) as excinfo:
        panels["crspq"]

    message = str(excinfo.value)
    assert "unknown panel 'crspq'" in message
    assert "['crsp', 'ff', 'link', 'sectors']" in message  # sorted, so it reads as a menu


def test_unknown_panel_error_is_both_a_panel_error_and_a_key_error() -> None:
    """A miss must satisfy the Mapping contract without losing its message.

    ``Mapping`` implements ``in`` and ``.get`` by catching ``KeyError``, so a
    lookup error that is not one turns probing for an optional panel into a crash.
    Being a ``PanelError`` too keeps it catchable alongside every other panel
    problem.
    """
    panels = make_panel_set()

    assert issubclass(UnknownPanelError, PanelError)
    assert issubclass(UnknownPanelError, KeyError)
    with pytest.raises(UnknownPanelError, match="unknown panel 'missing'"):
        panels["missing"]
    with pytest.raises(PanelError):
        panels["missing"]
    # The Mapping probes answer instead of raising.
    assert "missing" not in panels
    assert panels.get("missing") is None


def test_of_grain_selects_by_grain() -> None:
    panels = make_panel_set()
    assert list(panels.of_grain(PanelGrain.PANEL)) == ["crsp"]
    assert list(panels.of_grain(PanelGrain.TIME_SERIES, PanelGrain.CROSS_SECTION)) == [
        "ff",
        "sectors",
    ]
    assert panels.of_grain(PanelGrain.PANEL)["crsp"] is panels["crsp"]


def test_of_grain_with_no_grain_selects_nothing() -> None:
    assert make_panel_set().of_grain() == {}


def test_of_grain_returns_a_plain_dict_that_does_not_write_back() -> None:
    panels = make_panel_set()
    selected = panels.of_grain(PanelGrain.PANEL)
    selected.pop("crsp")
    assert "crsp" in list(panels)


def test_panel_set_repr_shows_grain_and_shape() -> None:
    text = repr(make_panel_set())
    assert text.startswith("PanelSet(")
    assert "crsp: panel(2, 1)" in text
    assert "link: table(2, 2)" in text


def test_validate_accepts_a_duplicated_key() -> None:
    """Uniqueness is not this layer's job, and enforcing it here breaks ingestion.

    ``validate`` runs on the panel a source has just been read into, which is
    upstream of the preprocessing ``deduplicate`` step that exists to merge rows
    sharing a key. A raw file with two rows for one ``(date, entity)`` is the case
    that step is configured for; rejecting it here makes the step unreachable.
    The merge is where a duplicate would actually corrupt a result, and that is
    where it is caught — see the two tests below.
    """
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(["2021-01-04", "2021-01-04"]), ["A", "A"]], names=["date", "entity"]
    )
    Panel(pd.DataFrame({"price": [1.0, 3.0]}, index=index)).validate()


def test_the_alignment_spine_still_rejects_a_duplicated_key() -> None:
    """The check that protects the numbers, at the point where it can.

    The spine is the merged panel's index, so a repeat there is the population
    every per-date cross-sectional statistic is taken over being wrong.
    """
    from nekron.align.base import Spine
    from nekron.frames import FrameError

    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(["2021-01-04", "2021-01-04"]), ["A", "A"]], names=["date", "entity"]
    )
    panel = Panel(pd.DataFrame({"price": [1.0, 3.0]}, index=index))
    # A bare FrameError, not an AlignmentError: the spine calls require_unique_index
    # directly where the join aligners wrap it. Pinned as-is rather than tidied.
    with pytest.raises(FrameError, match="must not contain duplicate keys"):
        Spine.from_panel(panel)


def test_a_deduplicated_panel_reaches_the_spine_intact() -> None:
    """The path the pipeline actually takes: ingest -> deduplicate -> align."""
    from nekron.align.base import Spine
    from nekron.preprocessing.deduplicate import DuplicateMerger

    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(["2021-01-04", "2021-01-04"]), ["A", "A"]], names=["date", "entity"]
    )
    panel = Panel(pd.DataFrame({"price": [1.0, 3.0]}, index=index))
    panel.validate()  # ingestion lets it through...
    merged = panel.with_frame(DuplicateMerger().apply(panel.frame))  # ...preprocessing merges it
    assert len(merged.frame) == 1
    assert merged.frame["price"].iloc[0] == 2.0  # NaN-skipping mean of 1.0 and 3.0
    Spine.from_panel(merged)  # ...and the merge accepts it
