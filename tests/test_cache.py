"""Tests for :mod:`nekron.cache` — stage keying and artifact fidelity.

A cache that returns a frame subtly unlike the one it was given is worse than no
cache, so most of what follows is dtype fidelity: the categorical vocabularies,
the datetime resolutions and the nullable dtypes that Parquet quietly rewrites
must survive an Arrow IPC round trip byte for byte. The rest pins the two ways a
key can be wrong — a digest that changes when nothing did, and a digest that does
not change when the data did.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nekron.cache import (
    CacheError,
    PanelCache,
    canonical_json,
    file_identity,
    index_names,
    load_frame,
    read_manifest,
    runtime_identity,
    save_frame,
    stage_key,
)
from nekron.panel import Panel, PanelGrain


def rich_frame() -> pd.DataFrame:
    """One column per dtype the pipeline produces that a format could degrade."""
    index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(["2021-01-04", "2021-01-04", "2021-01-05"]),
            np.array([10, 20, 10], dtype=np.int32),
        ],
        names=["date", "entity"],
    )
    return pd.DataFrame(
        {
            "cat_unused": pd.Categorical(["a", "b", "a"], categories=["a", "b", "c"]),
            "cat_missing": pd.Categorical(["x", None, "y"]),
            "cat_int": pd.Categorical([10, 20, 10], categories=[10, 20, 30]),
            "nullable": pd.array([1, None, 3], dtype="Int32"),
            "text": pd.array(["p", "q", None], dtype="string"),
            "f32": np.array([1.5, 2.5, 3.5], dtype=np.float32),
            "f64": np.array([1.0, np.nan, 3.0], dtype=np.float64),
            "when": pd.to_datetime(pd.Index(["2020-01-01", None, "2020-01-03"])),
            "flag": np.array([True, False, True]),
        },
        index=index,
    )


def panel_frame() -> pd.DataFrame:
    """A small ``(date, entity)`` frame."""
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(["2021-01-04", "2021-01-04"]), ["A", "B"]], names=["date", "entity"]
    )
    return pd.DataFrame({"price": [1.0, 2.0]}, index=index)


# --------------------------------------------------------------------------- #
# canonical_json
# --------------------------------------------------------------------------- #


def test_canonical_json_is_order_independent_and_compact() -> None:
    assert canonical_json({"b": 1, "a": [2, 3]}) == '{"a":[2,3],"b":1}'
    assert canonical_json({"a": [2, 3], "b": 1}) == canonical_json({"b": 1, "a": [2, 3]})


def test_canonical_json_rejects_a_set() -> None:
    # A set serializes in a different order under every PYTHONHASHSEED, so a key
    # built from one would not be reproducible across processes.
    with pytest.raises(TypeError, match="set"):
        canonical_json({"columns": {"a", "b"}})


def test_canonical_json_rejects_a_numpy_scalar() -> None:
    # numpy.int64(5) is not 5 to a JSON encoder with a `default=` fallback; two
    # equivalent configs would hash differently. numpy.bool_ is not a bool either.
    with pytest.raises(TypeError):
        canonical_json({"window": np.int64(5)})
    with pytest.raises(TypeError):
        canonical_json({"threshold": np.float32(0.5)})
    with pytest.raises(TypeError):
        canonical_json({"enabled": np.bool_(True)})


def test_numpy_float64_is_the_one_numpy_scalar_that_slips_through() -> None:
    # numpy.float64 subclasses float and numpy.str_ subclasses str, so json
    # encodes both natively. That is benign — the encoding is identical to the
    # builtin's — but it means "no numpy reaches canonical_json" is not something
    # this function can be relied on to enforce.
    assert canonical_json({"x": np.float64(0.5)}) == canonical_json({"x": 0.5})
    assert canonical_json({"x": np.str_("a")}) == canonical_json({"x": "a"})


def test_canonical_json_rejects_nan() -> None:
    # json's NaN literal is not valid JSON and NaN != NaN, so a NaN in a spec is
    # a key that can never match itself.
    with pytest.raises(ValueError, match="(?i)not json compliant"):
        canonical_json({"threshold": float("nan")})


def test_canonical_json_accepts_the_normalized_forms() -> None:
    payload = {"a": 1, "b": [1.5, None], "c": {"d": True}, "e": "text"}
    assert canonical_json(payload) == '{"a":1,"b":[1.5,null],"c":{"d":true},"e":"text"}'


# --------------------------------------------------------------------------- #
# file_identity and stage_key
# --------------------------------------------------------------------------- #


def write_source(tmp_path: Path, text: str = "a,b\n1,2\n") -> Path:
    """A stand-in source file to fingerprint."""
    path = tmp_path / "source.csv"
    path.write_text(text)
    return path


def test_file_identity_records_size_and_mtime_by_default(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    identity = file_identity(source)
    assert identity["mode"] == "stat"
    assert identity["size"] == source.stat().st_size
    assert identity["mtime_ns"] == source.stat().st_mtime_ns
    assert "sha256" not in identity


def test_file_identity_in_content_mode_records_a_digest(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    identity = file_identity(source, "content")
    assert identity["mode"] == "content"
    assert len(identity["sha256"]) == 64
    assert "mtime_ns" not in identity


def test_file_identity_names_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CacheError, match="cannot fingerprint source file"):
        file_identity(tmp_path / "absent.csv")


def test_stage_key_is_deterministic_for_equal_specs(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    left = stage_key({"a": 1, "b": [1, 2]}, upstream="up", sources={"s": source})
    right = stage_key({"b": [1, 2], "a": 1}, upstream="up", sources={"s": source})
    assert left == right
    assert len(left) == 64


def test_stage_key_changes_with_the_spec() -> None:
    assert stage_key({"window": 20}) != stage_key({"window": 21})
    assert stage_key({"window": 20}) != stage_key({"window": "20"})


def test_stage_key_changes_with_the_upstream_key() -> None:
    assert stage_key({"a": 1}, upstream="one") != stage_key({"a": 1}, upstream="two")
    assert stage_key({"a": 1}, upstream=None) != stage_key({"a": 1}, upstream="one")


def test_stage_key_changes_when_a_source_changes(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    before = stage_key({"a": 1}, sources={"s": source})
    source.write_text("a,b\n1,2\n3,4\n")  # a different size and a new mtime
    assert stage_key({"a": 1}, sources={"s": source}) != before


def test_stage_key_changes_when_a_source_is_added(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    assert stage_key({"a": 1}) != stage_key({"a": 1}, sources={"s": source})


def test_content_fingerprint_catches_an_edit_that_stat_misses(tmp_path: Path) -> None:
    # `cp -p`, `rsync -t`, `git checkout` and `tar -x` all restore size and mtime,
    # so an in-place edit of the same length is invisible to the stat mode. This
    # is the case that makes "content" worth its 0.34 s/GB.
    source = write_source(tmp_path, "a,b\n1,2\n")
    original = source.stat()
    stat_before = stage_key({"a": 1}, sources={"s": source})
    content_before = stage_key({"a": 1}, sources={"s": source}, fingerprint="content")

    source.write_text("a,b\n9,9\n")  # same byte count
    os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert source.stat().st_size == original.st_size
    assert source.stat().st_mtime_ns == original.st_mtime_ns

    assert stage_key({"a": 1}, sources={"s": source}) == stat_before  # the miss
    assert stage_key({"a": 1}, sources={"s": source}, fingerprint="content") != content_before


def test_stage_key_distinguishes_the_two_fingerprint_modes(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    assert stage_key({"a": 1}, sources={"s": source}) != stage_key(
        {"a": 1}, sources={"s": source}, fingerprint="content"
    )


def test_stage_key_rejects_an_unnormalized_spec() -> None:
    with pytest.raises(TypeError):
        stage_key({"columns": {"a", "b"}})


def test_runtime_identity_pins_the_libraries_that_decide_dtypes() -> None:
    identity = runtime_identity()
    assert identity["pandas"] == ".".join(pd.__version__.split(".")[:2])
    assert set(identity) == {"format", "pandas", "pyarrow"}


# --------------------------------------------------------------------------- #
# save_frame / load_frame fidelity
# --------------------------------------------------------------------------- #


def test_round_trip_is_exact_for_every_fragile_dtype(tmp_path: Path) -> None:
    frame = rich_frame()
    path = tmp_path / "artifact.arrow"
    save_frame(frame, path)

    back = load_frame(path)
    pd.testing.assert_frame_equal(frame, back, check_exact=True)
    # Spelled out, because "assert_frame_equal passed" hides which guarantee held.
    assert list(back["cat_unused"].dtype.categories) == ["a", "b", "c"]  # unused kept
    assert back["cat_missing"].isna().tolist() == [False, True, False]
    assert back["cat_int"].dtype.categories.dtype == np.int64  # Parquet returns codes
    assert back["nullable"].dtype == pd.Int32Dtype()
    assert back["f32"].dtype == np.float32
    assert back["flag"].dtype == np.bool_
    assert back.index.levels[1].dtype == np.int32  # the int32 key level, not int64


@pytest.mark.parametrize("compression", ["zstd", "lz4", "uncompressed"])
def test_round_trip_is_exact_under_every_codec(tmp_path: Path, compression: str) -> None:
    frame = rich_frame()
    path = tmp_path / f"{compression}.arrow"
    save_frame(frame, path, compression=compression)
    pd.testing.assert_frame_equal(frame, load_frame(path), check_exact=True)


def test_an_empty_frame_keeps_its_dtypes(tmp_path: Path) -> None:
    empty = rich_frame().iloc[:0]
    path = tmp_path / "empty.arrow"
    save_frame(empty, path)
    back = load_frame(path)

    categoricals = ["cat_unused", "cat_missing", "cat_int"]
    assert len(back) == 0
    assert list(back.columns) == list(empty.columns)
    # Every dtype survives, categoricals included — the case Parquet drops entirely.
    assert [str(dtype) for dtype in back.dtypes] == [str(dtype) for dtype in empty.dtypes]
    pd.testing.assert_frame_equal(
        empty.drop(columns=categoricals), back.drop(columns=categoricals), check_exact=True
    )
    assert back.index.names == ["date", "entity"]
    assert back.index.levels[1].dtype == np.int32
    # Known limitation: a zero-row table carries no record batch and therefore no
    # dictionary, so the *vocabulary* of a categorical is not restored even though
    # its `category` dtype is. Nothing downstream may assume that an empty panel's
    # categories are still there.
    assert list(back["cat_unused"].dtype.categories) == []
    assert list(back["cat_int"].dtype.categories) == []


def test_a_second_resolution_index_level_stays_seconds(tmp_path: Path) -> None:
    # Parquet promotes datetime64[s] to milliseconds; Arrow IPC does not.
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(["2021-01-04", "2021-01-05"]).as_unit("s"), ["A", "B"]],
        names=["date", "entity"],
    )
    frame = pd.DataFrame({"price": [1.0, 2.0]}, index=index)
    path = tmp_path / "seconds.arrow"
    save_frame(frame, path)

    back = load_frame(path)
    assert back.index.levels[0].dtype == np.dtype("datetime64[s]")
    pd.testing.assert_frame_equal(frame, back, check_exact=True)


def test_row_order_and_a_duplicated_unsorted_index_survive(tmp_path: Path) -> None:
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(["2021-01-05", "2021-01-04", "2021-01-05"]), ["B", "A", "B"]],
        names=["date", "entity"],
    )
    frame = pd.DataFrame({"price": [3.0, 1.0, 2.0]}, index=index)
    path = tmp_path / "unsorted.arrow"
    save_frame(frame, path)

    back = load_frame(path)
    pd.testing.assert_frame_equal(frame, back, check_exact=True)
    assert not back.index.is_unique  # the duplicate is preserved, not collapsed
    assert back["price"].tolist() == [3.0, 1.0, 2.0]  # and so is the order


def test_a_frame_with_no_index_names_round_trips(tmp_path: Path) -> None:
    frame = pd.DataFrame({"permno": [1, 2], "gvkey": ["a", "b"]})
    path = tmp_path / "table.arrow"
    save_frame(frame, path)
    pd.testing.assert_frame_equal(frame, load_frame(path), check_exact=True)
    # A frame with no index names is still written with its index materialized,
    # under Arrow's placeholder name, so column pruning has something to re-request.
    assert index_names(path) == ["__index_level_0__"]
    pruned = load_frame(path, columns=["gvkey"])
    assert list(pruned.columns) == ["gvkey"]
    assert pruned.index.tolist() == [0, 1]


# --------------------------------------------------------------------------- #
# Manifests, index names, column pruning
# --------------------------------------------------------------------------- #


def test_read_manifest_returns_what_was_embedded(tmp_path: Path) -> None:
    path = tmp_path / "artifact.arrow"
    manifest: dict[str, Any] = {"key": "abc", "rows": 3, "columns": ["price"], "spec": None}
    save_frame(panel_frame(), path, manifest=manifest)
    assert read_manifest(path) == manifest


def test_read_manifest_is_empty_when_none_was_embedded(tmp_path: Path) -> None:
    path = tmp_path / "artifact.arrow"
    save_frame(panel_frame(), path)
    assert read_manifest(path) == {}


def test_a_manifest_does_not_disturb_the_frame(tmp_path: Path) -> None:
    path = tmp_path / "artifact.arrow"
    frame = rich_frame()
    save_frame(frame, path, manifest={"key": "abc"})
    pd.testing.assert_frame_equal(frame, load_frame(path), check_exact=True)


def test_a_manifest_must_be_canonically_encodable(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        save_frame(panel_frame(), tmp_path / "artifact.arrow", manifest={"bad": {"a", "b"}})


def test_index_names_reports_the_level_names(tmp_path: Path) -> None:
    path = tmp_path / "artifact.arrow"
    save_frame(rich_frame(), path)
    assert index_names(path) == ["date", "entity"]


def test_column_pruning_keeps_the_multiindex(tmp_path: Path) -> None:
    frame = rich_frame()
    path = tmp_path / "artifact.arrow"
    save_frame(frame, path)

    pruned = load_frame(path, columns=["f32", "cat_int"])
    assert list(pruned.columns) == ["f32", "cat_int"]
    pd.testing.assert_index_equal(pruned.index, frame.index)
    pd.testing.assert_frame_equal(pruned, frame[["f32", "cat_int"]], check_exact=True)

    # The trap: an Arrow IPC read pruned without the index columns loses the index.
    naive = pd.read_feather(path, columns=["f32"])
    assert isinstance(naive.index, pd.RangeIndex)


def test_column_pruning_tolerates_naming_an_index_level(tmp_path: Path) -> None:
    path = tmp_path / "artifact.arrow"
    save_frame(rich_frame(), path)
    pruned = load_frame(path, columns=["f32", "date"])
    assert list(pruned.columns) == ["f32"]  # `date` is consumed as an index level
    assert pruned.index.names == ["date", "entity"]


# --------------------------------------------------------------------------- #
# Atomic writes
# --------------------------------------------------------------------------- #


def artifacts(directory: Path) -> list[str]:
    """Every file in ``directory``, dotfiles included."""
    return sorted(entry.name for entry in directory.iterdir())


def test_a_successful_write_leaves_no_temporary_behind(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "artifact.arrow"
    save_frame(panel_frame(), path, manifest={"key": "abc"})
    assert artifacts(path.parent) == ["artifact.arrow"]
    assert list(path.parent.glob("*.tmp")) == []


def test_a_failed_write_leaves_nothing_behind(tmp_path: Path) -> None:
    # The codec is rejected inside the write, i.e. after the temporary file has
    # been created — exactly the window that could leave a turd.
    path = tmp_path / "nested" / "artifact.arrow"
    with pytest.raises(ValueError, match="(?i)compression"):
        save_frame(panel_frame(), path, compression="bogus")

    assert artifacts(path.parent) == []
    assert not path.exists()


def test_a_rewrite_replaces_the_artifact_in_place(tmp_path: Path) -> None:
    path = tmp_path / "artifact.arrow"
    save_frame(panel_frame(), path, manifest={"version": 1})
    save_frame(panel_frame().head(1), path, manifest={"version": 2})

    assert artifacts(tmp_path) == ["artifact.arrow"]
    assert read_manifest(path) == {"version": 2}
    assert len(load_frame(path)) == 1


def test_a_written_artifact_is_readable_by_others(tmp_path: Path) -> None:
    # mkstemp creates 0600 and the mode survives the rename, so the write fixes it.
    path = tmp_path / "artifact.arrow"
    save_frame(panel_frame(), path)
    assert path.stat().st_mode & 0o044  # at least group/other readable, umask permitting


# --------------------------------------------------------------------------- #
# PanelCache
# --------------------------------------------------------------------------- #

KEY = "0f" + "a" * 62


def test_panel_cache_round_trips_a_panel(tmp_path: Path) -> None:
    cache = PanelCache(directory=tmp_path / "cache")
    panel = Panel(frame=rich_frame())
    cache.store(KEY, panel)

    loaded = cache.load(KEY)
    assert loaded is not None
    assert loaded.date_name == "date"
    assert loaded.entity_name == "entity"
    assert loaded.grain is PanelGrain.PANEL
    pd.testing.assert_frame_equal(panel.frame, loaded.frame, check_exact=True)


@pytest.mark.parametrize(
    ("date_name", "entity_name", "grain"),
    [
        ("date", "entity", PanelGrain.PANEL),
        ("date", None, PanelGrain.TIME_SERIES),
        (None, "entity", PanelGrain.CROSS_SECTION),
        (None, None, PanelGrain.TABLE),
    ],
)
def test_panel_cache_preserves_the_declared_key_names(
    tmp_path: Path, date_name: str | None, entity_name: str | None, grain: PanelGrain
) -> None:
    cache = PanelCache(directory=tmp_path / "cache")
    # The index must actually carry the declared key levels: the cache validates
    # what it loads, so an artifact whose manifest contradicts its index reads as
    # a miss rather than being handed on.
    levels = {
        "date": pd.to_datetime(["2021-01-04", "2021-01-05"]),
        "entity": ["A", "B"],
    }
    names = [name for name in (date_name, entity_name) if name is not None]
    if len(names) == 2:
        index: pd.Index = pd.MultiIndex.from_arrays([levels[n] for n in names], names=names)
    elif names:
        index = pd.Index(levels[names[0]], name=names[0])
    else:
        index = pd.RangeIndex(2)
    frame = pd.DataFrame({"price": [1.0, 2.0]}, index=index)
    cache.store(KEY, Panel(frame=frame, date_name=date_name, entity_name=entity_name))

    loaded = cache.load(KEY)
    assert loaded is not None
    assert loaded.date_name == date_name
    assert loaded.entity_name == entity_name
    assert loaded.grain is grain


def test_panel_cache_records_a_describing_manifest(tmp_path: Path) -> None:
    cache = PanelCache(directory=tmp_path / "cache")
    cache.store(KEY, Panel(frame=panel_frame()), spec={"window": 20})

    manifest = read_manifest(cache.path_for(KEY))
    assert manifest["key"] == KEY
    assert manifest["grain"] == "panel"
    assert manifest["rows"] == 2
    assert manifest["columns"] == ["price"]
    assert manifest["spec"] == {"window": 20}
    assert manifest["runtime"] == runtime_identity()


def test_panel_cache_shards_by_the_key_prefix(tmp_path: Path) -> None:
    cache = PanelCache(directory=tmp_path / "cache")
    assert cache.path_for(KEY) == tmp_path / "cache" / KEY[:2] / f"{KEY}.arrow"
    cache.store(KEY, Panel(frame=panel_frame()))
    assert cache.path_for(KEY).is_file()


def test_a_miss_returns_none(tmp_path: Path) -> None:
    cache = PanelCache(directory=tmp_path / "cache")
    assert cache.load(KEY) is None  # nothing written at all
    cache.store(KEY, Panel(frame=panel_frame()))
    assert cache.load("ff" + "b" * 62) is None  # a different key


def test_a_disabled_cache_neither_reads_nor_writes(tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    PanelCache(directory=directory).store(KEY, Panel(frame=panel_frame()))
    assert PanelCache(directory=directory).load(KEY) is not None

    disabled = PanelCache(directory=directory, enabled=False)
    assert disabled.load(KEY) is None  # the artifact is there and is still ignored

    disabled.store("ff" + "b" * 62, Panel(frame=panel_frame()))
    assert not disabled.path_for("ff" + "b" * 62).exists()
    assert artifacts(directory) == [KEY[:2]]


def test_a_truncated_artifact_reads_as_a_miss(tmp_path: Path) -> None:
    # The cache is regenerable by construction, so a half-written file must send
    # the caller back to the computation rather than blow the run up.
    cache = PanelCache(directory=tmp_path / "cache")
    cache.store(KEY, Panel(frame=rich_frame()))
    path = cache.path_for(KEY)

    payload = path.read_bytes()
    path.write_bytes(payload[: len(payload) // 2])
    assert cache.load(KEY) is None


def test_an_empty_artifact_reads_as_a_miss(tmp_path: Path) -> None:
    cache = PanelCache(directory=tmp_path / "cache")
    cache.store(KEY, Panel(frame=panel_frame()))
    cache.path_for(KEY).write_bytes(b"")
    assert cache.load(KEY) is None


def test_a_directory_in_place_of_an_artifact_reads_as_a_miss(tmp_path: Path) -> None:
    cache = PanelCache(directory=tmp_path / "cache")
    cache.path_for(KEY).mkdir(parents=True)
    assert cache.load(KEY) is None


def test_storing_twice_under_the_same_key_overwrites(tmp_path: Path) -> None:
    cache = PanelCache(directory=tmp_path / "cache")
    cache.store(KEY, Panel(frame=panel_frame()))
    cache.store(KEY, Panel(frame=panel_frame().head(1)))

    loaded = cache.load(KEY)
    assert loaded is not None
    assert len(loaded.frame) == 1
    assert artifacts(cache.path_for(KEY).parent) == [f"{KEY}.arrow"]


def test_the_cache_key_is_what_addresses_an_artifact(tmp_path: Path) -> None:
    # The end-to-end contract: an unchanged spec hits, a changed one misses.
    cache = PanelCache(directory=tmp_path / "cache")
    key = stage_key({"window": 20})
    cache.store(key, Panel(frame=panel_frame()))

    assert cache.load(stage_key({"window": 20})) is not None
    assert cache.load(stage_key({"window": 21})) is None


def test_store_never_aborts_the_run(tmp_path: Path) -> None:
    """A write-through cache must cost the next run its speed, not this run its answer."""
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(["2020-01-01"]), [1]], names=["date", "entity"]
    )
    cache = PanelCache(directory=tmp_path / "cache")

    # A dtype Arrow cannot encode.
    cache.store("a" * 64, Panel(pd.DataFrame({"z": pd.array([1 + 2j])}, index=index)))
    assert cache.load("a" * 64) is None


@pytest.mark.skipif(os.name == "nt", reason="chmod does not deny directory writes on Windows")
def test_store_never_aborts_the_run_on_a_read_only_directory(tmp_path: Path) -> None:
    """The same, for the failure that comes from the filesystem rather than the frame."""
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(["2020-01-01"]), [1]], names=["date", "entity"]
    )
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        readonly = PanelCache(directory=blocked / "cache")
        readonly.store("b" * 64, Panel(pd.DataFrame({"v": [1.0]}, index=index)))
        assert readonly.load("b" * 64) is None
    finally:
        blocked.chmod(0o700)
