"""Content-addressed caching of pipeline stages.

Re-parsing a multi-gigabyte CSV to recompute a panel that has not changed is the
single largest avoidable cost in this pipeline: on a two-million-row panel a
chunked CSV read takes ~9.5 s against ~0.17 s to read the same frame back from a
cached artifact. Every stage is a pure function of its configuration and its
input, so each one can be keyed by a digest of both and its result kept on disk.

Keying
------
:func:`stage_key` hashes four things, and all four are load-bearing:

* the stage's fully resolved configuration, as canonical JSON;
* the key of the stage upstream, which chains the stages together — a change to
  the ingestion spec invalidates the preprocessing and feature artifacts derived
  from it without those stages having to know anything about it;
* an identity record for each source file, since a config that reads
  ``data/crsp.csv`` says nothing about what that file currently contains;
* the runtime that produced the artifact. This is not paranoia: pandas 2.2 and
  3.0 produce *different dtypes* from identical code (datetime resolution and the
  default string dtype both changed), so an artifact written by one interpreter is
  not interchangeable with one written by the other.

File identity defaults to ``(size, mtime_ns)``, which costs under a microsecond.
That is a good change detector but not a sound one — ``cp -p``, ``rsync -t``,
``git checkout`` and ``tar -x`` can all reproduce both — so ``"content"`` mode
takes a SHA-256 of the file instead, at roughly 0.34 s per gigabyte.

Format
------
Artifacts are Arrow IPC (Feather v2), not Parquet. Parquet is smaller, but it
silently degrades three dtypes this pipeline actually produces: a categorical
with non-string categories comes back as its underlying integers, a
``datetime64[s]`` index level is promoted to milliseconds, and the empty-panel
path loses its categorical dtypes entirely. A cache that returns a frame subtly
unlike the one it was given is worse than no cache, and for a regenerable local
artifact the size difference does not buy anything.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from nekron.panel import Panel, PanelError

FingerprintMode = Literal["stat", "content"]

logger = logging.getLogger(__name__)

_MANIFEST_KEY = b"nekron_cache"
_PANDAS_KEY = b"pandas"
_FORMAT_VERSION = 1


class CacheError(Exception):
    """Base class for cache errors."""


# --------------------------------------------------------------------------- #
# Keying
# --------------------------------------------------------------------------- #


def canonical_json(payload: Any) -> str:
    """Serialize ``payload`` to the one JSON encoding a digest can rely on.

    Deliberately strict: no ``default=`` fallback, no NaN literals, sorted keys.
    A fallback encoder would let three real bugs through silently — a ``set``
    serializes in a different order under every ``PYTHONHASHSEED``, an unresolved
    OmegaConf node stringifies with its interpolations *intact* so two configs
    pointing at different files hash identically, and ``numpy.int64(5)`` encodes
    differently from ``5``. Normalize at the boundary (``OmegaConf.to_container``
    with ``resolve=True``, or :func:`dataclasses.asdict`) and let anything else
    raise.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def file_identity(path: str | os.PathLike[str], mode: FingerprintMode = "stat") -> dict[str, Any]:
    """Describe a source file well enough to detect that it changed.

    ``"stat"`` records size and modification time; ``"content"`` records size and
    a SHA-256 digest. SHA-256 rather than BLAKE2b because this is hardware
    accelerated on any current CPU and measured twice as fast here.
    """
    resolved = Path(path).resolve()
    try:
        stat = resolved.stat()
    except OSError as exc:
        raise CacheError(f"cannot fingerprint source file {str(resolved)!r}: {exc}") from exc
    identity: dict[str, Any] = {"path": str(resolved), "size": stat.st_size, "mode": mode}
    if mode == "content":
        with resolved.open("rb") as handle:
            identity["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    else:
        identity["mtime_ns"] = stat.st_mtime_ns
    return identity


def runtime_identity() -> dict[str, str]:
    """The library versions that determine an artifact's dtypes."""
    return {
        "format": str(_FORMAT_VERSION),
        "pandas": ".".join(pd.__version__.split(".")[:2]),
        "pyarrow": pa.__version__.split(".")[0],
    }


def stage_key(
    spec: Any,
    *,
    upstream: str | None = None,
    sources: Mapping[str, str | os.PathLike[str]] | None = None,
    fingerprint: FingerprintMode = "stat",
) -> str:
    """Digest a stage's configuration, its upstream key and its source files."""
    payload = {
        "spec": spec,
        "upstream": upstream,
        "sources": {
            name: file_identity(path, fingerprint) for name, path in sorted((sources or {}).items())
        },
        "runtime": runtime_identity(),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Artifact IO
# --------------------------------------------------------------------------- #


def save_frame(
    frame: pd.DataFrame,
    path: str | os.PathLike[str],
    *,
    manifest: Mapping[str, Any] | None = None,
    compression: str = "zstd",
) -> None:
    """Write ``frame`` to ``path`` atomically, with ``manifest`` embedded in it.

    The manifest travels inside the file's own schema metadata rather than in a
    sidecar, so it is committed by the same atomic rename. A sidecar would need an
    ordering rule, and the safe order is counter-intuitive: a crash between the
    two writes must leave an orphan artifact (which simply reads as a miss), never
    a manifest promising an artifact that is not there (which a reader trusts).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=True)
    if manifest is not None:
        metadata = dict(table.schema.metadata or {})
        metadata[_MANIFEST_KEY] = canonical_json(dict(manifest)).encode("utf-8")
        table = table.replace_schema_metadata(metadata)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            feather.write_feather(table, handle, compression=compression)
            handle.flush()
            os.fsync(handle.fileno())
        # mkstemp creates the file 0600 and that mode survives the rename, which
        # would leave every artifact unreadable by anyone else on the machine.
        os.chmod(temporary, 0o666 & ~_current_umask())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _fsync_directory(target.parent)


def load_frame(
    path: str | os.PathLike[str], *, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Read a cached frame, optionally reading only some of its columns.

    Column pruning must re-request the index levels: unlike Parquet, a pruned
    Arrow IPC read drops the index and hands back a ``RangeIndex``.
    """
    if columns is None:
        return pd.read_feather(path)
    wanted = list(dict.fromkeys([*columns, *index_names(path)]))
    return pd.read_feather(path, columns=wanted)


def read_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the manifest embedded in an artifact, or ``{}`` if it carries none.

    Reads the schema footer only, so this does not decode any data.
    """
    metadata = _schema(path).metadata or {}
    stored = metadata.get(_MANIFEST_KEY)
    if stored is None:
        return {}
    decoded: dict[str, Any] = json.loads(stored)
    return decoded


def index_names(path: str | os.PathLike[str]) -> list[str]:
    """Return the artifact's index level names, from its pandas footer metadata."""
    metadata = _schema(path).metadata or {}
    raw = metadata.get(_PANDAS_KEY)
    if raw is None:
        return []
    stored: dict[str, Any] = json.loads(raw)
    return [name for name in stored.get("index_columns", []) if isinstance(name, str)]


def _schema(path: str | os.PathLike[str]) -> pa.Schema:
    with pa.memory_map(str(path), "rb") as source:
        schema: pa.Schema = ipc.open_file(source).schema
    return schema


def _current_umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current


def _fsync_directory(directory: Path) -> None:
    """Persist the rename itself, not just the bytes it points at.

    POSIX only. Windows refuses to open a directory as a file descriptor and
    exposes no other handle ``os.fsync`` will take, so there the durability of
    the rename is left to the filesystem.
    """
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


# --------------------------------------------------------------------------- #
# Panel cache
# --------------------------------------------------------------------------- #


@dataclass
class CacheConfig:
    """Configuration for the stage cache.

    Parameters
    ----------
    enabled:
        When false the pipeline recomputes every stage and writes nothing, without
        any artifact having to be deleted.
    directory:
        Where artifacts live. Relative paths resolve against the working directory.
    compression:
        Arrow IPC codec: ``"zstd"``, ``"lz4"`` or ``"uncompressed"``.
    fingerprint:
        How a source file's identity enters the cache key. ``"stat"`` uses size and
        modification time, which costs under a microsecond but can be fooled by
        ``cp -p``, ``rsync -t``, ``git checkout`` and ``tar -x``. ``"content"``
        takes a SHA-256 of the file — sound, at roughly 0.34 s per gigabyte.
    """

    enabled: bool = True
    directory: str = ".nekron_cache"
    compression: str = "zstd"
    fingerprint: str = "stat"

    def __post_init__(self) -> None:
        if self.fingerprint not in ("stat", "content"):
            raise ValueError(
                f"cache.fingerprint must be 'stat' or 'content'; got {self.fingerprint!r}."
            )


@dataclass(frozen=True)
class PanelCache:
    """A directory of stage artifacts addressed by content digest.

    Parameters
    ----------
    directory:
        Where artifacts are written. Created on first write.
    enabled:
        When false every lookup misses and nothing is written, so a run can be
        forced to recompute without deleting anything.
    compression:
        Arrow IPC codec. ``"zstd"`` is ~1.6x smaller than uncompressed at
        comparable speed; ``"lz4"`` and ``"uncompressed"`` are the alternatives.
    """

    directory: Path
    enabled: bool = True
    compression: str = "zstd"

    def path_for(self, key: str) -> Path:
        """The artifact path for a stage key, sharded to keep directories small."""
        return self.directory / key[:2] / f"{key}.arrow"

    def load(self, key: str) -> Panel | None:
        """Return the cached panel for ``key``, or ``None`` on a miss.

        A corrupt or unreadable artifact is treated as a miss rather than an
        error: the cache is regenerable by construction, so the useful behaviour
        is to recompute and overwrite it.
        """
        if not self.enabled:
            return None
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            manifest = read_manifest(path)
            frame = load_frame(path)
        except (OSError, ValueError, pa.ArrowInvalid):
            return None
        panel = Panel(
            frame=frame,
            date_name=manifest.get("date_name"),
            entity_name=manifest.get("entity_name"),
        )
        try:
            panel.validate()
        except PanelError:
            # A missing or partial manifest yields a panel whose declared grain
            # contradicts its own index. Better to recompute than to hand that on
            # and have the contradiction surface somewhere with less context.
            logger.warning("cached artifact %s does not match its manifest; ignoring", key[:12])
            return None
        return panel

    def store(self, key: str, panel: Panel, *, spec: Any = None) -> None:
        """Write ``panel`` under ``key``.

        A failed write is logged and swallowed. This is a write-through cache of
        results that were just computed successfully, so a full disk, a read-only
        directory or a dtype Arrow cannot encode must cost the next run its
        speed-up — never this run its answer. Read failures are treated the same
        way, as a miss.
        """
        if not self.enabled:
            return
        manifest = {
            "key": key,
            "date_name": panel.date_name,
            "entity_name": panel.entity_name,
            "grain": panel.grain.value,
            "rows": int(len(panel.frame)),
            "columns": [str(column) for column in panel.frame.columns],
            "runtime": runtime_identity(),
            "spec": spec,
        }
        try:
            save_frame(
                panel.frame, self.path_for(key), manifest=manifest, compression=self.compression
            )
        except Exception:
            logger.warning("could not cache stage %s; continuing uncached", key[:12], exc_info=True)


def build_cache(cfg: CacheConfig) -> PanelCache:
    """Construct the :class:`PanelCache` a configuration describes."""
    return PanelCache(
        directory=Path(cfg.directory), enabled=cfg.enabled, compression=cfg.compression
    )
