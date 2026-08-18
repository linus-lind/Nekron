"""Remove disposable caches and build artifacts.

Referenced by ``make clean``. Everything listed here is regenerable, including the
pipeline's content-addressed stage cache — deleting it costs a recompute, never
data.
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directories removed wherever they appear, and top-level paths removed as a whole.
NESTED_DIRECTORIES = ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache")
TOP_LEVEL = (".nekron_cache", "build", "dist", "outputs", "multirun")


def _remove(path: Path) -> int:
    """Delete a file or directory, returning how many entries were removed."""
    if not path.exists():
        return 0
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
    print(f"removed {path.relative_to(PROJECT_ROOT)}")
    return 1


def main() -> None:
    removed = 0
    for name in TOP_LEVEL:
        removed += _remove(PROJECT_ROOT / name)
    for name in NESTED_DIRECTORIES:
        # rglob would descend into directories that are themselves being removed;
        # collecting first keeps the walk stable.
        for path in sorted(PROJECT_ROOT.rglob(name), reverse=True):
            if ".git" not in path.parts and ".venv" not in path.parts:
                removed += _remove(path)
    for path in sorted(PROJECT_ROOT.rglob("*.egg-info")):
        removed += _remove(path)
    print(f"{removed} path(s) removed")


if __name__ == "__main__":
    main()
