"""Cross-platform launcher for repository-local pre-commit hooks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMANDS: dict[str, tuple[str, ...]] = {
    "ruff": ("-m", "ruff", "check", "--fix"),
    "mypy": ("-m", "mypy", "src", "scripts"),
    "format": ("-m", "ruff", "format", "--check", "src", "tests", "scripts"),
    "tests": ("-m", "pytest", "tests", "-q"),
    "configs": ("scripts/check_configs.py",),
}


def _project_python() -> Path:
    candidates = (
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("project virtual environment is missing; run 'make install-dev'")


def main(arguments: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if arguments is None else arguments)
    if not args or args[0] not in COMMANDS:
        available = ", ".join(sorted(COMMANDS))
        raise SystemExit(f"usage: run_hook.py <command> [files...]; commands: {available}")
    command, *filenames = args
    suffix = tuple(filenames) if command == "ruff" else ()
    completed = subprocess.run(
        [str(_project_python()), *COMMANDS[command], *suffix],
        cwd=PROJECT_ROOT,
        check=False,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
