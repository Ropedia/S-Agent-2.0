"""Validate that every task directory under a dataset conforms to SAgent's
task layout and naming conventions.

Checks performed per task directory:
- Directory name uses only [A-Za-z0-9._-] (filesystem-safe; matches the kind of
  short names SAgent allows in package short_name).
- Required files exist: task.toml, instruction.md, environment/, tests/test.sh
- Optional but expected: solution/solve.sh (warning if missing)
- task.toml parses and validates against ``sagent.models.task.config.TaskConfig``.
- If [task] is present, its ``name`` must satisfy ORG_NAME_PATTERN.
- environment/ contains at least one recognised env file (Dockerfile,
  docker-compose.yaml, etc.). Missing env file is an error only when
  ``[environment].docker_image`` is also unset in ``task.toml``; otherwise
  the image is assumed to be supplied directly and the file is optional.

Usage:
    uv run python scripts/check_dataset_tasks.py datasets/swegen-selfmade-260415-260505
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from sagent.constants import ORG_NAME_PATTERN
from sagent.models.task.config import TaskConfig
from sagent.models.task.paths import TaskPaths

# Filesystem-safe short-name pattern. Mirrors the second half of ORG_NAME_PATTERN
# (a name segment) so a directory could legally be a SAgent package short_name.
DIR_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")

# Files SAgent recognises as defining an environment image.
ENV_FILE_CANDIDATES = (
    "Dockerfile",
    "docker-compose.yaml",
    "docker-compose.yml",
    "singularity-compose.yaml",
    "singularity.def",
)


def check_task(task_dir: Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a single task directory."""
    errors: list[str] = []
    warnings: list[str] = []

    name = task_dir.name
    if ".." in name:
        errors.append("directory name contains '..'")
    elif not DIR_NAME_PATTERN.match(name):
        errors.append(
            f"directory name '{name}' must match {DIR_NAME_PATTERN.pattern} "
            "(alphanumeric, '.', '_', '-'; cannot start with '.' or '-')"
        )

    paths = TaskPaths(task_dir)

    if not paths.config_path.exists():
        errors.append("missing task.toml")
    if not paths.instruction_path.exists():
        errors.append("missing instruction.md")
    if not paths.environment_dir.exists():
        errors.append("missing environment/ directory")
    env_files_present = paths.environment_dir.exists() and any(
        (paths.environment_dir / f).exists() for f in ENV_FILE_CANDIDATES
    )

    if not paths.tests_dir.exists():
        errors.append("missing tests/ directory")
    elif not paths.test_path.exists():
        errors.append("missing tests/test.sh")

    if not paths.solution_dir.exists():
        warnings.append("missing solution/ directory")
    elif not paths.solve_path.exists():
        warnings.append("missing solution/solve.sh")

    if paths.config_path.exists():
        try:
            config = TaskConfig.model_validate_toml(paths.config_path.read_text())
        except Exception as exc:
            errors.append(f"task.toml invalid: {exc.__class__.__name__}: {exc}")
        else:
            if config.task is not None:
                pkg_name = config.task.name
                if not re.match(ORG_NAME_PATTERN, pkg_name) or ".." in pkg_name:
                    errors.append(
                        f"[task].name='{pkg_name}' must match 'org/name' "
                        f"format ({ORG_NAME_PATTERN})"
                    )
                else:
                    short = pkg_name.split("/", 1)[1]
                    if short != name:
                        warnings.append(
                            f"[task].name short_name '{short}' does not match "
                            f"directory name '{name}'"
                        )
            if (
                paths.environment_dir.exists()
                and not env_files_present
                and not config.environment.docker_image
            ):
                errors.append(
                    "environment/ has no Dockerfile (or recognised env file) and "
                    "task.toml does not set [environment].docker_image"
                )

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to the dataset directory containing task subdirectories.",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Print per-task warnings in addition to errors.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=30,
        help="Maximum number of per-task issues to print verbatim.",
    )
    args = parser.parse_args()

    dataset_dir: Path = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        print(f"ERROR: {dataset_dir} is not a directory", file=sys.stderr)
        return 2

    task_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if not task_dirs:
        print(f"No task subdirectories under {dataset_dir}", file=sys.stderr)
        return 2

    total = len(task_dirs)
    bad_tasks: list[tuple[str, list[str]]] = []
    warn_tasks: list[tuple[str, list[str]]] = []
    error_counter: Counter[str] = Counter()
    warning_counter: Counter[str] = Counter()

    for task_dir in task_dirs:
        errors, warnings = check_task(task_dir)
        if errors:
            bad_tasks.append((task_dir.name, errors))
            for e in errors:
                error_counter[_normalise(e)] += 1
        if warnings:
            warn_tasks.append((task_dir.name, warnings))
            for w in warnings:
                warning_counter[_normalise(w)] += 1

    print(f"Dataset: {dataset_dir}")
    print(f"Total task directories: {total}")
    print(f"Tasks with errors:   {len(bad_tasks)}")
    print(f"Tasks with warnings: {len(warn_tasks)}")
    print()

    if error_counter:
        print("=== Error summary (count × kind) ===")
        for msg, count in error_counter.most_common():
            print(f"  {count:>6}  {msg}")
        print()

    if warning_counter:
        print("=== Warning summary (count × kind) ===")
        for msg, count in warning_counter.most_common():
            print(f"  {count:>6}  {msg}")
        print()

    if bad_tasks:
        print(f"=== First {min(args.max_print, len(bad_tasks))} tasks with errors ===")
        for name, errs in bad_tasks[: args.max_print]:
            print(f"- {name}")
            for e in errs:
                print(f"    ERROR: {e}")
        print()

    if args.show_warnings and warn_tasks:
        print(
            f"=== First {min(args.max_print, len(warn_tasks))} tasks with warnings ==="
        )
        for name, ws in warn_tasks[: args.max_print]:
            print(f"- {name}")
            for w in ws:
                print(f"    WARN: {w}")
        print()

    return 1 if bad_tasks else 0


def _normalise(msg: str) -> str:
    """Collapse task-specific details so we can group similar issues."""
    msg = re.sub(r"directory name '[^']+'", "directory name '<NAME>'", msg)
    msg = re.sub(r"\[task\]\.name='[^']+'", "[task].name='<VALUE>'", msg)
    msg = re.sub(r"short_name '[^']+'", "short_name '<NAME>'", msg)
    msg = re.sub(r"directory name '[^']+'", "directory name '<NAME>'", msg)
    return msg


if __name__ == "__main__":
    raise SystemExit(main())
