#!/usr/bin/env python3
"""Generate ERQA tasks in SAgent format from the local parquet.

Examples
--------
    # first 3 tasks, for inspection
    python3 run_adapter.py --output-dir /tmp/erqa_tasks --limit 3

    # specific questions
    python3 run_adapter.py --output-dir /tmp/erqa_tasks --task-ids ERQA_1 ERQA_154

    # the full benchmark
    python3 run_adapter.py --output-dir datasets/erqa_spatial --overwrite

Ordering is deterministic: tasks are emitted in parquet row order, which is
ERQA_1..ERQA_400. --task-ids preserves the order given on the command line.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import (  # noqa: E402
    DEFAULT_GATEWAY_WORK_DIR,
    RUNNABLE_VISION_MODES,
    ERQAAdapter,
    ERQATask,
    host_gid,
    host_uid,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# Override with $ERQA_PARQUET or --parquet. 2026-08-20: SpatialClaw was merged
# into the repro repo and the sibling SpatialClaw/ directory was deleted, so the
# default now points at repro/data/. Note this default SKIPS 18 tests silently
# when wrong (they assert "parquet absent" and pass as skipped), so a stale path
# here looks like a green run -- see docs/PITFALLS.md on silent-skip failures.
DEFAULT_PARQUET = Path(os.environ["ERQA_PARQUET"]) if os.environ.get("ERQA_PARQUET") else None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("question_id", "question", "question_type", "answer", "images")


def _as_list(value) -> list:
    """Normalize pandas/pyarrow nullable sequence values to a plain list.

    Truth-testing a numpy array raises, so ``value or []`` is not safe here.
    """
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def load_erqa_tasks(parquet_path: Path) -> list[ERQATask]:
    """Read the ERQA parquet into ERQATask objects, in row order."""
    import pandas as pd

    if not parquet_path.is_file():
        raise FileNotFoundError(f"ERQA parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"parquet is missing column(s): {', '.join(missing)}")

    tasks: list[ERQATask] = []
    for position, (_, row) in enumerate(df.iterrows(), start=1):
        question_id = str(row["question_id"]).strip()
        if not question_id:
            raise ValueError(f"row {position} has an empty question_id")

        answer = str(row["answer"]).strip()
        if not answer:
            raise ValueError(f"{question_id}: empty answer")

        image_bytes: list[bytes] = []
        for entry in _as_list(row["images"]):
            raw = entry.get("bytes") if isinstance(entry, dict) else entry
            if raw:
                image_bytes.append(bytes(raw))

        visual_indices = [int(i) for i in _as_list(row.get("visual_indices"))]

        tasks.append(
            ERQATask(
                question_id=question_id,
                question=str(row["question"]),
                question_type=str(row["question_type"] or "Unknown"),
                answer=answer,
                visual_indices=visual_indices,
                image_bytes=image_bytes,
            )
        )

    duplicates = len(tasks) - len({t.question_id for t in tasks})
    if duplicates:
        raise ValueError(f"parquet contains {duplicates} duplicate question_id(s)")

    logger.info("Loaded %d ERQA samples from %s", len(tasks), parquet_path)
    return tasks


def select_tasks(
    tasks: list[ERQATask], task_ids: list[str] | None, limit: int | None
) -> list[ERQATask]:
    """Apply --task-ids then --limit, deterministically."""
    if task_ids:
        by_id = {t.question_id: t for t in tasks}
        unknown = [tid for tid in task_ids if tid not in by_id]
        if unknown:
            raise ValueError(f"unknown task id(s): {', '.join(unknown)}")
        # Preserve command-line order, drop repeats.
        seen: set[str] = set()
        selected = []
        for tid in task_ids:
            if tid not in seen:
                seen.add(tid)
                selected.append(by_id[tid])
        tasks = selected

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        tasks = tasks[:limit]

    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate ERQA spatial reasoning tasks in SAgent format"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR.parent.parent / "datasets" / "erqa_spatial",
        help="Where to write task directories",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET,
        help="ERQA parquet path (default: $ERQA_PARQUET)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only generate the first N tasks"
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        metavar="ID",
        help="Only generate these question_ids (e.g. ERQA_1 ERQA_154)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing task directories instead of skipping them",
    )
    parser.add_argument(
        "--gateway-work-dir",
        type=Path,
        default=DEFAULT_GATEWAY_WORK_DIR,
        help="Host work dir the perception gateway reads images from "
        f"(default: {DEFAULT_GATEWAY_WORK_DIR})",
    )
    parser.add_argument(
        "--no-stage-images",
        action="store_true",
        help="Do not copy frames into the gateway work dir (paths are still "
        "baked into the instruction and helper)",
    )
    parser.add_argument(
        "--app-uid",
        type=int,
        default=None,
        help="uid the agent runs as inside the container, baked into the task "
        "Dockerfile and task.toml. Must match the uid the perception gateway "
        "runs as on this host, since both share the per-trial /perception "
        f"directory (default: this process's uid, {host_uid()})",
    )
    parser.add_argument(
        "--app-gid",
        type=int,
        default=None,
        help=f"gid to pair with --app-uid (default: {host_gid()})",
    )
    parser.add_argument(
        "--vision-mode",
        choices=list(RUNNABLE_VISION_MODES),
        default="disk",
        help="How the agent is told to obtain pixels. "
        "disk=frames on disk plus the perception API; "
        "disk_only=frames on disk and nothing else, no perception at all",
    )
    args = parser.parse_args()

    logger.info("=== ERQA -> SAgent adapter ===")
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Gateway work dir: %s", args.gateway_work_dir)

    try:
        if args.parquet is None:
            parser.error("no data location: set ERQA_PARQUET or pass --parquet")
        tasks = load_erqa_tasks(args.parquet)
        tasks = select_tasks(tasks, args.task_ids, args.limit)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Generating %d task(s)", len(tasks))

    adapter = ERQAAdapter(
        output_dir=args.output_dir,
        gateway_work_dir=args.gateway_work_dir,
        stage_images=not args.no_stage_images,
        vision_mode=args.vision_mode,
        app_uid=args.app_uid,
        app_gid=args.app_gid,
    )
    logger.info("Container agent uid:gid: %d:%d", adapter.app_uid, adapter.app_gid)
    success, skipped = adapter.generate_tasks(tasks, overwrite=args.overwrite)

    logger.info("Generated %d task(s)", len(success))
    if skipped:
        logger.info("Skipped %d task(s)", len(skipped))
        for reason in skipped[:5]:
            logger.info("  - %s", reason)
        if len(skipped) > 5:
            logger.info("  ... and %d more", len(skipped) - 5)

    multi = sum(1 for t in tasks if t.is_multi_image)
    logger.info(
        "Composition: %d single-image, %d multi-image", len(tasks) - multi, multi
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
