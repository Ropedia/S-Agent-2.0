#!/usr/bin/env python3
"""Load MMSI-Bench rows into the task objects the ERQA adapter already emits.

MMSI-Bench is multi-image spatial reasoning: several photos of one scene, asking
about positional relationships, camera/object motion, measured attributes, or
multi-step reasoning across views. It is the benchmark both SpatialClaw and
S-Agent report, so it is the one surface where a general-scaffold number is
directly comparable to published work.

Reusing `spatial.adapter` rather than forking it keeps a single code path
for image export, instruction rendering, the answer-file contract and scoring.
Only the *loader* differs, because MMSI's parquet schema differs from ERQA's:

| field          | ERQA                          | MMSI-Bench                     |
|----------------|-------------------------------|--------------------------------|
| id             | `question_id` (`"ERQA_1"`)    | `id` (int `0`)                 |
| images         | list of `{"bytes": ...}`      | list of raw `bytes`            |
| interleaving   | `visual_indices`              | absent — images all lead       |
| option syntax  | `A. text`                     | `A: text`                      |
| extras         | —                             | `thought`, `difficulty`        |

`thought` is a gold rationale. It is **never** exported into a task: doing so
would leak the answer. Only `question`, `images` and `answer` cross the boundary,
plus `question_type`/`difficulty` as metadata for slicing results afterwards.
ERQATask has no field for it, and load_mmsi_tasks_from_frame() names every field
it copies, so the omission is structural rather than a filter that could be
forgotten -- see tests/test_mmsi_arms.py for the regression that pins it.

Configuration selection mirrors run_adapter.py exactly (`--vision-mode`,
`--gateway-work-dir`, `--no-stage-images`, same choices, same `disk` default),
because scripts/run_job.sh hands the same arguments to whichever generator
`DATASET` selects.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# adapter.py sits beside this file (both moved into adapters/spatial/
# on 2026-08-21). Adding our own directory to sys.path keeps the plain
# `from adapter import ...` below working whether this is run as a script or
# imported -- there is no package __init__ here.
_ADAPTER_DIR = Path(__file__).resolve().parent
if str(_ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_DIR))

from adapter import (  # noqa: E402  (path set above)
    DEFAULT_GATEWAY_WORK_DIR,
    RUNNABLE_VISION_MODES,
    ERQAAdapter,
    ERQATask,
)

#: Seed for stratified sampling. Fixed so a given --limit always yields the
#: same question set -- an experiment series is only comparable if every version
#: saw identical questions, and `df.head(n)` used to provide that by accident.
SAMPLE_SEED = 20260822


def stratify(df, limit: int, seed: int = SAMPLE_SEED):
    """Sample `limit` rows keeping each question_type's share of the full set.

    WHY NOT head(n). `head` takes the parquet's own row order, and MMSI's is not
    shuffled: the first 100 rows are badly unrepresentative of the benchmark.
    Measured against all 1000 rows --

        question_type                    first 100    full 1000
        MSR                                 12%         19.8%
        Positional (Cam.-Obj.)              20%          8.6%
        Positional (Obj.-Obj.)               2%          9.4%
        Positional (Cam.-Reg.)               1%          8.3%

    -- so `Cam.-Obj.` was over-weighted 2.3x while `Obj.-Obj.` and `Cam.-Reg.`
    were nearly absent. Any prompt or skill tuned on that slice is tuned on a
    distribution the benchmark does not have, which is the overfitting risk this
    project is explicitly trying to avoid.

    Largest-remainder allocation, so the per-type counts sum to exactly `limit`
    rather than drifting by a few rows from independent rounding.

    ⚠️ A stratified set is NOT comparable with a head(n) set. Numbers from the
    two cannot go in one table, and a new baseline has to be run on the new
    sample before any version is compared against it.
    """
    import pandas as pd

    if limit >= len(df):
        return df
    shares = df["question_type"].value_counts()
    exact = {qt: n / len(df) * limit for qt, n in shares.items()}
    alloc = {qt: int(v) for qt, v in exact.items()}
    # Hand out the leftover rows to the largest fractional parts.
    short = limit - sum(alloc.values())
    for qt in sorted(exact, key=lambda q: exact[q] - alloc[q], reverse=True)[:short]:
        alloc[qt] += 1
    parts = [
        grp.sample(n=alloc[qt], random_state=seed)
        for qt, grp in df.groupby("question_type", sort=True)
        if alloc.get(qt)
    ]
    # Sorted by the original index so the task order is stable and readable,
    # not grouped by type (which would make a partial run cover only some types).
    return pd.concat(parts).sort_index()


def load_mmsi_tasks(
    parquet: Path, limit: int | None = None, stratified: bool = True
) -> list[ERQATask]:
    """Read MMSI-Bench rows into ERQATask objects.

    Raises on a schema surprise instead of silently producing a broken task: a
    wrong image type or a missing answer would otherwise become an unscorable
    trial that looks like an agent failure.

    `stratified` keeps each question_type's share of the benchmark (see
    `stratify`). Pass False only to reproduce a pre-2026-08-22 run, whose task
    set came from `df.head(limit)`.
    """
    import pandas as pd

    df = pd.read_parquet(parquet)
    if limit is not None:
        df = stratify(df, limit) if stratified else df.head(limit)
    return load_mmsi_tasks_from_frame(df)


def load_mmsi_tasks_from_manifest(parquet: Path, manifest: Path) -> list[ERQATask]:
    """Load exactly the task IDs recorded in a frozen manifest."""
    import pandas as pd

    payload = json.loads(manifest.read_text())
    task_ids = payload.get("task_ids")
    if not isinstance(task_ids, list) or not task_ids or not all(
        isinstance(item, str) for item in task_ids
    ):
        raise ValueError(f"manifest {manifest} must contain a non-empty task_ids list")

    frame = pd.read_parquet(parquet)
    by_id = {f"MMSI_{row.id}": row for row in frame.itertuples()}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"manifest contains IDs missing from parquet: {missing[:5]}")
    selected = frame.set_index(frame["id"].map(lambda value: f"MMSI_{value}"))
    return load_mmsi_tasks_from_frame(selected.loc[task_ids].reset_index(drop=True))


def load_mmsi_tasks_from_frame(df) -> list[ERQATask]:
    """Row-to-task conversion, split out from the parquet read.

    Separate so the `thought`-never-exported guarantee can be tested against a
    synthetic row: the real parquet is 704MB and not in the repo, and a leak
    test that skips when the data is absent is a leak test that never runs.
    """
    required = {"id", "images", "question", "answer", "question_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MMSI parquet missing columns: {sorted(missing)}")

    tasks: list[ERQATask] = []
    for _, row in df.iterrows():
        raw_images = list(row["images"])
        image_bytes: list[bytes] = []
        for idx, item in enumerate(raw_images):
            # MMSI stores raw bytes; tolerate the ERQA-style {"bytes": ...} too
            # so this loader keeps working if the dataset is ever re-exported.
            if isinstance(item, bytes):
                image_bytes.append(item)
            elif isinstance(item, dict) and isinstance(item.get("bytes"), bytes):
                image_bytes.append(item["bytes"])
            else:
                raise TypeError(
                    f"MMSI id={row['id']} image[{idx}] has unexpected type "
                    f"{type(item).__name__}; expected bytes or {{'bytes': ...}}"
                )

        answer = str(row["answer"]).strip()
        if not answer:
            raise ValueError(f"MMSI id={row['id']} has an empty answer")

        tasks.append(
            ERQATask(
                # Prefix so task dirs never collide with ERQA's and so the
                # benchmark of origin is readable straight off the trial name.
                question_id=f"MMSI_{row['id']}",
                question=str(row["question"]),
                question_type=str(row["question_type"]),
                answer=answer,
                visual_indices=[],  # MMSI does not interleave
                image_bytes=image_bytes,
            )
        )
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate SAgent tasks from MMSI-Bench.")
    ap.add_argument(
        "--parquet",
        type=Path,
        # Override with $MMSI_PARQUET or --parquet.
        default=(Path(os.environ["MMSI_PARQUET"]) if os.environ.get("MMSI_PARQUET") else None),
        help="MMSI-Bench parquet (default: $MMSI_PARQUET)",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Load exactly the MMSI task IDs from this JSON manifest",
    )
    ap.add_argument("--overwrite", action="store_true")
    # Configuration selection, deliberately the same flag names, choices and defaults as
    # adapters/spatial/run_adapter.py -- scripts/run_job.sh passes the
    # same arguments to whichever generator the dataset selects, so the two
    # CLIs have to stay interchangeable in this respect.
    ap.add_argument(
        "--vision-mode",
        choices=list(RUNNABLE_VISION_MODES),
        default="disk",
        help="How the agent is told to obtain pixels. "
        "disk=frames on disk plus the perception API; "
        "disk_only=frames on disk and nothing else, no perception at all",
    )
    ap.add_argument(
        "--gateway-work-dir",
        type=Path,
        default=DEFAULT_GATEWAY_WORK_DIR,
        help="Host work dir the perception gateway reads images from "
        f"(default: {DEFAULT_GATEWAY_WORK_DIR})",
    )
    ap.add_argument(
        "--no-stage-images",
        action="store_true",
        help="Do not copy frames into the gateway work dir",
    )
    ap.add_argument(
        "--head-order",
        action="store_true",
        help="Take the parquet's first --limit rows instead of a stratified "
        "sample. Only for reproducing runs made before 2026-08-22; the "
        "parquet's own order is badly unrepresentative (see stratify()).",
    )
    ap.add_argument(
        "--skills-dir",
        type=Path,
        default=None,
        help="Ship the SKILL.md cards from this directory instead of "
        "adapters/spatial/skills. For REMOVAL configurations: stage a copy with "
        "one card left out, so the configuration is a configuration rather than an edit "
        "to the tree. Editing the tree mid-run is worse than it sounds -- the "
        "image is built from the working tree, so a card removed and restored "
        "while a job is generating produces tasks that disagree with each "
        "other and with provenance.txt.",
    )
    ap.add_argument(
        "--prompt-dir",
        type=Path,
        default=None,
        help="Use prompt fragments from this directory instead of the legacy set",
    )
    ap.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        help="Ship this directory's perception.py (and geom.py / maskutil.py when "
        "present) instead of the task template's: a profile's own container "
        "CLI (configs/profile/<tag>/workspace). Unset = the template, as before.",
    )
    args = ap.parse_args()

    if args.parquet is None:
        ap.error("no data location: set MMSI_PARQUET or pass --parquet")
    if args.manifest is not None:
        tasks = load_mmsi_tasks_from_manifest(args.parquet, args.manifest)
    else:
        tasks = load_mmsi_tasks(args.parquet, args.limit, stratified=not args.head_order)
    if args.skills_dir is not None and not args.skills_dir.is_dir():
        # Fail here rather than let the adapter fall back to the default set:
        # a typo would silently produce a FULL-card configuration and the comparison
        # would read as "removing the card changed nothing".
        print(f"--skills-dir does not exist: {args.skills_dir}", file=sys.stderr)
        return 2

    adapter = ERQAAdapter(
        output_dir=args.output_dir,
        gateway_work_dir=args.gateway_work_dir,
        stage_images=not args.no_stage_images,
        vision_mode=args.vision_mode,
        prompt_dir=args.prompt_dir,
        skills_dir=args.skills_dir,
        workspace_dir=args.workspace_dir,
    )
    if args.skills_dir is not None:
        cards = sorted(p.parent.name for p in args.skills_dir.glob("*/SKILL.md"))
        print(f"shipping {len(cards)} cards from {args.skills_dir}: {cards}")
    created, skipped = adapter.generate_tasks(tasks, overwrite=args.overwrite)

    print(f"generated {len(created)} tasks -> {args.output_dir}")
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}")
    return 0 if created else 1


if __name__ == "__main__":
    raise SystemExit(main())
