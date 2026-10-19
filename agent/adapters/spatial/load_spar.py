#!/usr/bin/env python3
"""Load SPAR-Bench rows into the task objects the ERQA adapter already emits.

SPAR-Bench (Zhang et al., "From Flatland to Space: Teaching Vision-Language
Models to Perceive and Reason in 3D", NeurIPS 2025; `jasonzhango/SPAR-Bench`)
is 7,207 QA pairs over 20 spatial tasks, single- and multi-view, graded into
Low / Middle / High cognitive levels. It is one of the ten benchmarks in
SpatialClaw's Table 1 (47.8 no-tool -> 58.4 with tools on our Qwen3.6-35B-A3B
backbone; see docs/reference/SPATIALCLAW_PAPER_NUMBERS.md).

**Three answer contracts in one benchmark** -- one more than SPBench:

    select              multiple choice, gold is a letter        -> "letter"
    fill                a bare quantity, gold is a number        -> "numeric"
      ... on a multi-view task                                   -> "numeric_last"
    view_change_infer   a 5-axis camera motion vector, written
                        as "move_right:X,move_up:Y,..."          -> "vci"

The taxonomy below is copied from `repro/spatial_agent/evals/sparbench.py` so
the two harnesses agree on which task is scored how; that file in turn follows
the benchmark's own eval_scripts. `numeric_last` exists because the official
extractor takes the LAST number on multi-view NA tasks and the FIRST on
single-view ones -- a rule about how their prompt makes models answer, kept
here so a number produced under our answer-file contract is scored the way the
published column scored theirs.

Unlike MMSI or MindCube, **SPAR questions carry their own format instructions**
("Your answer can only include one of options A, B.", "Answer using a single
number and nothing else. The unit is meter."). The task instruction therefore
does not restate the shape beyond naming the answer file -- the benchmark's own
wording is part of the item and must reach the model intact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# adapter.py and load_mmsi.py sit beside this file; see load_mmsi.py for why the
# path is set rather than making this a package.
_ADAPTER_DIR = Path(__file__).resolve().parent
if str(_ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_DIR))

from adapter import (  # noqa: E402  (path set above)
    DEFAULT_GATEWAY_WORK_DIR,
    RUNNABLE_VISION_MODES,
    ERQAAdapter,
    ERQATask,
)
from load_mmsi import stratify  # noqa: E402

DEFAULT_SPAR_DIR = Path(os.environ["SPAR_DIR"]) if os.environ.get("SPAR_DIR") else None

#: The one task whose answer is neither a letter nor a single number.
VCI_TASK = "view_change_infer"

#: Multi-view numeric tasks: the official extractor reads the LAST number in the
#: prediction for these and the first for every other numeric task.
MV_NA_TASKS = frozenset(
    {
        "depth_prediction_oc_mv",
        "depth_prediction_oo_mv",
        "distance_prediction_oc_mv",
        "distance_prediction_oo_mv",
    }
)

#: Cognitive levels, verbatim from the benchmark's own grouping. Not used to
#: filter anything -- carried so results can be sliced the way the paper
#: reports them (Low / Middle / High averages) without re-deriving the mapping.
COGNITIVE_LEVELS = {
    "Low": {
        "depth_prediction_oc",
        "depth_prediction_oo",
        "depth_prediction_oc_mv",
        "depth_prediction_oo_mv",
        "distance_prediction_oc",
        "distance_prediction_oo",
        "distance_prediction_oc_mv",
        "distance_prediction_oo_mv",
    },
    "Middle": {"view_change_infer", "position_matching", "camera_motion_infer"},
    "High": {
        "obj_spatial_relation_oo",
        "obj_spatial_relation_oc_mv",
        "obj_spatial_relation_oo_mv",
        "spatial_imagination_oc",
        "spatial_imagination_oo",
        "spatial_imagination_oc_mv",
        "spatial_imagination_oo_mv",
        "distance_infer_center_oo",
        "distance_infer_center_oo_mv",
    },
}


def answer_kind_of(task: str, format_type: str) -> str:
    """The answer contract for one row.

    Keyed on the task first because `view_change_infer` is stored as a fill
    question but its gold is a five-axis vector, not a number: scoring it with
    Mean Relative Accuracy over "the number in the text" would read the first
    axis and discard the rest.
    """
    if task == VCI_TASK:
        return "vci"
    fmt = (format_type or "").strip().lower()
    if fmt == "select":
        return "letter"
    if fmt == "fill":
        return "numeric_last" if task in MV_NA_TASKS else "numeric"
    raise ValueError(f"unknown SPAR format_type {format_type!r} for task {task!r}")


def load_spar_tasks(
    data_dir: Path,
    limit: int | None = None,
    stratified: bool = True,
) -> list[ERQATask]:
    """Read every shard into ERQATask objects.

    `stratified` keeps each task's share of the benchmark, using
    load_mmsi.stratify's largest-remainder allocation. It matters more here
    than anywhere else: the parquet shards are ordered BY TASK (shard 0 is all
    single-view depth/distance), so a head(n) of any size below ~1800 would be
    one cognitive level and four of the twenty tasks.
    """
    import pandas as pd

    shards = sorted(data_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no SPAR-Bench parquet shards in {data_dir}")
    frame = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
    frame["question_type"] = frame["task"]
    if limit is not None:
        frame = stratify(frame, limit) if stratified else frame.head(limit)
    return load_spar_tasks_from_frame(frame)


def load_spar_tasks_from_frame(df) -> list[ERQATask]:
    """Row-to-task conversion, split out from the parquet read so the contract
    assignment can be tested against synthetic rows (the real shards are a
    1.8 GB download that is not in the repo)."""
    required = {"id", "task", "format_type", "question", "answer", "image"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SPAR-Bench parquet missing columns: {sorted(missing)}")

    tasks: list[ERQATask] = []
    for _, row in df.iterrows():
        task = str(row["task"])
        kind = answer_kind_of(task, str(row["format_type"]))

        image_bytes: list[bytes] = []
        for idx, item in enumerate(list(row["image"])):
            # HuggingFace image structs are {"bytes": ..., "path": ...}; tolerate
            # raw bytes too so a re-export does not break the loader.
            if isinstance(item, bytes):
                image_bytes.append(item)
            elif isinstance(item, dict) and isinstance(item.get("bytes"), bytes):
                image_bytes.append(item["bytes"])
            else:
                raise TypeError(
                    f"SPAR id={row['id']} image[{idx}] has unexpected type "
                    f"{type(item).__name__}; expected bytes or {{'bytes': ...}}"
                )

        answer = str(row["answer"]).strip()
        if not answer:
            raise ValueError(f"SPAR id={row['id']} has an empty answer")

        tasks.append(
            ERQATask(
                # Deliberately SHORT, and without the task name in it.
                #
                # SAgent truncates a trial name to 32 characters before adding
                # its `__<hash>` suffix. `SPAR_<task>_<id>` runs to 37, so 3164
                # of the 7211 ids would be cut -- collapsing them to 4099
                # distinct names, 3112 collisions. Worse than unreadable:
                # `SPAR_obj_spatial_relation_oo_1825` truncates to
                # `..._182`, which reads as a DIFFERENT question that also
                # exists. The per-trial result.json keeps the full task_name, so
                # nothing is unrecoverable, but the job-level reward_stats are
                # keyed by the truncated trial name and a per-question join off
                # them would silently mis-attribute.
                #
                # `id` is unique across the concatenated shards (checked:
                # 7211 rows, ids 0..7210), so this is both short and unambiguous.
                # Which of the twenty tasks a trial came from is carried in
                # question_type -> task.toml metadata + result.json, which is
                # where per-task slicing reads it from anyway.
                question_id=f"SPAR_{row['id']}",
                question=str(row["question"]),
                question_type=task,
                answer=answer,
                visual_indices=[],  # SPAR does not interleave
                image_bytes=image_bytes,
                answer_kind=kind,
            )
        )
    return tasks


def manifest_index(frame):
    """Index a raw shard frame by the SAME id `load_spar_tasks_from_frame` puts
    in `question_id`.

    These two used to be written out separately and drifted: the task builder
    was shortened to `SPAR_<id>` (SAgent's 32-char trial-name truncation, see
    the comment there) while the manifest filter kept `SPAR_<task>_<id>`. The
    failure was loud rather than silent -- every id missed and the run aborted
    -- but it aborted only after run_job.sh had done its setup. One expression,
    pinned by test_manifest_ids_are_the_ids_the_loader_emits.
    """
    return "SPAR_" + frame["id"].astype(str)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate SAgent tasks from SPAR-Bench.")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_SPAR_DIR,
        help=f"Dir holding the test-*.parquet shards (default: $SPAR_DIR)",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Load exactly the SPAR task ids listed in this JSON manifest",
    )
    ap.add_argument("--overwrite", action="store_true")
    # Same flag names, choices and defaults as run_adapter.py / load_mmsi.py:
    # scripts/run_job.sh passes the same arguments to whichever generator
    # $DATASET selects, so the CLIs have to stay interchangeable.
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
        help="Take the concatenated shards' first --limit rows instead of a "
        "stratified sample. The shards are ordered by task, so this is a very "
        "bad prefix -- it exists only for reproducing a run that used it.",
    )
    ap.add_argument(
        "--skills-dir",
        type=Path,
        default=None,
        help="Ship the SKILL.md cards from this directory instead of "
        "adapters/spatial/skills (for removal configurations).",
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
    if args.data_dir is None:
        ap.error("no data location: set SPAR_DIR or pass --data-dir")

    if args.manifest is not None:
        import pandas as pd

        wanted = json.loads(args.manifest.read_text()).get("task_ids")
        if not isinstance(wanted, list) or not wanted:
            print(
                f"manifest {args.manifest} must contain a non-empty task_ids list",
                file=sys.stderr,
            )
            return 2
        shards = sorted(args.data_dir.glob("*.parquet"))
        frame = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
        frame["question_type"] = frame["task"]
        frame.index = manifest_index(frame)
        missing = [task_id for task_id in wanted if task_id not in frame.index]
        if missing:
            print(
                f"manifest ids missing from the shards: {missing[:5]}", file=sys.stderr
            )
            return 2
        tasks = load_spar_tasks_from_frame(frame.loc[wanted])
    else:
        tasks = load_spar_tasks(
            args.data_dir, args.limit, stratified=not args.head_order
        )

    if args.skills_dir is not None and not args.skills_dir.is_dir():
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

    kinds: dict[str, int] = {}
    for task in tasks:
        kinds[task.answer_kind] = kinds.get(task.answer_kind, 0) + 1
    print(f"generated {len(created)} tasks -> {args.output_dir} ({kinds})")
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}")
    return 0 if created else 1


if __name__ == "__main__":
    raise SystemExit(main())
