#!/usr/bin/env python3
"""Load ReVSI (Rebuilding Visual Spatial Intelligence, ICML 2026) into the task
objects the ERQA adapter emits.

ReVSI is VSI-Bench rebuilt: 381 indoor scenes from ScanNet v2, ScanNet++ v2,
ARKitScenes, 3RScan and MultiScan, re-annotated, every QA pair regenerated and
human-verified, and -- the part that matters for a loader -- shipped in FOUR
frame-budget variants (16 / 32 / 64 / all frames per video). A question exists
in a variant only if it is answerable from that variant's frames, so the 16-frame
set has 4,568 questions, 32-frame 6,158, 64-frame 6,616, all-frame 6,808; ids are
shared across variants and the question text is identical.

HF `3dlg-hcvc/ReVSI`: `<budget>_frame/test-00000-of-00001.parquet` (QA only),
`metadata/sampled_video_frame_idx.json` (per scene, per budget, the frame indices
into the processed video), `video.zip` (one mp4 per scene, 640x480 / 480x640,
4-10 fps). The videos are decoded ONCE by ops/extract_revsi_frames.py into
`<data_dir>/frames_<budget>/<scene_id>/<k>_<src_idx>.png`; this loader only reads
those PNGs. Same reason as MMSI-Video: the perception stack consumes frames, and
decoding inside task generation would redo it per task.

Thirteen question types, two answer contracts (ANSWER_CONTRACTS in adapter.py):

| kind    | types                                                        | gold |
|---------|--------------------------------------------------------------|------|
| numeric | object_counting_single/multiple, object_size_estimation (cm), | number |
|         | object_abs_distance (m), room_size_estimation_single/multiple (m2) |   |
| letter  | object_rel_distance_closest/farthest, object_rel_direction_*  | A-D  |
|         | (forward/backward x easy/hard), route_planning               |      |

`options` is None exactly on the numeric types, so the kind is read off the row,
not off a type list. Numeric scoring is score_numeric.py's Mean Relative
Accuracy (thresholds 0.50..0.95 step 0.05) -- byte-for-byte the metric
lmms-eval's `revsi` task applies (`_mean_relative_accuracy(pred, gt, 0.5, 0.95,
0.05)`), counting included. Letter types are exact match.

Ids: `ReVSI_<id>` (id is the parquet's int64, unique across the set; 6 to 10
chars, well inside SAgent's 32-char trial-name cap).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ADAPTER_DIR = Path(__file__).resolve().parent
if str(_ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_DIR))

from adapter import (
    DEFAULT_GATEWAY_WORK_DIR,
    RUNNABLE_VISION_MODES,
    ERQAAdapter,
    ERQATask,
)
from load_mmsi import stratify

#: Local disk, never ossfs: ~19k PNG frames per 1000-question run.
DEFAULT_REVSI_DIR = Path(os.environ["REVSI_DIR"]) if os.environ.get("REVSI_DIR") else None
#: Which of ReVSI's variants to load. A benchmark parameter, not an
#: implementation detail: the frame budget changes which questions exist and
#: what the agent can see. 64 is the smallest variant that contains every id of
#: the ops/revsi_1000.json subset (the 32-frame variant is missing 71 of them).
DEFAULT_BUDGET = os.environ.get("REVSI_FRAME_BUDGET", "64")

PARQUET = "test-00000-of-00001.parquet"


def parquet_path(data_dir: Path, budget: str) -> Path:
    return data_dir / f"{budget}_frame" / PARQUET


def frames_dir(data_dir: Path, budget: str, scene_id: str) -> Path:
    return data_dir / f"frames_{budget}" / scene_id


def is_numeric(row) -> bool:
    opts = row["options"]
    return opts is None or len(opts) == 0


def build_question(row, n_frames: int) -> str:
    """Question (+ options for the letter kinds) after one line of framing.

    The framing line is the only thing added: without it the agent is told it
    has 64 "images" and nothing says they are consecutive views of one room,
    which is what makes `recon` over all of them the right first move.
    """
    lines = [
        (
            f"The {n_frames} images are frames sampled in chronological order from one "
            "continuous video walkthrough of a single indoor scene."
        ),
        "",
        str(row["question"]).strip(),
    ]
    if not is_numeric(row):
        lines.append("")
        lines.extend(str(o).strip() for o in row["options"])
    return "\n".join(lines)


def load_frames(data_dir: Path, budget: str, scene_id: str) -> list[bytes]:
    d = frames_dir(data_dir, budget, scene_id)
    files = sorted(d.glob("*.png")) if d.is_dir() else []
    if not files:
        raise FileNotFoundError(
            f"no extracted frames for scene {scene_id} under {d}; run "
            f"ops/extract_revsi_frames.py --budget {budget} first"
        )
    if len(files) != int(budget):
        # Loud: a task with fewer frames than the variant defines is a different
        # benchmark condition, not a warning.
        raise ValueError(
            f"scene {scene_id}: {len(files)} frames under {d}, expected {budget}"
        )
    return [f.read_bytes() for f in files]


def load_revsi_tasks_from_rows(rows, data_dir: Path, budget: str) -> list[ERQATask]:
    tasks: list[ERQATask] = []
    for row in rows:
        qid = int(row["id"])
        gold = str(row["ground_truth"]).strip()
        numeric = is_numeric(row)
        if numeric:
            try:
                float(gold)
            except ValueError as exc:
                raise ValueError(
                    f"ReVSI id={qid}: numeric gold {gold!r} is not a number"
                ) from exc
        elif gold not in "ABCD" or len(gold) != 1:
            raise ValueError(f"ReVSI id={qid}: letter gold {gold!r} is not A-D")
        image_bytes = load_frames(data_dir, budget, str(row["scene_id"]))
        tasks.append(
            ERQATask(
                question_id=f"ReVSI_{qid}",
                question=build_question(row, len(image_bytes)),
                question_type=str(row["question_type"]),
                answer=gold,
                visual_indices=[],  # frames are files, never interleaved
                image_bytes=image_bytes,
                answer_kind="numeric" if numeric else "letter",
                # 64 `[image: ...]` markers in a row would be noise; the image
                # block already lists every file (same call as MMSI-Video).
                interleave_images=False,
            )
        )
    return tasks


def read_table(data_dir: Path, budget: str):
    import pandas as pd

    p = parquet_path(data_dir, budget)
    if not p.is_file():
        raise FileNotFoundError(f"ReVSI parquet not found: {p}")
    df = pd.read_parquet(p)
    df["options"] = [None if o is None else list(o) for o in df["options"]]
    return df


def load_revsi_tasks(
    data_dir: Path, budget: str, limit: int | None = None, stratified: bool = True
) -> list[ERQATask]:
    df = read_table(data_dir, budget)
    if limit is not None and limit < len(df):
        df = stratify(df, limit) if stratified else df.head(limit)
    return load_revsi_tasks_from_rows(df.to_dict("records"), data_dir, budget)


def load_revsi_tasks_from_manifest(
    data_dir: Path, budget: str, manifest: Path
) -> list[ERQATask]:
    wanted = json.loads(manifest.read_text()).get("task_ids")
    if not isinstance(wanted, list) or not wanted:
        raise ValueError(f"manifest {manifest} must contain a non-empty task_ids list")
    df = read_table(data_dir, budget)
    by_id = {f"ReVSI_{int(i)}": r for i, r in zip(df["id"], df.to_dict("records"))}
    missing = [t for t in wanted if t not in by_id]
    if missing:
        raise ValueError(
            f"manifest has {len(missing)} ids absent from the {budget}-frame variant "
            f"(e.g. {missing[:5]}); use a larger budget or a different manifest"
        )
    return load_revsi_tasks_from_rows([by_id[t] for t in wanted], data_dir, budget)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate SAgent tasks from ReVSI.")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_REVSI_DIR,
        help=f"ReVSI dir with <budget>_frame/ parquets and frames_<budget>/ (default: $REVSI_DIR)",
    )
    ap.add_argument(
        "--budget",
        default=DEFAULT_BUDGET,
        choices=["16", "32", "64"],
        help=f"frame-budget variant to load (default: {DEFAULT_BUDGET}, env REVSI_FRAME_BUDGET)",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Load exactly the ReVSI_<id> ids listed in this JSON manifest, in order",
    )
    ap.add_argument("--overwrite", action="store_true")
    # Same flags as the other loaders: scripts/run_job.sh hands identical
    # arguments to whichever generator $DATASET selects.
    ap.add_argument(
        "--vision-mode", choices=list(RUNNABLE_VISION_MODES), default="disk"
    )
    ap.add_argument("--gateway-work-dir", type=Path, default=DEFAULT_GATEWAY_WORK_DIR)
    ap.add_argument("--no-stage-images", action="store_true")
    ap.add_argument(
        "--head-order",
        action="store_true",
        help="First --limit rows instead of a stratified sample",
    )
    ap.add_argument("--skills-dir", type=Path, default=None)
    ap.add_argument("--prompt-dir", type=Path, default=None)
    ap.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        help="profile's own container CLI directory (configs/profile/<tag>/workspace); "
        "run_job.sh passes $WORKSPACE_DIR",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes for task generation (64 PNGs per task; serial is ~5 s/task)",
    )
    args = ap.parse_args()
    if args.data_dir is None:
        ap.error("no data location: set REVSI_DIR or pass --data-dir")

    try:
        if args.manifest is not None:
            tasks = load_revsi_tasks_from_manifest(
                args.data_dir, args.budget, args.manifest
            )
        else:
            tasks = load_revsi_tasks(
                args.data_dir, args.budget, args.limit, stratified=not args.head_order
            )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.skills_dir is not None and not args.skills_dir.is_dir():
        print(f"--skills-dir does not exist: {args.skills_dir}", file=sys.stderr)
        return 2
    if args.workspace_dir is not None and not args.workspace_dir.is_dir():
        print(f"--workspace-dir does not exist: {args.workspace_dir}", file=sys.stderr)
        return 2

    kinds: dict[str, int] = {}
    for t in tasks:
        kinds[t.answer_kind] = kinds.get(t.answer_kind, 0) + 1
    print(f"ReVSI {args.budget}-frame: {len(tasks)} tasks, answer kinds {kinds}")

    adapter = ERQAAdapter(
        output_dir=args.output_dir,
        gateway_work_dir=args.gateway_work_dir,
        stage_images=not args.no_stage_images,
        vision_mode=args.vision_mode,
        prompt_dir=args.prompt_dir,
        skills_dir=args.skills_dir,
        # Only when given: an adapter predating the workspace_dir hook (wry2's
        # /sagent-repo-s4 tree) must keep generating scene-v4 tasks.
        **(
            {"workspace_dir": args.workspace_dir}
            if args.workspace_dir is not None
            else {}
        ),
    )
    if args.skills_dir is not None:
        cards = sorted(p.parent.name for p in args.skills_dir.glob("*/SKILL.md"))
        print(f"shipping {len(cards)} cards from {args.skills_dir}: {cards}")
    created, skipped = adapter.generate_tasks(
        tasks, overwrite=args.overwrite, workers=max(1, args.workers)
    )
    print(f"generated {len(created)} tasks -> {args.output_dir}")
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}")
    return 0 if created else 1


if __name__ == "__main__":
    raise SystemExit(main())
