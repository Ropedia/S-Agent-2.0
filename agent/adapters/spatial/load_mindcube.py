#!/usr/bin/env python3
"""Load MindCube rows into the task objects the ERQA adapter already emits.

MindCube is multi-view spatial reasoning: 2-4 photos of one small scene taken
from different viewpoints, asking what is where *from a stated viewpoint*. It is
one of the ten benchmarks in SpatialClaw's Table 1, and the one with that
paper's largest tool-use delta on our backbone (45.4 -> 61.9 for
Qwen3.6-35B-A3B; see docs/reference/SPATIALCLAW_PAPER_NUMBERS.md), which is why
it is worth running the harness against.

Reusing `spatial.adapter` rather than forking it keeps a single code path
for image export, instruction rendering, the answer-file contract and scoring --
the same reasoning load_mmsi.py gives. Only the *loader* differs, because
MindCube ships JSONL plus loose JPEGs rather than a parquet of embedded bytes:

| field         | MMSI-Bench                  | MindCube                        |
|---------------|-----------------------------|---------------------------------|
| id            | `id` (int `0`)              | `id` (`"among_group693_q1_5_2"`)|
| images        | raw `bytes` in the row      | paths relative to the data root |
| options       | inline in `question` (`A:`) | inline in `question` (`A.`)     |
| gold          | `answer`                    | `gt_answer`                     |
| type          | `question_type`             | derived from the id prefix      |

`category` and `meta_info` are **never** exported into a task. `meta_info`
carries the scene's object list *and their ground-truth relations* -- exporting
it would hand the agent the answer. Like load_mmsi_tasks_from_frame(), the
conversion below names every field it copies, so the omission is structural
rather than a filter someone could forget.

Configuration selection mirrors run_adapter.py and load_mmsi.py exactly (same flag names,
same choices, same `disk` default), because scripts/run_job.sh hands the same
arguments to whichever generator `DATASET` selects.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# adapter.py and load_mmsi.py sit beside this file. Adding our own directory to
# sys.path keeps the plain `from adapter import ...` below working whether this
# is run as a script or imported -- there is no package __init__ here.
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

#: Where the extracted MindCube release lives. The download is one data.zip from
#: `MLL-Lab/MindCube` that expands to data/raw/*.jsonl + data/other_all_image/.
DEFAULT_MINDCUBE_DIR = Path(os.environ["MINDCUBE_DIR"]) if os.environ.get("MINDCUBE_DIR") else None

#: Default split. The release ships three JSONLs: MindCube.jsonl (21154 rows),
#: MindCube_tinybench.jsonl (1050) and MindCube_train.jsonl (10000, TRAINING --
#: never evaluate on it). tinybench is the official small evaluation set and is
#: what "full MindCube" means here: 21154 agent trials at this project's ~30 s
#: -> minutes per trial is not a run anyone is going to make, and tinybench is
#: the subset the benchmark's own authors publish for exactly that reason.
DEFAULT_SPLIT = "MindCube_tinybench.jsonl"

#: Settings, taken from the id prefix. The published protocol excludes the
#: `translation` setting from overall accuracy -- worth knowing, but no row in
#: either evaluation split actually carries it (tinybench is among/around/
#: rotation; the full set adds `aroundnew`), so nothing is filtered here. If a
#: future release adds translation rows, slice them out at analysis time using
#: this same question_type rather than dropping them at generation time.
EXCLUDED_FROM_OVERALL = frozenset({"translation"})


def setting_of(sample_id: str) -> str:
    """The MindCube setting a sample belongs to, from its id prefix.

    Ids look like `among_group693_q1_5_2` / `rotation_group12_q0` -- the prefix
    before the first underscore is the setting. Used as `question_type`, so the
    per-type slicing the analysis scripts already do for MMSI works unchanged.
    """
    return sample_id.split("_", 1)[0]


def load_mindcube_tasks(
    jsonl: Path,
    image_root: Path,
    limit: int | None = None,
    stratified: bool = True,
) -> list[ERQATask]:
    """Read MindCube JSONL rows into ERQATask objects.

    `stratified` keeps each setting's share of the split, using the same
    largest-remainder allocation load_mmsi.stratify applies to MMSI question
    types -- imported rather than reimplemented so a MindCube subset and an MMSI
    subset are drawn by one rule. Pass False to take the file's own order, which
    is grouped by setting and therefore a bad prefix.
    """
    import pandas as pd

    records = [
        json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()
    ]
    if not records:
        raise ValueError(f"no rows in {jsonl}")
    frame = pd.DataFrame(records)
    frame["question_type"] = frame["id"].map(setting_of)
    if limit is not None:
        frame = stratify(frame, limit) if stratified else frame.head(limit)
    return load_mindcube_tasks_from_frame(frame, image_root)


def load_mindcube_tasks_from_frame(df, image_root: Path) -> list[ERQATask]:
    """Row-to-task conversion, split out from the JSONL read.

    Separate for the same reason load_mmsi has the split: the real data is a
    600 MB download that is not in the repo, and a leak test that skips when the
    data is absent is a leak test that never runs.
    """
    required = {"id", "question", "images", "gt_answer", "question_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MindCube rows missing fields: {sorted(missing)}")

    tasks: list[ERQATask] = []
    for _, row in df.iterrows():
        image_bytes: list[bytes] = []
        for idx, relative in enumerate(list(row["images"])):
            path = image_root / str(relative)
            if not path.is_file():
                # Loud, not skipped: a task generated with fewer frames than the
                # question refers to ("image 2") is unanswerable but still looks
                # like a normal trial, and would be scored as a model failure.
                raise FileNotFoundError(
                    f"MindCube id={row['id']} image[{idx}] not found: {path}"
                )
            image_bytes.append(path.read_bytes())

        answer = str(row["gt_answer"]).strip()
        if not answer:
            raise ValueError(f"MindCube id={row['id']} has an empty gt_answer")

        tasks.append(
            ERQATask(
                # Prefix so task dirs never collide with ERQA's or MMSI's, and so
                # the benchmark of origin is readable straight off the trial name.
                question_id=f"MindCube_{row['id']}",
                question=str(row["question"]),
                question_type=str(row["question_type"]),
                answer=answer,
                visual_indices=[],  # MindCube does not interleave
                image_bytes=image_bytes,
            )
        )
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate SAgent tasks from MindCube.")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_MINDCUBE_DIR,
        help=f"Extracted MindCube data/ dir (default: $MINDCUBE_DIR)",
    )
    ap.add_argument(
        "--split",
        default=os.environ.get("MINDCUBE_SPLIT", DEFAULT_SPLIT),
        help=f"JSONL under data/raw/ (default: {DEFAULT_SPLIT})",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Load exactly the MindCube ids listed in this JSON manifest",
    )
    ap.add_argument("--overwrite", action="store_true")
    # Configuration selection: same flag names, choices and defaults as run_adapter.py and
    # load_mmsi.py -- scripts/run_job.sh passes the same arguments to whichever
    # generator the dataset selects, so the CLIs have to stay interchangeable.
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
        help="Take the split's first --limit rows instead of a stratified "
        "sample. The file is grouped by setting, so this is a bad prefix; it "
        "exists only for reproducing a run that used it.",
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
        ap.error("no data location: set MINDCUBE_DIR or pass --data-dir")

    jsonl = args.data_dir / "raw" / args.split
    if not jsonl.is_file():
        print(f"MindCube split not found: {jsonl}", file=sys.stderr)
        return 2
    if args.split == "MindCube_train.jsonl":
        # The release ships the training split beside the evaluation ones.
        print(
            "refusing MindCube_train.jsonl: that is the training split", file=sys.stderr
        )
        return 2

    if args.manifest is not None:
        import pandas as pd

        wanted = json.loads(args.manifest.read_text()).get("task_ids")
        if not isinstance(wanted, list) or not wanted:
            print(
                f"manifest {args.manifest} must contain a non-empty task_ids list",
                file=sys.stderr,
            )
            return 2
        records = [
            json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()
        ]
        by_id = {f"MindCube_{record['id']}": record for record in records}
        missing = [task_id for task_id in wanted if task_id not in by_id]
        if missing:
            print(
                f"manifest contains ids missing from {args.split}: {missing[:5]}",
                file=sys.stderr,
            )
            return 2
        frame = pd.DataFrame([by_id[task_id] for task_id in wanted])
        frame["question_type"] = frame["id"].map(setting_of)
        tasks = load_mindcube_tasks_from_frame(frame, args.data_dir)
    else:
        tasks = load_mindcube_tasks(
            jsonl, args.data_dir, args.limit, stratified=not args.head_order
        )

    if args.skills_dir is not None and not args.skills_dir.is_dir():
        # Fail here rather than let the adapter fall back to the default set: a
        # typo would silently produce a FULL-card configuration and the comparison would
        # read as "removing the card changed nothing".
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
