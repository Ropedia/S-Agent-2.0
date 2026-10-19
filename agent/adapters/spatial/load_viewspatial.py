#!/usr/bin/env python3
"""Load ViewSpatial-Bench rows into the task objects the ERQA adapter emits.

ViewSpatial-Bench (Li et al., 2025, arXiv:2505.21500) is multi-perspective
spatial localization: 5,712 four-choice questions over ScanNet v2 val frames
and MS-COCO val2017, in five task types -- two from the camera's own viewpoint
and three from the viewpoint of a person in the scene (relative direction,
object facing, scene simulation). It is one of the ten benchmarks in
SpatialClaw's Table 1, and the one where tools helped it LEAST (No-tool 52.0 ->
SpatialClaw 52.9). That is exactly why it is worth running: it separates "the
gains come from multi-view machinery" from "the gains come from the agent
scaffold" -- a negative control for the +8-14pp we see on MMSI/MindCube.

Reusing `spatial.adapter` keeps one code path for image export,
instruction rendering, the answer-file contract and scoring; only the loader
differs:

| field    | MindCube                 | ViewSpatial-Bench                          |
|----------|--------------------------|--------------------------------------------|
| id       | `id` (str)               | row index in ViewSpatial-Bench.json        |
| images   | paths on disk            | paths on disk, `ViewSpatial-Bench/` prefix |
| options  | inline in question       | separate `choices` ("A. ...\nB. ...")      |
| gold     | `gt_answer`              | `answer` ("A. right" -> "A")               |
| type     | id prefix                | `question_type` (5 categories)             |

Row index as the id is deliberate: it is what SpatialClaw's own loader uses as
`sample_id` (repro/spatial_agent/evals/viewspatial.py, `sample_id=idx`), so the
paper-subset manifest (`ops/viewspatial_paper_1000.json`, seed-42 shuffle of the
row order, first 1000) addresses the same questions the paper evaluated.

**All images ship, no subsampling.** 210 of the 1000 paper-subset questions are
multi-image (the Scene Simulation type, up to 32 frames). SpatialClaw's loader
also passes every frame through, and 32 is inside both limits we run with
(gateway GATEWAY_MAX_IMAGES=32 is a `>` check; the Qwen3.8 deployment accepted
64 and 128 images per request when probed). Subsampling would be a benchmark
parameter, and there is no need to introduce one here.
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

#: Where the extracted release lives: ViewSpatial-Bench.json beside the
#: scannetv2_val/ and val2017/ trees the two zips expand to. On h20 this is the
#: LOCAL disk -- 23,805 JPEGs through ossfs is not usable.
DEFAULT_VIEWSPATIAL_DIR = Path(os.environ["VIEWSPATIAL_DIR"]) if os.environ.get("VIEWSPATIAL_DIR") else None

LETTERS = "ABCD"


def resolve_image(data_dir: Path, rel: str) -> Path:
    """Same rule as SpatialClaw's loader: drop the HF repo-name prefix."""
    rel = rel.lstrip("./")
    rel = rel.removeprefix("ViewSpatial-Bench/")
    return data_dir / rel


def gold_letter(answer: str) -> str:
    """`answer` is spelled "A. right"; the contract wants the bare letter."""
    letter = str(answer).strip()[:1].upper()
    if letter not in LETTERS:
        raise ValueError(f"unexpected ViewSpatial answer {answer!r}")
    return letter


def build_question(entry: dict) -> str:
    """Question text followed by the four options, one per line.

    The options come as one newline-separated string already in "A. ..."
    form; they are passed through verbatim so the letters the agent sees are the
    letters the gold answer uses.
    """
    q = str(entry.get("question", "")).strip()
    choices = str(entry.get("choices", "")).strip()
    return f"{q}\n\n{choices}" if choices else q


def load_viewspatial_tasks_from_records(
    records: list[tuple[int, dict]], data_dir: Path
) -> list[ERQATask]:
    """(row index, entry) pairs -> ERQATask objects."""
    tasks: list[ERQATask] = []
    for idx, entry in records:
        image_bytes: list[bytes] = []
        for rel in entry.get("image_path") or []:
            path = resolve_image(data_dir, str(rel))
            if not path.is_file():
                # Loud, not skipped: a task generated with fewer images than the
                # question refers to is unanswerable but still looks like a
                # normal trial, and would be scored as a model failure.
                raise FileNotFoundError(
                    f"ViewSpatial row {idx}: image not found: {path}"
                )
            image_bytes.append(path.read_bytes())
        if not image_bytes:
            raise ValueError(f"ViewSpatial row {idx} has no images")
        tasks.append(
            ERQATask(
                question_id=f"ViewSpatial_{idx}",
                question=build_question(entry),
                question_type=str(entry.get("question_type", "")) or "unknown",
                answer=gold_letter(entry.get("answer", "")),
                visual_indices=[],  # images are files, never interleaved
                image_bytes=image_bytes,
            )
        )
    return tasks


def load_viewspatial_tasks(
    data_dir: Path, limit: int | None = None, stratified: bool = True
) -> list[ERQATask]:
    """Read ViewSpatial-Bench.json into tasks, optionally a stratified subset.

    Row order is the file's order (that is what the paper-subset manifest is
    defined against). A `--limit` without a manifest keeps each of the five
    types' share via load_mmsi.stratify, so subsets of every benchmark here are
    drawn by one rule.
    """
    import pandas as pd

    entries = json.loads((data_dir / "ViewSpatial-Bench.json").read_text())
    if not entries:
        raise ValueError(f"no rows in {data_dir / 'ViewSpatial-Bench.json'}")
    frame = pd.DataFrame(
        {
            "row": range(len(entries)),
            "question_type": [e.get("question_type", "") for e in entries],
        }
    )
    if limit is not None:
        frame = stratify(frame, limit) if stratified else frame.head(limit)
    picked = [int(r) for r in frame["row"]]
    return load_viewspatial_tasks_from_records(
        [(i, entries[i]) for i in picked], data_dir
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate SAgent tasks from ViewSpatial-Bench."
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_VIEWSPATIAL_DIR,
        help=f"Extracted release dir (default: $VIEWSPATIAL_DIR)",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Load exactly the ViewSpatial_<row> ids listed in this JSON manifest",
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
        help="First --limit rows instead of a stratified sample (the file is "
        "grouped by type, so this is a bad prefix; for reproducing a run only)",
    )
    ap.add_argument("--skills-dir", type=Path, default=None)
    ap.add_argument("--prompt-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.data_dir is None:
        ap.error("no data location: set VIEWSPATIAL_DIR or pass --data-dir")

    json_path = args.data_dir / "ViewSpatial-Bench.json"
    if not json_path.is_file():
        print(f"ViewSpatial-Bench.json not found: {json_path}", file=sys.stderr)
        return 2

    if args.manifest is not None:
        wanted = json.loads(args.manifest.read_text()).get("task_ids")
        if not isinstance(wanted, list) or not wanted:
            print(
                f"manifest {args.manifest} must contain a non-empty task_ids list",
                file=sys.stderr,
            )
            return 2
        entries = json.loads(json_path.read_text())
        by_id = {f"ViewSpatial_{i}": (i, e) for i, e in enumerate(entries)}
        missing = [t for t in wanted if t not in by_id]
        if missing:
            print(
                f"manifest contains ids missing from the release: {missing[:5]}",
                file=sys.stderr,
            )
            return 2
        tasks = load_viewspatial_tasks_from_records(
            [by_id[t] for t in wanted], args.data_dir
        )
    else:
        tasks = load_viewspatial_tasks(
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
