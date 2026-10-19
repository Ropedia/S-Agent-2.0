#!/usr/bin/env python3
"""Load SPBench rows into the task objects the ERQA adapter already emits.

SPBench (from SpatialLadder, `hongxingli/SPBench` on HuggingFace) is 3D spatial
perception over ScanNet scenes: 1009 single-image rows (SI) plus 319 multi-view
rows (MV, eight frames each). It is one of the ten benchmarks in SpatialClaw's
Table 1 (52.4 no-tool -> 63.0 with tools on our Qwen3.6-35B-A3B backbone; see
docs/reference/SPATIALCLAW_PAPER_NUMBERS.md).

**Two answer kinds in one benchmark.** SPBench uses the VSI-Bench taxonomy:

    MCA  object_rel_direction, object_rel_distance          458 rows  (34%)
         multiple choice, options in the `options` column, gold is a letter
    NA   object_abs_distance, object_size_estimation,       870 rows  (66%)
         object_counting
         a bare quantity, `options` is null, gold is a number

The published SPBench score mixes exact-match accuracy over MCA with Mean
Relative Accuracy over NA, so **both halves have to run** for a number here to
be comparable with that column -- which is why this loader marks NA rows
`answer_kind="numeric"` and the adapter then gives them their own instruction
wording (`instruction_numeric.md`), their own scorer (`score_numeric.py`) and a
verifier that writes a fractional reward (`test_numeric.sh`). An MCA-only run is
a different quantity and must not be reported in the same column.

Reusing `spatial.adapter` rather than forking keeps a single code path
for image export, instruction rendering and staging -- same reasoning as
load_mmsi.py. Only the *loader* differs:

| field    | MMSI-Bench             | SPBench                                  |
|----------|------------------------|------------------------------------------|
| id       | `id`, unique           | `id`, restarts at 1 in EACH of SI and MV |
| images   | raw `bytes` in the row | filenames under `<root>/<scene_name>/`   |
| options  | inline in `question`   | separate `options` column, or null       |
| gold     | `answer`               | `ground_truth` (letter OR number)        |
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

DEFAULT_SPBENCH_DIR = Path(os.environ["SPBENCH_DIR"]) if os.environ.get("SPBENCH_DIR") else None

#: The two shipped parquets, and the subset tag each contributes to task ids.
#: Both are loaded by default: the published score is over the union, and `id`
#: is only unique *within* a parquet, which is what the tag disambiguates.
SUBSETS = (("SI", "SPBench-SI.parquet"), ("MV", "SPBench-MV.parquet"))

#: VSI-Bench taxonomy, verbatim from repro/spatial_agent/evals/spbench.py so the
#: two harnesses agree on which half of the benchmark a row belongs to.
MCA_QUESTION_TYPES = frozenset(
    {
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
        "object_rel_direction",
        "object_rel_distance",
    }
)
NA_QUESTION_TYPES = frozenset(
    {
        "object_abs_distance",
        "object_counting",
        "object_size_estimation",
        "room_size_estimation",
    }
)


def answer_kind_of(question_type: str, options) -> str:
    """ "letter" or "numeric" for one row.

    Decided from the question type, with the presence of options as a
    cross-check rather than the rule: a null `options` on an MCA type (or a
    populated one on an NA type) means the release changed shape, and silently
    guessing would produce tasks whose instruction asks for the wrong kind of
    answer and score 0 while looking like model failures.
    """
    has_options = options is not None and len(list(options)) > 0
    if question_type in MCA_QUESTION_TYPES:
        if not has_options:
            raise ValueError(f"MCA type {question_type!r} with no options")
        return "letter"
    if question_type in NA_QUESTION_TYPES:
        if has_options:
            raise ValueError(f"NA type {question_type!r} unexpectedly has options")
        return "numeric"
    raise ValueError(f"unknown SPBench question_type {question_type!r}")


def render_question(question: str, options) -> str:
    """Question text as the agent sees it, options appended one per line.

    SPBench keeps options out of the question string, but the adapter reads the
    offered letters back OUT of the rendered question (ERQAAdapter._choice_letters,
    so the instruction never advertises a choice that does not exist). Rendering
    them as `A. text` on their own lines is what that parser expects, and it is
    also the layout MMSI and ERQA questions already carry inline.
    """
    lines = [str(option).strip() for option in list(options)]
    return f"{question.strip()}\n\n" + "\n".join(lines)


def load_spbench_tasks(
    data_dir: Path,
    limit: int | None = None,
    stratified: bool = True,
) -> list[ERQATask]:
    """Read SI + MV into ERQATask objects.

    `stratified` keeps each (subset, question_type) cell's share of the union,
    using load_mmsi.stratify's largest-remainder allocation. The key is the
    PAIR, not the type alone: SI and MV have very different type mixes (MV holds
    every counting row and no SI row is multi-view), so stratifying on type
    alone could return a sample with almost no multi-view questions -- the half
    a perception harness is most likely to differ on.
    """
    import pandas as pd

    frames = []
    for subset, filename in SUBSETS:
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"SPBench parquet not found: {path}")
        frame = pd.read_parquet(path)
        frame["subset"] = subset
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)

    if limit is not None:
        if stratified:
            keyed = combined.assign(
                question_type=combined["subset"] + "/" + combined["question_type"]
            )
            combined = combined.loc[stratify(keyed, limit).index]
        else:
            combined = combined.head(limit)
    return load_spbench_tasks_from_frame(combined, data_dir)


def load_spbench_tasks_from_frame(df, data_dir: Path) -> list[ERQATask]:
    """Row-to-task conversion, split out from the parquet read so the shape
    guarantees above can be tested against synthetic rows (the real images are a
    190 MB download that is not in the repo)."""
    required = {
        "id",
        "subset",
        "scene_name",
        "question_type",
        "question",
        "ground_truth",
        "options",
        "images",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SPBench parquet missing columns: {sorted(missing)}")

    tasks: list[ERQATask] = []
    for _, row in df.iterrows():
        question_type = str(row["question_type"])
        kind = answer_kind_of(question_type, row["options"])

        scene_dir = data_dir / str(row["scene_name"])
        image_bytes: list[bytes] = []
        for idx, filename in enumerate(list(row["images"])):
            path = scene_dir / str(filename)
            if not path.is_file():
                # Loud, not skipped: a multi-view question that silently lost a
                # frame is unanswerable but still looks like a normal trial.
                raise FileNotFoundError(
                    f"SPBench {row['subset']}_{row['id']} image[{idx}] not found: {path}"
                )
            image_bytes.append(path.read_bytes())

        answer = str(row["ground_truth"]).strip()
        if not answer:
            raise ValueError(
                f"SPBench {row['subset']}_{row['id']} has an empty ground_truth"
            )

        question = (
            render_question(str(row["question"]), row["options"])
            if kind == "letter"
            else str(row["question"]).strip()
        )

        tasks.append(
            ERQATask(
                # Subset tag included because `id` restarts at 1 in each parquet:
                # without it SI_1 and MV_1 would collide on one task directory.
                question_id=f"SPBench_{row['subset']}_{row['id']}",
                question=question,
                question_type=question_type,
                answer=answer,
                visual_indices=[],  # SPBench does not interleave
                image_bytes=image_bytes,
                answer_kind=kind,
            )
        )
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate SAgent tasks from SPBench.")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_SPBENCH_DIR,
        help=f"Dir holding the parquets and scene image dirs (default: $SPBENCH_DIR)",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Load exactly the SPBench task ids listed in this JSON manifest",
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
        help="Take the union's first --limit rows instead of a stratified "
        "sample. The parquets are grouped by type, so this is a bad prefix.",
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
        ap.error("no data location: set SPBENCH_DIR or pass --data-dir")

    if args.manifest is not None:
        import pandas as pd

        wanted = json.loads(args.manifest.read_text()).get("task_ids")
        if not isinstance(wanted, list) or not wanted:
            print(
                f"manifest {args.manifest} must contain a non-empty task_ids list",
                file=sys.stderr,
            )
            return 2
        frames = []
        for subset, filename in SUBSETS:
            frame = pd.read_parquet(args.data_dir / filename)
            frame["subset"] = subset
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        combined.index = combined["subset"] + "_" + combined["id"].astype(str)
        keys = [task_id.removeprefix("SPBench_") for task_id in wanted]
        missing = [key for key in keys if key not in combined.index]
        if missing:
            print(
                f"manifest contains ids missing from the parquets: {missing[:5]}",
                file=sys.stderr,
            )
            return 2
        tasks = load_spbench_tasks_from_frame(combined.loc[keys], args.data_dir)
    else:
        tasks = load_spbench_tasks(
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

    numeric = sum(1 for task in tasks if task.answer_kind == "numeric")
    print(
        f"generated {len(created)} tasks -> {args.output_dir} "
        f"({numeric} numeric / {len(tasks) - numeric} multiple-choice)"
    )
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}")
    return 0 if created else 1


if __name__ == "__main__":
    raise SystemExit(main())
