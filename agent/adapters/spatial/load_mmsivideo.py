#!/usr/bin/env python3
"""Load MMSI-Video-Bench rows into the task objects the ERQA adapter emits.

MMSI-Video-Bench is video spatial & 4D reasoning: 1-2 videos per question,
shipped as **pre-extracted frames** with the capture time burned into each
frame's top-left corner. 1106 questions, 13 categories, multiple choice
(four options usually; 21 questions offer five and 103 six).

Reusing `spatial.adapter` rather than forking keeps one code path for
image export, instruction rendering, the answer-file contract and scoring --
same reasoning as load_mmsi.py / load_mindcube.py. Only the loader differs:

| field    | MMSI-Bench          | MindCube            | MMSI-Video-Bench          |
|----------|---------------------|---------------------|---------------------------|
| id       | `id` (int)          | `id` (str)          | `id` (`question_0000`)    |
| images   | bytes in the row    | paths on disk       | paths on disk, **per video** |
| options  | inline in question  | inline in question  | separate `options` list   |
| gold     | `answer`            | `gt_answer`         | `ground_truth` (A-F)      |
| type     | `question_type`     | id prefix           | `type` (13 categories)    |

Four things here have no counterpart in the other loaders.

**Frames are sampled, not shipped whole.** A question carries a median of 100
frames and up to 364 (129,242 across the set). Shipping them all would make a
task directory the size of a small dataset and hand the agent more images than
any context can hold, so each video is subsampled to $MMSIVIDEO_FRAMES_PER_VIDEO
(default 32, = SpatialClaw's num_key_frames) evenly-spaced frames, endpoints
included, chronological order preserved. This IS a benchmark parameter:
changing it changes what the agent can see, and the shipped/source counts are
recorded per video in the sidecar below.

**Files are named by video.** Frames ship as `v<K>_frame_<i>.png` -- the video
number in the name, the index `i` GLOBAL across the question (video 2 of a
two-video question continues where video 1 stopped) so the scene grammar
(`camera:f<i>`, `@f<i>`, `f<i>:label`) stays unique. Reference images ship as
`ref_<n>.png`: NOT a frame name, so the container CLI never feeds them to the
reconstruction (a reference image inside a DA3 batch corrupts the scene).

**Times go into a sidecar, not the question.** `frames.json` in the workspace
records every shipped frame's absolute clip time (the same clock as the
burned-in corner stamp) and which video it belongs to; the video-v1
perception.py `frames` command answers "which frame is nearest 1:14" / "which
frames fall in 0:19-0:20" from it. The question text carries only the
structural mapping (which files are video 1, video 2, reference image #n) and
that only when there is more than one video or any reference image -- a plain
single-video question is the original question plus its options, nothing else.

**Five- and six-option questions exist**, so gold can be E or F. The scorer's
extraction ladder was a-d until 2026-08-31 and would have scored every one of
those 0 no matter what the agent answered; see the note at `_ANSWER_TAG_RE` in
task_template/tests/score_erqa.py.

`videos/` (36.6 GB of mp4) is deliberately not required: the frames are what the
perception stack consumes, and the sagent configurations never call segment_video_*.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# adapter.py and load_mmsi.py sit beside this file. Adding our own directory to
# sys.path keeps the plain `from adapter import ...` below working whether this
# is run as a script or imported -- there is no package __init__ here.
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

#: Where the extracted MMSI-Video-Bench release lives. On h20 it is downloaded
#: from ModelScope (`xing0047/MMSI-Video-Bench`) onto the **local** disk --
#: /mnt/data is ossfs and reading 129k small JPEGs through it is unusable.
#: Locally: /media/gongrui/data/MMSI-Video-Bench (docs/DATA_MAP.md).
DEFAULT_MMSIVIDEO_DIR = Path(os.environ["MMSIVIDEO_DIR"]) if os.environ.get("MMSIVIDEO_DIR") else None

#: Frames kept per video. 32 is SpatialClaw's own budget, not a number picked
#: here: `spatial_agent/config.py` sets `num_key_frames = 32` and
#: `workflow.py` treats it as a PER-VIDEO allocation of evenly-spaced frames,
#: taking all of them when a video is shorter -- the same rule sample_evenly()
#: implements below. Keeping the two equal is what makes a number from this configuration
#: comparable to SpatialClaw's on the visible-frame axis.
#:
#: The alignment is only that axis. In SpatialClaw the 32 are the frames shown
#: to the main LLM while the tools can still reach every frame (and
#: `reconstruct_max_frames = 64` caps reconstruct separately); here what the
#: loader writes into the task dir is ALL the agent has, for both the LLM and
#: perception.
#:
#: A two-video question therefore ships 64 frames, one /recon batch. The
#: gateway's per-call cap is 64 for exactly this (GATEWAY_MAX_IMAGES); whether
#: DA3 / Pi3MOS fit 64 frames is a per-deployment GPU-memory measurement.
DEFAULT_FRAMES_PER_VIDEO = int(os.environ.get("MMSIVIDEO_FRAMES_PER_VIDEO", "32"))

#: Option letters, in order. Six is the maximum the released set uses.
LETTERS = "ABCDEF"

#: The timestamp sidecar, written into environment/workspace/ (so the Dockerfile
#: COPY lands it at /app/frames.json) and read by video-v1's `frames` command.
FRAMES_SIDECAR = "frames.json"
FRAMES_SCHEMA = "mmsivideo-frames-v1"

#: Released frame file names: `MM:SS.ss_frame_<interval>_<idx>.jpg` for 1268 of
#: the 1278 videos; the ten rh20t clips drop the timestamp prefix and are
#: `frame_<interval>_<idx>.jpg`. <idx> is the frame's index in the source video
#: at base_fps, so idx / base_fps is the exact time either way (the prefix is
#: that number rounded to 2 decimals, verified on every timestamped frame).
_FRAME_NAME_RE = re.compile(
    r"^(?:(?P<mm>\d+):(?P<ss>\d\d(?:\.\d+)?)_)?frame_(?P<interval>[\d.]+)_(?P<idx>\d+)\.jpe?g$"
)


def frame_time_s(rel_path: str, base_fps: float | None) -> tuple[float, int]:
    """(absolute clip time in seconds, source frame index) for one frame file.

    Prefers idx / base_fps (exact); falls back to the burned-in prefix when the
    row carries no fps, and to idx * interval when neither exists.
    """
    match = _FRAME_NAME_RE.match(os.path.basename(rel_path))
    if not match:
        raise ValueError(f"unrecognised MMSI-Video frame name: {rel_path}")
    idx = int(match.group("idx"))
    if base_fps:
        return idx / float(base_fps), idx
    if match.group("mm") is not None:
        return int(match.group("mm")) * 60 + float(match.group("ss")), idx
    return idx * float(match.group("interval")), idx


def sample_evenly(items: list, k: int) -> list:
    """`k` items spread over `items`, endpoints included, order preserved.

    Fewer than `k` available means take them all -- a short video is not padded.
    Indices are rounded onto a linear ramp, then de-duplicated, so a request for
    more frames than exist degrades to "all of them" instead of repeating.
    """
    if k <= 0 or len(items) <= k:
        return list(items)
    if k == 1:
        return [items[0]]
    idx = sorted({round(i * (len(items) - 1) / (k - 1)) for i in range(k)})
    return [items[i] for i in idx]


def build_question(
    item: dict, video_names: list[list[str]], ref_names: list[str]
) -> str:
    """Question text: structural file mapping (when needed), question, options.

    The mapping lines carry NO times: which files are video 1, which video 2,
    which are the reference images. They exist because the agent gets a flat
    directory and the question says "the first video" / "video 2"; a
    single-video question with no reference image needs none of it.
    """
    lines: list[str] = []
    if len(video_names) > 1 or ref_names:
        for k, names in enumerate(video_names, start=1):
            span = names[0] if len(names) == 1 else f"{names[0]} to {names[-1]}"
            lines.append(f"- {span}: video {k} ({len(names)} frames)")
        for n, name in enumerate(ref_names, start=1):
            lines.append(f"- {name}: reference image #{n}")

    question = item.get("ori_question", "")
    if ref_names and question.count("<image>") == len(ref_names):
        # The raw question interleaves `<image>` tags for its reference images.
        # Name them instead: the agent sees files, not an interleaved stream.
        for n in range(1, len(ref_names) + 1):
            question = question.replace("<image>", f"[reference image #{n}]", 1)

    options = item.get("options") or []
    rendered = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    body = f"{question}\n\n{rendered}"
    return f"{chr(10).join(lines)}\n\n{body}" if lines else body


def build_frames_sidecar(
    item: dict,
    kept_groups: list[list[str]],
    video_names: list[list[str]],
    ref_names: list[str],
) -> dict:
    """frames.json: per-video summary + per-frame absolute/relative times.

    Times are absolute clip seconds -- the clock the burned-in corner stamp
    and the questions use ("At 1 minute and 14 seconds"). `t_rel_s` counts
    from the video's own start for questions that say "7 seconds into the
    second video". Nothing here names the source dataset.
    """
    videos_meta = item.get("video_list") or []
    videos: list[dict] = []
    frames: list[dict] = []
    for k, (kept, names) in enumerate(zip(kept_groups, video_names), start=1):
        meta = videos_meta[k - 1] if k - 1 < len(videos_meta) else {}
        fps = meta.get("base_fps") or None
        times = [frame_time_s(rel, fps) for rel in kept]
        start = meta.get("start")
        start = float(start) if start is not None else times[0][0]
        end = meta.get("end")
        end = float(end) if end is not None else times[-1][0]
        first = int(names[0].rsplit("_", 1)[1].split(".")[0])
        for name, (t_s, src_idx) in zip(names, times):
            frames.append(
                {
                    "frame": int(name.rsplit("_", 1)[1].split(".")[0]),
                    "video": k,
                    "file": name,
                    "t_s": round(t_s, 3),
                    "t_rel_s": round(t_s - start, 3),
                    "source_index": src_idx,
                }
            )
        videos.append(
            {
                "video": k,
                "first_frame": first,
                "last_frame": frames[-1]["frame"],
                "n_shipped": len(kept),
                "n_source": len(item["frames_list"][k - 1]),
                "fps": fps,
                "start_s": round(start, 3),
                "end_s": round(end, 3),
            }
        )
    return {
        "schema": FRAMES_SCHEMA,
        "videos": videos,
        "frames": frames,
        "refs": [
            {"file": name, "label": f"reference image #{n}"}
            for n, name in enumerate(ref_names, start=1)
        ],
    }


def load_mmsivideo_tasks_from_records(
    records: list[dict], data_dir: Path, frames_per_video: int
) -> list[ERQATask]:
    """Record-to-task conversion, split out from the JSON read for testing."""
    tasks: list[ERQATask] = []
    for item in records:
        qid = str(item["id"])
        gold = str(item.get("ground_truth", "")).strip().upper()
        options = item.get("options") or []
        if gold not in LETTERS[: len(options)]:
            raise ValueError(
                f"MMSI-Video id={qid}: ground_truth {gold!r} is not one of the "
                f"{len(options)} options"
            )

        kept_groups: list[list[str]] = []
        video_names: list[list[str]] = []
        image_bytes: list[bytes] = []
        image_names: list[str] = []
        next_index = 0
        for gi, group in enumerate(item.get("frames_list") or []):
            kept = sample_evenly(list(group), frames_per_video)
            names: list[str] = []
            for rel in kept:
                path = data_dir / "frames" / rel
                if not path.is_file():
                    # Loud, not skipped: a task generated with missing frames is
                    # unanswerable but still looks like a normal trial, and would
                    # be scored as a model failure.
                    raise FileNotFoundError(
                        f"MMSI-Video id={qid} frame not found: {path}"
                    )
                image_bytes.append(path.read_bytes())
                name = f"v{gi + 1}_frame_{next_index}.png"
                names.append(name)
                image_names.append(name)
                next_index += 1
            kept_groups.append(kept)
            video_names.append(names)
        if not image_bytes:
            raise ValueError(f"MMSI-Video id={qid} has no frames")

        refs = [r for r in (item.get("ref_images") or []) if isinstance(r, str)]
        ref_names: list[str] = []
        for n, rel in enumerate(refs, start=1):
            path = data_dir / "ref_images" / rel
            if not path.is_file():
                raise FileNotFoundError(
                    f"MMSI-Video id={qid} reference image not found: {path}"
                )
            image_bytes.append(path.read_bytes())
            ref_names.append(f"ref_{n}.png")
            image_names.append(ref_names[-1])

        sidecar = build_frames_sidecar(item, kept_groups, video_names, ref_names)
        tasks.append(
            ERQATask(
                # Prefix so task dirs never collide with the other benchmarks',
                # and so the benchmark of origin is readable off the trial name.
                question_id=f"MMSIVideo_{qid}",
                question=build_question(item, video_names, ref_names),
                question_type=str(item.get("type", "")) or "unknown",
                answer=gold,
                visual_indices=[],  # frames are files, never interleaved
                image_bytes=image_bytes,
                image_names=image_names,
                workspace_files={FRAMES_SIDECAR: json.dumps(sidecar, indent=1) + "\n"},
                interleave_images=False,
            )
        )
    return tasks


def load_mmsivideo_tasks(
    data_dir: Path,
    limit: int | None = None,
    stratified: bool = True,
    frames_per_video: int = DEFAULT_FRAMES_PER_VIDEO,
) -> list[ERQATask]:
    """Read mmsivideo.json into ERQATask objects.

    `stratified` keeps each category's share of the set, using the same
    largest-remainder allocation load_mmsi.stratify applies to MMSI question
    types -- imported rather than reimplemented, so a subset of any of these
    benchmarks is drawn by one rule. The file is grouped by category, so the
    unstratified head is a bad prefix.
    """
    import pandas as pd

    records = json.loads((data_dir / "mmsivideo.json").read_text())
    if not records:
        raise ValueError(f"no rows in {data_dir / 'mmsivideo.json'}")
    frame = pd.DataFrame(records)
    frame["question_type"] = frame["type"]
    if limit is not None:
        frame = stratify(frame, limit) if stratified else frame.head(limit)
    picked = {str(r) for r in frame["id"]}
    chosen = [r for r in records if str(r["id"]) in picked]
    return load_mmsivideo_tasks_from_records(chosen, data_dir, frames_per_video)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate SAgent tasks from MMSI-Video-Bench."
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_MMSIVIDEO_DIR,
        help=f"Extracted release dir (default: $MMSIVIDEO_DIR)",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--frames-per-video",
        type=int,
        default=DEFAULT_FRAMES_PER_VIDEO,
        help="Frames kept per video, evenly spaced "
        f"(default: {DEFAULT_FRAMES_PER_VIDEO}, env "
        "MMSIVIDEO_FRAMES_PER_VIDEO)",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Load exactly the MMSI-Video ids listed in this JSON manifest",
    )
    ap.add_argument("--overwrite", action="store_true")
    # Task generation is single-core bound on PNG encoding (~100 min for the
    # 1106-question set: 40k frames, zlib level 6 dominates). Each task writes
    # only its own directory, so a process pool is safe; the output must stay
    # byte-identical to --workers 1, and that is what the tests assert.
    ap.add_argument("--workers", type=int, default=1)
    # Configuration selection: same flag names, choices and defaults as the other loaders --
    # scripts/run_job.sh hands the same arguments to whichever generator $DATASET
    # selects, so the CLIs have to stay interchangeable.
    ap.add_argument(
        "--vision-mode", choices=list(RUNNABLE_VISION_MODES), default="disk"
    )
    ap.add_argument("--gateway-work-dir", type=Path, default=DEFAULT_GATEWAY_WORK_DIR)
    ap.add_argument("--no-stage-images", action="store_true")
    ap.add_argument(
        "--head-order",
        action="store_true",
        help="Take the file's first --limit rows instead of a stratified "
        "sample. The file is grouped by category, so this is a bad "
        "prefix; it exists only for reproducing a run that used it.",
    )
    ap.add_argument("--skills-dir", type=Path, default=None)
    ap.add_argument("--prompt-dir", type=Path, default=None)
    ap.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        help="Ship this directory's perception.py (video-v1's `frames` "
        "command etc.) instead of the task template's",
    )
    args = ap.parse_args()
    if args.data_dir is None:
        ap.error("no data location: set MMSIVIDEO_DIR or pass --data-dir")

    json_path = args.data_dir / "mmsivideo.json"
    if not json_path.is_file():
        print(f"MMSI-Video JSON not found: {json_path}", file=sys.stderr)
        return 2
    if not (args.data_dir / "frames").is_dir():
        print(f"frames/ not found under {args.data_dir}", file=sys.stderr)
        return 2

    if args.manifest is not None:
        wanted = json.loads(args.manifest.read_text()).get("task_ids")
        if not isinstance(wanted, list) or not wanted:
            print(
                f"manifest {args.manifest} must contain a non-empty task_ids list",
                file=sys.stderr,
            )
            return 2
        records = json.loads(json_path.read_text())
        by_id = {f"MMSIVideo_{r['id']}": r for r in records}
        missing = [t for t in wanted if t not in by_id]
        if missing:
            print(
                f"manifest contains ids missing from the release: {missing[:5]}",
                file=sys.stderr,
            )
            return 2
        tasks = load_mmsivideo_tasks_from_records(
            [by_id[t] for t in wanted], args.data_dir, args.frames_per_video
        )
    else:
        tasks = load_mmsivideo_tasks(
            args.data_dir,
            args.limit,
            stratified=not args.head_order,
            frames_per_video=args.frames_per_video,
        )

    if args.skills_dir is not None and not args.skills_dir.is_dir():
        # Fail here rather than let the adapter fall back to the default set: a
        # typo would silently produce a FULL-card configuration and the comparison would
        # read as "removing the card changed nothing".
        print(f"--skills-dir does not exist: {args.skills_dir}", file=sys.stderr)
        return 2
    if args.workspace_dir is not None and not args.workspace_dir.is_dir():
        # Same reasoning: a video run on the template CLI has no `frames`
        # command, and the failure would look like the model ignoring time.
        print(f"--workspace-dir does not exist: {args.workspace_dir}", file=sys.stderr)
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
    created, skipped = adapter.generate_tasks(
        tasks, overwrite=args.overwrite, workers=args.workers
    )

    print(
        f"generated {len(created)} tasks -> {args.output_dir} "
        f"({args.frames_per_video} frames/video)"
    )
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}")
    return 0 if created else 1


if __name__ == "__main__":
    raise SystemExit(main())
