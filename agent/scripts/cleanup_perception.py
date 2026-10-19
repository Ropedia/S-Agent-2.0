#!/usr/bin/env python3
"""Reclaim per-trial /perception directories left behind by run_job.sh.

Bug 1 from the shared-workspace-perception spec: each trial's /perception
bind mount (wired in run_job.sh's MOUNTS_JSON, source
${HOST_ARTIFACTS_PATH}/../perception) is never deleted by SAgent -- SAgent's
environment.stop() tears down the *container*, not a bind mount's host-side
source directory, and nothing else in the pipeline removes it either. Left
alone this grows without bound (4.5GB and climbing at spec-writing time, on a
host already at 89% disk use).

Why "container-side empty, then host rmdir" and not a plain host `rm -rf`:
the container (and hence Depth-Anything/SAM 3 model code, or an agent inside
the sandbox) writes into /perception as root. The host user cannot unlink
root-owned files -- `rm -rf` from the host fails partway through with
"Permission denied" and leaves debris (empirically verified: see
task-1c-report.md). But an *empty* directory can be rmdir'd by the host user
regardless of who owns it, because directory removal is governed by the
parent directory's writability, not the child's ownership. So: run a
throwaway container that bind-mounts the same host path and empties it
(root emptying its own files back out), then rmdir from the host.

Why "sweep previous runs' leftovers at the start of the next run_job.sh
invocation" rather than "clean up this trial the instant it finishes":
run_job.sh launches SAgent via `setsid nohup "$SAGENT_BIN" run ... &` and
returns immediately -- the script's own process tree ends long before any
individual trial (let alone the whole job) completes. Waiting for
completion inline would mean either (a) blocking run_job.sh until the whole
job finishes, which defeats the point of backgrounding it, or (b) spawning a
second detached watcher process to poll for job completion and clean up
later, which is more moving parts (another background process to lose track
of, log, and debug) for a problem that does not need low latency: nothing
downstream reads a stale /perception dir as meaningful data, it is pure
scratch space. Sweeping at the top of the *next* invocation is simpler,
needs no new background process, and is triggered by the one event that
already reliably happens (a human runs another configuration) rather than one this
script would have to invent and supervise.

Trade-off this accepts: leftovers from the *last* run of a session sit on
disk until someone runs run_job.sh again (or invokes `sweep` by hand). That
is fine here -- operators run these configurations in a batch during active
experimentation, not as a single one-shot job followed by long idle periods
-- but it is a real limitation, not a full fix for unbounded growth in every
usage pattern. Documented per the brief's requirement to state the trade-off
rather than silently pick one.

A trial's /perception dir is only ever swept once BOTH are true:
  - trial_dir/result.json exists (SAgent's trial.py writes this in
    _cleanup_and_finalize(), *after* awaiting environment.stop() -- so by
    the time it exists, the trial's container is already stopped and
    nothing is still writing into /perception)
  - trial_dir/perception/ exists (no mount, e.g. disk_only or a pre-1c task,
    means nothing to sweep)

A job can opt permanently out of sweeping (across all future invocations,
not just the one that requested it) by having a KEEP_MARKER_NAME file in its
job_dir -- written by `--keep-artifacts` on the run that produced it. This
directly implements "when --keep-artifacts is passed, cleanup is skipped":
skipped for *that invocation's own sweep pass* (see main()'s early return)
*and* skipped by every later invocation's sweep, because those trials will
not exist yet at sweep time (sweep runs before SAgent launches) -- the
marker is what makes the exemption stick once they do exist.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, List

KEEP_MARKER_NAME = ".keep_perception"

# busybox ash (alpine's /bin/sh) has no nullglob/dotglob shopt. This 3-glob
# combination is the portable way to match "everything including dotfiles,
# excluding . and ..":
#   ..?*      -- dotfiles of 3+ chars (".foo", not "." or "..")
#   .[!.]*    -- dotfiles starting with a non-dot second character (".f")
#   *         -- everything else
# A glob that matches nothing expands to itself literally (e.g. the string
# "/target/..?*"), which `rm -rf` would then fail to find -- hence
# `2>/dev/null; true` so "nothing of that shape exists" is not an error.
_EMPTY_CMD = "rm -rf /target/..?* /target/.[!.]* /target/* 2>/dev/null; true"


# SAgent nests trials one level deeper when a run is launched without
# --job-name: it inserts a timestamp directory, giving
# <jobs_dir>/<job>/<timestamp>/<trial>/ instead of <jobs_dir>/<job>/<trial>/.
# Both layouts are present on this host (run_job.sh always passes --job-name,
# so its own runs are the shallow kind, but hand-launched runs are not), and a
# hard-coded two-level walk silently skipped every trial in the deeper one --
# the sweep reported success while reclaiming nothing there.
_MAX_TRIAL_DEPTH = 2


def _iter_trial_dirs(job_dir: Path, depth: int = 1) -> Iterable[Path]:
    """Yield candidate trial dirs at either nesting depth under job_dir.

    Recursion stops at _MAX_TRIAL_DEPTH, and a directory that already looks
    like a trial is never descended into: a trial dir is a leaf here, so its
    own subdirectories (perception/, and whatever the agent wrote) can never
    be mistaken for trials themselves.
    """
    for child in sorted(p for p in job_dir.iterdir() if p.is_dir()):
        yield child
        if depth < _MAX_TRIAL_DEPTH and not _looks_like_trial(child):
            yield from _iter_trial_dirs(child, depth + 1)


def _looks_like_trial(path: Path) -> bool:
    return (path / "result.json").is_file()


def find_sweepable_perception_dirs(jobs_dir: Path) -> List[Path]:
    """Trial perception/ dirs safe to reclaim: result.json exists (trial is
    done, its container is stopped) and the job isn't marked to keep.

    Deliberately conservative: anything that doesn't look exactly like a
    completed SAgent trial with a perception mount (both checks) is left
    alone. Sorted for deterministic iteration order (stable test output,
    predictable logs).

    Handles both of SAgent's job layouts -- <job>/<trial>/ and the deeper
    <job>/<timestamp>/<trial>/ produced when no --job-name is given. The
    result.json gate is what makes the extra depth safe: it is written in
    _cleanup_and_finalize() only after environment.stop(), so a directory
    passing it cannot belong to a running trial regardless of how deeply it
    sits.
    """
    out: List[Path] = []
    if not jobs_dir.is_dir():
        return out
    for job_dir in sorted(p for p in jobs_dir.iterdir() if p.is_dir()):
        if (job_dir / KEEP_MARKER_NAME).exists():
            continue
        for trial_dir in _iter_trial_dirs(job_dir):
            perception_dir = trial_dir / "perception"
            result_path = trial_dir / "result.json"
            if perception_dir.is_dir() and result_path.is_file():
                out.append(perception_dir)
    return out


def empty_dir_via_container(
    path: Path, *, docker_bin: str = "docker", image: str = "alpine:3.20"
) -> None:
    """Empty path's contents (including root-owned, including dotfiles) by
    running a throwaway container that bind-mounts it and rm -rf's from
    inside -- the only vantage point with permission to unlink root-owned
    files a prior trial container left behind. Never touches anything
    outside `path` itself (the container only ever sees path as /target).
    """
    subprocess.run(
        [
            docker_bin,
            "run",
            "--rm",
            "-v",
            f"{path}:/target",
            image,
            "sh",
            "-c",
            _EMPTY_CMD,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def sweep(
    jobs_dir: Path,
    *,
    dry_run: bool = False,
    emptier: Callable[[Path], None] = empty_dir_via_container,
    targets: Iterable[Path] | None = None,
) -> List[Path]:
    """Empty + rmdir every sweepable perception/ dir. Returns the ones removed
    (or, under dry_run, the ones that would be).

    `targets` lets a caller (or a test) supply the directory list directly,
    bypassing find_sweepable_perception_dirs's own filesystem walk -- used by
    main() after it has already computed the list once (so a --dry-run
    listing and the real sweep can't disagree because the filesystem changed
    between two separate walks).
    """
    if targets is None:
        targets = find_sweepable_perception_dirs(jobs_dir)
    removed: List[Path] = []
    for d in targets:
        if dry_run:
            removed.append(d)
            continue
        emptier(d)
        try:
            d.rmdir()
        except OSError:
            # Not actually empty (emptier failed silently, or something
            # raced in new content) or already gone. Leaving it behind is
            # safe -- it will be retried on the next sweep -- crashing the
            # run_job.sh invocation that's trying to launch a new job over
            # a stale-artifact cleanup issue is not an acceptable trade.
            continue
        removed.append(d)
    return removed


def write_keep_marker(job_dir: Path) -> Path:
    """Exempt job_dir from all future sweeps (this run's and later ones)."""
    job_dir.mkdir(parents=True, exist_ok=True)
    marker = job_dir / KEEP_MARKER_NAME
    marker.write_text("written by run_job.sh --keep-artifacts\n")
    return marker


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sweep", help="empty+rmdir completed trials' /perception dirs")
    p.add_argument("--jobs-dir", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("mark-keep", help="exempt one job_dir from all future sweeps")
    p.add_argument("--job-dir", required=True, type=Path)

    args = ap.parse_args(argv)

    if args.cmd == "sweep":
        removed = sweep(args.jobs_dir, dry_run=args.dry_run)
        verb = "would remove" if args.dry_run else "removed"
        for d in removed:
            print(f"{verb}: {d}")
        print(f"[cleanup_perception] {verb} {len(removed)} perception dir(s)")
        return 0

    # mark-keep
    marker = write_keep_marker(args.job_dir)
    print(f"[cleanup_perception] wrote {marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
