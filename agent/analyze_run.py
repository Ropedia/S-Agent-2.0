#!/usr/bin/env python3
"""Score and analyse a SAgent ERQA job.

    python3 analyze_run.py --job-dir /tmp/sagent_jobs_ywx/erqa100_run

Produces, on stdout, a JSON document with:
  * overall accuracy (n, correct, the exact id list)
  * accuracy broken down by ERQA ``question_type`` (joined from the parquet)
  * perception usage per task, derived from each trial's own trajectory
  * a failure taxonomy

Why usage stats come from trajectories, not logs/gateway.jsonl
-------------------------------------------------------------
The gateway's JSONL is a single append-only file shared by every client, and its
success records carry only ``{ts, request_id, method, path, status, latency}``
-- no task id. When two jobs run concurrently (which is the normal case on this
box) their requests interleave in that file and cannot be attributed. Each
trial's ``agent/trajectory.json``, by contrast, records the exact shell commands
that trial issued, so endpoint counts per task are recoverable exactly.

``--gateway-log`` additionally summarises the shared log for service-health
purposes (status-code mix, latency, upstream failures). Those totals cover ALL
clients in the window and are reported separately, never as per-task usage.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Override with $ERQA_PARQUET or --parquet. The default points at a sibling
# SpatialClaw checkout, which is where the benchmark data lands per README.
DEFAULT_PARQUET = Path(
    os.environ.get(
        "ERQA_PARQUET",
        Path(__file__).resolve().parent.parent
        / "SpatialClaw/data/ERQA/data/test-00000-of-00001.parquet",
    )
)

# Endpoints the gateway exposes; "health" is not evidence-gathering.
EVIDENCE_ENDPOINTS = ("reconstruct", "detect", "measure", "render")
ALL_ENDPOINTS = ("health",) + EVIDENCE_ENDPOINTS


def _sort_key(task_id: str):
    """ERQA_10 sorts after ERQA_9, not after ERQA_1."""
    m = re.match(r"^(.*?)(\d+)$", task_id)
    return (m.group(1), int(m.group(2))) if m else (task_id, 0)


# ---------------------------------------------------------------------------
# Trajectory mining
# ---------------------------------------------------------------------------


def _iter_commands(traj: Dict[str, Any]) -> Iterable[str]:
    """Every shell command string this trial executed."""
    for step in traj.get("steps") or []:
        for call in step.get("tool_calls") or []:
            args = call.get("arguments") or {}
            cmd = args.get("command")
            if isinstance(cmd, list):
                cmd = " ".join(str(c) for c in cmd)
            if isinstance(cmd, str) and cmd.strip():
                yield cmd


def endpoint_counts(commands: List[str]) -> Dict[str, int]:
    """Count gateway endpoint invocations across a trial's commands.

    Agents call the gateway in three observed shapes, all of which count:

      1. curl URL      curl ... http://<host>:8090/detect ...
      2. helper CLI    python3 /app/perception.py detect --image ...
      3. quoted argv   subprocess.run(['python3','/app/perception.py','detect',...])
                       ...including wrapper helpers where only ['detect', ...]
                       appears at the call site.

    Each pattern is matched separately and hits are then de-duplicated by the
    character span of the endpoint name itself, so a string matched by two
    patterns at once (shape 3 matches both the argv and quoted-token patterns)
    is counted once rather than twice.

    Bare prose is deliberately not counted: a comment like "# Reconstruct fresh"
    matches none of the three, because shape 2 permits only argument-like
    characters between ``perception.py`` and the endpoint name, and the
    lowercase-only match rejects capitalised prose.

    Caveat: counts for heredocs that loop over endpoints in Python are
    approximate -- the loop body is one static string. The derived booleans
    ("did this task use /detect at all", zero-evidence rate) are unaffected.
    """
    counts = {name: 0 for name in ALL_ENDPOINTS}
    for cmd in commands:
        for name in ALL_ENDPOINTS:
            spans = set()
            # 1. curl-style URL path component.
            for m in re.finditer(rf":\d{{2,5}}/({name})\b", cmd):
                spans.add(m.start(1))
            # 2/3. helper CLI, tolerating quotes and commas between tokens.
            pat = (
                rf"perception\.py['\"]?[\s,]+"
                rf"(?:['\"]?[-\w./=]+['\"]?[\s,]+)*?['\"]?({name})\b"
            )
            for m in re.finditer(pat, cmd):
                spans.add(m.start(1))
            # 3. quoted endpoint token in an argv list (wrapper helpers).
            for m in re.finditer(rf"['\"]({name})['\"]", cmd):
                spans.add(m.start(1))
            counts[name] += len(spans)
    return counts


def load_trajectory(trial_dir: Path) -> Optional[Dict[str, Any]]:
    path = trial_dir / "agent" / "trajectory.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Verifier output
# ---------------------------------------------------------------------------

# Emitted by tests/score_erqa.py, e.g.
#   reward=0 reason=extracted 'c' via bare-letter-last-line, expected 'a'
REASON_RE = re.compile(r"^reward=([01])\s+reason=(.*)$", re.MULTILINE)
EXTRACTED_RE = re.compile(r"extracted '([^']*)'")
# Non-greedy and newline-free: the next line is "expected answer: '<letter>'",
# and a greedy DOTALL match would swallow it into the raw answer.
ANSWER_RE = re.compile(r"agent answer \(first \d+ chars\): '(.*?)'\s*$", re.MULTILINE)
EXPECTED_RE = re.compile(r"expected answer: '(.*?)'\s*$", re.MULTILINE)


def parse_verifier(trial_dir: Path) -> Dict[str, Any]:
    """Pull the scorer's own verdict, extracted letter and raw answer."""
    out: Dict[str, Any] = {
        "verifier_ran": False,
        "reason": None,
        "extracted": None,
        "raw_answer": None,
        "answer_file_missing": False,
    }
    path = trial_dir / "verifier" / "test-stdout.txt"
    if not path.is_file():
        return out
    text = path.read_text(errors="replace")
    out["verifier_ran"] = True

    m = REASON_RE.search(text)
    if m:
        out["reason"] = m.group(2).strip()
        got = EXTRACTED_RE.search(out["reason"])
        if got:
            out["extracted"] = got.group(1)

    a = ANSWER_RE.search(text)
    if a:
        out["raw_answer"] = a.group(1)

    e = EXPECTED_RE.search(text)
    if e:
        out["expected"] = e.group(1).strip()

    # On a pass the scorer prints "reason=official exact match" with no
    # "extracted '<letter>'" clause, so recover the letter from the answer the
    # scorer echoed. Without this, every correct trial reports extracted=None.
    if out["extracted"] is None and out["raw_answer"] is not None:
        cleaned = str(out["raw_answer"]).replace(".", "").strip().lower()
        if len(cleaned) == 1:
            out["extracted"] = cleaned
    # The scorer says so explicitly when /app/answer.txt never appeared.
    if "no answer file" in text.lower() or "missing" in (out["reason"] or "").lower():
        out["answer_file_missing"] = True
    return out


def read_reward(trial_dir: Path, result: Dict[str, Any]) -> Optional[float]:
    """Prefer the structured result; fall back to reward.txt on disk."""
    vr = result.get("verifier_result") or {}
    rewards = vr.get("rewards") or {}
    if "reward" in rewards:
        try:
            return float(rewards["reward"])
        except (TypeError, ValueError):
            pass
    path = trial_dir / "verifier" / "reward.txt"
    if path.is_file():
        try:
            return float(path.read_text().strip())
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------

VALID_LETTERS = {"a", "b", "c", "d", "e"}


def classify(trial: Dict[str, Any]) -> str:
    """One bucket per trial. Order matters: harness faults outrank accuracy.

    Buckets
    -------
    correct            reward == 1
    harness_error      SAgent raised (exception_info set) -- not a model result
    timeout            agent or verifier hit its wall clock
    no_verifier        verifier never produced output
    empty_answer       /app/answer.txt missing or blank
    format_error       answer present but no valid letter could be extracted
    wrong_answer       a valid letter, but not the expected one
    """
    if trial.get("reward") == 1.0:
        return "correct"

    exc = trial.get("exception") or ""
    if "Timeout" in exc:
        return "timeout"
    if exc:
        return "harness_error"

    if not trial.get("verifier_ran"):
        return "no_verifier"

    raw = trial.get("raw_answer")
    if raw is None or not str(raw).strip():
        return "empty_answer"

    extracted = (trial.get("extracted") or "").strip().lower()
    if extracted not in VALID_LETTERS:
        return "format_error"
    return "wrong_answer"


def _pct(num: int, den: int) -> Optional[float]:
    return round(100.0 * num / den, 1) if den else None


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def collect_trials(job_dir: Path) -> List[Dict[str, Any]]:
    """One record per trial directory under the job dir."""
    trials: List[Dict[str, Any]] = []
    for result_path in sorted(job_dir.glob("*/result.json")):
        trial_dir = result_path.parent
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            trials.append(
                {
                    "task_id": trial_dir.name.split("__")[0],
                    "trial": trial_dir.name,
                    "reward": None,
                    "exception": f"unreadable result.json: {exc}",
                    "verifier_ran": False,
                    "raw_answer": None,
                    "extracted": None,
                    "reason": None,
                    "endpoints": {n: 0 for n in ALL_ENDPOINTS},
                    "n_steps": 0,
                }
            )
            continue

        exc_info = result.get("exception_info")
        exception = ""
        if exc_info:
            if isinstance(exc_info, dict):
                exception = str(
                    exc_info.get("exception_type")
                    or exc_info.get("type")
                    or exc_info.get("message")
                    or exc_info
                )
            else:
                exception = str(exc_info)

        traj = load_trajectory(trial_dir)
        commands = list(_iter_commands(traj)) if traj else []
        record: Dict[str, Any] = {
            "task_id": result.get("task_name") or trial_dir.name.split("__")[0],
            "trial": trial_dir.name,
            "reward": read_reward(trial_dir, result),
            "exception": exception,
            "endpoints": endpoint_counts(commands),
            "n_steps": len(traj.get("steps") or []) if traj else 0,
            "n_commands": len(commands),
            "has_trajectory": traj is not None,
        }
        record.update(parse_verifier(trial_dir))

        agent_res = result.get("agent_result") or {}
        record["n_input_tokens"] = agent_res.get("n_input_tokens")
        record["n_output_tokens"] = agent_res.get("n_output_tokens")

        record["bucket"] = classify(record)
        trials.append(record)
    return trials


def load_question_types(parquet: Path, task_ids: Iterable[str]) -> Dict[str, str]:
    """Join question_type back from the source parquet."""
    try:
        import pandas as pd
    except ImportError:
        return {}
    if not parquet.is_file():
        return {}
    df = pd.read_parquet(parquet, columns=["question_id", "question_type"])
    wanted = set(task_ids)
    return {
        str(r.question_id): str(r.question_type)
        for r in df.itertuples()
        if str(r.question_id) in wanted
    }


def summarize_gateway_log(
    log_path: Path, window: Optional[tuple] = None
) -> Dict[str, Any]:
    """Service-health view of the shared gateway log.

    NOT per-task usage: this file is shared by every client on the box, so these
    totals may include requests from other concurrent jobs.
    """
    if not log_path.is_file():
        return {"available": False}
    status: collections.Counter = collections.Counter()
    paths: collections.Counter = collections.Counter()
    latencies: List[float] = []
    errors: collections.Counter = collections.Counter()
    n = 0
    for line in log_path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = rec.get("ts") or ""
        if window and not (window[0] <= ts <= window[1]):
            continue
        n += 1
        status[rec.get("status")] += 1
        if rec.get("path"):
            paths[rec["path"]] += 1
        if isinstance(rec.get("latency_sec"), (int, float)):
            latencies.append(float(rec["latency_sec"]))
        err = rec.get("error") or {}
        if isinstance(err, dict) and err.get("code"):
            errors[err["code"]] += 1
    latencies.sort()
    return {
        "available": True,
        "path": str(log_path),
        "window": list(window) if window else None,
        "n_requests": n,
        "by_status": {
            str(k): v for k, v in sorted(status.items(), key=lambda kv: str(kv[0]))
        },
        "by_path": dict(paths.most_common()),
        "error_codes": dict(errors.most_common()),
        "latency_sec": {
            "mean": round(sum(latencies) / len(latencies), 4) if latencies else None,
            "p50": latencies[len(latencies) // 2] if latencies else None,
            "p95": latencies[int(len(latencies) * 0.95)] if latencies else None,
            "max": latencies[-1] if latencies else None,
        },
        "caveat": (
            "Shared append-only log with no task id on success records; totals "
            "may include other concurrent clients. Per-task usage is derived "
            "from trajectories instead."
        ),
    }


def build_report(
    trials: List[Dict[str, Any]],
    qtypes: Dict[str, str],
    gateway: Dict[str, Any],
) -> Dict[str, Any]:
    n = len(trials)
    scored = [t for t in trials if t.get("reward") in (0.0, 1.0)]
    correct = [t for t in scored if t["reward"] == 1.0]

    # --- accuracy by question_type ---
    by_type: Dict[str, Dict[str, Any]] = {}
    for t in trials:
        qt = qtypes.get(t["task_id"], "Unknown")
        slot = by_type.setdefault(qt, {"n": 0, "correct": 0, "scored": 0})
        slot["n"] += 1
        if t.get("reward") in (0.0, 1.0):
            slot["scored"] += 1
            slot["correct"] += int(t["reward"] == 1.0)
    for slot in by_type.values():
        slot["accuracy_pct"] = _pct(slot["correct"], slot["scored"])
    by_type = dict(sorted(by_type.items(), key=lambda kv: (-kv[1]["n"], kv[0])))

    # --- perception usage, from trajectories ---
    per_endpoint_tasks = {name: 0 for name in EVIDENCE_ENDPOINTS}
    per_endpoint_calls = {name: 0 for name in EVIDENCE_ENDPOINTS}
    no_evidence: List[str] = []
    total_evidence_calls = 0
    for t in trials:
        ep = t["endpoints"]
        used = 0
        for name in EVIDENCE_ENDPOINTS:
            c = ep.get(name, 0)
            per_endpoint_calls[name] += c
            if c:
                per_endpoint_tasks[name] += 1
            used += c
        total_evidence_calls += used
        if used == 0:
            no_evidence.append(t["task_id"])

    # Accuracy split by whether the task gathered any evidence at all.
    def _acc(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        sc = [t for t in subset if t.get("reward") in (0.0, 1.0)]
        cor = sum(1 for t in sc if t["reward"] == 1.0)
        return {
            "n": len(subset),
            "scored": len(sc),
            "correct": cor,
            "accuracy_pct": _pct(cor, len(sc)),
        }

    ne = set(no_evidence)
    usage = {
        "source": "per-trial agent/trajectory.json (exact attribution)",
        "tasks_calling_endpoint": per_endpoint_tasks,
        "total_calls_per_endpoint": per_endpoint_calls,
        "mean_evidence_calls_per_task": round(total_evidence_calls / n, 2)
        if n
        else None,
        "tasks_with_zero_evidence_calls": len(no_evidence),
        "look_and_guess_rate_pct": _pct(len(no_evidence), n),
        "zero_evidence_task_ids": sorted(no_evidence, key=_sort_key),
        "accuracy_when_evidence_gathered": _acc(
            [t for t in trials if t["task_id"] not in ne]
        ),
        "accuracy_when_no_evidence": _acc([t for t in trials if t["task_id"] in ne]),
        "tasks_calling_health_only": sum(
            1
            for t in trials
            if t["endpoints"].get("health", 0) > 0
            and all(t["endpoints"].get(e, 0) == 0 for e in EVIDENCE_ENDPOINTS)
        ),
    }

    buckets = collections.Counter(t["bucket"] for t in trials)
    taxonomy = {
        k: {"n": v, "pct_of_all": _pct(v, n)}
        for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])
    }

    steps = [t["n_steps"] for t in trials if t.get("has_trajectory")]
    return {
        "overall": {
            "n_tasks": n,
            "n_scored": len(scored),
            "n_correct": len(correct),
            "accuracy_pct": _pct(len(correct), len(scored)),
            "accuracy_pct_over_all_tasks": _pct(len(correct), n),
            "task_ids": [
                t["task_id"]
                for t in sorted(trials, key=lambda x: _sort_key(x["task_id"]))
            ],
            "correct_task_ids": sorted((t["task_id"] for t in correct), key=_sort_key),
        },
        "by_question_type": by_type,
        "perception_usage": usage,
        "failure_taxonomy": taxonomy,
        "agent_behaviour": {
            "mean_steps": round(sum(steps) / len(steps), 1) if steps else None,
            "max_steps": max(steps) if steps else None,
            "trials_without_trajectory": sum(
                1 for t in trials if not t.get("has_trajectory")
            ),
            "mean_input_tokens": (
                round(
                    sum(t["n_input_tokens"] for t in trials if t.get("n_input_tokens"))
                    / max(1, sum(1 for t in trials if t.get("n_input_tokens")))
                )
            ),
        },
        "gateway_service_log": gateway,
        "per_task": sorted(trials, key=lambda t: _sort_key(t["task_id"])),
    }


def print_text_report(rep: Dict[str, Any]) -> None:
    o = rep["overall"]
    w = sys.stderr.write
    w("\n=== OVERALL ===\n")
    w(
        f"tasks={o['n_tasks']} scored={o['n_scored']} correct={o['n_correct']} "
        f"accuracy={o['accuracy_pct']}%\n"
    )

    w("\n=== BY QUESTION TYPE ===\n")
    w(f"{'question_type':<24}{'n':>4}{'correct':>9}{'acc%':>8}\n")
    for qt, s in rep["by_question_type"].items():
        w(f"{qt:<24}{s['n']:>4}{s['correct']:>9}{str(s['accuracy_pct']):>8}\n")

    u = rep["perception_usage"]
    w("\n=== PERCEPTION USAGE (from trajectories) ===\n")
    for name in EVIDENCE_ENDPOINTS:
        w(
            f"  /{name:<12} tasks={u['tasks_calling_endpoint'][name]:>4}  "
            f"calls={u['total_calls_per_endpoint'][name]:>5}\n"
        )
    w(f"  mean evidence calls/task : {u['mean_evidence_calls_per_task']}\n")
    w(
        f"  zero-evidence tasks      : {u['tasks_with_zero_evidence_calls']} "
        f"({u['look_and_guess_rate_pct']}%)\n"
    )
    w(
        f"  acc when evidence used   : {u['accuracy_when_evidence_gathered']['accuracy_pct']}% "
        f"(n={u['accuracy_when_evidence_gathered']['scored']})\n"
    )
    w(
        f"  acc when no evidence     : {u['accuracy_when_no_evidence']['accuracy_pct']}% "
        f"(n={u['accuracy_when_no_evidence']['scored']})\n"
    )

    w("\n=== FAILURE TAXONOMY ===\n")
    for k, v in rep["failure_taxonomy"].items():
        w(f"  {k:<16}{v['n']:>4}  {v['pct_of_all']}%\n")

    g = rep["gateway_service_log"]
    if g.get("available"):
        w("\n=== GATEWAY SERVICE LOG (all clients) ===\n")
        w(f"  requests={g['n_requests']} by_status={g['by_status']}\n")
        w(f"  errors={g['error_codes']}\n")
        w(
            f"  latency mean={g['latency_sec']['mean']}s p95={g['latency_sec']['p95']}s\n"
        )
    w("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Score and analyse a SAgent ERQA job")
    ap.add_argument(
        "--job-dir",
        type=Path,
        required=True,
        help="Directory containing <trial>/result.json",
    )
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument(
        "--gateway-log",
        type=Path,
        default=None,
        help="Optional logs/gateway.jsonl for service-health stats",
    )
    ap.add_argument(
        "--window",
        nargs=2,
        metavar=("START_TS", "END_TS"),
        default=None,
        help="ISO timestamp bounds to filter the gateway log",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true", help="Suppress the text summary")
    args = ap.parse_args()

    if not args.job_dir.is_dir():
        print(f"error: not a directory: {args.job_dir}", file=sys.stderr)
        return 1

    trials = collect_trials(args.job_dir)
    if not trials:
        print(f"error: no */result.json under {args.job_dir}", file=sys.stderr)
        return 1

    qtypes = load_question_types(args.parquet, (t["task_id"] for t in trials))
    gateway = (
        summarize_gateway_log(
            args.gateway_log, tuple(args.window) if args.window else None
        )
        if args.gateway_log
        else {"available": False}
    )
    report = build_report(trials, qtypes, gateway)

    if not args.quiet:
        print_text_report(report)
    payload = json.dumps(report, indent=2, default=str)
    if args.json_out:
        args.json_out.write_text(payload + "\n")
        print(f"wrote {args.json_out}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
