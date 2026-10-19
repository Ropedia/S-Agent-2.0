#!/usr/bin/env python3
"""Report the metrics one sagent configuration run must have before it can be quoted.

    .venv/bin/python scripts/run_metrics.py /tmp/sagent-jobs/<job> [...]

Prints, per job: accuracy, unanswered rate, truncation, perception usage rate,
perception CALL SUCCESS rate, per-tool breakdown, entry cost, and (for skill
configurations) whether skills were actually registered and invoked.

Why a script and not ad-hoc greps
---------------------------------
PITFALLS 1.3: never text-grep a trajectory to count agent behaviour. The file
holds both what the agent emitted and the whole prompt that was sent to it, and
a substring scan cannot tell them apart -- that mistake has produced a fake
"100% perception usage" (the instruction's own `perception.py` examples), a
`perception_calls == 12` that was identical across every configuration, and a `find`
count of 22927 that all came from the OpenHands system prompt.

So every count here is parsed STRUCTURALLY out of
`response_body.choices[].message.tool_calls[].function.arguments.command`
in `agent/litellm-trajectory.jsonl` -- i.e. only strings the agent actually
emitted as a command, never prompt text.

Why "call success rate" is its own metric
-----------------------------------------
Usage rate says the agent reached for the tool; it does not say the tool
answered. Measured 2026-08-20, 73% of gateway 400s were the agent passing pixel
coordinates where the API wants 0-1 normalized values, so a run can look
"engaged" while most of its calls bounce. Those two numbers move independently
and a report needs both.

Why there is a PARSE HEALTH block
---------------------------------
Every count above is read out of `agent/litellm-trajectory.jsonl`. When that
file is absent, unreadable, or shaped differently than this parser expects,
every function here returns empty and the report prints `perception usage
0/100 = 0.0%` -- which is exactly what a real configuration that never touched the tools
would print. That is the failure shape PITFALLS chapter 1 is about: the number
looks like data, and nothing on the page says otherwise.

So the report refuses to be quotable until it can show its own reading was
sound. Two claims, both structural rather than about behaviour:

  * every trial has a trajectory this parser could read
  * across the job, the agent issued at least one command of ANY kind

The second is the discriminator. "No PERCEPTION calls" is a legitimate
finding; "no commands at all, in any trial" is not something an agent under
`max_iterations=80` does, so it means the parse is wrong. Either claim failing
makes the exit code non-zero, so a script cannot pipe these numbers onward
without noticing.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

#: Subcommands of the perception helper that reach the GPU gateway. `health` is
#: excluded on purpose -- it is a liveness probe, not perception work, and
#: counting it would inflate the usage rate of a configuration that merely checked the
#: service was up.
#:
#: 2026-09-04: the ten scene-era subcommands below were MISSING. seg-align added
#: its commands here as it went (there are tests pinning segment / match /
#: egomotion / gravity / bev), but the scene family -- scene-v1 through v4, i.e.
#: every configuration since 2026-09-01 -- added `recon`, `side`, `depth`, `distance`,
#: `elevation`, `motion`, `compass`, `add_instance`, `list_instance` and `drop`
#: without touching this tuple. The effect was silent and exactly the shape
#: PITFALLS chapter 1 is about: a scene trial that ran `recon` + `side` +
#: `motion` and nothing else counted as making NO perception call, so "perception
#: usage" under-reported and "call success" was computed over a fraction of the
#: calls. No published number is wrong (the scene experiment docs quote the
#: gateway event log, and the seg-align numbers that DO come from here only ever
#: used commands already listed), but every scene configuration reported from this script
#: from now on would have been.
#:
#: `reconstruct` stays ahead of `recon` for readability only -- the trailing \b
#: in _CALL_RE already stops `recon` from matching the first half of
#: `reconstruct` (the engine backtracks and tries the longer branch).
#:
#: video-v1's `frames` (time -> frame lookup) is deliberately ABSENT: it is a
#: pure local read of /app/frames.json and never reaches the gateway, so
#: counting it here would inflate the usage rate of a trial that only asked
#: what time a frame was -- the same reasoning that excludes `health`. Measuring
#: whether the agent adopts `frames` needs its own counter; see docs/TODO.md.
PERCEPTION_CMDS = (
    "detect",
    "reconstruct",
    "recon",
    "measure",
    "render",
    "segment",
    "match",
    "egomotion",
    "gravity",
    "bev",
    "side",
    "depth",
    "distance",
    "elevation",
    "motion",
    "compass",
    "add_instance",
    "list_instance",
    "drop",
    # video-v1-revsi: object size (OBB); the same `distance` command also
    # takes --closest/--farthest there.
    "size",
)

#: Match CLI calls under any path prefix. Library calls are detected separately
#: from the command that imported `perception.look`; keeping the two forms in
#: one structural call record avoids counting prompt text or repeated tool
#: observations as agent behaviour.
_CALL_RE = re.compile(
    r"(?<![\w/.-])(?:[\w./-]*/)?perception\.py\s+(" + "|".join(PERCEPTION_CMDS) + r")\b"
)
_LIBRARY_CALL_RE = re.compile(
    r"(?:from\s+perception\s+import\s+look|import\s+perception[\s;]|\bperception\.look\s*\(|(?<![\w.])look\s*\()"
)

#: Tool names that carry shell text, across harnesses. openhands-sdk names its
#: shell tool "terminal", mini-swe-agent "bash" (both put the text under
#: `command`); codex uses "exec_command" (text under `cmd`) or, with
#: unified_exec off, "shell_command" (under `command`). None is kept too: some
#: harnesses omit the function name on a tool call.
_SHELL_TOOL_NAMES = frozenset({"terminal", "bash", "exec_command", "shell_command"})


def _shell_command(call: dict) -> str | None:
    """The shell text a tool call carries, or None for a non-shell tool.

    Reads `command` first, then `cmd` (codex exec_command), so one code path
    covers every harness the analysis has to read.
    """
    name = (call.get("function") or {}).get("name")
    if name is not None and name not in _SHELL_TOOL_NAMES:
        return None
    args = _tool_arguments(call)
    for key in ("command", "cmd"):
        value = args.get(key)
        if isinstance(value, str):
            return value
    return None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def load_records(trial: Path) -> list[dict] | None:
    """Parsed lines of the trial's litellm trajectory, or None if unreadable.

    None and [] are different answers and the caller must keep them apart:
    None means this parser could not read the trial at all, [] means it read
    it and the agent did nothing. Returning [] for both -- which is what
    three separate `if not lt.exists(): return` guards used to do here -- is
    what makes a broken parse indistinguishable from a quiet agent.
    """
    lt = trial / "agent" / "litellm-trajectory.jsonl"
    if not lt.exists():
        return None
    out: list[dict] = []
    # Streamed, not read_text(). These files reach 663 MB (v12), and slurping
    # them measured 5.6 GB peak RSS -- enough to be killed on a loaded box,
    # which would surface as "trajectory unreadable" rather than as OOM.
    with lt.open(errors="ignore") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    # A file that exists but yields no record with a response_body is not a
    # trajectory this parser understands -- treat it as unreadable, not empty.
    if not any("response_body" in r for r in out):
        return None
    return out


def _tool_arguments(call: dict) -> dict:
    raw = (call.get("function") or {}).get("arguments")
    if not raw:
        return {}
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return {}
    return args if isinstance(args, dict) else {}


def _observation_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            block.get("text", "") for block in value if isinstance(block, dict)
        )
    return str(value or "")


def perception_calls(records: list[dict]) -> list[dict]:
    """Return each emitted perception command with its matching observation.

    The library API is normally invoked through a `terminal` tool call such as
    `from perception import look; scene = look(...)`, not through the legacy
    `perception.py <subcommand>` CLI. Tool observations are attached to later
    requests, so they must be joined by `tool_call_id`; scanning all request
    messages would otherwise duplicate one observation on every subsequent turn.
    """
    emitted: dict[str, dict] = {}
    anonymous: list[dict] = []
    observations: dict[str, str] = {}
    for rec in records:
        for choice in (rec.get("response_body") or {}).get("choices") or []:
            for call in (choice.get("message") or {}).get("tool_calls") or []:
                command = _shell_command(call)
                if command is None:
                    continue
                item = {"id": call.get("id"), "command": command}
                if call.get("id"):
                    emitted[call["id"]] = item
                else:
                    anonymous.append(item)
        for message in (rec.get("request_body") or {}).get("messages") or []:
            if message.get("role") != "tool":
                continue
            call_id = message.get("tool_call_id")
            if call_id and call_id not in observations:
                observations[call_id] = _observation_text(message.get("content"))

    calls = []
    seen: set[str] = set()
    for item in [*emitted.values(), *anonymous]:
        command = item["command"]
        cli = list(_CALL_RE.finditer(command))
        library = bool(_LIBRARY_CALL_RE.search(command))
        if not cli and not library:
            continue
        key = item.get("id") or f"anonymous:{len(calls)}:{command}"
        if key in seen:
            continue
        seen.add(key)
        observation = observations.get(item.get("id"), "")
        calls.append(
            {
                "id": item.get("id"),
                "command": command,
                "observation": observation,
                "tools": [match.group(1) for match in cli] or ["look"],
                "gateway_error": bool(
                    re.search(r"[\"']error[\"']\s*:\s*\{", observation)
                ),
            }
        )
    return calls


def agent_commands(records: list[dict]) -> list[str]:
    """Every shell command the agent actually emitted, in order."""
    out: list[str] = []
    for rec in records:
        for choice in (rec.get("response_body") or {}).get("choices") or []:
            for call in (choice.get("message") or {}).get("tool_calls") or []:
                command = _shell_command(call)
                if command is not None:
                    out.append(command)
    return out


def invoke_skill_calls(records: list[dict]) -> list[str]:
    """Skill names the agent passed to `invoke_skill`, in order.

    This is the M2 mechanism's behavioural signal and it is NOT the same as
    "skills loaded". Loading is infrastructure -- verified separately, in-image,
    by scripts/verify_skill_disclosure.py. What this measures is whether the
    agent, seeing only a name and a one-line description, chose to open one.

    An configuration can have every skill registered, `invoke_skill` mounted, and
    progressive disclosure working perfectly, and still show zero here. That
    combination is the finding M2 exists to detect, so it must be visible
    rather than inferred from a flat perception-usage number.
    """
    out: list[str] = []
    for rec in records:
        for choice in (rec.get("response_body") or {}).get("choices") or []:
            for call in (choice.get("message") or {}).get("tool_calls") or []:
                if (call.get("function") or {}).get("name") != "invoke_skill":
                    continue
                raw = (call.get("function") or {}).get("arguments")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except json.JSONDecodeError:
                    args = {}
                out.append(str(args.get("name") or args.get("skill_name") or "?"))
    return out


def _turn_key(rec: dict) -> tuple:
    """Identity of the CONVERSATION TURN a record belongs to.

    The trajectory logs one line per server-side completion, not per turn. When
    the client abandons a call and retries -- which it did constantly while the
    SDK's HTTP timeout sat at its 300s default against 2000s+ generations --
    the same turn appears several times, with identical history depth and an
    identical final message. Counting lines therefore multiplies one turn by
    the number of times it was retried.

    Measured on v12: 20 of 59 `length` records were such duplicates, which is
    what made "8 episodes died hitting the generation cap" look real. It was
    an artefact; deduplicated there are effectively none.
    """
    msgs = (rec.get("request_body") or {}).get("messages") or []
    return (len(msgs), str(msgs[-1])[:400] if msgs else "")


def finish_reasons(records: list[dict]) -> Counter:
    """finish_reason counts, ONE PER TURN.

    Deliberately deduped -- see `_turn_key`. Without this the counts include
    generations the client already walked away from, which inflates `length`
    and makes wasted work look like agent behaviour.

    ⚠️ Deduping by KEEPING THE FIRST RECORD, which an earlier version did, drops
    the turn entirely whenever that record carries no `choices`. Measured on
    arm1 (`mmsi_disk_200_skill_m2a1dirfrom`, MMSI_379__w3BvQvk rec 26-28): a
    runaway generation logs three records for one turn --

        rec 26  choices []            duration 2488885 ms
        rec 27  choices ['length']    duration 2488879 ms
        rec 28  choices ['tool_calls']  duration 2515 ms

    -- so the turn contributed nothing, `length` never appeared in this
    Counter, and the report printed `{'tool_calls': 4948, 'stop': 37}` on the
    same job where `length_durations` (which scans every record) found 24
    length turns. The two lines of the report contradicted each other, and 25
    turns were missing from this denominator.

    So: fold the whole record GROUP for a turn, and score the turn by its
    OUTCOME -- if any attempt ran to the token cap, the turn is `length`, even
    when a later retry succeeded. That is the fact worth counting: the cap was
    hit and the wall clock was spent. `length_durations` measures what it cost.
    """
    groups: dict[tuple, list[str]] = {}
    for rec in records:
        reasons = groups.setdefault(_turn_key(rec), [])
        for choice in (rec.get("response_body") or {}).get("choices") or []:
            reasons.append(choice.get("finish_reason"))

    c: Counter = Counter()
    for reasons in groups.values():
        if not reasons:
            # Every attempt for this turn logged an empty `choices`. Real (1
            # turn in the 12-trial subset), and it is not a finish_reason, so
            # give it its own bucket rather than silently dropping the turn.
            c["no_choices"] += 1
        elif "length" in reasons:
            c["length"] += 1
        else:
            # Retries of the same turn agree in practice; take the last
            # attempt, which is the one whose result the agent acted on.
            c[reasons[-1]] += 1
    return c


def length_durations(records: list[dict]) -> list[float]:
    """Wall-clock seconds for each `finish_reason=length` TURN.

    `length` alone does not tell you what went wrong, and reading it as
    "the answer got cut off" was wrong on mmsi_disk_100_m0a: none of those
    episodes lost an answer to truncation. What actually happened is that a
    response running to the full max_output_tokens takes far longer than the
    per-episode agent budget (`[agent] timeout_sec`), so the episode dies of
    AgentTimeoutError with no answer file. Median there was 1621 s against a
    900 s budget -- every single one blew it.

    ⚠️ TWO CAVEATS, both found by review after this function was already being
    quoted in experiment docs:

    1. `duration_ms` is the PROXY's round trip, so it includes queueing behind
       other concurrent requests, not just generation. On one v12 trial it ran
       6.57x the actual generation wall clock. So this is an upper bound; do
       not read a value near timeout_sec as proof of a runaway generation.
    2. Records are deduped per turn here (`_turn_key`), because client-retried
       calls otherwise appear as several independent slow generations.
    """
    out: list[float] = []
    seen: set = set()
    for rec in records:
        key = _turn_key(rec)
        if key in seen:
            continue
        rb = rec.get("response_body") or {}
        if any(c.get("finish_reason") == "length" for c in rb.get("choices") or []):
            seen.add(key)
            out.append((rec.get("duration_ms") or 0) / 1000)
    return out


def call_outcomes(records: list[dict]) -> tuple[int, int]:
    """Return `(gateway_successes, gateway_rejections)` for perception calls.

    Only a structured gateway error counts as a rejection. A normal terminal
    observation may contain the word `error` or a Python traceback because the
    agent used a library method incorrectly; that is not evidence that the
    perception gateway rejected the request.
    """
    calls = perception_calls(records)
    failed = sum(1 for call in calls if call["gateway_error"])
    return len(calls) - failed, failed


def report(job: Path) -> bool:
    """Print one job's metrics. Returns True if the parse health claims hold."""
    trials = sorted(p for p in job.glob("*__*") if p.is_dir())
    if not trials:
        print(f"{job.name}: no trials")
        return False

    correct = graded = unanswered = 0
    used_ep = 0
    tools: Counter = Counter()
    fr: Counter = Counter()
    calls_total = calls_failed = 0
    entry: list[int] = []
    skill_md = invoked = 0
    unreadable: list[str] = []
    any_cmd_ep = all_cmds = 0
    len_secs: list[float] = []
    inv_ep = inv_then_call_ep = 0
    inv_skills: Counter = Counter()

    for t in trials:
        rf = t / "verifier" / "reward.txt"
        if rf.exists():
            try:
                correct += int(float(rf.read_text().strip()))
                graded += 1
            except ValueError:
                pass
        so = t / "verifier" / "test-stdout.txt"
        if so.exists() and "answer file missing" in so.read_text(errors="ignore"):
            unanswered += 1

        records = load_records(t)
        if records is None:
            unreadable.append(t.name)
            records = []

        cmds = agent_commands(records)
        all_cmds += len(cmds)
        if cmds:
            any_cmd_ep += 1
        perception = perception_calls(records)
        for call in perception:
            tools.update(call["tools"])
        n = len(perception)
        if n:
            used_ep += 1
        calls_total += n
        _, bad = call_outcomes(records)
        calls_failed += bad

        invoked_names = invoke_skill_calls(records)
        if invoked_names:
            inv_ep += 1
            inv_skills.update(invoked_names)
            # The step-1 acceptance signal: the agent opened a skill AND then
            # got a perception call through. Either half alone is not it --
            # opening a skill it never uses, or calling from memory without
            # reading, both leave the mechanism unproven.
            if n > bad:
                inv_then_call_ep += 1

        fr.update(finish_reasons(records))
        len_secs.extend(length_durations(records))

        tj = t / "agent" / "trajectory.json"
        if tj.exists():
            try:
                steps = json.loads(tj.read_text()).get("steps", [])
            except json.JSONDecodeError:
                steps = []
            for s in steps:
                if s.get("source") == "user" and s.get("step_id") == 2:
                    entry.append(len(str(s.get("message", ""))))
                    break

        wd = t / "agent"
        skill_md += len(list(wd.rglob("SKILL.md")))
        if (t / "agent" / "invoked_skills.json").exists():
            invoked += 1

    n = len(trials)
    lo, hi = wilson(correct, graded or 1)
    ulo, uhi = wilson(used_ep, n)
    print(f"=== {job.name}  (n={n}) ===")
    print(
        f"  accuracy          {correct}/{graded} = {correct / max(graded, 1) * 100:5.1f}%"
        f"   95%CI [{lo * 100:.1f}, {hi * 100:.1f}]"
    )
    if graded < n:
        print(f"    !! only {graded}/{n} trials graded -- job may still be running")
    print(
        f"  unanswered        {unanswered}/{n} = {unanswered / n * 100:5.1f}%"
        "   <- counted as wrong; also the truncation indicator"
    )
    print(
        f"  perception usage  {used_ep}/{n} = {used_ep / n * 100:5.1f}%"
        f"   95%CI [{ulo * 100:.1f}, {uhi * 100:.1f}]"
    )
    ok = calls_total - calls_failed
    rate = ok / calls_total * 100 if calls_total else 0.0
    print(
        f"  call success      {ok}/{calls_total} = {rate:5.1f}%"
        f"   ({calls_failed} rejected)"
    )
    if calls_failed > calls_total:
        print(
            "    !! more rejections than parsed calls. The two come from"
            " different parsers -- the total from the command text, the"
            " rejections from the tool observations -- so this ratio is not"
            " trustworthy. Check _CALL_RE against how the agent actually"
            " spelled the call."
        )
    print(f"  per tool          {dict(sorted(tools.items())) or '{}'}")
    print(f"  finish_reason     {dict(fr)}")
    if len_secs:
        len_secs.sort()
        med = len_secs[len(len_secs) // 2]
        print(
            f"    finish_reason=length x{len(len_secs)}: generation time"
            f" median {med:.0f}s, max {len_secs[-1]:.0f}s"
        )
        print(
            "    Compare with [agent] timeout_sec. OVER budget = runaway"
            " generation, and raising max_output_tokens makes it strictly"
            " worse (the episode dies of AgentTimeoutError, not truncation)."
            " UNDER budget = the cap really is too small, PITFALLS 4.1d."
        )
    if entry:
        entry.sort()
        print(
            f"  entry cost        median {entry[len(entry) // 2]} chars (first user message)"
        )
    print(f"  SKILL.md in trial {skill_md}   invoked_skills files {invoked}")
    ilo, ihi = wilson(inv_ep, n)
    print(
        f"  invoke_skill      {inv_ep}/{n} = {inv_ep / n * 100:5.1f}%"
        f"   95%CI [{ilo * 100:.1f}, {ihi * 100:.1f}]"
    )
    print(f"    which skills    {dict(inv_skills.most_common()) or '{}'}")
    print(
        f"    invoked THEN made a successful perception call: {inv_then_call_ep}/{n}"
        "   <- step-1 acceptance signal"
    )

    # Parse health. Everything above is only as good as this block; print it
    # last so it is the final thing on screen, and make its failure the exit
    # code so a caller cannot pipe the numbers onward without seeing it.
    healthy = True
    print("  -- parse health --")
    print(f"  trajectories read {n - len(unreadable)}/{n}")
    if unreadable:
        healthy = False
        shown = ", ".join(unreadable[:5])
        more = f" (+{len(unreadable) - 5} more)" if len(unreadable) > 5 else ""
        print(f"    !!!! UNREADABLE: {shown}{more}")
        print("    Every count above silently omits these trials. Do not quote.")
    print(f"  trials w/ any cmd {any_cmd_ep}/{n}   ({all_cmds} commands total)")
    if all_cmds == 0:
        healthy = False
        print("    !!!! ZERO commands parsed across the whole job.")
        print("    'No perception calls' is a finding; 'no commands at all' is not")
        print("    something an agent does under max_iterations=80 -- the parser is")
        print("    reading the wrong field. Check the tool_calls shape before")
        print("    believing the 0.0% usage rate above.")
    print(
        "  VERDICT           " + ("ok, quotable" if healthy else "BROKEN, do not quote")
    )
    print()
    return healthy


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    all_ok = True
    for arg in sys.argv[1:]:
        if not report(Path(arg)):
            all_ok = False
    raise SystemExit(0 if all_ok else 1)
