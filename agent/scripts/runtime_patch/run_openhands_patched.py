#!/usr/bin/env python3
"""Versioned SAgent runtime entrypoint for OpenHands SDK 1.33.0."""

import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

PATCH_BEGIN_MARKER = "# sagent patch: send_reasoning_content_models begin"
PATCH_END_MARKER = "# sagent patch: send_reasoning_content_models end"


def _normalize_model_entries(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        candidate = value.strip()
        return [candidate] if candidate else []

    normalized: list[str] = []
    for item in value:
        candidate = str(item).strip()
        if candidate:
            normalized.append(candidate)
    return normalized


def parse_send_reasoning_content_models(raw: str | None) -> list[str]:
    if not raw:
        return []

    stripped = raw.strip()
    if not stripped:
        return []

    parsed: Any
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in stripped.split(",")]

    deduped_models: list[str] = []
    seen: set[str] = set()
    for candidate in _normalize_model_entries(parsed):
        if candidate not in seen:
            seen.add(candidate)
            deduped_models.append(candidate)
    return deduped_models


def parse_json_dict_env(raw: str | None, env_name: str) -> dict[str, Any]:
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{env_name} must decode to a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{env_name} must decode to a JSON object")
    return parsed


def _build_send_reasoning_content_patch(models: list[str]) -> str:
    return "\n".join(
        [
            PATCH_BEGIN_MARKER,
            f"_SAGENT_EXTRA_SEND_REASONING_CONTENT_MODELS = {models!r}",
            "_SAGENT_EXISTING_SEND_REASONING_CONTENT_MODELS = globals().get(",
            '    "SEND_REASONING_CONTENT_MODELS", ()',
            ")",
            "SEND_REASONING_CONTENT_MODELS = tuple(",
            "    dict.fromkeys(",
            "        [",
            "            *list(_SAGENT_EXISTING_SEND_REASONING_CONTENT_MODELS),",
            "            *_SAGENT_EXTRA_SEND_REASONING_CONTENT_MODELS,",
            "        ]",
            "    )",
            ")",
            "del _SAGENT_EXTRA_SEND_REASONING_CONTENT_MODELS",
            "del _SAGENT_EXISTING_SEND_REASONING_CONTENT_MODELS",
            PATCH_END_MARKER,
            "",
        ]
    )


def patch_send_reasoning_content_models(extra_models: list[str]) -> list[str]:
    if not extra_models:
        return []

    model_features = importlib.import_module("openhands.sdk.llm.utils.model_features")
    existing_models = _normalize_model_entries(
        getattr(model_features, "SEND_REASONING_CONTENT_MODELS", ())
    )

    merged_models: list[str] = []
    seen: set[str] = set()
    for candidate in [*existing_models, *extra_models]:
        if candidate not in seen:
            seen.add(candidate)
            merged_models.append(candidate)

    # Runtime directories are mounted read-only into task containers, so the
    # compatibility patch must stay in-memory for the current process instead
    # of trying to rewrite the installed SDK source file.
    setattr(model_features, "SEND_REASONING_CONTENT_MODELS", tuple(merged_models))

    return merged_models


class SkillLoadError(RuntimeError):
    """Raised when skills on disk would load into a silently degraded state.

    Every failure mode guarded here is *silent* in the SDK, and each one
    produces the same outcome: a run that completes normally, reports
    ``n_errors=0``, and measures the wrong thing. That is indistinguishable
    from a real negative result ("progressive disclosure did not help"), so
    the entrypoint refuses to start instead.

    The guarded modes, in the order they bite:

    * a skill path that does not exist (a typo in ``SKILL_PATHS``) -- loads
      zero skills
    * ``SKILL_PATHS`` set but nothing loaded -- same end state, any cause
    * ``SKILL.md`` nested deeper than one level -- the SDK's discovery only
      scans immediate subdirectories, so all three buckets come back empty
    * frontmatter ``name`` failing validation -- dropped with only a
      ``logger.warning``
    * a skill landing in the legacy repo bucket -- full text resident in
      ``<REPO_CONTEXT>``
    * ``disable-model-invocation: true`` -- loads fine, but is filtered out of
      both ``<available_skills>`` and ``InvokeSkillTool``
    * no ``description`` -- ``to_prompt()`` falls back to the first line of
      content, which looks exactly like working progressive disclosure
    """


def _expected_skill_names(base_path: Path) -> set[str]:
    """Directory names under ``base_path`` that hold a ``SKILL.md``.

    Mirrors the SDK's discovery rule (``skills/utils.py``
    ``find_skill_md_directories`` -> ``find_skill_md``): immediate
    subdirectories only, filename matched case-insensitively. In strict mode
    the SDK requires frontmatter ``name`` to equal the directory name, so
    these directory names are exactly the keys we expect back -- which is
    what lets us detect a skill that was dropped rather than loaded.
    """
    expected: set[str] = set()
    if not base_path.is_dir():
        return expected

    for subdir in sorted(base_path.iterdir()):
        if not subdir.is_dir():
            continue
        if any(
            item.is_file() and item.name.lower() == "skill.md"
            for item in sorted(subdir.iterdir())
        ):
            expected.add(subdir.name)
    return expected


def _unreachable_skill_md(base_path: Path) -> list[Path]:
    """``SKILL.md`` files too deep for the SDK to ever find.

    ``find_skill_md_directories`` scans *immediate* subdirectories only, so a
    file at ``<base>/<group>/<skill>/SKILL.md`` is invisible: the SDK returns
    all three buckets empty and logs nothing. The count guard cannot catch this
    on its own, because ``_expected_skill_names`` mirrors the same one-level
    rule and therefore also sees nothing -- expected and actual agree, both
    empty.

    This matters concretely: the two-layer skill grouping under evaluation in
    task 4 of the spec lays skills out at exactly this depth.
    """
    if not base_path.is_dir():
        return []

    loadable_dirs = {
        subdir
        for subdir in base_path.iterdir()
        if subdir.is_dir()
        and any(
            item.is_file() and item.name.lower() == "skill.md"
            for item in subdir.iterdir()
        )
    }

    unreachable: list[Path] = []
    for path in base_path.rglob("*"):
        if not path.is_file() or path.name.lower() != "skill.md":
            continue
        # Reachable: <base>/<skill-name>/SKILL.md, exactly one level down.
        if path.parent in loadable_dirs:
            continue
        # A skill that *is* loadable may carry scripts/, references/ or assets/
        # subdirectories per the AgentSkills spec, and those may legitimately
        # contain their own files. Anything under a loadable skill is that
        # skill's business, not a missed top-level skill.
        if any(parent in loadable_dirs for parent in path.parents):
            continue
        unreachable.append(path)
    return sorted(unreachable)


def _has_body(content: str | None) -> bool:
    """True if ``content`` has substance beyond markdown headings.

    ``invoke_skill`` returns the body verbatim, so a file that is only
    frontmatter -- or only a ``# title`` line -- costs the agent a tool call
    and teaches it nothing, while still advertising normally in the prompt.
    """
    for line in (content or "").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return True
    return False


def discover_skills(
    skill_paths: list[str],
    load_skills_from_dir: Any,
    logger: Any,
    required: bool = False,
) -> list[Any]:
    """Load AgentSkills-format skills from each path in ``skill_paths``.

    Delegates to the SDK's ``load_skills_from_dir`` instead of constructing
    ``Skill(...)`` by hand. That matters because the AgentSkills branch is
    selected by *filename* (``skills/skill.py`` ``Skill.load`` dispatches on
    ``path.name.lower() == "skill.md"``), and only that branch sets
    ``is_agentskills_format=True``. A hand-built ``Skill`` leaves the field at
    its ``False`` default, so ``AgentContext._partition_skills`` files it under
    ``repo_skills`` and the full text becomes permanently resident in
    ``<REPO_CONTEXT>`` -- no progressive disclosure, plus an
    ``<UNTRUSTED_CONTENT>`` wrapper and a risk-level bump.

    ``load_skills_from_dir`` takes a single directory and returns
    ``(repo_skills, knowledge_skills, agent_skills)`` as dicts keyed by skill
    name, so this flattens across paths itself. First path wins on a name
    collision, matching the previous behavior.

    The loader is injected rather than imported at module scope because the
    SDK only exists inside the runtime image.

    Raises:
        SkillLoadError: if any skill would load as a legacy repo skill, if a
            ``SKILL.md`` on disk did not come back, if any loaded skill has
            no ``description``, or -- when ``required`` is set -- if a named
            path is absent or the whole set yields nothing.
    """
    seen_names: set[str] = set()
    skills: list[Any] = []
    missing: list[Path] = []

    for base_path_str in skill_paths:
        base_path = Path(base_path_str).expanduser()
        # A missing path is COLLECTED, not raised on. The original guard read
        # "naming a path is a claim that it holds skills", which is false here:
        # SAgent sets SKILL_PATHS unconditionally on every run to eight
        # speculative agent-home locations it does not create --
        # ~/.openhands-sdk/skills, ~/.claude/skills, ~/.codex/skills, ... (see
        # sagent/agents/custom/openhands_sdk.py DEFAULT_SKILL_PATHS, assigned at
        # :456 whether or not the caller asked for skills). None of them exist in
        # our runtime image, so raising on the first one killed every M0 trial
        # before the agent started: 100/100 NonZeroAgentExitCodeError.
        #
        # The claim the guard should enforce belongs to the *set*, not to each
        # entry: if NOTHING loads from ANY path, that is still fatal (the
        # backstop at the end of this function). A single absent speculative
        # default alongside a real one is not.
        #
        # Configurations that pass their own path (M2: --ae SKILL_PATHS=/app/skills) are
        # unaffected either way -- _extra_env is merged last
        # (agents/installed/base.py:316), so an explicit value replaces the
        # defaults rather than adding to them.
        #
        # ``required`` (SKILL_PATHS_REQUIRED=1, set by run_job.sh only when
        # SKILL_MODE=skill) restores the strict per-path claim for exactly the
        # case where it is true. Without it there is a hole: under M0 the
        # absence of /app/skills is CORRECT (inline mode never stages it), so
        # "path missing" cannot be fatal in general -- and that makes an M2 run
        # whose image build silently dropped /app/skills indistinguishable
        # from a healthy M0 run. The loader would report zero skills, the
        # agent would run with no perception documentation, and the job would
        # still exit 0. Since run_job.sh knows which configuration it launched, it tells
        # the loader, and the loader can be strict without tripping on
        # SAgent's speculative defaults (which are replaced, not merged, in
        # exactly this mode).
        if not base_path.is_dir():
            if required:
                raise SkillLoadError(
                    f"SKILL_PATHS_REQUIRED=1 but {base_path} does not exist. "
                    "This configuration was launched with SKILL_MODE=skill, so the task "
                    "image must carry <skill-name>/SKILL.md there; a missing "
                    "directory means the build dropped them and the agent "
                    "would run with no perception documentation while the run "
                    "still reported success."
                )
            missing.append(base_path)
            continue

        unreachable = _unreachable_skill_md(base_path)
        if unreachable:
            raise SkillLoadError(
                f"{base_path}: {[str(p) for p in unreachable]} are nested too "
                "deep for the SDK to discover. It only scans immediate "
                "subdirectories, so these load as zero skills with no warning. "
                "Flatten each one to <skills-root>/<skill-name>/SKILL.md."
            )

        expected = _expected_skill_names(base_path)
        repo_skills, knowledge_skills, agent_skills = load_skills_from_dir(base_path)

        # A non-empty repo bucket is the exact regression this change fixes:
        # those skills go into <REPO_CONTEXT> in full, always.
        if repo_skills:
            raise SkillLoadError(
                f"{base_path}: {sorted(repo_skills)} loaded as legacy repo skills, "
                "so their full text would stay resident in <REPO_CONTEXT> instead "
                "of being progressively disclosed. Move each one into "
                "<skill-name>/SKILL.md with AgentSkills frontmatter."
            )

        missing = expected - set(agent_skills)
        if missing:
            raise SkillLoadError(
                f"{base_path}: {sorted(missing)} have a SKILL.md on disk but were "
                "not loaded. The SDK only logs a warning when frontmatter "
                "validation fails, so check that each frontmatter 'name' equals "
                "its directory name and matches ^[a-z0-9]+(-[a-z0-9]+)*$."
            )

        for name, skill in sorted(agent_skills.items()) + sorted(
            knowledge_skills.items()
        ):
            if name in seen_names:
                logger.debug(f"Skipping duplicate skill name: {name}")
                continue
            seen_names.add(name)
            skills.append(skill)
            logger.debug(f"Loaded skill: {name} from {skill.source}")

    # to_prompt() falls back to the first line of content when description is
    # empty, which still yields a one-line entry -- indistinguishable from
    # working progressive disclosure, but advertising the wrong text.
    undescribed = sorted(s.name for s in skills if not (s.description or "").strip())
    if undescribed:
        raise SkillLoadError(
            f"{undescribed} have no 'description' in frontmatter. The prompt would "
            "silently fall back to the first line of content, so the skill list "
            "would look right while advertising the wrong text. description is the "
            "agent's only basis for choosing a skill; state what it does and when "
            "to use it."
        )

    # disable_model_invocation passes every check above -- it loads cleanly, is
    # AgentSkills format, and has a description -- but both
    # AgentContext._partition_skills and the InvokeSkillTool auto-attach test
    # are gated on `not s.disable_model_invocation`, so the skill vanishes from
    # <available_skills> and the tool never mounts. Present on disk, absent from
    # the prompt.
    uninvocable = sorted(
        s.name for s in skills if getattr(s, "disable_model_invocation", False)
    )
    if uninvocable:
        raise SkillLoadError(
            f"{uninvocable} set 'disable-model-invocation: true', which removes "
            "them from <available_skills> and suppresses InvokeSkillTool, so the "
            "agent can neither see nor invoke them. Remove that field to use "
            "progressive disclosure."
        )

    # An empty body advertises correctly in <available_skills> but makes
    # invoke_skill return nothing, so the agent pays a tool call to learn
    # nothing. Cheap to check, and it catches a frontmatter-only file.
    empty_body = sorted(s.name for s in skills if not _has_body(s.content))
    if empty_body:
        raise SkillLoadError(
            f"{empty_body} have frontmatter but no usable body, so invoke_skill "
            "would return nothing while the skill still appears in the prompt. "
            "Write the actual instructions below the frontmatter."
        )

    # Final backstop: every guard above compares against what is on disk, so
    # they all stay silent when there is nothing on disk to compare against.
    #
    # Fires only when at least one named path EXISTED and still yielded nothing
    # -- that is a real claim gone wrong (a directory is there, someone put it
    # in SKILL_PATHS, and it holds no <skill-name>/SKILL.md). When every path
    # was merely absent, this is SAgent's unconditional DEFAULT_SKILL_PATHS on
    # a configuration that ships no skills at all (M0/disk_only), which is correct and
    # must not abort the run.
    #
    # The distinction matters because both states look identical from the
    # outside -- "zero skills loaded" -- but only one of them is a defect. An
    # configuration whose skills went missing must still fail loudly; that case is caught
    # at generation time instead (adapters/spatial/adapter.py step 3b,
    # which refuses to emit a perception task with no SKILL.md in either
    # delivery mode), and at runtime by SKILL_PATHS_REQUIRED for the configuration that
    # is supposed to have them.
    existing = [p for p in skill_paths if Path(p).expanduser().is_dir()]
    if not skills and (existing or required):
        raise SkillLoadError(
            f"SKILL_PATHS={skill_paths} loaded 0 skills"
            + (" (SKILL_PATHS_REQUIRED=1)" if required else "")
            + f". {existing} exist but "
            "hold no <skill-name>/SKILL.md, so the agent would run with no "
            "skills at all while the run still reports success. Unset "
            "SKILL_PATHS if that is what you want."
        )

    if missing:
        logger.debug(
            "SKILL_PATHS entries that do not exist (ignored): "
            f"{[str(p) for p in missing]}"
        )

    return skills


def build_trajectory(
    events: list[dict[str, Any]],
    llm_metrics: dict[str, Any],
    model_name: str,
    system_prompt: str | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an ATIF-format trajectory from conversation events."""
    steps: list[dict[str, Any]] = []
    step_id = 1

    for event in events:
        event_type = event.get("type", "")

        if event_type == "user_message":
            steps.append(
                {
                    "step_id": step_id,
                    "timestamp": event.get("timestamp"),
                    "source": "user",
                    "message": event.get("content", ""),
                }
            )
            step_id += 1

        elif event_type == "assistant_message":
            step: dict[str, Any] = {
                "step_id": step_id,
                "timestamp": event.get("timestamp"),
                "source": "agent",
                "message": event.get("content", ""),
                "model_name": model_name,
            }

            tool_calls = event.get("tool_calls", [])
            if tool_calls:
                step["tool_calls"] = [
                    {
                        "tool_call_id": tc.get("id", ""),
                        "function_name": tc.get("name", ""),
                        "arguments": tc.get("arguments", {}),
                    }
                    for tc in tool_calls
                ]

            token_data = event.get("token_ids")
            if token_data:
                step["metrics"] = {
                    "prompt_token_ids": token_data.get("prompt_token_ids", []),
                    "completion_token_ids": token_data.get("response_token_ids", []),
                }

            steps.append(step)
            step_id += 1

        elif event_type == "tool_result":
            if steps and steps[-1].get("source") == "agent":
                steps[-1]["observation"] = {
                    "results": [
                        {
                            "source_call_id": event.get("tool_call_id"),
                            "content": event.get("content", ""),
                        }
                    ]
                }

    if system_prompt:
        system_step: dict[str, Any] = {
            "step_id": 0,
            "timestamp": steps[0]["timestamp"] if steps else None,
            "source": "system",
            "message": system_prompt,
        }
        steps.insert(0, system_step)

    for i, step in enumerate(steps):
        step["step_id"] = i + 1

    return {
        "schema_version": "ATIF-v1.5",
        "session_id": os.environ.get("SESSION_ID", "sagent-session"),
        "agent": {
            "name": "openhands-sdk",
            "tool_definitions": tool_definitions if tool_definitions else None,
            "version": "unknown",
        },
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": llm_metrics.get("prompt_tokens", 0),
            "total_completion_tokens": llm_metrics.get("completion_tokens", 0),
            "total_cached_tokens": llm_metrics.get("cached_tokens", 0),
            "total_cost_usd": llm_metrics.get("cost_usd", 0.0),
            # run-integrity column, not a token stat: True when the agent
            # finished without the answer file and one recovery message was
            # sent (see maybe_nudge_answer_file). Ships with the data so
            # nudged answers can be split out in analysis.
            "nudged": llm_metrics.get("nudged", False),
            # Run-integrity columns for the XML-leak salvage. They ship as data
            # because the leak rate and the rescue rate both move accuracy:
            # xml_leak alone sizes the defect, xml_leak AND NOT salvaged is the
            # unsalvageable variant (a degenerate turn that never named an
            # option), and salvaged marks a trial the harness submitted for --
            # compliance would be overstated if those were counted as the model
            # submitting on its own.
            "salvaged": llm_metrics.get("salvaged", False),
            "salvage_letter": llm_metrics.get("salvage_letter"),
            "salvage_stage": llm_metrics.get("salvage_stage"),
            # Which contract the trial was under, so an analysis can separate a
            # recovered letter from a recovered number.
            "salvage_format": llm_metrics.get("salvage_format"),
            "xml_leak": llm_metrics.get("xml_leak", False),
        },
    }


# --- XML-leak salvage -------------------------------------------------------
#
# The extraction ladder below mirrors the verifier's scorer
# (adapters/spatial/task_template/tests/score_erqa.py). It is copied,
# not imported: the scorer lives in the verifier container under /tests, next
# to expected_answer.txt, and mounting that into the agent container would leak
# gold answers. tests/test_xml_salvage.py pins the copy by asserting both
# extractors return the same letter and stage on the same text -- salvage must
# write exactly what the scorer would have read, or the harness and the grader
# disagree about what the model answered.

_MAX_BARE_LINE_LEN = 12

# Written as escapes, not as the literal characters the scorer spells them
# with: a literal zero-width space in a source file is itself a lint error, and
# an invisible character is not something a later reader can proofread. Same
# five code points, same order.
_INVISIBLES = (
    "\ufeff"  # BOM / zero-width no-break space
    "\u200b"  # zero-width space
    "\u200c"  # zero-width non-joiner
    "\u200d"  # zero-width joiner
    "\u2060"  # word joiner
    "\xa0"  # non-breaking space (folded to a plain space, see scrub)
)

_WRAPPERS = " \t\r\n*_`'\"()[]{}<>.:;,-–—!?"

_ANSWER_TAG_RE = re.compile(
    r"""
    (?: final \s+ answer | answer )
    ["'*_`]{0,3}
    (?: \s+ is \b | \s* [:=] | \s* -+ > )
    [\s*_`"'(\[]{0,6}
    ( [abcd] )
    (?![a-z])
    """,
    re.IGNORECASE | re.VERBOSE,
)

_OPTION_VERDICT_RE = re.compile(
    r"""
    (?: option | choice | selection )
    \s* [:=]? \s*
    [*_`"'(\[]{0,3}
    ( [abcd] )
    (?![a-z])
    [*_`"')\]]{0,3}
    \s+ (?: is | was ) \s+ (?: the \s+ )?
    (?: correct | right | answer | best )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Qwen's tool-call XML epilogue arriving as literal prose. All five tags were
# observed closing a terminal turn across the eight leaked trials: v4/MMSI_329
# ended on </parameter></thought>, v5a/MMSI_333 and v15/MMSI_771 on </think>,
# the rest on </parameter></function></tool_call>. Opening tags never appear --
# the model emits the epilogue it was fine-tuned to end on without ever opening
# a call, which is why replaying the leaked XML as a tool call is impossible
# here and prose extraction is the only route.
_XML_LEAK_RE = re.compile(r"</(?:tool_call|function|parameter|thought|think)>")

# How much of the turn's end counts as its conclusion.
#
# The scorer reads an answer FILE -- a short text whose whole purpose is to
# state an answer, where "last match wins" is safe. Salvage reads a whole
# reasoning turn, up to 147k characters of the model arguing with itself, and
# there "last match wins" is not safe. Two measured cases say so:
#
#   v1_seg10/MMSI_225  the last recognized tag is "the answer is B" inside
#                      "I should consider the possibility that the answer is B
#                      rather than D" -- a hypothesis the model then rejected.
#                      The turn ends "Answer D: Northeast corner", which is
#                      gold. Writing B would replace a right answer with a
#                      wrong one.
#   v1_seg10/MMSI_98   the last tag sits 109,639 characters from the end; the
#                      turn ends "let me just write a script to test all four
#                      options". The model never committed to anything.
#
# Across the 18 trials in tests/golden/xml_leak_corpus.json, every commitment
# the model actually ended on lies within 2,853 characters of the end, and the
# nearest mis-read is at 4,160. 3000 sits in that gap, deliberately near the
# bottom of it: too small a window only ever declines a rescue, while too large
# a one writes a letter the model had moved on from.
_CONCLUSION_WINDOW = 3000


def scrub(text: str) -> str:
    """Remove invisible characters and fold NBSP to a normal space."""
    if not text:
        return text
    text = text.replace("\xa0", " ")
    for ch in _INVISIBLES:
        if ch != "\xa0":
            text = text.replace(ch, "")
    return text


def extract_answer_letter(raw: str) -> tuple[str, str]:
    """Return (letter, stage); letter is "" when nothing was extracted.

    Anchored on an explicit commitment phrase, never on length, so a 140k-char
    reasoning dump is as safe to feed in as a one-line answer. The last match
    wins: an agent's concluding statement is its answer.
    """
    raw = scrub(raw)
    if not raw or not raw.strip():
        return "", "empty"

    matches = _ANSWER_TAG_RE.findall(raw)
    if matches:
        return matches[-1].lower(), "answer-tag"

    matches = _OPTION_VERDICT_RE.findall(raw)
    if matches:
        return matches[-1].lower(), "option-verdict"

    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if len(last.strip()) <= _MAX_BARE_LINE_LEN:
            stripped = last.strip(_WRAPPERS)
            if len(stripped) == 1 and stripped.lower() in "abcd":
                return stripped.lower(), "bare-letter-last-line"

    return "", "no-match"


#: Task-side marker the adapter writes for numeric-answer tasks (SPBench's NA
#: half). Absent means the original letter contract, so every letter task keeps
#: its exact behaviour and no existing task directory changes.
ANSWER_FORMAT_PATH = Path("/app/.answer_format")


#: Contracts the marker can name. "number_last" differs from "number" only in
#: the SCORER (SPAR-Bench reads the last number on multi-view tasks); salvage
#: treats the two identically, because it extracts the model's own commitment
#: rather than scanning for numbers.
_ANSWER_FORMATS = ("letter", "number", "number_last", "vci")


def answer_format(path: Path | None = None) -> str:
    """The answer contract this task is under, read from its marker file.

    Defaults to "letter" on anything unreadable or unknown: the letter path is
    the one every prior run used, so a bad marker degrades to the old behaviour
    rather than to a new one.
    """
    marker = ANSWER_FORMAT_PATH if path is None else path
    try:
        value = marker.read_text().strip().lower()
    except OSError:
        return "letter"
    return value if value in _ANSWER_FORMATS else "letter"


# The numeric counterparts of _ANSWER_TAG_RE / _OPTION_VERDICT_RE. Same
# discipline and the same reason for it: anchored on an explicit commitment
# phrase, never on "there is a number nearby". A reasoning turn about distances
# is *full* of numbers -- every intermediate measurement is one -- so an
# unanchored search would salvage a scratch value with total confidence. The
# scorer (score_numeric.py) may take the first number it sees because it reads
# a short answer FILE; salvage reads up to 147k characters of the model arguing
# with itself, which is the same asymmetry _CONCLUSION_WINDOW exists for.
_NUMBER = r"([-+]?\d+(?:\.\d+)?)"
_APPROX = r"(?:about|approximately|around|roughly|~|≈)?\s*"

_NUMBER_TAG_RE = re.compile(
    r"""
    (?: final \s+ answer | answer )
    ["'*_`]{0,3}
    (?: \s+ is \b | \s* [:=] | \s* -+ > )
    [\s*_`"'(\[]{0,6}
    """
    + _APPROX
    + _NUMBER
    + r"""
    (?! \s* [-–] \s* \d )   # "2-3" is a range, not a commitment
    """,
    re.IGNORECASE | re.VERBOSE,
)

_NUMBER_VERDICT_RE = re.compile(
    r"""
    (?: I \s* (?: '|’ ) ll \s+ go \s+ with
      | I \s+ will \s+ go \s+ with
      | I \s* (?: '|’ ) ll \s+ (?: say | answer )
      | going \s+ with
      | settle \s+ on
      | my \s+ (?: best \s+ )? (?: estimate | answer ) \s+ is
      | the \s+ (?: value | number | measurement ) \s+ is
    )
    \s* [:=]? \s*
    [*_`"'(\[]{0,3}
    """
    + _APPROX
    + _NUMBER
    + r"""
    (?! \s* [-–] \s* \d )
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: A bare final line may carry the unit the question asked in; the number is
#: what gets written either way. Salvage never CONVERTS units -- writing a
#: converted value would be inventing an answer the model did not give, and the
#: verifier reads the number the same way whatever unit the model typed after
#: it.
_BARE_NUMBER_LINE_RE = re.compile(
    r"^" + _NUMBER + r"\s*(?:cm|centimet(?:er|re)s?|m|met(?:er|re)s?)?$",
    re.IGNORECASE,
)
#: Longer than the letter path's 12 because a legitimate bare numeric answer is
#: longer than one character ("120.5 centimeters" is 17).
_MAX_BARE_NUMBER_LINE_LEN = 24


def extract_answer_number(raw: str) -> tuple[str, str]:
    """Return (number, stage); number is "" when nothing was extracted.

    Mirrors extract_answer_letter's ladder rung for rung -- commitment tag,
    verdict phrase, bare last line -- and refuses in exactly the same way when
    the turn never commits.
    """
    raw = scrub(raw)
    if not raw or not raw.strip():
        return "", "empty"

    matches = _NUMBER_TAG_RE.findall(raw)
    if matches:
        return matches[-1], "answer-tag"

    matches = _NUMBER_VERDICT_RE.findall(raw)
    if matches:
        return matches[-1], "number-verdict"

    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if lines:
        last = lines[-1].strip(_WRAPPERS)
        if len(last) <= _MAX_BARE_NUMBER_LINE_LEN:
            match = _BARE_NUMBER_LINE_RE.match(last.strip())
            if match:
                return match.group(1), "bare-number-last-line"

    return "", "no-match"


#: A run of at least three `name:value` pairs -- the shape SPAR-Bench's
#: view_change_infer answer has, and essentially nothing else in a reasoning
#: turn does. Three rather than five so a model that omits a zero axis is still
#: rescued; the scorer reads a missing axis as 0, exactly as the gold does.
_VCI_LINE_RE = re.compile(
    r"((?:[a-z_]+\s*:\s*[-+]?\d+(?:\.\d+)?\s*,\s*){2,}"
    r"[a-z_]+\s*:\s*[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def extract_answer_vci(raw: str) -> tuple[str, str]:
    """Return (line, stage) for a view-change answer; line is "" when absent.

    No commitment phrase is required here, unlike the letter and number
    ladders: the format is distinctive enough that a multi-axis list sitting in
    the conclusion IS the answer. Last match wins, same rule as the other two.
    """
    raw = scrub(raw)
    if not raw or not raw.strip():
        return "", "empty"
    matches = _VCI_LINE_RE.findall(raw)
    if not matches:
        return "", "no-match"
    return re.sub(r"\s+", "", matches[-1]), "vci-line"


def terminal_agent_message(conversation) -> str | None:
    """Text of the assistant turn that ended the episode, if it was a message.

    Walks back to the most recent agent-sourced event and stops there. If the
    episode ended on a tool call the submission was delivered and there is
    nothing to salvage; only when the last thing the agent did was *talk* can a
    write have been eaten by the wire. Duck-typed because the SDK event classes
    are imported inside main().
    """
    events = getattr(getattr(conversation, "state", None), "events", None) or []
    for event in reversed(list(events)):
        if getattr(event, "source", None) != "agent":
            continue
        if getattr(event, "tool_name", None) or getattr(event, "tool_call", None):
            return None
        message = getattr(event, "llm_message", None)
        if message is None:
            return None
        content = getattr(message, "content", None)
        if isinstance(content, list):
            return "\n".join(
                getattr(part, "text", "")
                for part in content
                if getattr(part, "text", None)
            )
        return str(content) if content else None
    return None


def _empty_salvage() -> dict[str, Any]:
    # "letter" keeps its name because it ships as the salvage_letter data
    # column that existing analysis reads; on a numeric task it carries the
    # number instead, and "format" is what tells the two apart.
    return {
        "salvaged": False,
        "letter": None,
        "stage": None,
        "xml_leak": False,
        "format": "letter",
    }


def maybe_salvage_answer_file(conversation) -> dict[str, Any]:
    """Write the answer the model reached when its submitting call leaked.

    Measured shape across eight trials (v4 through v15): the terminal assistant
    turn arrives with finish_reason=stop, no tool_calls, and Qwen's tool-call
    XML epilogue as literal text. The write never happens, the answer file is
    missing, and the trial scores 0 however good the reasoning was -- twice out
    of eight on reasoning that had already reached the gold letter
    (v6a/MMSI_466, v15/MMSI_98). Without this, every configuration's score carries that
    as a floor of random noise.

    This repairs the transport failure and nothing else: it writes the letter
    the model already committed to in prose, read with the verifier's own
    ladder, and refuses when there is no explicit commitment. Two of the eight
    are exactly that -- a degenerate turn that never named an option (v5a/333,
    v15/771) cannot be rescued and must not be guessed at. Runs BEFORE
    maybe_nudge_answer_file: the nudge exists to ask a model that forgot, while
    a leaked write means the model did not forget, and every turn after the
    leak only exists because of the bug. v15/MMSI_98 shows why the order
    matters -- its pre-nudge turn concluded A (gold), its nudge reply concluded
    D, and repairing at the point of breakage is what keeps the recovered
    answer the one the episode actually ended on.

    Gated by SALVAGE_ANSWER_FILE (unset/"0" = off, "1" = /app/answer.txt,
    anything else = that path), same convention as NUDGE_ANSWER_FILE. Returns
    the record that ships as final_metrics data columns.
    """
    record = _empty_salvage()
    raw = (os.environ.get("SALVAGE_ANSWER_FILE") or "").strip()
    if not raw or raw == "0":
        return record
    answer_path = Path("/app/answer.txt" if raw == "1" else raw)
    if answer_path.is_file():
        return record

    text = terminal_agent_message(conversation)
    if not text:
        return record

    record["xml_leak"] = bool(_XML_LEAK_RE.search(text))
    fmt = answer_format()
    record["format"] = fmt
    window = text[-_CONCLUSION_WINDOW:]
    if fmt in ("number", "number_last"):
        value, stage = extract_answer_number(window)
    elif fmt == "vci":
        value, stage = extract_answer_vci(window)
    else:
        value, stage = extract_answer_letter(window)
        value = value.upper()
    if not value:
        print(
            f"salvage: terminal turn has no explicit {fmt} answer ({stage}, "
            f"xml_leak={record['xml_leak']}); refusing to guess"
        )
        return record

    try:
        answer_path.parent.mkdir(parents=True, exist_ok=True)
        answer_path.write_text(value)
    except OSError as exc:
        print(f"salvage: could not write {answer_path} ({exc!r})")
        return record

    record.update(salvaged=True, letter=value, stage=stage)
    print(
        f"salvage: wrote {value} to {answer_path} via {stage} "
        f"(format={fmt}, xml_leak={record['xml_leak']}); "
        "the model's submitting call never landed"
    )
    return record


def _merge_salvage(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Combine a pre-nudge and post-nudge salvage attempt into one record."""
    return {
        "salvaged": first["salvaged"] or second["salvaged"],
        "letter": first["letter"] or second["letter"],
        "stage": first["stage"] or second["stage"],
        "xml_leak": first["xml_leak"] or second["xml_leak"],
        # Same task either way; take whichever attempt read the marker.
        "format": second.get("format") or first.get("format") or "letter",
    }


def maybe_nudge_answer_file(conversation) -> bool:
    """One recovery message when the agent finished without its answer file.

    The SDK treats any assistant reply without tool calls as task completion,
    and the runtime is one-shot — so a single degenerate completion (MMSI_98
    in _segv9a: 2 tokens, ".") ends the trial with nothing written, which
    scores a guaranteed 0. Gated by NUDGE_ANSWER_FILE (unset/"0" = off,
    "1" = /app/answer.txt, anything else = that path), set per-configuration via
    run_job.sh --ae so non-spatial sagent tasks never see it. ONE message,
    ONE extra run(), never more: a model that stalls twice stays stalled,
    and unbounded nudging could loop forever. Returns whether it fired —
    recorded as final_metrics.nudged so analysis can split forced answers.
    """
    raw = (os.environ.get("NUDGE_ANSWER_FILE") or "").strip()
    if not raw or raw == "0":
        return False
    answer_path = Path("/app/answer.txt" if raw == "1" else raw)
    if answer_path.is_file():
        return False
    # The shape asked for has to match the task. Telling a model that just
    # measured a table in centimetres to write "a single option letter" is a
    # question it cannot answer, and the observed result is that it writes
    # nothing at all -- a guaranteed 0 caused by the nudge itself.
    shape = {
        "number": "a single number, digits only, no units",
        "number_last": "a single number, digits only, no units",
        "vci": "the one comma-separated name:value line the question specifies",
    }.get(answer_format(), "a single option letter")
    print(f"nudge: agent finished without {answer_path}; sending ONE recovery message")
    conversation.send_message(
        "You finished without writing the answer file. Write your current "
        f"best answer to {answer_path} now ({shape}), using "
        "the terminal, then stop."
    )
    conversation.run()
    print(f"nudge: recovery run done, answer_file_exists={answer_path.is_file()}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenHands SDK agent")
    parser.add_argument("--instruction", required=True, help="Task instruction")
    parser.add_argument("--logs-dir", required=True, help="Directory for logs")
    parser.add_argument(
        "--trajectory-path", required=True, help="Path to save trajectory"
    )
    args = parser.parse_args()

    model = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4-5-20250929")
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    send_reasoning_content_models = parse_send_reasoning_content_models(
        os.environ.get("SEND_REASONING_CONTENT_MODELS")
    )

    if not api_key:
        print("Error: LLM_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    patched_models = patch_send_reasoning_content_models(send_reasoning_content_models)

    from openhands.sdk import (
        LLM,
        Agent,
        AgentContext,
        Conversation,
        Tool,
        get_logger,
    )
    from openhands.sdk.context import Skill, load_skills_from_dir
    from openhands.sdk.event import (
        ActionEvent,
        MessageEvent,
        ObservationEvent,
        TokenEvent,
    )
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    logger = get_logger(__name__)

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    litellm_extra_body = parse_json_dict_env(
        os.environ.get("LITELLM_EXTRA_BODY"),
        "LITELLM_EXTRA_BODY",
    )
    if litellm_extra_body:
        logger.debug(f"LiteLLM extra body: {litellm_extra_body}")

    litellm_extra_headers = parse_json_dict_env(
        os.environ.get("LITELLM_EXTRA_HEADERS"),
        "LITELLM_EXTRA_HEADERS",
    )
    if litellm_extra_headers:
        logger.debug(f"LiteLLM extra headers: {litellm_extra_headers}")

    # Let the model see its own prior thinking.
    #
    # Qwen3.6's chat template renders an assistant turn's reasoning ONLY when
    # `preserve_thinking` is true, or when the turn sits after the last real user
    # message (chat_template.jinja:100). Every earlier assistant turn is
    # therefore stripped, so with `--reasoning-parser qwen3` the agent starts
    # each turn having forgotten what it worked out in the previous ones.
    #
    # Measured on the real stack (a 3555-char reasoning_content in history):
    #     preserve_thinking off -> prompt_tokens  36
    #     preserve_thinking on  -> prompt_tokens 761   (+725)
    #
    # And measured on the runs before this flag existed, same 100 questions:
    #     assistant history in the final request  38405 chars (v9, reasoning in
    #     content) -> 232 chars (v16, reasoning split out and dropped)
    #     tokens per turn 1631 -> 559, turns 15 -> 17, accuracy 51% -> 34%
    #
    # ⚠️ THIS IS ONLY HALF THE GATE. openhands-sdk independently refuses to
    # serialize reasoning unless the model name matches
    # SEND_REASONING_CONTENT_MODELS (model_features.py:222-230, substring
    # match) -- a list of kimi/minimax/deepseek spellings that no spelling of
    # ours matches. So this flag alone changes nothing for SDK-generated
    # history: verified 2x2, template-gate-only was +0 tokens. The SDK half
    # needs a separate change; see docs/experiments/
    # 2026-08-22-reasoning-not-replayed.md. Setting this now is still correct
    # and harmless -- it is the half that needs no code change and it must be
    # on for the other half to have any effect.
    #
    # Cost: history now grows with reasoning. Estimated +16k-23k tokens at 18
    # turns against a 262144 input budget, so no wall, but `run_job.sh`
    # configures no condenser, so growth is unbounded in principle. The 80-turn
    # worst case is UNVERIFIED and should be measured on the next run.
    extra_body = dict(litellm_extra_body or {})
    template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    # Switchable, because whether replaying reasoning HELPS is an open question,
    # not a settled fix. First measurement, same 100 questions:
    #     v16 (replay off)  accuracy 47.3%  CI [37.3, 57.4]
    #     v17 (replay on)   accuracy 38.3%  CI [29.1, 48.4]
    # 9.0pp apart, below this project's 15pp threshold, so unresolved -- but the
    # direction is the opposite of what the fix intended, and Qwen's own chat
    # template strips prior-turn thinking BY DEFAULT, which is itself evidence
    # that dropping it may be the intended regime. Being able to flip this from
    # the environment makes the A/B one variable instead of an image rebuild.
    _replay = os.environ.get("PRESERVE_THINKING", "1").strip().lower()
    if _replay not in {"0", "false", "no"}:
        template_kwargs.setdefault("preserve_thinking", True)
    extra_body["chat_template_kwargs"] = template_kwargs

    llm_kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "litellm_extra_body": extra_body,
    }
    if litellm_extra_headers:
        llm_kwargs["extra_headers"] = {
            str(key): str(value) for key, value in litellm_extra_headers.items()
        }
    temperature_raw = os.environ.get("LLM_TEMPERATURE")
    if temperature_raw:
        llm_kwargs["temperature"] = float(temperature_raw)

    # HTTP timeout for one LLM call. MUST be set explicitly: the SDK's default
    # is 300s (openhands/sdk/llm/llm.py:316-320) with num_retries=5, and this
    # code never passed it, so every generation longer than five minutes was
    # abandoned by the client and silently retried.
    #
    # That default is incompatible with this workload by a factor of ~7. Our
    # long generations take a median of 2076s (max 2511s), so the client gave
    # up mid-generation while the server kept going to completion. Measured
    # consequences on the v12 run:
    #
    #   * 20 of 59 `length` turns in the trajectory are DUPLICATES -- same
    #     history depth, same final message, logged again because litellm
    #     records each server-side completion while the client had already
    #     walked away. Deduplicated, v12 has effectively zero accepted
    #     cap-truncations, so the "8 episodes died hitting the generation cap"
    #     reading was an artefact of counting abandoned retries.
    #   * the SDK's own final_metrics accounts for 1.6M completion tokens
    #     against 10.5M in the proxy log -- roughly 85% of all generation was
    #     paid for and thrown away.
    #
    # ⚠️ This is coupled to `[agent] timeout_sec` in task.toml and must be read
    # together with it. While timeout_sec was 900 the episode ended before a
    # third 300s window could open, so the defect was INVISIBLE: v9 shows only
    # 2% unanswered because the 900s wall closed episodes early, not because
    # the pipeline was healthy. Raising timeout_sec to 7200 (cdb12ee) removed
    # the wall and exposed this. Do not tune one of the two alone.
    #
    # 7200 matches timeout_sec: the episode budget should be what bounds a run,
    # not an HTTP client default nobody chose.
    llm_kwargs["timeout"] = int(os.environ.get("LLM_TIMEOUT_SEC", "7200"))

    llm = LLM(**llm_kwargs)

    tools = [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ]

    skills: list[Skill] = []
    if os.environ.get("LOAD_SKILLS", "1") == "1":
        skill_paths_str = os.environ.get("SKILL_PATHS", "")
        if skill_paths_str:
            # Configurations that run without skills (disk_only, none) leave SKILL_PATHS
            # unset and never reach discover_skills, whose guards all require
            # at least one skill. A value of only separators is the same intent.
            skill_paths = [p for p in skill_paths_str.split(":") if p.strip()]
            if skill_paths:
                # run_job.sh sets this only for SKILL_MODE=skill (M2), where
                # every named path is a hard requirement rather than one of
                # SAgent's speculative agent-home defaults. See discover_skills.
                required = os.environ.get("SKILL_PATHS_REQUIRED", "0") == "1"
                skills = discover_skills(
                    skill_paths, load_skills_from_dir, logger, required=required
                )
            logger.debug(f"Loaded {len(skills)} skills")

    agent_context = AgentContext(skills=skills)

    mcp_config = None
    mcp_servers_raw = os.environ.get("MCP_SERVERS_JSON")
    if mcp_servers_raw:
        mcp_servers = json.loads(mcp_servers_raw)
        mcp_config = {"mcpServers": {}}
        for mcp in mcp_servers:
            server_name = mcp.get("name", "mcp-server")
            transport = mcp.get("transport", "stdio")
            server_cfg: dict[str, Any] = {}
            if transport == "stdio":
                if mcp.get("command"):
                    server_cfg["command"] = mcp["command"]
                if mcp.get("args"):
                    server_cfg["args"] = mcp["args"]
            else:
                if mcp.get("url"):
                    server_cfg["url"] = mcp["url"]
            mcp_config["mcpServers"][server_name] = server_cfg
        logger.debug(f"MCP config: {json.dumps(mcp_config, indent=2)}")

    agent_kwargs: dict[str, Any] = {
        "llm": llm,
        "tools": tools,
        "agent_context": agent_context,
    }
    if mcp_config:
        agent_kwargs["mcp_config"] = mcp_config

    condenser_raw = os.environ.get("CONDENSER_CONFIG_JSON")
    condenser_summary: str | None = None
    if condenser_raw:
        try:
            cfg = json.loads(condenser_raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "CONDENSER_CONFIG_JSON is invalid JSON (%s); no condenser will be attached.",
                exc,
            )
            cfg = {"type": "<invalid_json>"}
        if not isinstance(cfg, dict):
            logger.warning(
                "CONDENSER_CONFIG_JSON must decode to a JSON object; no condenser will be attached."
            )
            cfg = {"type": "<invalid_json>"}
        condenser_type = cfg.get("type")
        if condenser_type == "summarizer":
            from openhands.sdk.context.condenser import LLMSummarizingCondenser

            llm_cfg = cfg.get("llm") or {}
            cond_temperature_raw = llm_cfg.get("temperature")
            cond_llm_kwargs: dict[str, Any] = {
                "model": llm_cfg.get("model") or model,
                "api_key": llm_cfg.get("api_key") or api_key,
                "base_url": llm_cfg.get("base_url") or base_url,
                # Must differ from the agent LLM's default ("default") or the SDK
                # raises ValueError when both are added to llm_registry.
                "usage_id": "condenser",
            }
            if cond_temperature_raw is not None:
                cond_llm_kwargs["temperature"] = float(cond_temperature_raw)
            else:
                cond_llm_kwargs["temperature"] = 0.0
            # Same 300s default trap as the agent LLM above -- see the long
            # comment there. Condenser calls are shorter, but "shorter" is not
            # a guarantee and a silent client-side retry here would corrupt the
            # condensed history rather than just waste tokens.
            cond_llm_kwargs["timeout"] = int(os.environ.get("LLM_TIMEOUT_SEC", "7200"))
            cond_llm = LLM(**cond_llm_kwargs)

            cond_kwargs: dict[str, Any] = {
                "llm": cond_llm,
                "max_size": int(cfg.get("max_size", 120)),
                "keep_first": int(cfg.get("keep_first", 4)),
                "minimum_progress": float(cfg.get("minimum_progress", 0.1)),
            }
            if cfg.get("max_tokens") is not None:
                cond_kwargs["max_tokens"] = int(cfg["max_tokens"])

            agent_kwargs["condenser"] = LLMSummarizingCondenser(**cond_kwargs)
            condenser_summary = (
                "Enabled LLMSummarizingCondenser: "
                f"model={cond_llm.model}, max_size={cond_kwargs['max_size']}, "
                f"keep_first={cond_kwargs['keep_first']}, "
                f"max_tokens={cond_kwargs.get('max_tokens')}, "
                f"minimum_progress={cond_kwargs['minimum_progress']}, "
                f"temperature={cond_llm_kwargs['temperature']}"
            )
        else:
            logger.warning(
                "CONDENSER_CONFIG_JSON specified unknown type=%r; "
                "no condenser will be attached.",
                condenser_type,
            )

    agent = Agent(**agent_kwargs)

    workspace = os.getcwd()
    conv_kwargs: dict[str, Any] = {"agent": agent, "workspace": workspace}
    max_iter_raw = os.environ.get("MAX_ITERATIONS")
    if max_iter_raw:
        conv_kwargs["max_iteration_per_run"] = int(max_iter_raw)
        logger.debug(f"Max iterations per run: {max_iter_raw}")
    conversation = Conversation(**conv_kwargs)

    print(f"Starting agent with instruction: {args.instruction[:200]}...")
    print(f"Using model: {model}")
    print(f"Python executable: {sys.executable}")
    if temperature_raw:
        print(f"Temperature: {temperature_raw}")
    if max_iter_raw:
        print(f"Max iterations per run: {max_iter_raw}")
    print(f"Loaded {len(skills)} skills")
    if patched_models:
        print("Patched SEND_REASONING_CONTENT_MODELS: " + ", ".join(patched_models))
    if mcp_config:
        print(f"MCP servers: {list(mcp_config['mcpServers'].keys())}")
    if condenser_summary:
        print(condenser_summary)

    conversation.send_message(args.instruction)
    conversation.run()
    # Repair a leaked submission first, then ask a model that forgot; a nudge
    # reply can leak the same way, so try once more after it (see
    # maybe_salvage_answer_file for why the order is not the other way round).
    salvage = maybe_salvage_answer_file(conversation)
    nudged = maybe_nudge_answer_file(conversation)
    if nudged:
        salvage = _merge_salvage(salvage, maybe_salvage_answer_file(conversation))

    token_usage = llm.metrics.accumulated_token_usage
    metrics = {
        "prompt_tokens": token_usage.prompt_tokens if token_usage else 0,
        "completion_tokens": token_usage.completion_tokens if token_usage else 0,
        "cached_tokens": token_usage.cache_read_tokens if token_usage else 0,
        "cost_usd": llm.metrics.accumulated_cost,
        "nudged": nudged,
        "salvaged": salvage["salvaged"],
        "salvage_letter": salvage["letter"],
        "salvage_stage": salvage["stage"],
        "salvage_format": salvage["format"],
        "xml_leak": salvage["xml_leak"],
    }

    system_prompt = None
    tool_definitions: list[dict[str, Any]] = []
    try:
        system_prompt = agent.static_system_message
    except Exception as e:
        logger.warning(f"Could not extract system prompt: {e}")
    try:
        for tool_name, tool_obj in agent.tools_map.items():
            tool_definitions.append(tool_obj.to_openai_tool())
    except Exception as e:
        logger.warning(f"Could not extract tool definitions: {e}")

    if system_prompt:
        print(f"Captured system prompt ({len(system_prompt)} chars)")
    print(f"Captured {len(tool_definitions)} tool definitions")

    events_list: list[dict[str, Any]] = []
    last_agent_timestamp: str | None = None
    for event in conversation.state.events:
        if isinstance(event, MessageEvent):
            content = ""
            if event.llm_message:
                msg_content = getattr(event.llm_message, "content", None)
                if isinstance(msg_content, list):
                    content = "\n".join(
                        getattr(c, "text", str(c))
                        for c in msg_content
                        if getattr(c, "text", None)
                    )
                elif msg_content:
                    content = str(msg_content)
            if event.source == "user":
                events_list.append(
                    {
                        "type": "user_message",
                        "content": content,
                        "timestamp": event.timestamp,
                    }
                )
            elif event.source == "agent":
                events_list.append(
                    {
                        "type": "assistant_message",
                        "content": content,
                        "timestamp": event.timestamp,
                    }
                )
                last_agent_timestamp = event.timestamp
        elif isinstance(event, ActionEvent):
            tool_call_args: dict[str, Any] = {}
            if event.tool_call and hasattr(event.tool_call, "function"):
                raw_args = getattr(event.tool_call.function, "arguments", None)
                if isinstance(raw_args, str):
                    try:
                        tool_call_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        tool_call_args = {"raw": raw_args}
                elif isinstance(raw_args, dict):
                    tool_call_args = raw_args
            if not tool_call_args and event.action:
                try:
                    action_dict = (
                        event.action.model_dump()
                        if hasattr(event.action, "model_dump")
                        else vars(event.action)
                    )
                    tool_call_args = {
                        k: v
                        for k, v in action_dict.items()
                        if k != "kind" and v is not None
                    }
                except Exception:
                    pass
            events_list.append(
                {
                    "type": "assistant_message",
                    "content": "",
                    "timestamp": event.timestamp,
                    "tool_calls": [
                        {
                            "id": event.tool_call_id,
                            "name": event.tool_name,
                            "arguments": tool_call_args,
                        }
                    ],
                }
            )
            last_agent_timestamp = event.timestamp
        elif isinstance(event, ObservationEvent):
            obs_content = ""
            if event.observation:
                obs_raw = getattr(event.observation, "content", None)
                if isinstance(obs_raw, list):
                    obs_content = "\n".join(
                        getattr(c, "text", str(c))
                        for c in obs_raw
                        if getattr(c, "text", None)
                    )
                elif obs_raw:
                    obs_content = str(obs_raw)
                else:
                    obs_content = str(event.observation)
            events_list.append(
                {
                    "type": "tool_result",
                    "tool_call_id": event.tool_call_id,
                    "content": obs_content,
                    "timestamp": event.timestamp,
                }
            )
        elif isinstance(event, TokenEvent):
            if last_agent_timestamp and events_list:
                for ev in reversed(events_list):
                    if ev.get("timestamp") == last_agent_timestamp:
                        ev["token_ids"] = {
                            "prompt_token_ids": getattr(event, "prompt_token_ids", []),
                            "response_token_ids": getattr(
                                event, "response_token_ids", []
                            ),
                        }
                        break

    trajectory = build_trajectory(
        events_list,
        metrics,
        model,
        system_prompt=system_prompt,
        tool_definitions=tool_definitions,
    )

    trajectory_path = Path(args.trajectory_path)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trajectory_path, "w") as f:
        json.dump(trajectory, f, indent=2)

    print(f"Agent completed. Trajectory saved to {trajectory_path}")
    print(f"Total cost: ${metrics['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
