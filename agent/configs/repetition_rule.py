"""litellm post-call rule: reject degenerate repetition so the Router retries.

Wiring (configs/litellm_qwen_spatial.yaml):

    litellm_settings:
      post_call_rules: repetition_rule.repetition_rule
    router_settings:
      num_retries: 2

`get_instance_fn` resolves that dotted path relative to THIS directory
(litellm/proxy/types_utils/utils.py:38-47), same as trajectory_logger.

WHY A POST-CALL RULE AND NOT THE THINGS WE ALREADY TRIED
--------------------------------------------------------
Two previous attempts failed, both recorded at length in
configs/litellm_qwen_spatial.yaml. Read that before touching this.

  v10  capped max_output_tokens at 16384.  accuracy 51%->27%, unanswered
       2%->57%.  It truncated ordinary long reasoning mid-episode; the cap
       cannot tell a loop from real work.
  v11  enabled vLLM's built-in repetition_detection.  accuracy 51%->9%,
       unanswered 2%->74%.  The detector worked, but vLLM reports it as
       finish_reason="repetition", litellm's map_finish_reason() does not
       know that value and rewrites it to "stop"
       (litellm_core_utils/core_helpers.py:109-116), and openhands-sdk
       declares an episode FINISHED on any non-blank text with no tool_calls
       (response_dispatch.py:290-294).  So a mid-sentence truncation read as
       "the agent is done".

The v11 failure is not fixable by picking a different finish_reason: the SDK
never inspects finish_reason anywhere (grepped the whole package; the only
occurrence is a test fixture). So the bad response must never leave the proxy.

That is exactly what this hook does. Returning {"decision": False} raises
APIResponseValidationError (rules.py:48-54), which carries status_code=500
(exceptions.py:750), which _should_retry() treats as retryable
(utils.py:6930-6956), so the Router regenerates IN-PROCESS and hands the
client a normal 200 (router.py:6822-6842). The SDK learns nothing happened.

If every retry is exhausted the client does get a 500 -- and openhands-sdk
retries InternalServerError (llm.py:130-138, num_retries=5). So the terminal
failure mode is another retry, never a silently-completed episode. That is
the precise inverse of v11.

WHAT IT DOES NOT TOUCH
----------------------
Nothing in the sampling path. A healthy generation is bit-identical to today,
not "close" -- the code producing it is untouched. This is why the hook is
preferable to repetition_penalty / frequency_penalty / no_repeat_ngram_size,
all of which reshape logits on every decode step and would break
comparability with v1-v12.

KNOWN LIMITS, both verified rather than assumed
-----------------------------------------------
* The rule only sees `choices[0].message.content`, and is skipped entirely
  when content is None (rules.py:40-41, utils.py:1334). A loop inside
  tool-call ARGUMENTS is therefore invisible here. Fail-safe: not firing is
  today's behaviour. Upgrade path if it matters is
  async_post_call_success_hook, which sees the whole response object.
* Only ONE rule is ever loaded: proxy_server.py:4079 assigns a
  single-element list rather than appending. All logic must live in this one
  function.
* Retries are independent draws (temperature=0.6, no seed), but a prompt that
  reliably induces looping may loop again. Router retries (2) compose with
  the SDK's (5); whether that is enough is the open question a smoke run has
  to answer, not something to assert here.
"""

from __future__ import annotations

import os
import sys

# The predicate lives in the sweep script so the offline validation and the
# live rule cannot drift apart -- if the thresholds move, they move for both.
#
# Two candidate locations, because this file is copied out of the repo in at
# least one place: the proxy loads it relative to the CONFIG file's directory
# (litellm/proxy/types_utils/utils.py:38-47), and the Stage 2 harness stages
# a copy in a scratch dir. Looking only at <parent>/../scripts made the module
# unimportable there, and litellm reports that as a bare
# "Could not import repetition_rule from repetition_rule" with the real
# ModuleNotFoundError swallowed -- an hour of confusion if it happens on the
# live proxy. Stage 2 caught it; keep both paths.
_here = os.path.dirname(os.path.abspath(__file__))
for _candidate in (
    os.path.join(os.path.dirname(_here), "scripts"),  # repo layout
    os.path.join(_here, "scripts"),  # staged copy
):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

try:
    from repetition_detector_sweep import (
        MAX_ZLIB_RATIO,
        MIN_TOKENS,
        zlib_ratio,
    )
except ImportError as exc:  # pragma: no cover - deployment error, not logic
    # Say WHERE we looked. litellm's own error message names neither the
    # missing module nor the search path.
    raise ImportError(
        "repetition_rule needs scripts/repetition_detector_sweep.py (it owns "
        "the thresholds this rule enforces). Looked in: "
        + ", ".join(sys.path[:2])
    ) from exc

#: Measured over 3016 archived responses: median 2.56, max 6.49. See the
#: docstring of repetition_rule() for why this is the median and not a
#: "conservative" round number.
CHARS_PER_TOKEN = 2.56


def is_degenerate_text(text: str) -> str:
    """Shared predicate. Returns a reason string when degenerate, else "".

    Used by BOTH entry points so they cannot disagree:
      * `repetition_rule` below (litellm post_call_rules) -- sees `content`
        only, which since `--reasoning-parser qwen3` is usually empty;
      * `TrajectoryLogger.async_post_call_success_hook` -- sees the whole
        response, including `reasoning`, which is where the loops actually are.

    The second exists because the first went blind: a real response with
    completion_tokens=131072 had content of length 0, so the rule passed it.
    Keeping one predicate means a threshold change lands in both places, and
    the offline sweep keeps licensing both.
    """
    if not text or len(text) < MIN_TOKENS * CHARS_PER_TOKEN:
        return ""
    ratio = zlib_ratio(text)
    if ratio >= MAX_ZLIB_RATIO:
        return ""
    return (
        f"degenerate repetition detected: {len(text)} chars compress to "
        f"{ratio:.4f} of their size (threshold {MAX_ZLIB_RATIO}; the lowest "
        f"ratio seen on healthy output is 0.0403). Regenerating rather than "
        f"returning a decode loop."
    )


def repetition_rule(input: str) -> dict:
    """Return {"decision": False} to make the Router regenerate this response.

    `input` is the completion text only -- litellm passes
    choices[0].message.content (utils.py:1330-1337), so there is no token
    count available here and the length gate has to be applied on characters.

    The conversion is MEASURED, not assumed: over 3016 archived responses the
    chars-per-token ratio is median 2.56, max 6.49. An earlier draft of this
    file used 3.5 "conservatively" without checking, which put the effective
    gate at ~22400 tokens rather than the 16384 the sweep validated -- i.e. it
    silently disagreed with its own offline validation. Using the median keeps
    the live gate at the token count the sweep actually tested.

    Direction of error, if the ratio drifts: a LOWER chars-per-token constant
    lets shorter text through the gate, so the safe direction is to keep this
    at or above the true median. Degenerate loops sit far above the gate
    anyway (the shortest firing response in the sweep is ~25000 chars), so
    the gate's job is excluding short healthy text, not catching loops.
    """
    reason = is_degenerate_text(input)
    if not reason:
        return {"decision": True}

    # The message reaches the client on retry exhaustion, and lands in the
    # trajectory's failure record, so make it say what happened -- v11's core
    # problem was that an early stop was indistinguishable from a normal one
    # downstream.
    return {"decision": False, "message": reason}
