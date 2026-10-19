#!/usr/bin/env python3
"""Official-ERQA-compatible scorer with a bounded defensive extraction ladder.

Reads the agent's answer file and the gold answer file, prints a short verdict
to stdout, and exits 0 when the answer is correct, 1 otherwise. The calling
test.sh turns that exit code into a SAgent reward of 1 or 0.

Scoring contract
----------------
Stage 0 (official). ERQA's own eval_harness compares
``prediction.replace(".", "").strip().lower() == answer.strip().lower()``.
That is tried first, verbatim, on the whole file contents, so any answer the
official harness accepts is accepted here. It already covers "B", "B.", " b ".

Stage 1 (explicit answer tag). Search the whole text for an explicit
answer-tag phrase such as "Answer: B", "final answer = (c)", "the answer is D".
The *last* match wins, matching the convention that an agent's concluding
statement is its answer. Safe on long text because the tag itself is the
anchor, not the length.

Stage 2 (bare letter on the last line). The last non-blank line, once stripped
of surrounding whitespace, parentheses, quotes, and trailing punctuation, must
reduce to exactly one letter A-F (see the note at _ANSWER_TAG_RE). A pre-normalization length cap keeps this
from firing on prose.

Anything else scores 0. In particular a full reasoning dump that never states
its answer in one of the above forms is NOT accepted -- string-containment
matching a whole essay against a single letter is exactly the bug this ladder
exists to prevent.
"""

from __future__ import annotations

import os
import re
import sys

# Pre-normalization length cap for the stage-2 bare-letter line. Generous
# enough for '**(B)**' or '"b".' and far too small for a sentence.
_MAX_BARE_LINE_LEN = 12

# Invisible characters that carry no meaning but defeat naive stripping.
# Editors and some agent tool-call paths prepend a UTF-8 BOM, and models
# occasionally emit zero-width or non-breaking spaces around a bare letter.
# Left in place these produce false negatives: a file whose entire content is
# "﻿C" is unambiguously the letter C but matches neither the official
# comparison nor a whitespace strip. They are deleted before any scoring.
_INVISIBLES = (
    "﻿"  # BOM / zero-width no-break space
    "​"  # zero-width space
    "‌"  # zero-width non-joiner
    "‍"  # zero-width joiner
    "⁠"  # word joiner
    "\xa0"    # non-breaking space (normalized to a plain space, see below)
)

# Wrapping characters stripped from a candidate bare-letter line.
_WRAPPERS = " \t\r\n*_`'\"()[]{}<>.:;,-–—!?"


def scrub(text: str) -> str:
    """Remove invisible characters and fold NBSP to a normal space.

    Applied once, up front, so every scoring stage sees the same clean text.
    This only ever removes characters that a human reader cannot see; it can
    never turn a wrong letter into a right one.
    """
    if not text:
        return text
    text = text.replace("\xa0", " ")
    for ch in _INVISIBLES:
        if ch != "\xa0":
            text = text.replace(ch, "")
    return text

# Stage 1a: an explicit answer tag followed by a letter. Permissive about the
# connector ("is", ":", "=", "->") and about wrapping punctuation/markdown
# around both the tag and the letter, so "Answer: B", "**Answer:** **B**",
# 'answer -> B' and '{"answer": "B"}' all match. The letter must not be glued
# to a longer word -- (?![a-z]) rejects "Answer: Because ...".
#
# Note the anchor is only "answer" / "final answer". A bare "option X" is
# deliberately NOT an anchor here: reasoning text enumerates "option A ...
# option B ..." as a matter of course, so treating that as a decision would
# let an essay score by accident.
# The letter class is a-f, not a-d. MMSI-Video-Bench ships 21 five-option and
# 103 six-option questions (of 1106), whose gold is E or F -- with a-d those
# questions could never be scored right no matter what the agent answered.
# Widening is behaviour-preserving for the four-option benchmarks: stage 2 is
# the LAST stage, so an extracted 'e'/'f' cannot preempt a later rule, and
# against an A-D gold it scores 0 exactly as "no-match" did. The (?![a-z])
# lookahead still keeps prose out ("answer is enough" does not yield 'e').
_ANSWER_TAG_RE = re.compile(
    r"""
    (?: final \s+ answer | answer )
    ["'*_`]{0,3}                    # closing wrapper on the tag word
    # The connector is REQUIRED. When it was optional, "answer" followed by any
    # whitespace and a letter a-d matched, so ordinary prose like
    # "To answer a question about depth..." extracted 'a'. Because this stage runs
    # before the bare-letter stage and keeps the LAST match, that silently
    # overrode the agent's real answer -- verified live: a file ending in a correct
    # "C" was graded 0 with reason "extracted 'a' via answer-tag". It also fails
    # the other way, awarding 1 when gold happens to be the prose letter.
    (?: \s+ is \b | \s* [:=] | \s* -+ > )
    [\s*_`"'(\[]{0,6}               # opening wrappers before the letter
    ( [abcdef] )
    (?![a-z])
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Stage 1b: "Option C is correct." A bare option mention is not enough; the
# sentence must assert that this option is the chosen one.
_OPTION_VERDICT_RE = re.compile(
    r"""
    (?: option | choice | selection )
    \s* [:=]? \s*
    [*_`"'(\[]{0,3}
    ( [abcdef] )
    (?![a-z])
    [*_`"')\]]{0,3}
    \s+ (?: is | was ) \s+ (?: the \s+ )?
    (?: correct | right | answer | best )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def official_normalize(text: str) -> str:
    """Exactly ERQA's own prediction normalization."""
    return text.replace(".", "").strip().lower()


def extract_letter(raw: str) -> tuple[str, str]:
    """Return (letter, stage) where letter is '' when nothing was extracted.

    ``letter`` is lowercase. ``stage`` names the rule that fired, for logging.
    """
    if not raw or not raw.strip():
        return "", "empty"

    # Stage 1a: explicit answer tag, last occurrence wins.
    matches = _ANSWER_TAG_RE.findall(raw)
    if matches:
        return matches[-1].lower(), "answer-tag"

    # Stage 1b: "Option C is correct."
    matches = _OPTION_VERDICT_RE.findall(raw)
    if matches:
        return matches[-1].lower(), "option-verdict"

    # Stage 2: bare letter on the last non-blank line.
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if len(last.strip()) <= _MAX_BARE_LINE_LEN:
            stripped = last.strip(_WRAPPERS)
            if len(stripped) == 1 and stripped.lower() in "abcdef":
                return stripped.lower(), "bare-letter-last-line"

    return "", "no-match"


def score(raw_prediction: str, gold: str) -> tuple[int, str]:
    """Return (reward, reason). reward is 1 or 0."""
    raw_prediction = scrub(raw_prediction)
    gold_norm = scrub(gold).strip().lower()
    if not gold_norm:
        return 0, "gold answer is empty (task generation bug)"

    # Stage 0: official ERQA exact match on the whole file.
    if official_normalize(raw_prediction) == gold_norm:
        return 1, "official exact match"

    letter, stage = extract_letter(raw_prediction)
    if not letter:
        return 0, f"no answer letter could be extracted ({stage})"
    if letter == gold_norm:
        return 1, f"extracted '{letter}' via {stage}"
    return 0, f"extracted '{letter}' via {stage}, expected '{gold_norm}'"


def main() -> int:
    answer_file = os.environ.get("ANSWER_FILE", "/app/answer.txt")
    expected_file = os.environ.get("EXPECTED_FILE", "/tests/expected_answer.txt")

    if not os.path.isfile(expected_file):
        print(f"ERROR: expected answer file not found: {expected_file}")
        return 1
    gold = _read(expected_file)

    if not os.path.isfile(answer_file):
        print(f"ERROR: answer file not found: {answer_file}")
        print("reward=0 reason=answer file missing")
        return 1

    raw = _read(answer_file)
    # Emptiness is judged after scrubbing, so a file holding only a BOM or a
    # zero-width space is reported as empty rather than as an unparseable answer.
    if not scrub(raw).strip():
        print(f"ERROR: answer file is empty: {answer_file}")
        print("reward=0 reason=answer file empty")
        return 1

    reward, reason = score(raw, gold)
    print(f"agent answer (first 300 chars): {raw[:300]!r}")
    print(f"expected answer: {gold.strip()!r}")
    print(f"reward={reward} reason={reason}")
    return 0 if reward == 1 else 1


def _read(path: str) -> str:
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8", errors="replace")


if __name__ == "__main__":
    sys.exit(main())
