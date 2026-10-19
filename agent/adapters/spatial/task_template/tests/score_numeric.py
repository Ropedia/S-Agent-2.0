#!/usr/bin/env python3
"""Scorer for numeric-answer tasks (SPBench's NA half).

Prints one float in [0, 1] to stdout and exits 0. The reward is **fractional**,
so the exit code carries no score -- tests/test_numeric.sh reads stdout. That is
the whole reason this is a separate scorer from score_erqa.py, whose contract is
"exit 0 = right, exit 1 = wrong".

The metric is Mean Relative Accuracy (MRA), the VSI-Bench measure SPBench and
SpatialLadder report and the one `repro/spatial_agent/evals/spbench.py` already
implements. It is reproduced here rather than imported because this file runs
*inside the task container*, which ships no numpy-free copy of that tree and no
SpatialClaw install -- only python3, numpy, pillow and requests (see
environment/Dockerfile). Reproduced means byte-for-byte equivalent behaviour:

    thresholds = linspace(0.50, 0.95, 10)          # 0.50, 0.55, ..., 0.95
    score      = mean(|pred - gold| / gold <= 1 - threshold)

so a prediction within 5% of gold scores 1.0, within 50% scores 0.1, and
anything off by more than half scores 0. Keeping the same thresholds is what
makes an SPBench number here comparable with the paper's Table 1 column; a
"close enough" rule of our own would not be.

Prediction extraction mirrors `_fuzzy_matching_num` in that same file: spelled
numbers first ("three" -> 3), then the first decimal in the text. Anything with
no number in it, and any gold of 0 (undefined relative error), scores 0.

Env (same names/defaults as score_erqa.py):
    ANSWER_FILE    default /app/answer.txt
    EXPECTED_FILE  default /tests/expected_answer.txt
"""

from __future__ import annotations

import os
import re
import sys

#: Same ladder as VSI-Bench: start 0.5, end 0.95, step 0.05 (10 thresholds).
MRA_START = 0.5
MRA_END = 0.95
MRA_INTERVAL = 0.05

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def scrub(text: str) -> str:
    """Drop the invisible characters that survive .strip().

    Same treatment score_erqa.py gives them, and for the same reason: a file
    holding only a BOM plus "3" is unambiguously 3, and used to score 0.
    """
    return text.replace("﻿", "").replace("​", "").replace("⁠", "").replace("\xa0", " ")


#: Task-side marker (written by the adapter, same file the container runtime
#: reads). "number_last" selects the last number in the answer instead of the
#: first -- SPAR-Bench's official extractor does that on its multi-view NA
#: tasks, and a number produced here has to be read the way the published
#: column read theirs. Anything else, including an unreadable marker, means
#: "first".
ANSWER_FORMAT_PATH = "/app/.answer_format"


def answer_format(path: str | None = None) -> str:
    try:
        with open(path or ANSWER_FORMAT_PATH, encoding="utf-8") as handle:
            value = handle.read().strip().lower()
    except OSError:
        return "number"
    return value if value in ("number", "number_last") else "number"


def extract_number(prediction: str, take_last: bool = False) -> float | None:
    """First (or last) number in `prediction`, spelled or written, or None.

    Mirrors repro's `_fuzzy_matching_num`: spelled digits win over digits,
    because "two meters (2m)" and "two" must score the same.
    """
    text = scrub(prediction).strip().lower()
    if not text:
        return None
    for word, digit in NUMBER_WORDS.items():
        if re.search(r"\b" + word + r"\b", text):
            return float(digit)
    matches = re.findall(r"(\d+(?:\.\d+)?)", text)
    if not matches:
        return None
    try:
        return float(matches[-1] if take_last else matches[0])
    except ValueError:
        return None


def mean_relative_accuracy(pred: float, gold: float) -> float:
    """Fraction of the threshold ladder this prediction clears.

    A gold of exactly 0 makes relative error undefined, so the error is taken
    ABSOLUTE there -- the rule SPAR-Bench's own metric uses (an exact 0 then
    scores 1.0, as it should). SPBench never has a zero gold (its smallest is
    0.3, checked against both parquets), so this is behaviour-neutral for it.
    """
    error = abs(pred - gold) if gold == 0 else abs(pred - gold) / abs(gold)
    num_points = int((MRA_END - MRA_START) / MRA_INTERVAL + 2)
    step = (MRA_END - MRA_START) / (num_points - 1)
    thresholds = [MRA_START + step * i for i in range(num_points)]
    hits = sum(1 for threshold in thresholds if error <= 1 - threshold)
    return hits / len(thresholds)


def score(
    prediction: str, expected: str, marker_path: str | None = None
) -> tuple[float, str]:
    """Return (reward, reason). Never raises -- an unreadable pair scores 0."""
    gold = extract_number(expected)
    if gold is None:
        return 0.0, f"unparseable gold {expected!r}"
    take_last = answer_format(marker_path) == "number_last"
    pred = extract_number(prediction, take_last=take_last)
    if pred is None:
        return 0.0, "no number in answer"
    reward = mean_relative_accuracy(pred, gold)
    # A gold of 0 makes relative error undefined; mean_relative_accuracy()
    # already returns 0 for it, and the reason string must not divide by it
    # either -- an exception here would turn a scored 0 into a harness error.
    detail = (
        f"abs_err={abs(pred - gold):.3f} (gold is 0)"
        if gold == 0
        else f"rel_err={abs(pred - gold) / abs(gold):.3f}"
    )
    return reward, f"pred={pred} gold={gold} {detail}{' [last]' if take_last else ''}"


def main() -> int:
    answer_file = os.environ.get("ANSWER_FILE", "/app/answer.txt")
    expected_file = os.environ.get("EXPECTED_FILE", "/tests/expected_answer.txt")

    try:
        with open(answer_file, encoding="utf-8", errors="replace") as handle:
            prediction = handle.read()
    except OSError as exc:
        print("0.0")
        print(f"reason: cannot read {answer_file}: {exc}", file=sys.stderr)
        return 0
    try:
        with open(expected_file, encoding="utf-8", errors="replace") as handle:
            expected = handle.read()
    except OSError as exc:
        print("0.0")
        print(f"reason: cannot read {expected_file}: {exc}", file=sys.stderr)
        return 0

    reward, reason = score(prediction, expected)
    # stdout is the reward channel; everything else goes to stderr so a stray
    # print can never be parsed as a score.
    print(f"{reward:.6f}")
    print(f"reason: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
