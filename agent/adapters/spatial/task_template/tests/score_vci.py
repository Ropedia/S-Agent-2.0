#!/usr/bin/env python3
"""Scorer for SPAR-Bench's `view_change_infer` task.

The answer is not a letter or a number but a five-axis camera motion, written
in the format the question itself dictates:

    move_right:1.6,move_down:0,move_backward:0,rotate_up:15,rotate_left:75

Three translations in metres and two rotations in degrees, each axis named by
its DIRECTION -- so "move_left:2.6" and "move_right:-2.6" are the same motion.
Parsing therefore folds each opposing pair into one signed axis before scoring.

Score = Mean Relative Accuracy averaged over the five axes, the same threshold
ladder score_numeric.py uses (0.50..0.95 by 0.05). An axis both sides agree is
zero scores 1.0: relative error is undefined at a target of 0, and the
benchmark's own metric treats an exact zero as exact, so |pred - target| is
used there instead of |pred - target| / target.

⚠️ AXIS NAMES. The rotation axes in the released data are `rotate_up` /
`rotate_down` / `rotate_right` / `rotate_left`. `repro/spatial_agent/evals/
sparbench.py` parses `turn_*` instead, which matches nothing in the data: both
prediction and gold come back 0.0 on those two axes, MRA(0, 0) is 1.0, and
every sample collects a free 2/5 whatever it answered. This file follows the
DATA. `turn_*` is accepted as a synonym only on the prediction side, because a
model that writes "turn_left" plainly means the rotation axis and the transport
should not decide the question.

Prints one float in [0, 1] to stdout and exits 0; tests/test_vci.sh reads
stdout, the same fractional-reward contract score_numeric.py has.

Env (same names/defaults as the other scorers):
    ANSWER_FILE    default /app/answer.txt
    EXPECTED_FILE  default /tests/expected_answer.txt
"""

from __future__ import annotations

import os
import re
import sys

MRA_START = 0.5
MRA_END = 0.95
MRA_INTERVAL = 0.05

#: Canonical signed axes, and the direction word that counts as negative.
#: `turn_*` mirrors `rotate_*` so a model's wording cannot cost it the axis.
AXES = {
    "move_right": ("move_right", "move_left"),
    "move_up": ("move_up", "move_down"),
    "move_forward": ("move_forward", ("move_backward", "move_back")),
    "rotate_up": (("rotate_up", "turn_up"), ("rotate_down", "turn_down")),
    "rotate_right": (("rotate_right", "turn_right"), ("rotate_left", "turn_left")),
}

_PAIR_RE = re.compile(r"([a-z_]+)\s*:\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)


def _names(spec) -> tuple[str, ...]:
    return (spec,) if isinstance(spec, str) else tuple(spec)


def parse_vci(text: str) -> dict[str, float]:
    """Fold "move_left:2.6,rotate_up:10,..." into the five signed axes.

    Missing axes read 0.0, which is what the benchmark's gold does too -- an
    axis the camera did not move along is simply written as 0 or omitted.
    """
    raw: dict[str, float] = {}
    for key, value in _PAIR_RE.findall(text or ""):
        try:
            raw[key.strip().lower()] = float(value)
        except ValueError:
            continue

    folded: dict[str, float] = {}
    for axis, (positive, negative) in AXES.items():
        value = 0.0
        for name in _names(positive):
            value += raw.get(name, 0.0)
        for name in _names(negative):
            value -= raw.get(name, 0.0)
        folded[axis] = value
    return folded


def mean_relative_accuracy(pred: float, target: float) -> float:
    """MRA for one axis. |error| is absolute when the target is exactly 0."""
    error = abs(pred - target) if target == 0 else abs(pred - target) / abs(target)
    num_points = int((MRA_END - MRA_START) / MRA_INTERVAL + 2)
    step = (MRA_END - MRA_START) / (num_points - 1)
    thresholds = [MRA_START + step * i for i in range(num_points)]
    hits = sum(1 for threshold in thresholds if error <= 1 - threshold)
    return hits / len(thresholds)


def score(prediction: str, expected: str) -> tuple[float, str]:
    """Return (reward, reason). Never raises -- an unreadable pair scores 0."""
    gold = parse_vci(expected)
    if not _PAIR_RE.search(expected or ""):
        return 0.0, f"unparseable gold {expected[:60]!r}"
    if not _PAIR_RE.search(prediction or ""):
        return 0.0, "no axis:value pair in answer"
    pred = parse_vci(prediction)
    per_axis = {axis: mean_relative_accuracy(pred[axis], gold[axis]) for axis in AXES}
    reward = sum(per_axis.values()) / len(per_axis)
    detail = " ".join(f"{axis}={per_axis[axis]:.1f}" for axis in AXES)
    return reward, detail


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
    print(f"{reward:.6f}")
    print(f"reason: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
