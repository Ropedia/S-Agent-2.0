#!/bin/bash
# Numeric-answer verifier (SPBench NA). Writes a FRACTIONAL reward in [0,1] to
# $REWARD_DIR/reward.txt and always exits 0, so a scoring failure is recorded as
# reward 0 rather than a harness error -- same fail-closed posture as test.sh.
#
# Why this is a separate file from test.sh rather than a flag inside it: test.sh
# maps the scorer's EXIT CODE to 1 or 0, which cannot express Mean Relative
# Accuracy. Here the scorer prints the reward on stdout instead, and the exit
# code only says whether it ran at all.
#
# All paths are overridable so this script can be exercised outside a container:
#   ANSWER_FILE    default /app/answer.txt
#   EXPECTED_FILE  default /tests/expected_answer.txt
#   REWARD_DIR     default /logs/verifier
#   SCORER         default <dir of this script>/score_numeric.py

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ANSWER_FILE="${ANSWER_FILE:-/app/answer.txt}"
EXPECTED_FILE="${EXPECTED_FILE:-/tests/expected_answer.txt}"
REWARD_DIR="${REWARD_DIR:-/logs/verifier}"
SCORER="${SCORER:-$SCRIPT_DIR/score_numeric.py}"

echo "=== SPBENCH NUMERIC VERIFIER (MRA) ==="
echo "answer_file=$ANSWER_FILE"
echo "expected_file=$EXPECTED_FILE"
echo "reward_dir=$REWARD_DIR"

mkdir -p "$REWARD_DIR"

# Fail closed: any early exit below leaves reward 0 on disk.
echo 0 > "$REWARD_DIR/reward.txt"

PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "ERROR: no python interpreter found; cannot score"
    echo "=== REWARD: 0 ==="
    exit 0
fi

if [ ! -f "$SCORER" ]; then
    echo "ERROR: scorer not found at $SCORER"
    echo "=== REWARD: 0 ==="
    exit 0
fi

REWARD=$(ANSWER_FILE="$ANSWER_FILE" EXPECTED_FILE="$EXPECTED_FILE" "$PY" "$SCORER")
SCORE_RC=$?

# Only a well-formed float is accepted. A scorer that crashed after printing
# something, or printed nothing, must not be able to write junk into reward.txt
# -- SAgent parses that file as a number and a bad value poisons the job's mean.
if [ "$SCORE_RC" -eq 0 ] && [[ "$REWARD" =~ ^(0|1)(\.[0-9]+)?$ ]]; then
    echo "$REWARD" > "$REWARD_DIR/reward.txt"
    echo "=== REWARD: $REWARD ==="
else
    echo "ERROR: scorer rc=$SCORE_RC output=${REWARD:-<empty>}"
    echo 0 > "$REWARD_DIR/reward.txt"
    echo "=== REWARD: 0 ==="
fi

exit 0
