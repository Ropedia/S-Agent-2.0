#!/bin/bash
# ERQA verifier. Writes 1 or 0 to $REWARD_DIR/reward.txt and always exits 0,
# so a scoring failure is recorded as reward 0 rather than a harness error.
#
# All paths are overridable so this script can be exercised outside a container:
#   ANSWER_FILE    default /app/answer.txt
#   EXPECTED_FILE  default /tests/expected_answer.txt
#   REWARD_DIR     default /logs/verifier
#   SCORER         default <dir of this script>/score_erqa.py

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ANSWER_FILE="${ANSWER_FILE:-/app/answer.txt}"
EXPECTED_FILE="${EXPECTED_FILE:-/tests/expected_answer.txt}"
REWARD_DIR="${REWARD_DIR:-/logs/verifier}"
SCORER="${SCORER:-$SCRIPT_DIR/score_erqa.py}"

echo "=== ERQA VERIFIER ==="
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

ANSWER_FILE="$ANSWER_FILE" EXPECTED_FILE="$EXPECTED_FILE" \
    "$PY" "$SCORER"
SCORE_RC=$?

if [ "$SCORE_RC" -eq 0 ]; then
    echo 1 > "$REWARD_DIR/reward.txt"
    echo "=== REWARD: 1 ==="
else
    echo 0 > "$REWARD_DIR/reward.txt"
    echo "=== REWARD: 0 ==="
fi

exit 0
