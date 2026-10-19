#!/bin/bash
# Oracle solution: write the gold answer letter. Proves the harness plumbing
# (agent phase -> /app/answer.txt -> verifier -> reward.txt) independent of
# any model.

set -euo pipefail

ANSWER_FILE="${ANSWER_FILE:-/app/answer.txt}"

mkdir -p "$(dirname "$ANSWER_FILE")"
printf '%s' '{answer}' > "$ANSWER_FILE"

echo "oracle wrote '{answer}' to $ANSWER_FILE"
