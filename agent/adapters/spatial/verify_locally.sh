#!/bin/bash
# Exercise a generated task's tests/test.sh outside any container, using the
# ANSWER_FILE / EXPECTED_FILE / REWARD_DIR overrides. Prints the reward each
# scenario produces.
#
# Usage: ./verify_locally.sh /tmp/erqa_gen3/ERQA_1

set -uo pipefail

TASK_DIR="${1:?usage: verify_locally.sh <generated-task-dir>}"
TEST_SH="$TASK_DIR/tests/test.sh"
GOLD="$(cat "$TASK_DIR/tests/expected_answer.txt")"

if [ ! -x "$TEST_SH" ]; then
    echo "ERROR: $TEST_SH not found or not executable"
    exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "task: $TASK_DIR"
echo "gold answer: '$GOLD'"
echo

PASS=0
FAIL=0

# run_case <label> <expected_reward> <mode> [payload]
# mode: write  -> write payload into the answer file
#       missing-> do not create the answer file at all
run_case() {
    local label="$1" want="$2" mode="$3" payload="${4-}"
    local case_dir="$TMP/case_$((PASS + FAIL))"
    mkdir -p "$case_dir/app" "$case_dir/logs/verifier"

    if [ "$mode" = "write" ]; then
        printf '%s' "$payload" > "$case_dir/app/answer.txt"
    fi

    ANSWER_FILE="$case_dir/app/answer.txt" \
    EXPECTED_FILE="$TASK_DIR/tests/expected_answer.txt" \
    REWARD_DIR="$case_dir/logs/verifier" \
        "$TEST_SH" > "$case_dir/stdout.txt" 2>&1
    local rc=$?

    local reward
    reward="$(cat "$case_dir/logs/verifier/reward.txt" 2>/dev/null || echo MISSING)"

    local verdict="ok  "
    if [ "$reward" != "$want" ] || [ "$rc" -ne 0 ]; then
        verdict="FAIL"
        FAIL=$((FAIL + 1))
    else
        PASS=$((PASS + 1))
    fi

    printf '[%s] %-34s reward=%s want=%s exit=%s\n' \
        "$verdict" "$label" "$reward" "$want" "$rc"
    grep -m1 '^reward=' "$case_dir/stdout.txt" | sed 's/^/        /'
}

echo "--- should score 1 ---"
run_case "exact gold"                1 write "$GOLD"
run_case "gold with trailing period" 1 write "$GOLD."
run_case "gold lowercase padded"     1 write " $(printf '%s' "$GOLD" | tr '[:upper:]' '[:lower:]') "
run_case "gold with newline"         1 write "$GOLD
"
run_case "Answer: <gold>"            1 write "Answer: $GOLD"
run_case "essay then Answer: <gold>" 1 write "I considered every option carefully.
The gripper trajectory arcs upward and terminates above the surface,
which rules out the lateral interpretations entirely.
Answer: $GOLD"
run_case "reasoning then bare gold"  1 write "Looking at the scene and the trajectory.
$GOLD"

echo
echo "--- should score 0 ---"
run_case "missing answer file"       0 missing
run_case "empty answer file"         0 write ""
run_case "whitespace only"           0 write "   "
run_case "long essay, no letter"     0 write "Let me think about this step by step.
The gripper starts above the object, so the first reading is plausible.
But looking at the trajectory more carefully, the arc terminates below
the top of the structure, which rules that out. The remaining readings
describe a lateral move and a purely vertical move respectively.
Considering the shape of the curve and where it ends, the most
consistent reading is that the object is placed on the structure.
This is my complete reasoning about the spatial configuration."
run_case "essay quoting all letters" 0 write "Options A, B, C and D all describe plausible outcomes here."

echo
WRONG=$(printf '%s' "$GOLD" | tr 'ABCD' 'BCDA')
run_case "wrong letter ($WRONG)"     0 write "$WRONG"
run_case "Answer: $WRONG (wrong)"    0 write "Answer: $WRONG"

echo
echo "passed=$PASS failed=$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
