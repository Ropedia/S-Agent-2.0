#!/usr/bin/env bash
# Run S-Agent 2.0 on a benchmark.
#
#   scripts/run.sh <dataset> [n_tasks|all] [n_concurrent] [--no-tools] [--keep-artifacts]
#
#   dataset      mmsi | revsi | spar | mindcube | mmsivideo | viewspatial | spbench | erqa
#   n_tasks      "all" (default): the items of the reported runs when
#                configs/items/<dataset>.json exists, else the whole benchmark;
#                a number samples that many items from the whole benchmark
#   --no-tools   the same agent without the perception toolkit (control)
#
# Environment (all optional): LLM_MODEL_NAME, LITELLM_PORT, PERCEPTION_PORT,
# JOBS_DIR, JOB_SUFFIX, ITEMS (full, or another item list), REVSI_FRAME_BUDGET.
# See docs/SETUP.md.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODE=disk
args=()
for a in "$@"; do
  case "$a" in
    --no-tools) MODE=disk_only ;;
    *) args+=("$a") ;;
  esac
done
DATASET="${args[0]:?usage: scripts/run.sh <dataset> [n_tasks|all] [n_concurrent] [--no-tools]}"
N_TASKS="${args[1]:-all}"
N_CONC="${args[2]:-16}"
export DATASET
export PROMPT_DIR="${PROMPT_DIR:-$REPO_ROOT/configs/profile/prompt}"
export SKILLS_SOURCE_DIR="${SKILLS_SOURCE_DIR:-$REPO_ROOT/configs/profile/skills}"
export JOBS_DIR="${JOBS_DIR:-$REPO_ROOT/jobs}"
extra=()
for a in "${args[@]:3}"; do extra+=("$a"); done
exec bash "$SCRIPT_DIR/run_job.sh" "$MODE" "$N_TASKS" "$N_CONC" ${extra[@]+"${extra[@]}"}
