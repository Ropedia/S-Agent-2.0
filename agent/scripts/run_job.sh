#!/usr/bin/env bash
# Generate a task set for one benchmark and launch the job. Normally invoked
# through scripts/run.sh, which fills in the profile.
#
#   scripts/run_job.sh <disk|disk_only> [n_tasks|all] [n_concurrent] [--keep-artifacts]
#
#   disk       frames on disk, the agent opens them itself and has the
#              perception toolkit (the S-Agent 2.0 configuration)
#   disk_only  frames on disk and nothing else: no perception.py, no gateway
#              mount (the no-tools control)
#
#   $DATASET   erqa (default), mmsi, mindcube, spbench, spar, mmsivideo,
#              viewspatial or revsi.
#   all        the items of the reported runs when configs/items/<dataset>.json
#              exists, else the whole benchmark; ITEMS=full forces the whole
#              benchmark and ITEMS=<file> another list.
#   --keep-artifacts  exempt this job's /perception trial dirs from
#              cleanup_perception.py sweeps.
#
# Other knobs (environment): PROMPT_DIR, SKILLS_SOURCE_DIR, WORKSPACE_DIR
# (profile files handed to the generator), ITEMS, GEN_WORKERS, JOBS_DIR, JOB_SUFFIX,
# ALLOW_RESUME, LLM_MODEL_NAME, LITELLM_PORT, LLM_TEMPERATURE,
# PERCEPTION_PORT, PERCEPTION_URL, RUNTIME_IMAGE or RUNTIME_DIR,
# AGENT_TIMEOUT_MULTIPLIER, PRESERVE_THINKING, NUDGE_ANSWER_FILE,
# SALVAGE_ANSWER_FILE.
set -euo pipefail

KEEP_ARTIFACTS=0
positional=()
for arg in "$@"; do
  if [ "$arg" = "--keep-artifacts" ]; then
    KEEP_ARTIFACTS=1
  else
    positional+=("$arg")
  fi
done
set -- "${positional[@]+"${positional[@]}"}"

MODE="${1:?usage: run_job.sh <disk|disk_only> [n_tasks|all] [n_concurrent] [--keep-artifacts]}"
case "$MODE" in
  disk | disk_only) ;;
  *) echo "unknown mode: $MODE (expected disk or disk_only)" >&2; exit 2 ;;
esac
N_TASKS="${2:-all}"
LIMIT_ARGS=()
if [ "$N_TASKS" != all ]; then LIMIT_ARGS=(--limit "$N_TASKS"); fi
N_CONC="${3:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET="${DATASET:-erqa}"

# "all" runs the item list of the reported runs when configs/items/<dataset>.json
# exists (ReVSI, SPAR-Bench, MMSI-Video-Bench, ViewSpatial-Bench), and the whole
# benchmark otherwise. ITEMS=<file> runs another list; ITEMS=full runs the
# whole benchmark regardless. A numeric n_tasks samples n items from the whole
# benchmark instead.
ITEMS="${ITEMS:-}"
if [ -z "$ITEMS" ] && [ "$N_TASKS" = all ] && [ -f "$REPO_ROOT/configs/items/$DATASET.json" ]; then
  ITEMS="$REPO_ROOT/configs/items/$DATASET.json"
fi
if [ "$ITEMS" = full ]; then ITEMS=""; fi
if [ -n "$ITEMS" ] && [ ! -f "$ITEMS" ]; then
  echo "ITEMS does not exist: $ITEMS" >&2
  exit 2
fi

PROMPT_DIR="${PROMPT_DIR:-}"
SKILLS_SOURCE_DIR="${SKILLS_SOURCE_DIR:-}"
WORKSPACE_DIR="${WORKSPACE_DIR:-}"
GEN_WORKERS="${GEN_WORKERS:-}"

# The repository's own .venv (sagent installed editable from src/), so edits to
# the source take effect without a reinstall.
SAGENT_BIN="${SAGENT_BIN:-$REPO_ROOT/.venv/bin/sagent}"
if [ ! -x "$SAGENT_BIN" ]; then
  echo "FATAL: $SAGENT_BIN not found" >&2
  echo "  This repository uses its own .venv (sagent installed editable from src/). Create it with:" >&2
  echo "    cd $REPO_ROOT && uv venv --python 3.12 && uv pip install -e \".[spatial]\"" >&2
  echo "  See docs/SETUP.md. To use another sagent install, set SAGENT_BIN explicitly." >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "FATAL: $PYTHON_BIN not found" >&2
  echo "  Set PYTHON_BIN, or create the repository .venv with the command above." >&2
  exit 2
fi
CLEANUP_PY="${CLEANUP_PY:-$PYTHON_BIN}"

# Silent failures cost hours; assert the services first.
bash "$SCRIPT_DIR/preflight.sh"

case "$DATASET" in
  erqa)        GENERATOR="$REPO_ROOT/adapters/spatial/run_adapter.py" ;;
  mmsi)        GENERATOR="$REPO_ROOT/adapters/spatial/load_mmsi.py" ;;
  mindcube)    GENERATOR="$REPO_ROOT/adapters/spatial/load_mindcube.py" ;;
  spbench)     GENERATOR="$REPO_ROOT/adapters/spatial/load_spbench.py" ;;
  spar)        GENERATOR="$REPO_ROOT/adapters/spatial/load_spar.py" ;;
  mmsivideo)   GENERATOR="$REPO_ROOT/adapters/spatial/load_mmsivideo.py" ;;
  viewspatial) GENERATOR="$REPO_ROOT/adapters/spatial/load_viewspatial.py" ;;
  revsi)       GENERATOR="$REPO_ROOT/adapters/spatial/load_revsi.py" ;;
  *) echo "unknown DATASET: $DATASET (expected erqa, mmsi, mindcube, spbench, spar, mmsivideo, viewspatial or revsi)" >&2; exit 2 ;;
esac

TASK_DIR="$REPO_ROOT/tasks/${DATASET}_${MODE}_${N_TASKS}"
# $JOB_SUFFIX names a re-run: the runner refuses to reuse a job dir whose config
# differs, and resumes one whose config matches.
JOB_NAME="${DATASET}_${MODE}_${N_TASKS}${JOB_SUFFIX:-}"
JOBS_DIR="${JOBS_DIR:-/tmp/sagent-jobs}"

# Trials are keyed on the task PATH, not its content: a job dir that already
# holds finished trials would be resumed and hand back the old results even if
# the tasks on disk changed. Refuse unless the caller says so.
if [ -d "$JOBS_DIR/$JOB_NAME" ]; then
  _done=$(find "$JOBS_DIR/$JOB_NAME" -maxdepth 2 -name result.json 2>/dev/null | wc -l)
  if [ "$_done" -gt 0 ] && [ "${ALLOW_RESUME:-0}" != 1 ]; then
    echo "FATAL: $JOBS_DIR/$JOB_NAME already holds $_done finished trial(s)." >&2
    echo "  Trials are keyed on task PATH, not content, so the run would resume" >&2
    echo "  those and return the OLD results even though the tasks changed." >&2
    echo "  Set JOB_SUFFIX to name this run, e.g. JOB_SUFFIX=_v2, or ALLOW_RESUME=1" >&2
    echo "  to resume after a crash with unchanged tasks." >&2
    exit 2
  fi
fi

# Reclaim /perception dirs of trials that are provably finished (result.json
# written). Jobs launched with --keep-artifacts are exempt from every sweep.
"$CLEANUP_PY" "$SCRIPT_DIR/cleanup_perception.py" sweep --jobs-dir "$JOBS_DIR"
if [ "$KEEP_ARTIFACTS" = 1 ]; then
  "$CLEANUP_PY" "$SCRIPT_DIR/cleanup_perception.py" mark-keep \
    --job-dir "$JOBS_DIR/$JOB_NAME"
fi

# Record which version of the repository this job ran (the working tree is what
# gets baked into the task image, so a dirty tree is recorded too).
mkdir -p "$JOBS_DIR/$JOB_NAME"
{
  echo "job_name=$JOB_NAME"
  echo "started_at=$(date -Is)"
  if git -C "$REPO_ROOT" rev-parse HEAD >/dev/null 2>&1; then
    echo "git_head=$(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
    _dirty=$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | wc -l)
    echo "git_dirty_files=$_dirty"
    if [ "$_dirty" -gt 0 ]; then
      # awk, not head: head closes the pipe early and `set -o pipefail` would
      # abort the whole script on git's SIGPIPE.
      git -C "$REPO_ROOT" status --porcelain 2>/dev/null \
        | sed 's/^/dirty: /' | awk 'NR<=50'
      git -C "$REPO_ROOT" diff --binary > "$JOBS_DIR/$JOB_NAME/provenance.diff" 2>/dev/null || true
    fi
  else
    echo "git_head=unavailable"
  fi
} > "$JOBS_DIR/$JOB_NAME/provenance.txt"

echo "[run_job] generating $N_TASKS $DATASET tasks in mode=$MODE"
GEN_EXTRA=()
if [ -n "$ITEMS" ]; then
  if [ "$DATASET" = "erqa" ]; then
    echo "ITEMS is not wired for DATASET=erqa" >&2
    exit 2
  fi
  GEN_EXTRA+=(--manifest "$ITEMS")
  echo "[run_job] ITEMS=$ITEMS"
fi
if [ -n "$SKILLS_SOURCE_DIR" ]; then
  if [ "$DATASET" = "erqa" ]; then
    echo "SKILLS_SOURCE_DIR is not wired for DATASET=erqa" >&2
    exit 2
  fi
  GEN_EXTRA+=(--skills-dir "$SKILLS_SOURCE_DIR")
  echo "[run_job] SKILLS_SOURCE_DIR=$SKILLS_SOURCE_DIR"
fi
if [ -n "$WORKSPACE_DIR" ]; then
  if [ "$DATASET" = "erqa" ]; then
    echo "WORKSPACE_DIR is not wired for DATASET=erqa (run_adapter.py has no such flag)" >&2
    exit 2
  fi
  if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "WORKSPACE_DIR does not exist: $WORKSPACE_DIR" >&2
    exit 2
  fi
  GEN_EXTRA+=(--workspace-dir "$WORKSPACE_DIR")
  echo "[run_job] WORKSPACE_DIR=$WORKSPACE_DIR"
fi
if [ -n "$PROMPT_DIR" ]; then
  GEN_EXTRA+=(--prompt-dir "$PROMPT_DIR")
  echo "[run_job] PROMPT_DIR=$PROMPT_DIR"
fi
if [ -n "$GEN_WORKERS" ]; then
  if [ "$DATASET" != "mmsivideo" ] && [ "$DATASET" != "revsi" ]; then
    echo "GEN_WORKERS is only wired for DATASET=mmsivideo|revsi" >&2
    exit 2
  fi
  GEN_EXTRA+=(--workers "$GEN_WORKERS")
  echo "[run_job] GEN_WORKERS=$GEN_WORKERS"
fi
# Start from an empty task dir, but never wipe one a running job is reading.
if pgrep -f -- "sagent run .*-p $TASK_DIR( |\$)" > /dev/null 2>&1; then
  echo "[run_job] refusing to regenerate $TASK_DIR: a running sagent job uses it" >&2
  exit 2
fi
rm -rf "$TASK_DIR"
"$PYTHON_BIN" "$GENERATOR" \
  --output-dir "$TASK_DIR" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
  --vision-mode "$MODE" --overwrite ${GEN_EXTRA[@]+"${GEN_EXTRA[@]}"}

# The address containers use to reach LiteLLM on the host: the default-route
# source address, which does not change when docker bridges come and go.
HOST_IP="$(ip route get 1.1.1.1 2> /dev/null | awk '{for (i=1;i<NF;i++) if ($i=="src") print $(i+1)}' | head -1)"
if [ -z "$HOST_IP" ]; then
  HOST_IP="$(hostname -I | awk '{print $1}')"
  echo "[run_job] WARNING: no default route; falling back to hostname -I ($HOST_IP)" >&2
fi
LLM_MODEL_NAME="${LLM_MODEL_NAME:-qwen3.8-27b-fp8}"
export LLM_MODEL_NAME
LITELLM_PORT="${LITELLM_PORT:-4011}"
# No /v1 suffix: litellm_proxy/ adds it.
LLM_BASE_URL="http://${HOST_IP}:${LITELLM_PORT}"

# The agent runtime: the openhands-sdk image with the patched entrypoint
# (scripts/build_runtime.sh), mounted read-only into every trial container,
# either from the image or from a directory tree ($RUNTIME_DIR).
CONTAINER_RUNTIME_ROOT=/opt/custom-agent-runtime/oh-sdk
RUNTIME_IMAGE="${RUNTIME_IMAGE:-sagent-oh-sdk-1.33.0:v1}"
{
  echo "runtime_image=$RUNTIME_IMAGE"
  echo "runtime_mount=${RUNTIME_DIR:+bind:$RUNTIME_DIR}${RUNTIME_DIR:-image}"
  echo "task_network_mode=${SAGENT_TASK_NETWORK_MODE:-default}"
  echo "runtime_image_id=$(docker image inspect "$RUNTIME_IMAGE" --format '{{.Id}}' 2>/dev/null || echo unavailable)"
  if [ -n "$PROMPT_DIR" ] && [ -d "$PROMPT_DIR" ]; then
    echo "prompt_dir=$PROMPT_DIR"
    echo "prompt_dir_sha256=$(find "$PROMPT_DIR" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
  fi
  if [ -n "$SKILLS_SOURCE_DIR" ] && [ -d "$SKILLS_SOURCE_DIR" ]; then
    echo "skills_source_dir=$SKILLS_SOURCE_DIR"
    echo "skills_source_dir_sha256=$(find "$SKILLS_SOURCE_DIR" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
  fi
  if [ -n "$WORKSPACE_DIR" ] && [ -d "$WORKSPACE_DIR" ]; then
    echo "workspace_dir=$WORKSPACE_DIR"
    echo "workspace_dir_sha256=$(find "$WORKSPACE_DIR" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
  fi
  if [ -n "$ITEMS" ]; then
    echo "items=$ITEMS"
    echo "items_sha256=$(sha256sum "$ITEMS" | awk '{print $1}')"
  fi
  echo "task_dir=$TASK_DIR"
  echo "task_dir_sha256=$(find "$TASK_DIR" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
} >> "$JOBS_DIR/$JOB_NAME/provenance.txt"

# Mounts: the runtime, and (with the toolkit) the per-trial /perception dir the
# gateway reads and writes; ${HOST_ARTIFACTS_PATH} is interpolated per trial.
MOUNTS_JSON="$(RUNTIME_SOURCE_IMAGE="$RUNTIME_IMAGE" \
  RUNTIME_DIR="${RUNTIME_DIR:-}" \
  CONTAINER_RUNTIME_ROOT="$CONTAINER_RUNTIME_ROOT" \
  RUN_MODE="$MODE" "$PYTHON_BIN" -c 'import json, os
runtime_dir = os.environ.get("RUNTIME_DIR") or ""
if runtime_dir:
    mounts = [{
      "type": "bind",
      "source": runtime_dir,
      "target": os.environ["CONTAINER_RUNTIME_ROOT"],
      "read_only": True,
    }]
else:
    mounts = [{
      "type": "image",
      "source": os.environ["RUNTIME_SOURCE_IMAGE"],
      "target": os.environ["CONTAINER_RUNTIME_ROOT"],
      "read_only": True,
      "image": {"subpath": os.environ["CONTAINER_RUNTIME_ROOT"].lstrip("/")},
    }]
if os.environ["RUN_MODE"] != "disk_only":
    mounts.append({
        "type": "bind",
        "source": "${HOST_ARTIFACTS_PATH}/../perception",
        "target": "/perception",
    })
print(json.dumps(mounts))')"

REASONING_AE=(--ae "PRESERVE_THINKING=${PRESERVE_THINKING:-1}")
# One recovery turn when the agent stops without an answer file, and the
# repair of a leaked submission; both record themselves in final_metrics.
NUDGE_AE=(--ae "NUDGE_ANSWER_FILE=${NUDGE_ANSWER_FILE:-/app/answer.txt}")
SALVAGE_AE=(--ae "SALVAGE_ANSWER_FILE=${SALVAGE_ANSWER_FILE:-/app/answer.txt}")

# Only the toolkit mode is told where the gateway is.
PERCEPTION_AE=()
if [ "$MODE" != disk_only ]; then
  PERCEPTION_AE=(--ae "PERCEPTION_HOST_JOBS_DIR=$JOBS_DIR" --ae "PERCEPTION_PORT=${PERCEPTION_PORT:-8090}")
  if [ -n "${PERCEPTION_URL:-}" ]; then
    case "$PERCEPTION_URL" in
      *://127.*|*://localhost*|*://0.0.0.0*)
        if [ "${SAGENT_TASK_NETWORK_MODE:-}" != host ]; then
          echo "PERCEPTION_URL=$PERCEPTION_URL is a loopback address but SAGENT_TASK_NETWORK_MODE is not 'host';" >&2
          echo "  trial containers on a bridge network cannot reach it. Unset PERCEPTION_URL (perception.py probes" >&2
          echo "  host.docker.internal -> default-route gateway -> 127.0.0.1 by itself) or set the host's IP." >&2
          exit 2
        fi
        ;;
    esac
    PERCEPTION_AE+=(--ae "PERCEPTION_URL=$PERCEPTION_URL")
  fi
fi

AGENT_TIMEOUT_ARGS=()
if [ -n "${AGENT_TIMEOUT_MULTIPLIER:-}" ]; then
  AGENT_TIMEOUT_ARGS=(--agent-timeout-multiplier "$AGENT_TIMEOUT_MULTIPLIER")
  echo "[run_job] AGENT_TIMEOUT_MULTIPLIER=$AGENT_TIMEOUT_MULTIPLIER (task.toml base 7200s)"
fi

echo "[run_job] job=$JOB_NAME image=$RUNTIME_IMAGE base_url=$LLM_BASE_URL conc=$N_CONC preserve_thinking=${PRESERVE_THINKING:-1}"

mkdir -p "$REPO_ROOT/logs"
setsid nohup "$SAGENT_BIN" run \
  -o "$JOBS_DIR" \
  --job-name "$JOB_NAME" \
  -p "$TASK_DIR" \
  --agent-import-path sagent.agents.custom.openhands_sdk:CustomOpenHandsSDK \
  --mounts-json "$MOUNTS_JSON" \
  --model "litellm_proxy/${LLM_MODEL_NAME}" \
  --n-concurrent "$N_CONC" \
  --ak "version=1.33.0" \
  --ak "temperature=${LLM_TEMPERATURE:-1.0}" \
  --ak "max_iterations=80" \
  --ae "LLM_BASE_URL=$LLM_BASE_URL" \
  --ae "LLM_API_KEY=dummy-key-cf" \
  --ae "CUSTOM_AGENT_RUNTIME_ROOT=$CONTAINER_RUNTIME_ROOT" \
  --ae "CUSTOM_AGENT_PYTHON=$CONTAINER_RUNTIME_ROOT/bin/python" \
  ${AGENT_TIMEOUT_ARGS[@]+"${AGENT_TIMEOUT_ARGS[@]}"} \
  ${PERCEPTION_AE[@]+"${PERCEPTION_AE[@]}"} \
  ${REASONING_AE[@]+"${REASONING_AE[@]}"} \
  ${NUDGE_AE[@]+"${NUDGE_AE[@]}"} \
  ${SALVAGE_AE[@]+"${SALVAGE_AE[@]}"} \
  > "$REPO_ROOT/logs/run_${JOB_NAME}.log" 2>&1 < /dev/null &

echo "[run_job] launched; log: $REPO_ROOT/logs/run_${JOB_NAME}.log"
