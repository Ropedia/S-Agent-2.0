#!/usr/bin/env bash
# Launch the SpatialClaw perception REST gateway.
#
# The gateway loads no models: it proxies to the already running
# launch_gpu_server process discovered via spatial_agent/logs/gpu_server.json.
#
# `conda activate` does not work in non-interactive shells here, so the
# interpreter is referenced by absolute path.
set -euo pipefail

PYTHON="${PYTHON:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/bin/python}"
GATEWAY_ROOT="${GATEWAY_ROOT:-/scratch/ywxzml3j/wry/action/agent}"
SPATIALCLAW_ROOT="${SPATIALCLAW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/repro}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8090}"

# A stray .pth in the base env prints import noise unless user site is off.
export PYTHONNOUSERSITE=1
export SPATIALCLAW_ROOT
export GATEWAY_ROOT
export GATEWAY_WORK_DIR="${GATEWAY_WORK_DIR:-${GATEWAY_ROOT}/work}"
export GATEWAY_LOG="${GATEWAY_LOG:-${GATEWAY_ROOT}/logs/gateway.jsonl}"
# Per-request work_dir isolation (task 1c): a request may point reads/writes
# at a per-trial directory instead of the global work dir, but only if that
# directory resolves under one of these roots (see resolve_work_base() in
# perception_gateway.py -- this is the gateway's only defense against a
# container directing it to write anywhere on the host).
#
# run_job.sh mounts each trial's /perception from SAgent's own per-trial
# artifacts dir (HOST_ARTIFACTS_PATH/../perception), and SAgent trial dirs
# live under JOBS_DIR=/tmp/sagent-jobs (see run_job.sh), i.e. paths shaped
# /tmp/sagent-jobs/<job>/<trial>/perception -- which is NOT under
# GATEWAY_WORK_DIR, hence the second root below. ${GATEWAY_WORK_DIR}/trials
# is kept as the first root because it is perception_gateway.py's own
# built-in default (used when GATEWAY_TRIAL_ROOTS is unset at all) and other
# tooling may still rely on it; this just makes it explicit and adds the
# SAgent root alongside it, colon-separated per TRIAL_ROOTS' own parsing.
#
# GATEWAY_TRIAL_ROOTS is read once at process start (see the module-level
# TRIAL_ROOTS list in perception_gateway.py) -- changing it here has NO
# effect on a gateway that is already running. The live gateway on :8090
# must not be restarted as a side effect of deploying this script; a human
# operator restarts it (picking up this new root) on their own schedule.
export GATEWAY_TRIAL_ROOTS="${GATEWAY_TRIAL_ROOTS:-${GATEWAY_WORK_DIR}/trials:/tmp/sagent-jobs}"

mkdir -p "${GATEWAY_WORK_DIR}" "$(dirname "${GATEWAY_LOG}")"

if ! [ -x "${PYTHON}" ]; then
  echo "error: interpreter not found: ${PYTHON}" >&2
  exit 1
fi

if ss -ltn 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"; then
  echo "error: port ${PORT} is already in use" >&2
  exit 1
fi

echo "gateway  : http://${HOST}:${PORT}"
echo "upstream : $(grep -o '"http_port": *[0-9]*' \
  "${SPATIALCLAW_ROOT}/spatial_agent/logs/gpu_server.json" | head -1 || echo unknown)"
echo "work dir : ${GATEWAY_WORK_DIR}"
echo "trial roots: ${GATEWAY_TRIAL_ROOTS}"
echo "log      : ${GATEWAY_LOG}"

exec "${PYTHON}" "${GATEWAY_ROOT}/gateway/perception_gateway.py" \
  --host "${HOST}" --port "${PORT}"
