#!/bin/bash

set -euo pipefail

# ==============================================================================
# LiteLLM Proxy Launcher
#
# This script only runs the LiteLLM proxy. The upstream vLLM/model server is
# configured in a local litellm_config.yaml and does not need to be installed here.
# ==============================================================================

# ------------------------------------------------------------------------------
# One-time environment setup
#
# Run these commands once before using the script on a new machine/environment.
# ------------------------------------------------------------------------------
#    conda create -n litellm_1.83.9 python=3.13
#    conda activate litellm_1.83.9
#    pip install "litellm[proxy]==1.83.9"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------------------
# Optional local activation
#
# Uncomment these lines if you want this script to activate the environment for
# you. Leave them commented if you activate conda before running the script.
# ------------------------------------------------------------------------------
# source /path/to/miniconda/etc/profile.d/conda.sh
# conda activate litellm_1.83.9

# ------------------------------------------------------------------------------
# Frequently changed settings
#
# LITELLM_CONFIG points to an explicit config file. If it is not set,
# CONFIG_NAME selects the matching local ${CONFIG_NAME}.yaml file.
# If CONFIG_NAME is unset with LITELLM_CONFIG set, the log prefix is derived
# from the config file basename.
# LITELLM_LOG_FOLDER points to an explicit log directory. If it is not set,
# logs are written to ./litellm_log next to this script.
# LITELLM_STICKY_ROUTING_ALIASES maps a public model name to backend-specific
# aliases. The trajectory callback hashes each task id to one alias so repeated
# turns for the same task keep hitting the same upstream KV cache.
# HOST, LITELLM_PORT, LITELLM_NUM_WORKERS, LITELLM_LOG_LEVEL, and API_KEY can be
# overridden from the shell:
#   LITELLM_CONFIG=/path/to/config.yaml LITELLM_LOG_FOLDER=/path/to/logs LITELLM_PORT=4002 LITELLM_NUM_WORKERS=4 bash serve_litellm.sh
# ------------------------------------------------------------------------------
if [[ -n "${LITELLM_CONFIG:-}" ]]; then
  CONFIG_NAME="${CONFIG_NAME:-$(basename "${LITELLM_CONFIG}" .yaml)}"
else
  CONFIG_NAME="${CONFIG_NAME:-litellm_config}"
  LITELLM_CONFIG="$SCRIPT_DIR/${CONFIG_NAME}.yaml"
fi
LITELLM_LOG_FOLDER="${LITELLM_LOG_FOLDER:-$SCRIPT_DIR/litellm_log}"
mkdir -p "$LITELLM_LOG_FOLDER"

HOST="${HOST:-0.0.0.0}"
LITELLM_PORT="${LITELLM_PORT:-4001}" # LiteLLM proxy port
LITELLM_NUM_WORKERS="${LITELLM_NUM_WORKERS:-4}"
LITELLM_LOG_LEVEL="${LITELLM_LOG_LEVEL:-INFO}"
API_KEY="${API_KEY:-dummy-key-cf}"
if [[ -z "${LITELLM_STICKY_ROUTING_ALIASES:-}" && -f "${LITELLM_CONFIG}" ]]; then
  LITELLM_STICKY_ROUTING_ALIASES="$(python - "${LITELLM_CONFIG}" <<'PY'
import re
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
model_names: list[str] = []
for line in config_path.read_text().splitlines():
    stripped = line.strip()
    if stripped.startswith("#"):
        continue
    match = re.match(r"""-\s*model_name:\s*["']?([^"']+)["']?\s*$""", stripped)
    if match:
        model_names.append(match.group(1).strip())

public_models = {name for name in model_names if "@" not in name}
aliases_by_model: dict[str, list[str]] = {}
for name in model_names:
    model, separator, _alias = name.partition("@")
    if not separator or model not in public_models:
        continue
    aliases_by_model.setdefault(model, []).append(name)

print(
    ";".join(
        f"{model}={','.join(aliases)}"
        for model, aliases in aliases_by_model.items()
        if aliases
    )
)
PY
)"
fi
export LITELLM_STICKY_ROUTING_ALIASES

# ------------------------------------------------------------------------------
# Rarely changed safety requirements
# ------------------------------------------------------------------------------
MIN_LITELLM_VERSION="1.83.9"

if [[ ! -f "${LITELLM_CONFIG}" ]]; then
  echo "ERROR: LiteLLM config not found: ${LITELLM_CONFIG}"
  echo "Create a local config from the tracked example, then fill in your endpoint and credentials:"
  echo "  cp ${SCRIPT_DIR}/litellm_config.example.yaml ${LITELLM_CONFIG}"
  exit 1
fi

# ------------------------------------------------------------------------------
# Stable helper functions
#
# These checks keep failures explicit before the proxy starts. They usually do not
# need edits unless the environment requirements or port diagnostics change.
# ------------------------------------------------------------------------------
check_litellm_version() {
  local required="$1"
  local installed
  local status

  if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python is not available. Activate the LiteLLM conda environment first:"
    echo "  conda activate litellm_1.83.9"
    exit 1
  fi

  if ! command -v litellm >/dev/null 2>&1; then
    echo "ERROR: litellm CLI is not available in the current environment."
    echo "Install the required proxy package:"
    echo "  pip install \"litellm[proxy]>=${required}\""
    exit 1
  fi

  # Use Python's installed package metadata so the check follows the active
  # conda environment instead of parsing CLI output.
  set +e
  installed="$(python - "${required}" <<'PY'
import importlib.metadata
import re
import sys

required_version = sys.argv[1]


def parse_numeric_version(value: str) -> list[int]:
    parts = []
    for chunk in re.split(r"[-+]", value, maxsplit=1)[0].split("."):
        match = re.match(r"(\d+)", chunk)
        if match is None:
            break
        parts.append(int(match.group(1)))
    return parts


try:
    installed_version = importlib.metadata.version("litellm")
except importlib.metadata.PackageNotFoundError:
    print("missing")
    sys.exit(2)

installed_parts = parse_numeric_version(installed_version)
required_parts = parse_numeric_version(required_version)
width = max(len(installed_parts), len(required_parts))
installed_tuple = tuple(installed_parts + [0] * (width - len(installed_parts)))
required_tuple = tuple(required_parts + [0] * (width - len(required_parts)))

print(installed_version)
if installed_tuple < required_tuple:
    sys.exit(3)
PY
)"
  status=$?
  set -e

  case "${status}" in
    0)
      echo "LiteLLM version check passed: ${installed} >= ${required}"
      ;;
    2)
      echo "ERROR: litellm is not installed in the current Python environment."
      echo "Activate the LiteLLM conda environment, then install:"
      echo "  conda activate litellm_1.83.9"
      echo "  pip install \"litellm[proxy]>=${required}\""
      exit 1
      ;;
    3)
      echo "ERROR: litellm ${installed} is too old. Required version: >= ${required}."
      echo "Upgrade the current environment:"
      echo "  pip install --upgrade \"litellm[proxy]>=${required}\""
      exit 1
      ;;
    *)
      echo "ERROR: Failed to check litellm version."
      [[ -n "${installed}" ]] && echo "${installed}"
      exit 1
      ;;
  esac
}

show_port_usage() {
  local port="$1"

  if command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :${port}" 2>/dev/null || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true
  else
    echo "Neither ss nor lsof is available, so process details cannot be shown."
  fi
}

is_port_in_use() {
  local port="$1"

  if command -v ss >/dev/null 2>&1; then
    [[ -n "$(ss -H -ltn "sport = :${port}" 2>/dev/null)" ]]
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
  else
    return 1
  fi
}

check_port_available() {
  local port="$1"
  local suggested_port="<another-port>"

  if [[ "${port}" =~ ^[0-9]+$ ]]; then
    suggested_port=$((port + 1))
  fi

  if ! is_port_in_use "${port}"; then
    return 0
  fi

  echo "ERROR: Port ${port} is already in use on this node."
  echo ""
  echo "Current listener information:"
  show_port_usage "${port}"
  echo ""
  echo "Suggested fixes:"
  echo "  1. Use another LiteLLM port:"
  echo "     LITELLM_PORT=${suggested_port} $0"
  echo "  2. If you are sure no one else on this node is using the service, stop the existing listener:"
  if command -v lsof >/dev/null 2>&1; then
    echo "     kill \$(lsof -t -iTCP:${port} -sTCP:LISTEN)"
  else
    echo "     kill <PID from the listener information above>"
  fi
  echo ""
  exit 1
}

cleanup() {
  echo ""
  echo "Stopping services..."
  [[ -n "${LITELLM_PID:-}" ]] && kill "${LITELLM_PID}" 2>/dev/null
  wait 2>/dev/null
  echo "Stopped."
}
trap cleanup INT TERM

# ------------------------------------------------------------------------------
# Core startup flow
# ------------------------------------------------------------------------------
check_litellm_version "${MIN_LITELLM_VERSION}"
check_port_available "${LITELLM_PORT}"

echo "=========================================="
echo "Starting LiteLLM Proxy"
echo "=========================================="
echo ""
echo "Endpoints:"
echo "  - LiteLLM (Anthropic format): <http://${HOST}:${LITELLM_PORT}/v1/messages>"
echo "  - LiteLLM (OpenAI format):    <http://${HOST}:${LITELLM_PORT}/v1/chat/completions>"
echo ""
echo "API key: configured"
echo "Config: ${LITELLM_CONFIG}"
echo "Log file: ${LITELLM_LOG_FOLDER}/${CONFIG_NAME}_<timestamp>.log"
echo "Workers: ${LITELLM_NUM_WORKERS}"
echo "LiteLLM log level: ${LITELLM_LOG_LEVEL}"
echo "Callback module path: ${SCRIPT_DIR}/trajectory_logger.py"
if [[ -n "${LITELLM_STICKY_ROUTING_ALIASES}" ]]; then
  echo "Sticky task routing: enabled"
  echo "Sticky task routing aliases: ${LITELLM_STICKY_ROUTING_ALIASES}"
fi
echo "=========================================="
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LITELLM_LOG_FOLDER}/${CONFIG_NAME}_${TIMESTAMP}.log"

echo "[1/1] Starting LiteLLM proxy (Anthropic + OpenAI format)..."
LITELLM_LOG="${LITELLM_LOG_LEVEL}" litellm \
    --config "${LITELLM_CONFIG}" \
    --port "${LITELLM_PORT}" \
    --host "${HOST}" \
    --num_workers "${LITELLM_NUM_WORKERS}" 2>&1 | tee "${LOG_FILE}" &

LITELLM_PID=$!

echo ""
echo "LiteLLM started!"
echo "  - LiteLLM PID: ${LITELLM_PID}"
echo ""

wait
