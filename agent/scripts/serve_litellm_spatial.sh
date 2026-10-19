#!/usr/bin/env bash
# Start the LiteLLM proxy in front of the local vLLM for the agent.
#
# Why a proxy at all: openhands-sdk gates image delivery on
# LLM.vision_is_active(), which consults litellm.supports_vision(). That
# returns False for every spelling of our self-hosted model, and when the gate
# is closed ImageContent is dropped from the request payload with no error and
# no warning. Declaring model_info.supports_vision on a litellm proxy is the
# supported way to tell the SDK the truth. Measured: 0 -> 1 image_url parts.
#
# Second benefit: trajectory_logger writes every call to
# jobs/<job>/<trial>/agent/litellm-trajectory.jsonl, which is protocol-level
# ground truth independent of whatever a scaffold claims in its own trajectory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PORT="${LITELLM_PORT:-4011}"
WORKERS="${LITELLM_NUM_WORKERS:-4}"
CONFIG="${LITELLM_CONFIG:-$REPO_ROOT/configs/litellm_qwen_spatial.yaml}"

# Sampling. The yaml reads these via os.environ/..., so they MUST be exported or
# litellm receives the literal string and rejects the request -- hence defaults
# here rather than in the yaml, which has no default syntax.
#
# Chosen after a one-variable-per-configuration sweep (see the long comment in the yaml):
# temp 1.0 and top_p 0.95 each pointed up, presence_penalty 1.5 pointed down, so
# it is omitted. This pair is exactly what the model card's SWE-Bench and
# Terminal-Bench evaluation notes specify, so it is upstream's own agentic
# configuration rather than a mix of our choosing.
export SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-1.0}"
export SAMPLE_TOP_P="${SAMPLE_TOP_P:-0.95}"
export SAMPLE_TOP_K="${SAMPLE_TOP_K:-20}"
export SAMPLE_PRESENCE_PENALTY="${SAMPLE_PRESENCE_PENALTY:-0.0}"
LOG_DIR="$REPO_ROOT/logs"
LOG="$LOG_DIR/litellm_spatial.log"
mkdir -p "$LOG_DIR"

# trajectory_logger is referenced by bare module name in the config's
# `callbacks`. LiteLLM 1.89.1 resolves `callbacks:` module names relative to
# the config file's directory, not the process CWD (see get_instance_fn in
# litellm/proxy/types_utils/utils.py), so the callback module must sit beside
# the config.
#
# It used to be a symlink into a sagent checkout. That made the repo unusable
# for anyone without access to the private sagent repo -- and worse, git stores
# a symlink as its literal target path, so a fresh clone got a dangling link to
# somebody else's machine. The file is now committed at configs/ (Apache-2.0,
# provenance in vendor/sagent/README.md).
TL="$REPO_ROOT/configs/trajectory_logger.py"
if [ ! -f "$TL" ]; then
  echo "ERROR: $TL missing (expected a real file, not a symlink)" >&2
  exit 1
fi
if [ -L "$TL" ]; then
  echo "ERROR: $TL is a symlink; it must be a real file so clones work" >&2
  exit 1
fi

if ss -lnt 2>/dev/null | grep -q ":${PORT} "; then
  echo "ERROR: port ${PORT} already in use" >&2
  exit 1
fi

echo "[serve_litellm_spatial] config=$CONFIG port=$PORT workers=$WORKERS"
echo "[serve_litellm_spatial] log -> $LOG"

cd "$REPO_ROOT/configs"
setsid nohup litellm --config "$CONFIG" --port "$PORT" --host 0.0.0.0 \
  --num_workers "$WORKERS" >> "$LOG" 2>&1 < /dev/null &

for i in $(seq 1 90); do
  if curl -sf --connect-timeout 2 -H "Authorization: Bearer dummy-key-cf" \
       "http://127.0.0.1:${PORT}/health/readiness" >/dev/null 2>&1; then
    echo "[serve_litellm_spatial] ready after ${i}s"
    exit 0
  fi
  sleep 1
done

echo "ERROR: proxy did not become ready in 90s; see $LOG" >&2
exit 1
