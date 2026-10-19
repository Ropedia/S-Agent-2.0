#!/usr/bin/env bash
# Build the spatial runtime image: stock openhands-sdk 1.33.0 with the
# entrypoint patch and the CPU dependency used by the task-side object API.
#
# Baseline configurations select this image explicitly so delivery mechanism is the only
# runtime difference under study.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Public base image (openhands-sdk 1.33.0 runtime). To build it yourself instead:
#   bash scripts/agent_runtimes/openhands-sdk/1.33.0/build-runtime-image.sh --image <tag>
#   BASE_IMAGE=<tag> bash scripts/build_runtime.sh
BASE="${BASE_IMAGE:-docker.io/jierun/c-oh-sdk-1.33.0:v0.1}"
TAG="${TARGET_IMAGE:-sagent-oh-sdk-1.33.0:v1}"
# The custom agent executes /opt/custom-agent-runtime/oh-sdk/runtime/run_openhands.py
# inside the mounted runtime; the patched entrypoint is written there.
ENTRYPOINT_REL="${ENTRYPOINT_REL:-opt/custom-agent-runtime/oh-sdk/runtime/run_openhands.py}"

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

cp "$SCRIPT_DIR/runtime_patch/run_openhands_patched.py" \
   "$BUILD_DIR/run_openhands.py"

# Declare that Qwen 3.x models carry their reasoning back into history.
#
# WHY THIS IS NEEDED. openhands-sdk decides whether to serialize an assistant
# turn's `reasoning` field by substring-matching the model name against
# SEND_REASONING_CONTENT_MODELS (llm/utils/model_features.py:222-230, matcher at
# :10-21, consumed at :259 -> llm.py:2639 -> the drop point message.py:324-326).
# That list holds only kimi / minimax / deepseek spellings, so no spelling of
# ours matches and the model's own thinking is dropped from every replayed turn.
#
# Measured, same 100 questions, before this fix:
#     assistant history in the final request  38405 chars (reasoning inside
#     content) -> 232 chars (reasoning split out by --reasoning-parser and then
#     discarded); tokens per turn 1631 -> 559; turns 15 -> 17; accuracy 51% -> 34%
#
# WHY A REGISTRY EDIT AND NOT AN ALIAS. The alternative was to point
# `model_canonical_name` at a fake proxy alias whose name embeds a listed
# substring. Rejected: it puts a model called `...kimi-k2.6` in our config for
# no reason a future reader could infer, and it is booby-trapped -- `model_info`
# is fetched with the CANONICAL name (llm.py:2347-2351) and the proxy match is
# EXACT (model_info.py:39-49), so a canonical name that is not a real proxy
# entry silently loses `supports_vision` and every image gets dropped
# (docs/PITFALLS.md 3.4, the same failure we already hit once).
#
# This is a CAPABILITY DECLARATION, not a change to harness behaviour logic --
# the same kind of statement as `supports_vision: true` in our litellm config.
# CHARTER's "do not fork the harness" is about control flow; this adds one
# pattern to a lookup table so the SDK stops mis-describing our model.
#
# Pattern choice, verified against the installed matcher:
#     qwen3.6-35b-a3b, litellm_proxy/qwen3.6-35b-a3b, hosted_vllm/qwen3.5-72b,
#     Qwen3.6-27B  -> True
#     qwen3-next-80b, qwen3-coder, qwen2.5-vl                       -> False
# So it covers current and future Qwen 3.x without catching unrelated Qwen
# lines. The upstream fix is this same one-line addition; when it lands, this
# patch becomes a no-op and can be deleted.
#
# ⚠️ This is HALF the fix. Qwen's chat template renders a prior turn's reasoning
# only when `preserve_thinking` is set (chat_template.jinja:100), so without the
# `chat_template_kwargs` in runtime_patch/run_openhands_patched.py this
# registry edit changes nothing -- measured +0 tokens on its own. Both or
# neither.
FEATURES_REL="opt/custom-agent-runtime/oh-sdk/python-home/glibc/lib/python3.12/site-packages/openhands/sdk/llm/utils/model_features.py"

cat > "$BUILD_DIR/Dockerfile" <<EOF
FROM ${BASE}
COPY run_openhands.py /${ENTRYPOINT_REL}
# Insert "qwen3." into SEND_REASONING_CONTENT_MODELS, then FAIL THE BUILD if it
# did not take -- a silently-unpatched image would look identical and score
# worse, which is the failure shape this project keeps hitting.
RUN sed -i 's|^SEND_REASONING_CONTENT_MODELS: list\[str\] = \[|SEND_REASONING_CONTENT_MODELS: list[str] = [\n    "qwen3.",  # agent: Qwen 3.5/3.6 emit reasoning we must replay|' \
      /${FEATURES_REL} \
 && grep -q '"qwen3\.",' /${FEATURES_REL} \
 && /opt/custom-agent-runtime/oh-sdk/bin/python -c "\
import sys; sys.path.insert(0, '/${FEATURES_REL}'.rsplit('/', 6)[0]); \
from openhands.sdk.llm.utils.model_features import get_features; \
    assert get_features('litellm_proxy/qwen3.6-35b-a3b').send_reasoning_content, 'patch did not take'; \
    assert not get_features('qwen3-next-80b').send_reasoning_content, 'pattern too broad'; \
    print('[build] send_reasoning_content patch verified')"
RUN /opt/custom-agent-runtime/oh-sdk/bin/python -m pip install --break-system-packages --no-cache-dir \
      "numpy==2.2.6"
EOF

echo "[build_runtime] ${BASE} -> ${TAG}"
docker build -t "$TAG" "$BUILD_DIR"
docker run --rm --entrypoint /opt/custom-agent-runtime/oh-sdk/bin/python "$TAG" \
  -c 'import numpy; print(f"[build] numpy {numpy.__version__} verified")'
echo "[build_runtime] built $TAG"
