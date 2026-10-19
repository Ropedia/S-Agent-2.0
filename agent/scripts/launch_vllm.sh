#!/usr/bin/env bash
# Serve the backbone with vLLM for the agent.
#
# Forked from repro/scripts/launch_vllm.sh (SpatialClaw's launcher, left untouched
# so that experiment stays reproducible). All the CUDA/FlashInfer workarounds below
# are inherited verbatim from it -- they are load-bearing, see the comments.
#
# Deltas vs the SpatialClaw launcher, and why:
#
#   --max-model-len 262144   (was 65536)
#       Model's native max_position_embeddings. At 65536, 6/100 of the measured
#       Opus ERQA trajectories already exceeded the window (p95 68k, max 80k), and
#       the vision configurations add image tokens on top. KV cache is 4.57M tokens total and
#       does not shrink with this flag; it only caps how much ONE request may take,
#       so the cost is max concurrency at full length: 69.7x -> 17.4x. Still well
#       above --max-num-seqs 40.
#
#   --max-num-seqs 40        (was unset -> vLLM default 1024)
#       SAgent runs one in-flight request per trial container, so this is a ceiling,
#       not a target. Measured throughput was still scaling linearly at concurrency
#       48 (5,785 tok/s aggregate), so 40 is not the bottleneck. SAgent-side -n is
#       the real knob; start at 8 and watch the perception gateway, which was the
#       first thing to degrade under load in earlier runs.
#
#   --enable-auto-tool-choice --tool-call-parser qwen3_xml   (new)
#       Without these, vLLM 400s on any tool_choice:auto request:
#         '"auto" tool choice requires --enable-auto-tool-choice and
#          --tool-call-parser to be set'
#       OpenHands defaults to native function calling, so every trial would die.
#       Parser choice: in vLLM 0.27.1 the names qwen3_coder / qwen3_xml / mimo all
#       resolve to the SAME class (Qwen3EngineToolParser, an 8-line subclass of
#       Qwen3ParserToolAdapter), so they are behaviourally identical here. We use
#       qwen3_xml because that is what vLLM's own tool_calling.md points at for
#       Qwen3 models; the model card still says qwen3_coder and the two are out of
#       sync upstream. Naming it qwen3_xml means a future vLLM bump (where they may
#       diverge) lands on the documented one.
#
#   --limit-mm-per-prompt image:160  (kept from the SpatialClaw launcher)
#       This is a cap, not a preallocation, so it is free. It matters because the
#       on-disk and rendered vision configurations let the agent pull images repeatedly
#       (view / render), so per-prompt image count accumulates. Hitting the default
#       cap produces 400 "At most N image(s) may be provided in one prompt" plus a
#       client retry storm -- see the original comment about video benchmarks
#       scoring 0 samples for exactly this reason.
#
# prefix caching: OPT-IN via PREFIX_CACHING=1, off by default.
#
#   Every trial shares a long fixed prefix (system prompt + perception API doc +
#   tool definitions), so the prefill saving is real and worth having. But this
#   model is a Mamba/linear-attention hybrid:
#     layer_types = [linear_attention x3, full_attention] x N,
#     full_attention_interval = 4, mamba_ssm_dtype = float32
#   which is the exact architecture in vLLM issue #43587 ("Prefix caching fails for
#   incremental multimodal requests on Mamba-Attention hybrid models (Qwen3.5)",
#   num_cached_tokens stuck at 0). More seriously, issue #20261 reports that with
#   prefix caching on, identical text prompts with DIFFERENT images produced garbled
#   output under concurrency -- which is precisely the load shape of a run
#   (many tasks, one shared prompt, different images each, 8-32 concurrent).
#   Enabling it also switches hybrid models onto the mamba_cache_mode code path
#   (default "none" -> "all"/"align"), unverified on this build.
#
#   ⚠️ A silent image mix-up would corrupt the dependent variable of the whole
#   study while looking like a merely surprising score. So turning this on is NOT
#   complete until the correctness gate has run:
#       .venv/bin/python scripts/prefix_cache_correctness_check.py
#   which compares per-question answers ON vs OFF at temperature 0 under
#   concurrency. Speed alone is not acceptance -- that is the whole failure mode.
PREFIX_CACHING=${PREFIX_CACHING:-0}

set -uo pipefail
export PYTHONNOUSERSITE=1

VENV="${VENV:-/scratch/ywxzml3j/wry/envs/vllm27}"

# vLLM's MoE path compiles kernels at startup and needs nvcc. The system has no
# /usr/local/cuda, but vllm27 ships the pip CUDA toolkit, so point CUDA_HOME at
# it. Use the CUDA 12.9 toolkit from the sibling conda env rather than vllm27's
# in-venv cu13 one: FlashInfer JIT-compiles kernels (sampling AND the TP all-reduce
# `trtllm_mnnvl_comm`) against its own bundled CUDA-12-era CCCL headers, and nvcc
# 13.3 rejects those with
#   cuda_toolkit.h: "CUDA compiler and CUDA toolkit headers are incompatible".
export CUDA_HOME="${CUDA_HOME:-/scratch/ywxzml3j/wry/miniconda3/envs/vllm}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

# FlashInfer JIT is used for BOTH sampling and the TP all-reduce path, so disabling
# the sampler alone is not enough -- the all-reduce kernel hits the same
# header/compiler mismatch. Disable every entry point and use vLLM's native paths.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_INT=0
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}

MODEL=Qwen/Qwen3.6-35B-A3B
PORT=${PORT:-8077}
# ⚠️ ONLY GPUs 0-3 MAY BE USED (2026-08-23, user constraint). Both this server
# AND the perception GPU server must live on those four cards; 4-7 are off
# limits. Pinned here rather than left to the caller so a forgotten export
# cannot silently spill onto a card we are not allowed to touch.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TP=${TP:-4}                       # the four cards we are allowed
MAX_LEN=${MAX_LEN:-262144}
MAX_SEQS=${MAX_SEQS:-40}
# 0.85 -> 0.78 because the perception server now shares these same four cards.
# Measured: perception peaks at 13.67 GB on its primary card (plus ~0.6 GB on
# each of the others). At 0.85 vLLM takes 67.7 GB of 79.6 GB usable and leaves
# 11.9 GB -- less than perception's measured peak, so the two would race and
# whichever started second would OOM. 0.78 leaves 17.5 GB, i.e. the peak plus
# ~28% headroom.
#
# Cost of the reduction: KV cache shrinks proportionally, so max concurrency at
# full context drops. It was never the binding limit -- `--max-num-seqs 40` and
# sagent's own -n 32 cap in-flight requests well below what the cache allows.
GPU_UTIL=${GPU_UTIL:-0.78}
TOOL_PARSER=${TOOL_PARSER:-qwen3_xml}

# Split the model's reasoning out of `content` into `reasoning_content`.
#
# WHY THIS IS REQUIRED, not optional. Qwen3.5's chat template puts the opening
# `<think>` in the PROMPT, so the model emits only the closing `</think>`
# (vllm/reasoning/qwen3_reasoning_parser.py:20-24). With no reasoning parser
# configured, vLLM's parser engine forces initial_state=CONTENT on the
# assumption that reasoning was already stripped -- nothing stripped it, so the
# tool parser silently absorbs `</think>` and the model's whole reasoning trace
# lands in `content`.
#
# Measured consequence on v12/v13: `reasoning_content` was non-empty in 0 of
# 234 responses, and the assistant prose we replay into history was 83.2%
# reasoning. Worse, the affected responses arrive with EMPTY tool_calls and
# orphan `</parameter></function></tool_call>` text, and openhands-sdk treats a
# non-blank response with no tool_calls as "the agent is finished"
# (response_dispatch.py:290-294). So the episode ends there. Deduplicated, 21 of
# 21 such leaks in v12 were the episode's LAST response -- this defect is
# essentially always fatal -- and it accounts for 46 of v13's 67 unanswered
# episodes, 26 of which had already stated their answer in prose.
#
# ⚠️ This changes SERVER launch args, so runs before and after are NOT pairable
# with each other. A version comparison across this boundary is meaningless;
# start a new configuration.
REASONING_PARSER=${REASONING_PARSER:-qwen3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG=${LOG:-"$SCRIPT_DIR/../logs/vllm_sagent.log"}

mkdir -p "$(dirname "$LOG")"

PREFIX_ARGS=()
if [ "$PREFIX_CACHING" = "1" ]; then
  PREFIX_ARGS+=(--enable-prefix-caching)
else
  PREFIX_ARGS+=(--no-enable-prefix-caching)
fi

echo "[launch_vllm] model=$MODEL port=$PORT tp=$TP max_len=$MAX_LEN max_seqs=$MAX_SEQS parser=$TOOL_PARSER reasoning=$REASONING_PARSER prefix_cache=$PREFIX_CACHING"
echo "[launch_vllm] log -> $LOG"

"$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen3.6-35b-a3b \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_PARSER" \
  --reasoning-parser "$REASONING_PARSER" \
  "${PREFIX_ARGS[@]}" \
  --limit-mm-per-prompt "{\"image\":${MM_IMAGES:-160},\"video\":1}" \
  --trust-remote-code \
  >> "$LOG" 2>&1
