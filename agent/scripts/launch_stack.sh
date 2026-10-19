#!/usr/bin/env bash
# Bring up the whole serving stack on GPUs 0-3 only, in the one order that works.
#
#   bash scripts/launch_stack.sh
#
# ⚠️ ONLY GPUs 0-3 MAY BE USED (2026-08-23, standing constraint). Cards 4-7 are
# off limits for this project. Everything below is pinned so that a forgotten
# export cannot silently spill onto a card we are not allowed to touch, and the
# script asserts the constraint at the end rather than trusting itself.
#
# WHAT LIVES WHERE, and why this exact split
#
#   GPU 0   vLLM shard + SAM3          (detection / segmentation)
#   GPU 1   vLLM shard + DA3           (depth / 3D reconstruction)
#   GPU 2   vLLM shard
#   GPU 3   vLLM shard
#
# The perception server loads TWO models, and `--num_gpus 4` only makes Ray
# *claim* four cards -- measured, both models originally landed on the first one
# (11.6 GB on the primary, 1.4 GB of bare CUDA context on each of the others).
# `DA3Model.DEVICE` reads SPATIAL_AGENT_DA3_DEVICE (upstream's own hook,
# da3_model.py:86, applied at `.to(self.DEVICE)`), so DA3 can be moved without
# forking anything. SAM3's DEVICE is hardcoded to 'cuda', so it stays on 0.
#
# Splitting them is not just about fitting: it unlocked real parallelism.
# Measured after the split --
#
#     detect alone        3.03 s
#     reconstruct alone   2.73 s
#     both serially       5.76 s
#     both in parallel    2.24 s     <- 2.58x, and below max(3.03, 2.73)
#
# which retires an earlier conclusion in docs/INFRA_SETTINGS.md that "the
# upstream perception server is serial, so raising the gateway's
# UPSTREAM_CONCURRENCY buys nothing". That measurement was taken while both
# models shared one card and were genuinely contending.
#
# GPU_UTIL is 0.78, not 0.85, because perception now shares these cards. At 0.85
# vLLM takes 67.7 GB of 79.6 GB usable and leaves 11.9 GB -- less than
# perception's measured peak, so whichever service started second would OOM.
# Under real load (200-question run, concurrency 32) the observed peak is
# 70.5 GB on GPU 1, leaving ~10.8 GB, and zero OOM lines in either log.
#
# ORDER MATTERS: perception first. It grabs a fixed ~10 GB and cannot be told to
# fit into what is left, whereas vLLM sizes its KV cache against whatever free
# memory it finds at startup. Reversed, vLLM would take 78% of a nearly-empty
# card and leave perception nothing.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPRO_ROOT="${SPATIALCLAW_ROOT:-$(cd "$REPO_ROOT/../repro" && pwd)}"

GPUS="${STACK_GPUS:-0,1,2,3}"
DA3_DEVICE="${SPATIAL_AGENT_DA3_DEVICE:-cuda:1}"

log() { echo "[launch_stack] $*"; }

# --- 0. tear down whatever is holding the GPUs --------------------------------
#
# Kill by GPU pid, not by process name. `pkill -f vllm` misses the four
# VLLM::Worker_TP* children -- they do not carry "vllm" in argv -- and those are
# what actually hold 71 GB each. The criterion for "clean" is nvidia-smi
# reporting zero, not a port being free or pgrep coming up empty
# (docs/PITFALLS.md 3.1, which cost a whole debugging session).
log "stopping anything on the GPUs"
pkill -9 -f "sagent run" 2>/dev/null || true
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
  kill -9 "$pid" 2>/dev/null || true
done
sleep 15

held=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null |
       head -4 | awk '{s+=$1} END {print s+0}')
if [ "$held" -gt 2000 ]; then
  log "FATAL: GPUs 0-3 still hold ${held} MiB after the kill sweep."
  log "  Check: nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
  exit 1
fi
log "GPUs 0-3 clear (${held} MiB residual)"

# --- 1. perception server (two models, split across 0 and 1) ------------------
log "starting perception server: SAM3 on cuda:0, DA3 on ${DA3_DEVICE}"
( cd "$REPRO_ROOT" && \
  GPU_DEVICES="$GPUS" SPATIAL_AGENT_DA3_DEVICE="$DA3_DEVICE" \
  setsid nohup bash scripts/launch_gpu_server.sh \
    > /tmp/gpu_server.log 2>&1 < /dev/null & )

for _ in $(seq 1 60); do
  if curl -s -m 5 http://127.0.0.1:8078/health 2>/dev/null | grep -q '"status":"ok"'; then
    log "perception ready"
    break
  fi
  sleep 10
done
curl -s -m 10 http://127.0.0.1:8078/health 2>/dev/null | grep -q '"status":"ok"' || {
  log "FATAL: perception server never became healthy; see /tmp/gpu_server.log"
  exit 1
}

# --- 2. vLLM (TP=4 across all four, 0.78 to leave room for perception) -------
log "starting vLLM (TP=4, util 0.78) on ${GPUS}"
CUDA_VISIBLE_DEVICES="$GPUS" setsid nohup bash "$SCRIPT_DIR/launch_vllm.sh" \
  > /dev/null 2>&1 < /dev/null &

for _ in $(seq 1 90); do
  curl -s -m 5 http://127.0.0.1:8077/health >/dev/null 2>&1 && { log "vLLM ready"; break; }
  sleep 10
done
curl -s -m 10 http://127.0.0.1:8077/health >/dev/null 2>&1 || {
  log "FATAL: vLLM never became healthy; see logs/vllm_sagent.log"
  exit 1
}

# --- 3. litellm proxy --------------------------------------------------------
#
# Restarted with the stack rather than left running: it holds connections to the
# vLLM that was just replaced, and its sampling defaults are exported here (the
# yaml reads them via os.environ/ and has no default syntax of its own).
log "restarting litellm proxy"
for pid in $(ss -tlnp 2>/dev/null | grep 4011 | grep -oP 'pid=\K[0-9]+' | sort -u); do
  kill -9 "$pid" 2>/dev/null || true
done
sleep 5
setsid nohup bash "$SCRIPT_DIR/serve_litellm_spatial.sh" >/dev/null 2>&1 < /dev/null &
sleep 45

# --- 4. assert the constraint, do not assume it ------------------------------
spill=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null |
        tail -4 | awk '{s+=$1} END {print s+0}')
if [ "$spill" -gt 500 ]; then
  log "FATAL: GPUs 4-7 hold ${spill} MiB -- something ignored the device pinning."
  exit 1
fi

log "stack up. GPUs 4-7 clear (${spill} MiB)."
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  GPU /'
log "verify end to end before trusting a run: bash scripts/preflight.sh"
