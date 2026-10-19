#!/usr/bin/env bash
# Start the perception server: Reconstruct (Depth Anything 3), SAM3, Match (LoMa),
# Gravity (GeoCalib), Dynamic (Pi3MOS). The gateway discovers it through
# SpatialClaw/spatial_agent/logs/gpu_server.json.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES:-0,1}"
cd "$HERE/SpatialClaw"
exec "${PERCEPTION_PYTHON:-python}" -m spatial_agent.entrypoints.launch_gpu_server   --num_gpus "${NUM_GPUS:-2}"   --reconstruct_backend "${RECON_BACKEND:-da3}"   --http_port "${GPU_PORT:-8078}"
