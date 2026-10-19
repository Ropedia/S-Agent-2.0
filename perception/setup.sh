#!/usr/bin/env bash
# One-time setup of the perception side: fetch SpatialClaw (pinned), apply the
# S-Agent 2.0 overlay (three extra GPU tools: Match=LoMa, Gravity=GeoCalib,
# Dynamic=Pi3MOS), and fetch the model repositories those tools wrap.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SC="$HERE/SpatialClaw"
# SpatialClaw: a submodule in a git checkout; cloned directly from a plain
# (zip) copy of this repository.
if [ -e "$SC/.git" ]; then
  :
elif git -C "$HERE/.." rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$HERE/.." submodule update --init perception/SpatialClaw
else
  rmdir "$SC" 2>/dev/null || true
  git clone -q https://github.com/NVlabs/SpatialClaw.git "$SC"
  git -C "$SC" checkout -q b062f82962549391a02cc70e022641c1f4d8ac2b
fi
cd "$SC"
git submodule update --init            # Depth-Anything-3, sam3, Pi3, map-anything
if git apply --check "$HERE/overlay/spatialclaw.patch" 2>/dev/null; then
  git apply "$HERE/overlay/spatialclaw.patch"
  echo "[setup] overlay patch applied"
else
  echo "[setup] overlay patch already applied (or SpatialClaw diverged from b062f82)"
fi
cp "$HERE"/overlay/spatial_agent/gpu_models/*.py spatial_agent/gpu_models/
mkdir -p tools/third_party
if [ ! -d tools/third_party/GeoCalib ]; then
  git clone -q https://github.com/cvg/GeoCalib.git tools/third_party/GeoCalib
  git -C tools/third_party/GeoCalib checkout -q 97b8968e7798a66bf04fcf791fb535624241bda7
fi
if [ ! -d tools/third_party/LoMa ]; then
  git clone -q https://github.com/davnords/LoMa tools/third_party/LoMa
  git -C tools/third_party/LoMa checkout -q 5e541b8e39b5e0b2fd7d6b74ced583061a3e805a
fi
if [ ! -d tools/third_party/Pi3MOS ]; then
  git clone -q https://github.com/PRBonn/Pi3MOS-SLAM.git tools/third_party/Pi3MOS-SLAM
  git -C tools/third_party/Pi3MOS-SLAM checkout -q 5e709d27324a935e364e01efc56f3128ac00bf24
  mkdir -p tools/third_party/Pi3MOS
  cp -r tools/third_party/Pi3MOS-SLAM/pi3 tools/third_party/Pi3MOS/pi3mos   # renamed so it coexists with Pi3
fi
cat <<EOF
[setup] done. Weights:
  Depth Anything 3 / Pi3 / map-anything : pre-download per SpatialClaw/docs (HF cache)
  SAM 3                                  : SpatialClaw/tools/third_party/sam3/weights/sam3.pt
  GeoCalib, LoMa                         : auto-downloaded via torch.hub on first use
  Pi3MOS                                 : SPATIAL_AGENT_PI3MOS_CKPT=<path to model.safetensors>
                                           (default SpatialClaw/weights/pi3mos/model.safetensors)
EOF
