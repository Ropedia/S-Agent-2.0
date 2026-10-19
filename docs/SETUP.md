# Setup

S-Agent 2.0 runs as four services plus the task runner. Everything below is
configured through environment variables; no code edits are needed to move
machines.

## 1. Python environments

```bash
# agent side (task generation, gateway, runner)
cd agent
uv venv --python 3.12 && uv pip install -e ".[spatial]"

# perception side (GPU server)
cd ../perception
bash setup.sh                        # SpatialClaw submodule + overlay + model repos
# then create SpatialClaw's environment: see SpatialClaw/docs (installation)
```

Model weights: Depth Anything 3, SAM 3, Pi3 and map-anything follow
SpatialClaw's pre-download instructions; GeoCalib and LoMa download through
torch.hub on first use; Pi3MOS's checkpoint path is `SPATIAL_AGENT_PI3MOS_CKPT`.

Agent runtime image: trial containers mount an OpenHands-SDK 1.33.0 runtime
with S-Agent 2.0's patched entrypoint (answer-file recovery, reasoning
preservation). Build it once with docker; by default this pulls the public
base image `docker.io/jierun/c-oh-sdk-1.33.0:v0.1` and layers the entrypoint
on it:

```bash
cd agent
bash scripts/build_runtime.sh          # -> sagent-oh-sdk-1.33.0:v1 (RUNTIME_IMAGE selects another tag)
```

To build the base yourself instead of pulling it (same package versions,
see `scripts/agent_runtimes/openhands-sdk/1.33.0/manifest.json`):

```bash
bash scripts/agent_runtimes/openhands-sdk/1.33.0/build-runtime-image.sh --image my-oh-sdk-base:1.33.0
BASE_IMAGE=my-oh-sdk-base:1.33.0 bash scripts/build_runtime.sh
```

## 2. Perception server (GPU)

```bash
cd perception
GPU_DEVICES=0,1 PERCEPTION_PYTHON=/path/to/spatialclaw-env/bin/python bash launch_gpu_server.sh   # serves :8078
```

## 3. Gateway (scene memory and readouts)

```bash
cd agent
GATEWAY_ROOT=$PWD \
GATEWAY_WORK_DIR=/path/to/sagent-work \
GATEWAY_TRIAL_ROOTS=/path/to/sagent-work:/path/to/jobs \
PERCEPTION_PORT=8090 bash gateway/launch_gateway.sh    # serves :8090
```

`GATEWAY_STRIP_NOTES` defaults to 1 (replies carry numbers, `conclusion` and
`units`; the reading guidance lives in `configs/profile/prompt/perception_api.md`).

## 4. Backbone

Any OpenAI-compatible endpoint behind LiteLLM works. For a local open model:

```bash
cd agent
bash scripts/launch_vllm.sh                            # vLLM, e.g. Qwen3.8-27B, --max-model-len 262144
bash scripts/serve_litellm_spatial.sh configs/litellm_qwen_spatial_27b.yaml   # LiteLLM on :4011
```

For Gemini or other hosted models use the matching `configs/litellm_*.yaml`
(keys are read from environment variables named inside each file).

## 5. Benchmarks

Point the loaders at the benchmark files through these variables (or the
matching `--parquet` / `--data-dir` flag); a loader refuses to run without
one:

| benchmark | variable | points at |
|---|---|---|
| MMSI-Bench | `MMSI_PARQUET` | the `MMSI_Bench.parquet` file |
| ReVSI | `REVSI_DIR`, `REVSI_FRAME_BUDGET` (default 64) | the ReVSI release dir with `frames_<budget>/` |
| SPAR-Bench | `SPAR_DIR` | the SPAR-Bench dataset dir |
| MindCube | `MINDCUBE_DIR`, `MINDCUBE_SPLIT` | the MindCube data dir |
| MMSI-Video-Bench | `MMSIVIDEO_DIR`, `MMSIVIDEO_FRAMES_PER_VIDEO` (default 32) | the MMSI-Video dir with official frames |
| ViewSpatial-Bench | `VIEWSPATIAL_DIR` | the ViewSpatial dataset dir |
| SPBench | `SPBENCH_DIR` | the SPBench dataset dir |
| ERQA | `ERQA_PARQUET` | the ERQA test parquet |

See `adapters/spatial/load_*.py` for each loader's flags.

## 6. Run

```bash
cd agent
LLM_MODEL_NAME=qwen3.8-27b-fp8 scripts/run.sh mmsi all 16
LLM_MODEL_NAME=qwen3.8-27b-fp8 scripts/run.sh revsi all 16
scripts/run.sh mmsi all 16 --no-tools        # control without the toolkit
```

`all` runs the items of the reported runs: the whole benchmark for MMSI-Bench
and MindCube, and the 1000 items in `configs/items/<dataset>.json` for ReVSI
(64-frame variant), SPAR-Bench, MMSI-Video-Bench and ViewSpatial-Bench.
`ITEMS=full` runs the whole benchmark and `ITEMS=<file>` another list; a
number in place of `all` samples that many items from the whole benchmark.
`scripts/preflight.sh` checks that vLLM, LiteLLM and the gateway answer
before a job starts. Results land under `agent/jobs/<job>/<task>__*/`
(`result.json`, `agent/trajectory.json`, `perception/artifacts/scene_state.json`).
Trial containers reach the gateway through `host.docker.internal` or the
default-route address; set `SAGENT_TASK_NETWORK_MODE=host` on hosts where
bridge networks cannot reach it.
