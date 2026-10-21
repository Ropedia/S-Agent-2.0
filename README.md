<div align="center">

# S-Agent 2.0: Spatial Reasoning with a General Coding Agent

[![Paper](https://img.shields.io/badge/Paper-coming%20soon-lightgrey)](#)
[![Project Page](https://img.shields.io/badge/Project%20Page-coming%20soon-lightgrey)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0%20%2F%20NVIDIA%20NC-blue)](#license)

</div>



https://github.com/user-attachments/assets/d540ddeb-f531-48ef-afc9-e3347c83ca48



## Overview

S-Agent 2.0 is a spatial reasoning agent built on an off-the-shelf general
coding agent (OpenHands) and a command-line perception toolkit. The toolkit
is designed around one principle, closing the **evidence loop**: the agent
**acquires** scene evidence with 3D perception tools, **remembers** it in a
persistent scene memory, and **interprets** every reading in plain spatial
language. No training and no custom controller: the toolkit is one
executable the agent calls from its shell, and every image it renders is a
file the agent opens with its ordinary file viewer.

This repository contains the full system: the task instruction and toolkit
tutorial, the perception gateway (scene memory and readouts), the benchmark
loaders, the agent harness, and the GPU perception server. Model weights are
not included; see [docs/SETUP.md](docs/SETUP.md).

## News

- **2026-10** Code release.

## Results

Accuracy with and without the toolkit, on the same agent and the same items
(full ReVSI and MMSI-Bench). SpatialClaw runs on the same backbone and the
same perception backends.

| | Qwen3.8-27B ReVSI | Qwen3.8-27B MMSI | Gemini-3.8-Flash ReVSI | Gemini-3.8-Flash MMSI |
|---|:---:|:---:|:---:|:---:|
| SpatialClaw, no tools | 61.3 | 50.4 | 77.0 | 60.0 |
| SpatialClaw | 74.5 | 62.6 | 79.7 | 67.0 |
| General coding agent, no tools | 61.8 | 52.6 | 80.9 | 62.9 |
| **S-Agent 2.0** | **82.3** | **66.5** | **90.5** | **74.1** |

Comparison with agentic spatial reasoning baselines, Qwen3.8-27B backbone:

| Method | MMSI-Bench | MindCube | SPAR | MMSI-Video | ReVSI |
|---|:---:|:---:|:---:|:---:|:---:|
| pySpatial | 36.3 | 66.3 | 42.1 | 26.9 | 39.3 |
| SpaceTools | 43.2 | 58.6 | 55.3 | 39.3 | 64.4 |
| S-Agent | 48.3 | 75.5 | 63.0 | 37.9 | 59.5 |
| SpatialClaw | 62.6 | 83.2 | 70.2 | 53.1 | 74.5 |
| **S-Agent 2.0** | **66.5** | **88.3** | **72.4** | **56.4** | **82.3** |

## Installation

Four services plus the task runner; every path and port is set through
environment variables. Step-by-step instructions, including weights and the
benchmark data variables, are in [docs/SETUP.md](docs/SETUP.md).

```bash
git clone --recurse-submodules <this repository> && cd S-Agent-2.0

# agent side
cd agent && uv venv --python 3.12 && uv pip install -e ".[spatial]"
bash scripts/build_runtime.sh            # the OpenHands runtime image with our entrypoint

# perception side (GPU): SpatialClaw submodule + overlay + model repositories
cd ../perception && bash setup.sh        # then create SpatialClaw's environment per its docs
```

Then start the perception server (`perception/launch_gpu_server.sh`), the
gateway (`agent/gateway/launch_gateway.sh`) and a LiteLLM endpoint for the
backbone (`agent/scripts/launch_vllm.sh` + `agent/scripts/serve_litellm_spatial.sh`,
or a hosted model through `agent/configs/litellm_*.yaml`).

## Evaluation

Point a loader at the benchmark data (`MMSI_PARQUET`, `REVSI_DIR`, `SPAR_DIR`,
`MINDCUBE_DIR`, `MMSIVIDEO_DIR`, `VIEWSPATIAL_DIR`, `SPBENCH_DIR`,
`ERQA_PARQUET`) and run:

```bash
cd agent
LLM_MODEL_NAME=qwen3.8-27b-fp8 scripts/run.sh mmsi all 16        # MMSI-Bench, 16 concurrent trials
LLM_MODEL_NAME=qwen3.8-27b-fp8 scripts/run.sh revsi all 16       # ReVSI
LLM_MODEL_NAME=qwen3.8-27b-fp8 scripts/run.sh spar all 16        # SPAR-Bench
LLM_MODEL_NAME=qwen3.8-27b-fp8 scripts/run.sh mindcube all 16    # MindCube
LLM_MODEL_NAME=qwen3.8-27b-fp8 scripts/run.sh mmsivideo all 16   # MMSI-Video-Bench
scripts/run.sh mmsi all 16 --no-tools                            # the same agent without the toolkit
```

`scripts/run.sh` generates one task per item from `configs/profile`, checks
the services (`scripts/preflight.sh`) and launches the job. Results land under
`agent/jobs/<job>/<item>__*/`: `result.json` (score), `agent/trajectory.json`
(the full episode) and `perception/artifacts/scene_state.json` (the final
scene memory).

## The toolkit

Inside the task container the agent runs `python3 /app/perception.py <command>`:

| | commands |
|---|---|
| tool perception | `recon` (metric, gravity-aligned reconstruction of all frames), `segment` (SAM 3 by text prompt; every label becomes a named instance) |
| model perception | `add_instance` (register a point or an oriented observer the question implies), `drop` |
| readouts | `list_instance`, `bev`, `distance`, `depth`, `side`, `elevation`, `motion`, `compass` |

Every reply is JSON with numeric fields in metres, a per-target `conclusion`
sentence in the options' own words, and `units`. The tutorial the agent
reads is `agent/configs/profile/prompt/perception_api.md`.

## Repository layout

```
agent/                 the agent side
  configs/profile/     the task instruction, toolkit tutorial, workflow and skill card
  adapters/spatial/    benchmark loaders, task template, perception.py (the CLI)
  gateway/             the perception gateway: scene memory, readouts, replies
  src/sagent/          the evaluation harness; agents/custom/ drives the OpenHands SDK
  scripts/             run.sh, service launchers, the runtime image build
perception/            SpatialClaw's GPU perception server as a pinned submodule, plus
                       an overlay adding GeoCalib, LoMa and Pi3-MOS tools
docs/SETUP.md          how to bring the services up and run a benchmark
```

## Citation

```bibtex
@article{sagent2,
  title   = {S-Agent 2.0: Spatial Reasoning with a General Coding Agent},
  year    = {2026}
}
```

## Acknowledgements

The perception server builds on [SpatialClaw](https://github.com/NVlabs/SpatialClaw)
and the models it serves: [Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3),
[SAM 3](https://github.com/facebookresearch/sam3), [GeoCalib](https://github.com/cvg/GeoCalib),
[LoMa](https://github.com/davnords/LoMa) and [Pi3MOS](https://github.com/PRBonn/Pi3MOS-SLAM).
The agent runs on the [OpenHands](https://github.com/OpenHands/OpenHands) SDK.

## License

Code under `agent/` and `docs/` is released under the Apache License 2.0
(see `LICENSE`). `perception/` wraps NVIDIA's SpatialClaw (a pinned submodule)
and keeps its NVIDIA Source Code License-NC (non-commercial); the model
repositories carry their own licenses. See `NOTICE`.
