# Perception server

S-Agent 2.0 uses the GPU perception server of
[SpatialClaw](https://github.com/NVlabs/SpatialClaw) (pinned at `b062f82`,
a git submodule under `SpatialClaw/`) and adds three tools to it:

| tool | model | file |
|---|---|---|
| Match | LoMa keypoint matcher | `overlay/spatial_agent/gpu_models/loma_model.py` |
| Gravity | GeoCalib single-image gravity | `overlay/spatial_agent/gpu_models/geocalib_model.py` |
| Dynamic | Pi3MOS per-pixel motion probability | `overlay/spatial_agent/gpu_models/pi3mos_model.py` |

`overlay/spatialclaw.patch` registers them in SpatialClaw's tool table
(`types.py`, `entrypoints/launch_gpu_server.py`). `setup.sh` fetches the
submodule, applies the overlay and clones the three model repositories;
`launch_gpu_server.sh` starts the server. The gateway under `../agent/gateway`
talks to it over SpatialClaw's RPC and needs nothing else from here.

License: the overlay derives from SpatialClaw and is distributed under the
NVIDIA Source Code License-NC (`overlay/LICENSE`); non-commercial use only.
The model repositories carry their own licenses (GeoCalib Apache-2.0, LoMa
MIT, Pi3MOS-SLAM MIT).
