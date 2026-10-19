"""GeoCalib single-image gravity estimator as a GPU server tool.

Wraps https://github.com/cvg/GeoCalib (vendored under tools/third_party/GeoCalib,
installed editable into the perception venv) behind the same RPC surface as
SAM3/DA3/LoMa. One method: ``estimate`` — given one image, return the unit
gravity direction in that camera's coordinates plus the optimizer's
uncertainties.

Convention (pinned empirically on the repo's own assets — the church interior
shot looking ~38° up at the ceiling): GeoCalib's ``Gravity.vec3d`` is the
WORLD-UP direction in OpenCV camera coordinates (x right, y down, z forward);
an upright level camera gives (0, -1, 0) and its ``pitch`` is positive for a
camera looking up. The perception gateway's /gravity artifact stores the DOWN
direction (level camera -> (0, 1, 0)), so ``estimate`` returns the NEGATED
vector. Do not "fix" the sign here without re-checking both ends.

Weights: geocalib-pinhole.tar (~111 MB), auto-downloaded via torch.hub on
first construction into ~/.cache/torch/hub/geocalib; offline afterwards.
"""

import math
import os
import sys
from typing import Dict

import numpy as np
from PIL import Image

from spatial_agent.gpu_models.base import AgentTool, gpu_inference_lock

ImageLoader = None  # stub: PIL images arrive directly, same as SAM3Model

__all__ = ["GeoCalibModel"]

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_GEOCALIB_PATH = os.path.join(_PROJECT_ROOT, "tools", "third_party", "GeoCalib")
if _GEOCALIB_PATH not in sys.path:
    sys.path.insert(0, _GEOCALIB_PATH)


class GeoCalibModel(AgentTool):
    CPU_CONSUMED = 0.5
    VRAM_CONSUMED = 0.5  # MSCAN-B backbone + LM optimizer, ~111 MB weights
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 1

    DEVICE = "cuda"

    def __init__(self, image_loader: ImageLoader = None) -> None:
        super().__init__()
        import torch
        from geocalib import GeoCalib

        print("[GeoCalib] Loading pinhole calibration model...", flush=True)
        self._torch = torch
        self.model = GeoCalib(weights="pinhole").to(self.DEVICE)
        print("[GeoCalib] Ready.", flush=True)
        self.image_loader = image_loader

    def estimate(self, image_source) -> Dict:
        """Gravity direction (and camera intrinsics) from a single image.

        Args:
            image_source: PIL image (the gateway's load path) or a file path
                readable by this process.

        Returns:
            {"gravity_xyz": (3,) float64 unit DOWN direction in camera coords
                 (x right, y down, z forward; level upright camera -> (0,1,0)),
             "roll_deg": float, "pitch_deg": float  (GeoCalib's own angles,
                 pitch > 0 = camera looking up),
             "gravity_uncertainty_deg": float,
             "roll_uncertainty_deg": float, "pitch_uncertainty_deg": float,
             "focal_px": float, "vfov_deg": float}
        """
        image = self._as_pil(image_source)
        array = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, 3), RGB
        tensor = (
            self._torch.from_numpy(array).permute(2, 0, 1).to(self.DEVICE)
        )
        with gpu_inference_lock(), self._torch.no_grad():
            result = self.model.calibrate(tensor)

        gravity = result["gravity"]
        up = gravity.vec3d.reshape(3).double().cpu().numpy()
        down = -up / np.linalg.norm(up)
        camera = result["camera"]

        def _deg(key: str) -> float:
            value = result.get(key)
            return float("nan") if value is None else math.degrees(float(value))

        return {
            "gravity_xyz": down.astype(np.float64),
            "roll_deg": math.degrees(float(gravity.roll)),
            "pitch_deg": math.degrees(float(gravity.pitch)),
            "gravity_uncertainty_deg": _deg("gravity_uncertainty"),
            "roll_uncertainty_deg": _deg("roll_uncertainty"),
            "pitch_uncertainty_deg": _deg("pitch_uncertainty"),
            "focal_px": float(camera.f.reshape(-1)[-1]),
            "vfov_deg": math.degrees(float(camera.vfov.reshape(-1)[0])),
        }

    @staticmethod
    def _as_pil(image) -> Image.Image:
        if isinstance(image, str):
            with Image.open(image) as im:
                return im.convert("RGB")
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"estimate expects a PIL image or a path, got {type(image).__name__}"
            )
        return image.convert("RGB") if image.mode != "RGB" else image
