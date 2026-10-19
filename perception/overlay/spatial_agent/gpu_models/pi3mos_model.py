"""Pi3MOS per-pixel motion probability (moving-object segmentation).

Independent deployment (``spatial_Dynamic``): rides along with every
reconstruct backend the way SAM3/Match/Gravity do, and serves exactly one
method, ``dynamic_masks`` — the per-frame motion probability map from the
Pi3MOS model (PRBonn/Pi3MOS-SLAM: the original Pi3 plus a frozen-backbone
MOS head). Reconstruction outputs of the model (points/poses/conf) are NOT
exposed here: geometry stays with the Reconstruct deployment so the two
backends can be swapped independently.

Preprocessing is a WHOLE-FRAME resize (multiple-of-14 grid under a pixel
budget, LANCZOS) — deliberately no aspect crop, unlike DA3/Pi3
reconstruction: the consumer (perception gateway) maps the probability map
back onto the original frame with a plain resize and then applies the
reconstruction's own crop_box, so keeping this side crop-free removes one
source of misalignment.

The checkpoint (``SPATIAL_AGENT_PI3MOS_CKPT``) is loaded lazily on the
first call: the deployment registers even when the weights are absent, and
only ``dynamic_masks`` calls fail — a missing side-channel must never take
down the shared GPU server that also hosts Reconstruct/SAM3.
"""

import asyncio
import os
import sys
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF

from spatial_agent.gpu_models.base import (
    AgentTool,
    AgentToolOutput,
    gpu_inference_lock,
)
from spatial_agent.gpu_models.types import Pi3MosDynamicOutput

__all__ = ["Pi3MosModel", "Pi3MosDynamicOutput"]

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PI3MOS_PATH = os.path.join(_PROJECT_ROOT, "tools", "third_party", "Pi3MOS")
if _PI3MOS_PATH not in sys.path:
    sys.path.insert(0, _PI3MOS_PATH)

#: Same training-distribution pixel budget the vendored code uses.
_PIXEL_LIMIT = int(os.environ.get("SPATIAL_AGENT_PI3MOS_PIXEL_LIMIT", "255000"))


def _target_grid(width: int, height: int) -> tuple:
    """Multiple-of-14 (W, H) under the pixel budget, whole-frame (no crop)."""
    scale = (_PIXEL_LIMIT / float(width * height)) ** 0.5
    tw, th = width * scale, height * scale
    k, m = round(tw / 14), round(th / 14)
    while (k * 14) * (m * 14) > _PIXEL_LIMIT:
        if k / max(m, 1) > tw / th:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


class Pi3MosModel(AgentTool):
    """Pi3MOS motion-probability model (lazy-loaded)."""

    CPU_CONSUMED = 0.5
    VRAM_CONSUMED = 4.0
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 2

    CKPT = os.environ.get(
        "SPATIAL_AGENT_PI3MOS_CKPT",
        os.path.join(_PROJECT_ROOT, "weights", "pi3mos", "model.safetensors"),
    )
    DEVICE = os.environ.get("SPATIAL_AGENT_PI3MOS_DEVICE", "cuda")

    def __init__(self, image_loader=None) -> None:
        super().__init__()
        self.image_loader = image_loader
        self.model = None
        self._load_error: Optional[str] = None
        self.infer_lock = asyncio.Lock()

    def _ensure_model(self) -> Optional[str]:
        """Load once; return an error string instead of raising."""
        if self.model is not None:
            return None
        if self._load_error is not None:
            return self._load_error
        try:
            if not os.path.isfile(self.CKPT):
                raise FileNotFoundError(
                    f"Pi3MOS checkpoint not found: {self.CKPT} "
                    "(set SPATIAL_AGENT_PI3MOS_CKPT)"
                )
            from safetensors.torch import load_file
            from pi3mos.models.pi3 import Pi3 as Pi3Mos

            model = Pi3Mos().to(self.DEVICE).eval()
            model.load_state_dict(load_file(self.CKPT))
            self.model = model
        except Exception as exc:  # noqa: BLE001 - report, don't kill the server
            self._load_error = f"Pi3MOS load failed: {exc}"
        return self._load_error

    @torch.no_grad()
    def _dynamic_masks(
        self,
        images: List[Image.Image],
        frame_indices: Optional[List[int]],
    ) -> Pi3MosDynamicOutput:
        to_tensor = TF.ToTensor()
        first = images[0].convert("RGB")
        tw, th = _target_grid(first.width, first.height)
        tensors = []
        for img in images:
            img = img.convert("RGB")
            if img.size != (tw, th):
                img = img.resize((tw, th), Image.LANCZOS)
            tensors.append(to_tensor(img))
        batch = torch.stack(tensors).unsqueeze(0).to(self.DEVICE)  # (1, N, 3, H, W)

        use_bf16 = (
            self.DEVICE.startswith("cuda")
            and torch.cuda.get_device_capability()[0] >= 8
        )
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.amp.autocast("cuda", dtype=dtype, enabled=self.DEVICE.startswith("cuda")):
            result = self.model(batch)

        prob = (
            result["dynamic_masks"][0]
            .reshape(len(images), th, tw)
            .float()
            .cpu()
            .numpy()
            .astype(np.float16)
        )
        if frame_indices is None:
            frame_indices = list(range(len(images)))
        return Pi3MosDynamicOutput(
            dynamic_prob=prob,
            frame_indices=[int(i) for i in frame_indices],
            num_frames=len(images),
            input_size=[tw, th],
        )

    def _dynamic_masks_locked(
        self,
        images: List[Image.Image],
        frame_indices: Optional[List[int]],
    ) -> Pi3MosDynamicOutput:
        with gpu_inference_lock():
            return self._dynamic_masks(images, frame_indices)

    async def dynamic_masks(
        self,
        video_frames: List[Image.Image],
        frame_indices: Optional[List[int]] = None,
    ) -> AgentToolOutput:
        """Per-pixel motion probability for each frame.

        Args:
            video_frames: List of PIL images (one joint forward pass).
            frame_indices: Absolute video frame indices.
        """
        if not isinstance(video_frames, list):
            video_frames = [video_frames]
        if not video_frames:
            return self.error("dynamic_masks needs at least one frame")

        err = self._ensure_model()
        if err is not None:
            return self.error(err)

        async with self.infer_lock:
            output = await asyncio.to_thread(
                self._dynamic_masks_locked, video_frames, frame_indices
            )
        return self.success(result=output)
