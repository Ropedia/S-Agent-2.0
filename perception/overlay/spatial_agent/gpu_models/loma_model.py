"""LoMa-B keypoint matcher as a GPU server tool.

Wraps https://github.com/davnords/LoMa (vendored under tools/third_party/LoMa,
installed editable into the perception venv) behind the same RPC surface as
SAM3/DA3. One method: ``match`` — given two images, return the matched keypoint
pairs in each image's own pixel coordinates.

Why paths-through-tempfiles instead of tensors: LoMa's ``match`` accepts either
a file path or an already-normalized ``(B, C, H, W)`` tensor, and the two paths
do NOT share preprocessing — the path branch runs the detector's own loading
(resize + normalization). Re-implementing that here would silently drift the
next time upstream changes it; a temp PNG costs ~10 ms against an ~800 ms match.

Weights: loma_B.pt + DaD detector + DeDoDe-G descriptor + DINOv2 ViT-L backbone,
all auto-downloaded by upstream via torch.hub on first construction (~2 GB,
cached under ~/.cache/torch/hub/checkpoints). Construction therefore needs
network the first time; after that it is offline.
"""

import os
import sys
import tempfile
from typing import Dict, Optional

import numpy as np
from PIL import Image

from spatial_agent.gpu_models.base import AgentTool, gpu_inference_lock

ImageLoader = None  # stub: PIL images arrive directly, same as SAM3Model

__all__ = ["LoMaModel"]

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LOMA_PATH = os.path.join(_PROJECT_ROOT, "tools", "third_party", "LoMa", "src")
if _LOMA_PATH not in sys.path:
    sys.path.insert(0, _LOMA_PATH)


class LoMaModel(AgentTool):
    CPU_CONSUMED = 0.5
    VRAM_CONSUMED = 4.0  # DINOv2 ViT-L backbone dominates (~1.2 GB weights + activations)
    AUTOSCALING_MIN_REPLICAS = 1
    AUTOSCALING_MAX_REPLICAS = 1

    DEVICE = "cuda"

    def __init__(self, image_loader: ImageLoader = None) -> None:
        super().__init__()
        from loma import LoMa, LoMaB

        print("[LoMa] Loading LoMa-B (DaD detector + DeDoDe-G descriptor)...", flush=True)
        self.model = LoMa(LoMaB())
        print("[LoMa] Ready.", flush=True)
        self.image_loader = image_loader

    def match(
        self,
        image_a,
        image_b,
        num_keypoints: Optional[int] = None,
    ) -> Dict:
        """Matched keypoints between two images, in pixel coordinates.

        Args:
            image_a / image_b: PIL images (the gateway's load path) or file
                paths readable by this process.
            num_keypoints: Optional override of the per-image keypoint budget
                (upstream default: cfg.num_keypoints).

        Returns:
            {"kpts_a": (N, 2) float32, "kpts_b": (N, 2) float32,
             "n": N, "size_a": [W, H], "size_b": [W, H]}
            Row i of kpts_a corresponds to row i of kpts_b. Coordinates are on
            each image's ORIGINAL pixel grid (x, y).
        """
        tmp_paths = []
        try:
            path_a, size_a = self._as_path(image_a, tmp_paths)
            path_b, size_b = self._as_path(image_b, tmp_paths)
            with gpu_inference_lock():
                kpts_a, kpts_b = self.model.match(
                    path_a, path_b, num_keypoints=num_keypoints
                )
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

        kpts_a = np.asarray(kpts_a, dtype=np.float32).reshape(-1, 2)
        kpts_b = np.asarray(kpts_b, dtype=np.float32).reshape(-1, 2)
        return {
            "kpts_a": kpts_a,
            "kpts_b": kpts_b,
            "n": int(kpts_a.shape[0]),
            "size_a": list(size_a),
            "size_b": list(size_b),
        }

    @staticmethod
    def _as_path(image, tmp_paths) -> tuple:
        """Return (path, (W, H)) for a PIL image or an existing path."""
        if isinstance(image, str):
            with Image.open(image) as im:
                return image, im.size
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"match expects a PIL image or a path, got {type(image).__name__}"
            )
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        fd, path = tempfile.mkstemp(suffix=".png", prefix="loma_")
        os.close(fd)
        rgb.save(path, "PNG")
        tmp_paths.append(path)
        return path, rgb.size
