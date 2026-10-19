#!/usr/local/bin/python3
# Interpreter pinned to the TASK image's python, not `env python3`.
# The agent runs with the openhands-sdk runtime mounted at
# /opt/custom-agent-runtime/oh-sdk, whose bin/ comes FIRST on PATH and
# ships its own python 3.12.8 WITHOUT numpy. `env python3` resolves there,
# so `python3 /app/geom.py` died with ModuleNotFoundError: numpy on 17
# calls in the m0a run alone -- while `pip show numpy` inside the task
# image says 2.2.6 is installed, which is why it looked impossible.
# /usr/local/bin/python3 is the task image's own interpreter (3.11.16).
"""Mask utility functions (CPU-only, no external dependencies).

Operates on binary masks in either of the two shapes that actually occur here:
- (H, W): a single mask.
- (n_instances, H, W): exactly what the gateway's /detect saves (its sidecar
  says `layout: "(n_instances, H, W) boolean instance masks"`), so a mask .npy
  taken straight out of /perception/artifacts/ works with no reshaping. One
  instance is unwrapped automatically; with several, pass --instance to pick
  one (same flag and meaning as perception.py's --instance).

Coordinate conventions:
- Masks: True = foreground, False = background
- Pixel coordinates: (x, y) where x is column (horizontal), y is row (vertical)
- Bounding boxes: (x1, y1, x2, y2) in pixel coordinates, inclusive of endpoints
- Centroid: (cx, cy) in pixel coordinates, computed as median of True pixels

Usage:
    # As a library:
    import maskutil
    iou_score = maskutil.iou(mask_a, mask_b)

    # As a CLI (outputs JSON):
    /app/maskutil.py iou --mask-a /path/to/a.npy --mask-b /path/to/b.npy
    /app/maskutil.py centroid --mask /path/to/mask.npy
    # multi-instance masks: select one
    /app/maskutil.py centroid --mask /perception/artifacts/masks_ab12.npy --instance 2
"""

import argparse
import json
import sys

import numpy as np


def as_mask_2d(mask, name: str = "mask", instance: int | None = None) -> np.ndarray:
    """Coerce a gateway mask array to a single 2D (H, W) mask.

    Accepts (H, W) unchanged, and (n_instances, H, W) -- the layout /detect
    actually writes -- by selecting one instance:

    * n_instances == 1: unwrapped automatically. There is no ambiguity, and
      this is the common case (a single-prompt detect).
    * n_instances > 1 with `instance` given: that index is used.
    * n_instances > 1 with no `instance`: raises, rather than silently
      defaulting to index 0. A silent default would return a confident number
      about the wrong object, which is worse than an error the caller can act
      on. (The gateway's own /render and /measure default instance to 0; the
      difference is deliberate -- there, picking a mask is a visible, checkable
      side effect, whereas here the mask index vanishes into a scalar.)

    `instance` also accepts 0 for a single-instance array, so a caller can pass
    --instance unconditionally.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(mask).__name__}")
    if mask.ndim == 2:
        if instance not in (None, 0):
            raise ValueError(
                f"{name} is a single (H, W) mask, so --instance {instance} "
                f"does not apply (only 0 is valid)"
            )
        return mask
    if mask.ndim == 3:
        n = mask.shape[0]
        if instance is None:
            if n == 1:
                return mask[0]
            raise ValueError(
                f"{name} holds {n} instances (shape {mask.shape}); pass "
                f"--instance <0..{n - 1}> to choose one"
            )
        if not 0 <= instance < n:
            raise ValueError(
                f"--instance {instance} out of range for {name} with "
                f"{n} instances (shape {mask.shape})"
            )
        return mask[instance]
    raise ValueError(
        f"{name} must be 2D (H, W) or 3D (n_instances, H, W), "
        f"got shape {mask.shape}"
    )


def _check_mask_2d(mask: np.ndarray, name: str = "mask") -> None:
    """Validate that mask is a 2D numpy array.

    Kept for callers that have already narrowed to 2D; the public functions go
    through as_mask_2d() so they also accept the gateway's 3D layout.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(mask).__name__}")
    if mask.ndim != 2:
        raise ValueError(f"{name} must be 2D (H, W), got shape {mask.shape}")


def iou(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    instance_a: int | None = None,
    instance_b: int | None = None,
) -> float:
    """Intersection over Union between two binary masks.

    Args:
        mask_a: (H, W) or (n_instances, H, W) boolean array, first mask
        mask_b: same, second mask (selected instance must match mask_a's shape)
        instance_a/instance_b: instance to select when that array is 3D with
            more than one instance (see as_mask_2d)

    Returns:
        float: IoU score in [0, 1], or 0.0 if both masks are empty

    Formula: intersection_area / union_area
    """
    mask_a = as_mask_2d(mask_a, "mask_a", instance_a)
    mask_b = as_mask_2d(mask_b, "mask_b", instance_b)
    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"mask_a shape {mask_a.shape} and mask_b shape {mask_b.shape} must match"
        )

    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)

    if union == 0:
        return 0.0
    return float(intersection / union)


def intersection(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    instance_a: int | None = None,
    instance_b: int | None = None,
) -> np.ndarray:
    """Element-wise AND (intersection) of two binary masks.

    Args:
        mask_a: (H, W) or (n_instances, H, W) boolean array, first mask
        mask_b: same, second mask (selected instance must match mask_a's shape)
        instance_a/instance_b: instance to select when 3D (see as_mask_2d)

    Returns:
        (H, W) boolean array: True where both masks are True
    """
    mask_a = as_mask_2d(mask_a, "mask_a", instance_a)
    mask_b = as_mask_2d(mask_b, "mask_b", instance_b)
    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"mask_a shape {mask_a.shape} and mask_b shape {mask_b.shape} must match"
        )
    return mask_a.astype(bool) & mask_b.astype(bool)


def centroid(mask: np.ndarray, instance: int | None = None) -> tuple[float, float]:
    """Compute (cx, cy) centroid of a binary mask.

    Args:
        mask: (H, W) or (n_instances, H, W) boolean array
        instance: instance to select when 3D (see as_mask_2d)

    Returns:
        (cx, cy) tuple: centroid in pixel coordinates (column, row).
        Returns (nan, nan) if the mask is empty (all False).

    The centroid is computed as the median of x and y coordinates of all True pixels.
    Median is more robust to outliers than mean.
    """
    mask = as_mask_2d(mask, "mask", instance)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (float('nan'), float('nan'))
    # Centroid as median of pixel coordinates: (x, y) = (column, row)
    return (float(np.median(xs)), float(np.median(ys)))


def area(mask: np.ndarray, instance: int | None = None) -> int:
    """Count of True pixels in the mask.

    Args:
        mask: (H, W) or (n_instances, H, W) boolean array
        instance: instance to select when 3D (see as_mask_2d)

    Returns:
        int: number of foreground pixels
    """
    mask = as_mask_2d(mask, "mask", instance)
    return int(np.sum(mask))


def bounding_box(
    mask: np.ndarray,
    instance: int | None = None,
    percentile: float | None = None,
) -> tuple[int, int, int, int] | None:
    """Compute the tight bounding box around True pixels.

    Args:
        mask: (H, W) or (n_instances, H, W) boolean array
        instance: instance to select when 3D (see as_mask_2d)
        percentile: None (default) for a truly tight box -- the exact min/max
            of the True pixels, containing every foreground pixel. Pass a
            value p in (0, 50) to instead trim to the [p, 100-p] percentile
            range of pixel coordinates, which discards speckle at the cost of
            clipping genuine thin protrusions.

    Returns:
        (x1, y1, x2, y2) tuple in pixel coordinates (inclusive), or None if
        mask is empty. x1, x2 are column indices (horizontal), y1, y2 are row
        indices (vertical).

    Tight is the default because the returned box is usually treated as the
    object's extent, and a silently-shrunk extent is an error that propagates
    quietly (an object measured narrower than it is). Percentile trimming used
    to be applied automatically to any mask over 100 px, which contradicted
    this docstring: on a segmentation with a thin protrusion -- a handle, a
    robot arm, shapes SAM produces routinely -- the 1/99 box clipped several
    pixels of real object. That was invisible to regular-rectangle tests,
    where floor/ceil happens to land back on the true edges.

    If you do want speckle rejection, ask for it explicitly (`percentile=1`)
    and treat the result as an approximate, deliberately-shrunk box.
    """
    mask = as_mask_2d(mask, "mask", instance)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    if percentile is not None:
        if not 0 < percentile < 50:
            raise ValueError(
                f"percentile must be in (0, 50), got {percentile}"
            )
        lo, hi = percentile, 100.0 - percentile
        x1, x2 = np.percentile(xs, [lo, hi])
        y1, y2 = np.percentile(ys, [lo, hi])
        return (int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2)))

    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Masks saved by /detect are (n_instances, H, W); --instance picks one
    # (same meaning as perception.py's --instance). Omit it for a single-mask
    # or single-instance array.
    inst_help = (
        "Instance index for a multi-instance (n_instances, H, W) mask, as "
        "saved by /detect. Optional when the array holds one mask."
    )

    p = sub.add_parser("iou", help="Intersection over Union between two masks")
    p.add_argument("--mask-a", required=True, help="Path to first mask .npy file")
    p.add_argument("--mask-b", required=True, help="Path to second mask .npy file")
    p.add_argument("--instance-a", type=int, default=None, help=inst_help)
    p.add_argument("--instance-b", type=int, default=None, help=inst_help)

    p = sub.add_parser("intersection", help="Element-wise AND of two masks")
    p.add_argument("--mask-a", required=True, help="Path to first mask .npy file")
    p.add_argument("--mask-b", required=True, help="Path to second mask .npy file")
    p.add_argument("--output", required=True, help="Path to save intersection mask .npy")
    p.add_argument("--instance-a", type=int, default=None, help=inst_help)
    p.add_argument("--instance-b", type=int, default=None, help=inst_help)

    p = sub.add_parser("centroid", help="Compute mask centroid")
    p.add_argument("--mask", required=True, help="Path to mask .npy file")
    p.add_argument("--instance", type=int, default=None, help=inst_help)

    p = sub.add_parser("area", help="Count True pixels in mask")
    p.add_argument("--mask", required=True, help="Path to mask .npy file")
    p.add_argument("--instance", type=int, default=None, help=inst_help)

    p = sub.add_parser("bbox", help="Compute tight bounding box")
    p.add_argument("--mask", required=True, help="Path to mask .npy file")
    p.add_argument("--instance", type=int, default=None, help=inst_help)
    p.add_argument(
        "--percentile",
        type=float,
        default=None,
        help=(
            "Optional: trim to the [p, 100-p] percentile of pixel coordinates "
            "instead of the exact extent. Rejects speckle but clips thin "
            "protrusions. Default is a truly tight box."
        ),
    )

    args = ap.parse_args()

    try:
        if args.cmd == "iou":
            mask_a = np.load(args.mask_a)
            mask_b = np.load(args.mask_b)
            result = {"iou": iou(mask_a, mask_b, args.instance_a, args.instance_b)}

        elif args.cmd == "intersection":
            mask_a = np.load(args.mask_a)
            mask_b = np.load(args.mask_b)
            result_mask = intersection(
                mask_a, mask_b, args.instance_a, args.instance_b
            )
            np.save(args.output, result_mask)
            result = {
                "output": args.output,
                "shape": list(result_mask.shape),
                "area": int(result_mask.sum()),
            }

        elif args.cmd == "centroid":
            mask = np.load(args.mask)
            cx, cy = centroid(mask, args.instance)
            if np.isnan(cx):
                result = {"centroid": None, "empty": True}
            else:
                result = {"centroid": {"x": cx, "y": cy}, "empty": False}

        elif args.cmd == "area":
            mask = np.load(args.mask)
            result = {"area": area(mask, args.instance)}

        elif args.cmd == "bbox":
            mask = np.load(args.mask)
            bbox = bounding_box(mask, args.instance, args.percentile)
            if bbox is None:
                result = {"bbox": None, "empty": True}
            else:
                result = {
                    "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
                    "empty": False,
                }

        else:
            return 1

        print(json.dumps(result, indent=2))
        return 0

    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
