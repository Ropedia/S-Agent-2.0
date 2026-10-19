#!/usr/local/bin/python3
# Interpreter pinned to the TASK image's python, not `env python3`.
# The agent runs with the openhands-sdk runtime mounted at
# /opt/custom-agent-runtime/oh-sdk, whose bin/ comes FIRST on PATH and
# ships its own python 3.12.8 WITHOUT numpy. `env python3` resolves there,
# so `python3 /app/geom.py` died with ModuleNotFoundError: numpy on 17
# calls in the m0a run alone -- while `pip show numpy` inside the task
# image says 2.2.6 is installed, which is why it looked impossible.
# /usr/local/bin/python3 is the task image's own interpreter (3.11.16).
"""Geometric computation utilities (CPU-only, no external dependencies).

All operations work on numpy arrays and return native Python types or numpy arrays.
Coordinate conventions:
- 3D points/vectors: (x, y, z) in world or camera coordinates (meters)
- Camera coordinates: +X right, +Y down, +Z forward (camera looking along +Z)
- World coordinates: application-dependent; extrinsics define camera-to-world transform
- Pixel coordinates: (u, v) where u is horizontal (column), v is vertical (row), origin at top-left
- Intrinsics: fx, fy (focal lengths in pixels), cx, cy (principal point in pixels)

Usage:
    # As a library:
    import geom
    dist = geom.euclidean_distance([0, 0, 1], [0, 0, 2])

    # As a CLI (outputs JSON):
    /app/geom.py distance --point-a 0 0 1 --point-b 0 0 2
    /app/geom.py fit-ground-plane --points /path/to/points.npy --confidence /path/to/conf.npy
"""

import argparse
import json
import sys

import numpy as np


def euclidean_distance(p1, p2) -> float:
    """Euclidean distance between two 3D points.

    Args:
        p1: (3,) array-like, first point in meters
        p2: (3,) array-like, second point in meters

    Returns:
        float: distance in meters
    """
    p1 = np.asarray(p1, dtype=np.float64).ravel()
    p2 = np.asarray(p2, dtype=np.float64).ravel()
    if p1.shape != p2.shape or p1.shape[0] != 3:
        raise ValueError(f"Both points must be 3D, got shapes {p1.shape} and {p2.shape}")
    return float(np.linalg.norm(p1 - p2))


def angle_between_vectors(v1, v2) -> float:
    """Angle in degrees between two 3D vectors.

    Args:
        v1: (3,) array-like vector
        v2: (3,) array-like vector

    Returns:
        float: angle in degrees [0, 180]

    Raises:
        ValueError: if either vector has zero length
    """
    v1 = np.asarray(v1, dtype=np.float64).ravel()
    v2 = np.asarray(v2, dtype=np.float64).ravel()
    if v1.shape[0] != 3 or v2.shape[0] != 3:
        raise ValueError(f"Both vectors must be 3D, got shapes {v1.shape} and {v2.shape}")
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        raise ValueError(f"Cannot compute angle with zero-length vector: ||v1||={n1:.2e}, ||v2||={n2:.2e}")
    cos_angle = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def select_pose(poses, index: int | None = None) -> np.ndarray:
    """Pick one (4,4) camera-to-world matrix out of a saved pose array.

    /reconstruct saves `arrays.camera_poses` as an (N, 4, 4) stack, one row per
    entry of the reply's `frame_indices`, so a file loaded from there is 3D
    even for a single-image reconstruction. Accepts either form:

    * (4, 4): returned as-is.
    * (N, 4, 4): row `index`; N == 1 is unwrapped without needing an index.
      With N > 1 and no index this raises rather than assuming row 0 -- the
      wrong extrinsic yields a plausible-looking pixel that is simply wrong.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape == (4, 4):
        if index not in (None, 0):
            raise ValueError(
                f"extrinsic is a single (4,4) matrix, so --pose-index {index} "
                f"does not apply (only 0 is valid)"
            )
        return poses
    if poses.ndim == 3 and poses.shape[1:] == (4, 4):
        n = poses.shape[0]
        if index is None:
            if n == 1:
                return poses[0]
            raise ValueError(
                f"extrinsic holds {n} poses (shape {poses.shape}); pass "
                f"--pose-index <0..{n - 1}> to choose one"
            )
        if not 0 <= index < n:
            raise ValueError(
                f"--pose-index {index} out of range for {n} poses "
                f"(shape {poses.shape})"
            )
        return poses[index]
    raise ValueError(
        f"extrinsic must be (4,4) or (N,4,4) camera-to-world, got shape {poses.shape}"
    )


def project_point_to_camera(
    point_3d,
    extrinsic_c2w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float] | None:
    """Project a 3D world point onto the camera image plane.

    Args:
        point_3d: (3,) world coordinates in meters
        extrinsic_c2w: (4, 4) camera-to-world SE(3) transformation matrix
        fx: focal length in pixels (horizontal)
        fy: focal length in pixels (vertical)
        cx: principal point x-coordinate in pixels
        cy: principal point y-coordinate in pixels

    Returns:
        (u, v) pixel coordinates, or None if the point is behind the camera (z_cam <= 0)

    Coordinate conventions:
    - World point is transformed to camera coordinates via world-to-camera (inverse of c2w)
    - Camera coordinates: +Z forward (looking direction), +X right, +Y down
    - Point must have positive Z in camera space to be in front of camera
    - Pixel (u, v): u is column (horizontal), v is row (vertical), origin at top-left
    """
    point_3d = np.asarray(point_3d, dtype=np.float64).ravel()
    if point_3d.shape[0] != 3:
        raise ValueError(f"point_3d must be (3,), got shape {point_3d.shape}")
    extrinsic_c2w = np.asarray(extrinsic_c2w, dtype=np.float64)
    if extrinsic_c2w.shape != (4, 4):
        raise ValueError(f"extrinsic_c2w must be (4, 4), got shape {extrinsic_c2w.shape}")

    # World to camera: invert the camera-to-world transform
    w2c = np.linalg.inv(extrinsic_c2w)
    p_homo = np.append(point_3d, 1.0)
    p_cam = w2c @ p_homo

    # Behind camera check
    if p_cam[2] <= 0:
        return None

    # Perspective projection
    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy
    return (float(u), float(v))


#: Cap on points used per RANSAC iteration. See the comment in
#: fit_ground_plane_ransac() -- the inner loop is O(points * iterations), and
#: the uncapped version did not finish inside the agent's 120 s tool timeout on
#: a real 4-frame reconstruction.
_RANSAC_MAX_FIT_POINTS = 50_000


def fit_ground_plane_ransac(
    points: np.ndarray,
    confidence: np.ndarray,
    conf_threshold: float = 0.3,
    n_iterations: int = 1000,
    inlier_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """RANSAC-based ground plane fitting.

    Args:
        points: (H, W, 3) or (N, 3) point cloud in meters
        confidence: matching shape without last dim, confidence score per point [0, 1]
        conf_threshold: minimum confidence to include a point in fitting
        n_iterations: number of RANSAC iterations
        inlier_threshold: maximum distance (meters) from plane to be considered inlier

    Returns:
        (plane_normal, inlier_mask) where:
            - plane_normal: (3,) unit vector perpendicular to the plane
            - inlier_mask: boolean array same shape as flattened valid_pts
        Or (None, None) if fitting fails (too few valid points or inliers)

    The plane normal sign is ambiguous (both n and -n represent the same plane).
    Caller should disambiguate using domain knowledge (e.g., camera "up" direction).

    Method: Fit a plane ax + by + cz = d to the point cloud. The normal is (a, b, c).
    Uses RANSAC to reject outliers, then refines with SVD least-squares on all inliers.
    """
    pts = points.reshape(-1, 3)
    conf = confidence.reshape(-1)
    if pts.shape[0] != conf.shape[0]:
        raise ValueError(
            f"points and confidence must have matching counts: "
            f"{pts.shape[0]} vs {conf.shape[0]}"
        )

    mask = conf > conf_threshold
    valid_pts = pts[mask]

    if len(valid_pts) < 100:
        return None, None

    best_normal = None
    best_inliers = 0
    best_mask = None
    best_origin = None
    rng = np.random.default_rng(42)

    # Subsample for the ITERATION phase. The inner loop projects every point
    # onto a candidate plane, so its cost is O(n_points * n_iterations): a real
    # 4-frame reconstruction is 115 MB / ~47M points, and 1000 iterations over
    # that did not finish in 120 s (the agent's tool timeout) -- measured on
    # this project's own artifacts. It looked fast in testing only because the
    # artifact used there was 1.4 MB.
    #
    # RANSAC does not need every point to FIND a plane; it needs enough to make
    # the dominant plane the most-supported hypothesis. 50k random points is
    # several orders of magnitude above the 3 needed per hypothesis and keeps
    # the dominant-surface ordering intact. Inliers are then counted on the
    # FULL set once, so the returned mask and n_inliers still describe all
    # points -- callers that threshold on n_inliers keep their meaning.
    fit_pts = valid_pts
    if len(valid_pts) > _RANSAC_MAX_FIT_POINTS:
        sel = rng.choice(len(valid_pts), size=_RANSAC_MAX_FIT_POINTS, replace=False)
        fit_pts = valid_pts[sel]

    for _ in range(n_iterations):
        idx = rng.choice(len(fit_pts), size=3, replace=False)
        p0, p1, p2 = fit_pts[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm

        # Scored on the subsample; the winner is re-scored on everything below.
        dists = np.abs(np.dot(fit_pts - p0, normal))
        inlier_mask = dists < inlier_threshold
        n_inliers = int(inlier_mask.sum())

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_normal = normal
            best_origin = p0

    if best_normal is None or best_inliers < 50:
        return None, None

    # Re-score the winning hypothesis on EVERY valid point, so the returned
    # mask and inlier count describe the whole cloud rather than the subsample.
    # Callers threshold on n_inliers (perception.py's up_vector() requires
    # >= 1000), so this has to be the full-set number or that gate shifts
    # meaning with cloud size.
    best_mask = np.abs(np.dot(valid_pts - best_origin, best_normal)) < inlier_threshold
    if int(best_mask.sum()) < 50:
        return None, None

    # Refit plane from all inliers using SVD (least-squares)
    inlier_pts = valid_pts[best_mask]
    centroid = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    best_normal = vh[-1]  # last right singular vector = plane normal

    return best_normal, best_mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("distance", help="Euclidean distance between two 3D points")
    p.add_argument("--point-a", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--point-b", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))

    p = sub.add_parser("angle", help="Angle in degrees between two 3D vectors")
    p.add_argument("--vector-a", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--vector-b", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))

    p = sub.add_parser("project", help="Project 3D world point to 2D pixel coordinates")
    p.add_argument("--point", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    p.add_argument(
        "--extrinsic",
        required=True,
        help=(
            "Path to a camera-to-world .npy file: either a single (4,4) matrix "
            "or the (N,4,4) stack /reconstruct saves as arrays.camera_poses "
            "(use --pose-index to pick a row)"
        ),
    )
    p.add_argument(
        "--pose-index",
        type=int,
        default=None,
        help=(
            "Row to use from an (N,4,4) extrinsic stack. This is a position in "
            "the array, matching frame_indices[i] from the /reconstruct reply, "
            "not an absolute frame index. Optional when N == 1."
        ),
    )
    p.add_argument("--fx", type=float, required=True, help="Focal length X (pixels)")
    p.add_argument("--fy", type=float, required=True, help="Focal length Y (pixels)")
    p.add_argument("--cx", type=float, required=True, help="Principal point X (pixels)")
    p.add_argument("--cy", type=float, required=True, help="Principal point Y (pixels)")

    p = sub.add_parser("fit-ground-plane", help="RANSAC ground plane fitting")
    p.add_argument("--points", required=True, help="Path to points .npy array (H,W,3) or (N,3)")
    p.add_argument("--confidence", required=True, help="Path to confidence .npy array")
    p.add_argument("--conf-threshold", type=float, default=0.3)
    p.add_argument("--n-iterations", type=int, default=1000)
    p.add_argument("--inlier-threshold", type=float, default=0.05)

    args = ap.parse_args()

    try:
        if args.cmd == "distance":
            result = {"distance": euclidean_distance(args.point_a, args.point_b)}
        elif args.cmd == "angle":
            result = {"angle_degrees": angle_between_vectors(args.vector_a, args.vector_b)}
        elif args.cmd == "project":
            extrinsic = select_pose(np.load(args.extrinsic), args.pose_index)
            uv = project_point_to_camera(
                args.point, extrinsic, args.fx, args.fy, args.cx, args.cy
            )
            if uv is None:
                result = {"pixel": None, "behind_camera": True}
            else:
                result = {"pixel": {"u": uv[0], "v": uv[1]}, "behind_camera": False}
        elif args.cmd == "fit-ground-plane":
            points = np.load(args.points)
            confidence = np.load(args.confidence)
            normal, mask = fit_ground_plane_ransac(
                points,
                confidence,
                conf_threshold=args.conf_threshold,
                n_iterations=args.n_iterations,
                inlier_threshold=args.inlier_threshold,
            )
            if normal is None:
                result = {"success": False, "normal": None, "n_inliers": 0}
            else:
                result = {
                    "success": True,
                    "normal": normal.tolist(),
                    "n_inliers": int(mask.sum()),
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
