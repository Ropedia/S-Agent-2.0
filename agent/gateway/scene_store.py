"""Per-trial 3D scene state: named instances over one reconstruction.

The scene is the persistent counterpart of the stateless artifact pipeline:
/recon creates it, /segment and /add_instance populate it, and the readout
endpoints (/distance, /depth, /compass, /list_instance) answer questions
from it by NAME instead of by artifact path.

Storage model (decided against a pure in-memory store on purpose):

* The source of truth is ONE JSON snapshot per trial,
  ``<work_dir>/artifacts/scene_state.json``, rewritten atomically on every
  mutation. The gateway is a single long-lived process shared by every
  concurrent trial and is REQUIRED to be restarted whenever its code
  changes (docs/RUNBOOK.md); state that lives only in memory would
  silently vanish for every trial in flight at that moment.
* The in-process cache in this module is exactly that -- a cache. Entries
  are keyed by the resolved artifacts directory, evicted LRU, and always
  reloadable from the snapshot.
* Heavy arrays (point maps, masks) are NOT duplicated into the scene.
  The snapshot stores their artifact paths; instance positions derived
  from masks are recomputed from the newest masks artifacts on read, so
  a /drop or a re-/segment can never leave the scene stale.

This module is deliberately gateway-agnostic: numpy plus the standard
library, no imports from perception_gateway (which imports us). Anything
that reads artifacts or talks to the GPU stays on the gateway side.
Validation failures raise ``SceneError``; the gateway maps it to HTTP 400.
"""

from __future__ import annotations

import copy
import json
import math
import os
import threading
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCENE_FILENAME = "scene_state.json"
SCENE_VERSION = 1

#: 5 cm occupancy grid for footprint centroids, matching /bev's
#: ``_GROUND_CELL_M`` -- one vote per occupied ground spot, so a vertical
#: face's stacked points cannot drag a centroid off the 2D footprint.
GROUND_CELL_M = 0.05

#: Same-label instances from different frames within this ground distance
#: are one physical object (matches /bev's merge_radius_m default).
MERGE_RADIUS_M = 0.75

_CACHE_MAX = 64


class SceneError(ValueError):
    """A caller-facing validation failure (gateway maps it to HTTP 400)."""


# ---------------------------------------------------------------------------
# Persistence: one JSON snapshot per trial, atomic replace, LRU memory cache.
# ---------------------------------------------------------------------------

_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


def _key(artifacts_dir: Path) -> str:
    return str(Path(artifacts_dir).resolve())


def lock_for(artifacts_dir: Path) -> threading.Lock:
    """Per-trial mutation lock. Handlers run in a threadpool, so scene
    mutations must serialize; the lock is per trial, not global."""
    key = _key(artifacts_dir)
    with _registry_lock:
        lock = _locks.get(key)
        if lock is None:
            lock = _locks[key] = threading.Lock()
        return lock


def reset_cache() -> None:
    """Test hook: drop every cached scene (snapshots on disk survive)."""
    with _registry_lock:
        _cache.clear()
        _locks.clear()


def new_scene() -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "version": SCENE_VERSION,
        "created": now,
        "updated": now,
        "frames": {},
        "recon": None,
        "camera_poses": {},
        "manual_instances": {},
        "gravity": None,
        "compass": None,
        "scale": {"factor": 1.0, "history": []},
    }


def load(artifacts_dir: Path) -> dict[str, Any] | None:
    """The trial's scene, or None if /recon never ran for it.

    Returns a deep copy: readers may inspect it freely, and mutators must
    go back through save() so the snapshot and cache stay in step.
    """
    key = _key(artifacts_dir)
    with _registry_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return copy.deepcopy(cached)
    path = Path(artifacts_dir) / SCENE_FILENAME
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise SceneError(f"scene snapshot at {path} is unreadable: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != SCENE_VERSION:
        raise SceneError(
            f"scene snapshot at {path} has an unsupported layout "
            f"(version {data.get('version') if isinstance(data, dict) else '?'})"
        )
    with _registry_lock:
        _cache[key] = copy.deepcopy(data)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return data


def save(artifacts_dir: Path, data: dict[str, Any]) -> None:
    """Atomically replace the snapshot and refresh the cache.

    The write mirrors the gateway's ArtifactWriter paranoia at the level a
    fixed-name mutable file allows: the directory is opened once with
    O_NOFOLLOW|O_DIRECTORY, the temp file is created O_CREAT|O_EXCL|
    O_NOFOLLOW relative to that descriptor, and the rename happens via the
    same descriptor -- a symlink swapped in after validation cannot
    redirect the write.
    """
    data = copy.deepcopy(data)
    data["updated"] = datetime.now(UTC).isoformat()
    payload = json.dumps(data, indent=2)
    key = _key(artifacts_dir)
    tmp_name = f".{SCENE_FILENAME}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        dir_fd = os.open(
            str(artifacts_dir), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
    except OSError as exc:
        raise SceneError(f"could not open {artifacts_dir} for writing: {exc}") from exc
    try:

        def _opener(_path: str, flags: int) -> int:
            return os.open(
                tmp_name, flags | os.O_NOFOLLOW | os.O_EXCL, 0o644, dir_fd=dir_fd
            )

        try:
            with open(tmp_name, "w", opener=_opener, encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_name, SCENE_FILENAME, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        except OSError as exc:
            with _suppress_oserror():
                os.unlink(tmp_name, dir_fd=dir_fd)
            raise SceneError(f"could not write scene snapshot: {exc}") from exc
    finally:
        os.close(dir_fd)
    with _registry_lock:
        _cache[key] = data
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)


class _suppress_oserror:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, OSError)


def scale_factor(scene: dict[str, Any] | None) -> float:
    if not scene:
        return 1.0
    try:
        return float((scene.get("scale") or {}).get("factor", 1.0))
    except (TypeError, ValueError):
        return 1.0


# ---------------------------------------------------------------------------
# Geometry helpers. World conventions: positions in the reconstruction's
# (raw, uncalibrated) world coordinates; "down" is the unit gravity
# direction in world coordinates. Instance orientation is the orthonormal
# triple (forward, right, up) with down = forward x right -- the same
# handedness as camera axes (+x right, +y down, +z forward).
# ---------------------------------------------------------------------------


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-9:
        raise SceneError("degenerate direction vector (zero length)")
    return v / n


def horizontal(v: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Component of v perpendicular to gravity (NOT normalized)."""
    v = np.asarray(v, dtype=np.float64).ravel()
    down = np.asarray(down, dtype=np.float64).ravel()
    return v - down * float(v @ down)


def frame_from_forward(
    forward: np.ndarray, up_hint: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (forward, right, up) from a forward direction and an up
    hint (typically -down from gravity). down = forward x right."""
    f = unit(forward)
    u_hint = unit(up_hint)
    if abs(float(f @ u_hint)) > 0.999:
        raise SceneError(
            "forward direction is (anti)parallel to up -- the orientation "
            "is underdetermined; pick a target that is not straight above "
            "or below the instance"
        )
    r = unit(np.cross(f, u_hint))
    u = np.cross(r, f)
    return f, r, u


_AXIS_SIGNS = {
    "forward": ("forward", 1.0),
    "back": ("forward", -1.0),
    "right": ("right", 1.0),
    "left": ("right", -1.0),
    "top": ("up", 1.0),
    "bottom": ("up", -1.0),
}


def frame_from_primary_secondary(
    primary_axis: str,
    primary_dir: np.ndarray,
    secondary_axis: str,
    secondary_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (forward, right, up) from a PRIMARY axis (taken exactly)
    and a SECONDARY direction (Gram-Schmidt projected onto the plane
    orthogonal to the primary). Axis names take either member of each pair
    (back/left/bottom flip the sign); the two must belong to different
    pairs. The third axis completes the right-handed triple
    (down = forward x right)."""
    canon_p, sign_p = _AXIS_SIGNS[primary_axis]
    canon_s, sign_s = _AXIS_SIGNS[secondary_axis]
    if canon_p == canon_s:
        raise SceneError(
            f"primary {primary_axis} and secondary {secondary_axis} lie on "
            "the same body axis -- they cannot span a frame; pick the "
            "secondary from a different pair"
        )
    p = unit(np.asarray(primary_dir, dtype=np.float64)) * sign_p
    s_raw = unit(np.asarray(secondary_dir, dtype=np.float64)) * sign_s
    if abs(float(p @ s_raw)) > 0.999:
        raise SceneError(
            "the secondary direction is (anti)parallel to the primary axis "
            "-- it cannot fix the frame's remaining rotation; pick a "
            "secondary target off the primary's line"
        )
    s = unit(s_raw - p * float(s_raw @ p))
    axes = {canon_p: p, canon_s: s}
    if "forward" in axes and "up" in axes:
        axes["right"] = np.cross(axes["forward"], axes["up"])
    elif "forward" in axes and "right" in axes:
        axes["up"] = np.cross(axes["right"], axes["forward"])
    else:
        axes["forward"] = np.cross(axes["up"], axes["right"])
    return axes["forward"], axes["right"], axes["up"]


def frame_from_targets(
    forward_dir: np.ndarray, right_dir: np.ndarray, top_dir: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (forward, right, up) from three rough target directions.

    forward is primary; right is orthogonalized against it; up completes
    the right-handed triple (down = forward x right) and must agree with
    the caller's top direction -- a triple that comes out left-handed is a
    caller error worth stopping on, not silently flipping.
    """
    f_raw, r_raw, t_raw = unit(forward_dir), unit(right_dir), unit(top_dir)
    if abs(float(f_raw @ r_raw)) > 0.966:
        raise SceneError(
            "forward and right targets are nearly collinear with the "
            "instance (under ~15 deg apart) -- they cannot span a "
            "horizontal frame; pick two targets in clearly different "
            "directions"
        )
    f = f_raw
    r = unit(r_raw - f * float(r_raw @ f))
    u = np.cross(r, f)
    if float(u @ t_raw) < 0.0:
        raise SceneError(
            "the forward/right/top targets describe a LEFT-handed frame: "
            "with that forward and right, 'up' points away from the top "
            "target. Swap right, or pick a top target genuinely above the "
            "instance"
        )
    if float(u @ t_raw) < 0.5:
        # > 60 deg off: honest but suspicious; still an error -- a top
        # anchor this far from the derived up usually means a mis-picked
        # reference, and a silently accepted one poisons every /depth
        # readout made from this instance.
        raise SceneError(
            "the top target is over 60 deg away from the up direction "
            "implied by forward/right -- the three references are "
            "inconsistent; re-pick them"
        )
    return f, r, u


def ground_cells(us: np.ndarray, vs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate ground samples onto the 5 cm grid (one vote per cell)."""
    cells = np.unique(
        np.stack([np.floor(us / GROUND_CELL_M), np.floor(vs / GROUND_CELL_M)], axis=1),
        axis=0,
    )
    return (
        (cells[:, 0] + 0.5) * GROUND_CELL_M,
        (cells[:, 1] + 0.5) * GROUND_CELL_M,
    )


def horizontal_basis(down: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic orthonormal basis (e1, e2) of the ground plane with
    e2 = down x e1. Any fixed choice works -- bearings are always taken
    RELATIVE to a north vector expressed in the same basis."""
    down = unit(down)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ down)) > 0.9:
        seed = np.array([0.0, 0.0, 1.0])
    e1 = unit(horizontal(seed, down))
    e2 = np.cross(down, e1)
    return e1, e2


def to_ground(p: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).ravel()
    return np.array([float(p @ e1), float(p @ e2)])


def east_2d(north_2d: np.ndarray) -> np.ndarray:
    """East in ground coordinates: cross(down, north) expressed in the
    (e1, e2) basis with e2 = down x e1 comes out as (-n2, n1)."""
    return np.array([-north_2d[1], north_2d[0]])


def bearing_deg_2d(vec_2d: np.ndarray, north_2d: np.ndarray) -> float:
    """Compass bearing of a ground vector: degrees clockwise from north,
    viewed from above."""
    n = vec_2d[0] * north_2d[0] + vec_2d[1] * north_2d[1]
    e = vec_2d[0] * east_2d(north_2d)[0] + vec_2d[1] * east_2d(north_2d)[1]
    return math.degrees(math.atan2(e, n)) % 360.0


def north_2d_from_arrow(arrow_2d: np.ndarray, offset_deg: float) -> np.ndarray:
    """North on the ground plane, given an arrow whose declared compass
    bearing is offset_deg (e.g. 225 for southwest)."""
    w = np.asarray(arrow_2d, dtype=np.float64)
    norm = float(np.hypot(w[0], w[1]))
    if norm < 1e-9:
        raise SceneError("compass anchor arrow has no horizontal component")
    w = w / norm
    c, s = math.cos(math.radians(offset_deg)), math.sin(math.radians(offset_deg))
    # rotate w by -offset (clockwise bearings): north = cos*w - sin*east(w)
    e = east_2d(w)
    return np.array([c * w[0] - s * e[0], c * w[1] - s * e[1]])


def robust_position(pts: np.ndarray, down: np.ndarray | None) -> np.ndarray:
    """Robust 3D representative of an instance's world points.

    With gravity known: footprint-cell median on the ground plane (one
    vote per occupied 5 cm cell, /bev's convention -- a vertical face's
    point pile must not drag the centroid) plus the median height. Without
    gravity: the plain coordinate-wise median.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        raise SceneError("no 3D points to position an instance on")
    if down is None:
        return np.median(pts, axis=0)
    d = unit(down)
    e1, e2 = horizontal_basis(d)
    us, vs = pts @ e1, pts @ e2
    cu, cv = ground_cells(us, vs)
    u_med, v_med = float(np.median(cu)), float(np.median(cv))
    h_med = float(np.median(pts @ d))
    return e1 * u_med + e2 * v_med + d * h_med


def ground_line_fit(
    pts: np.ndarray, down: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """PCA line through an instance's ground footprint: (p0_2d, dir_2d,
    t_lo, t_hi) in the horizontal_basis(down) coordinates, percentile
    trimmed like /bev's plane fit."""
    d = unit(down)
    e1, e2 = horizontal_basis(d)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    us, vs = ground_cells(pts @ e1, pts @ e2)
    p0 = np.array([float(np.median(us)), float(np.median(vs))])
    du, dv = us - us.mean(), vs - vs.mean()
    cov = np.array(
        [
            [float(np.mean(du * du)), float(np.mean(du * dv))],
            [float(np.mean(du * dv)), float(np.mean(dv * dv))],
        ]
    )
    evecs = np.linalg.eigh(cov)[1]
    axis = evecs[:, -1]
    t = (np.stack([us, vs], axis=1) - p0) @ axis
    return p0, axis, float(np.percentile(t, 2.5)), float(np.percentile(t, 97.5))


def aggregate_gravity(
    per_frame: list[dict[str, Any]], top_k: int = 3
) -> dict[str, Any]:
    """Fuse per-frame GeoCalib estimates into one world down vector.

    Ranks frames by GeoCalib's own uncertainty (confidence), keeps the
    top_k best, and averages their world-frame down vectors. The spread
    (max pairwise angle among the kept vectors) is recorded as an honest
    quality fact -- it is NOT surfaced as a reply field the model could
    treat as licence to override readouts.
    """
    usable = [
        f
        for f in per_frame
        if f.get("down_world") is not None and f.get("error") is None
    ]
    if not usable:
        raise SceneError("gravity estimation failed on every frame")

    def _unc(f: dict[str, Any]) -> float:
        u = f.get("uncertainty_deg")
        try:
            u = float(u)
        except (TypeError, ValueError):
            return 1e6
        return u if math.isfinite(u) else 1e6

    ranked = sorted(usable, key=_unc)
    kept = ranked[: max(1, top_k)]
    vecs = [unit(np.asarray(f["down_world"], dtype=np.float64)) for f in kept]
    mean = np.mean(np.stack(vecs), axis=0)
    down = unit(mean)
    spread = 0.0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dot = max(-1.0, min(1.0, float(vecs[i] @ vecs[j])))
            spread = max(spread, math.degrees(math.acos(dot)))
    return {
        "down_world": [round(float(x), 6) for x in down],
        "used_frames": [int(f["frame"]) for f in kept],
        "spread_deg": round(spread, 2),
        "method": f"geocalib, top {len(kept)} of {len(usable)} frames by uncertainty",
    }


# ---------------------------------------------------------------------------
# Extents and merging over instance point clouds (/size, /distance
# --closest/--farthest, /count). Pure numpy, unit-tested without the gateway.
# ---------------------------------------------------------------------------

#: Fraction trimmed at EACH end of a cloud before an extent is read
#: (2.5 % / 97.5 % percentiles, ground_line_fit's convention). A segment mask
#: bleeds a thin tail of far points onto whatever stands behind the object
#: (flying pixels at depth edges); a raw min/max would measure the tail.
#: Used by the closest/farthest search.
CLOUD_TRIM = 0.025
#: /size's box uses a lighter trim: on 34 real ReVSI objects the 2.5 % box
#: read a median 0.91 of the annotated longest dimension (8 of 34 under
#: 0.8, 2 over 1.25) -- the trim, not the tail, was setting the length.
BOX_TRIM = 0.01

#: Points kept per cloud for the pairwise closest/farthest search: two
#: clouds of this size are a 6M-entry distance matrix, computed in chunks.
EXTREMES_MAX_PTS = 2500


def _trimmed_span(values: np.ndarray, trim: float = CLOUD_TRIM) -> tuple[float, float]:
    lo = float(np.percentile(values, 100.0 * trim))
    hi = float(np.percentile(values, 100.0 * (1.0 - trim)))
    return lo, hi


def convex_hull_2d(pts2: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain: hull vertices, counter-clockwise, (k, 2)."""
    p = np.unique(np.asarray(pts2, dtype=np.float64).reshape(-1, 2), axis=0)
    if p.shape[0] <= 2:
        return p
    order = np.lexsort((p[:, 1], p[:, 0]))
    p = p[order]

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for q in p:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list[np.ndarray] = []
    for q in p[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.asarray(lower[:-1] + upper[:-1])


def min_area_rect_axis(pts2: np.ndarray) -> np.ndarray:
    """Unit direction of the minimum-area bounding rectangle of 2D points
    (rotating calipers over the convex hull). Degenerate inputs (a point,
    a line) fall back to the principal axis."""
    p = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    hull = convex_hull_2d(p)
    best_axis = None
    best_area = math.inf
    n = hull.shape[0]
    if n >= 3:
        for i in range(n):
            e = hull[(i + 1) % n] - hull[i]
            norm = float(np.hypot(e[0], e[1]))
            if norm < 1e-9:
                continue
            u = e / norm
            v = np.array([-u[1], u[0]])
            pu, pv = hull @ u, hull @ v
            area = float((pu.max() - pu.min()) * (pv.max() - pv.min()))
            if area < best_area - 1e-12:
                best_area, best_axis = area, u
    if best_axis is None:
        c = p - p.mean(axis=0)
        if c.shape[0] < 2 or float(np.abs(c).max()) < 1e-9:
            return np.array([1.0, 0.0])
        cov = c.T @ c
        best_axis = np.linalg.eigh(cov)[1][:, -1]
        best_axis = best_axis / max(float(np.hypot(*best_axis)), 1e-12)
    return best_axis


def oriented_bbox(pts: np.ndarray, down: np.ndarray | None) -> dict[str, Any]:
    """Oriented bounding box of an instance cloud, trimmed extents.

    With gravity known the box is GRAVITY-ALIGNED: one axis is vertical
    (`height`), the other two are the minimum-area rectangle of the ground
    footprint (`length` >= `width`). That is what "length, width or height"
    means for anything standing in a room -- a full 3D PCA box would tilt
    with a mask's noise and mix the axes. Without gravity: PCA axes.

    Extents are 1-99 percentile spans along each axis, never raw min/max
    (see BOX_TRIM; a uniform solid reads ~2 % short, a thin tail of flying
    pixels is ignored). Returns axes (3, 3)
    world unit vectors (rows: length, width, height), extents (3,) in the
    cloud's own units, centre (3,), and n_points used.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        raise SceneError("no 3D points to fit a box to")
    if down is not None:
        d = unit(down)
        e1, e2 = horizontal_basis(d)
        g = np.stack([pts @ e1, pts @ e2], axis=1)
        u = min_area_rect_axis(g)
        v = np.array([-u[1], u[0]])
        gu, gv, h = g @ u, g @ v, pts @ d
        (u_lo, u_hi), (v_lo, v_hi), (h_lo, h_hi) = (
            _trimmed_span(gu, BOX_TRIM),
            _trimmed_span(gv, BOX_TRIM),
            _trimmed_span(h, BOX_TRIM),
        )
        axes_w = [u[0] * e1 + u[1] * e2, v[0] * e1 + v[1] * e2, -d]
        extents = [u_hi - u_lo, v_hi - v_lo, h_hi - h_lo]
        if extents[1] > extents[0]:
            axes_w[0], axes_w[1] = axes_w[1], axes_w[0]
            extents[0], extents[1] = extents[1], extents[0]
            u_lo, u_hi, v_lo, v_hi = v_lo, v_hi, u_lo, u_hi
        mid = (
            e1 * 0.0
            + (axes_w[0] * ((u_lo + u_hi) / 2.0))
            + (axes_w[1] * ((v_lo + v_hi) / 2.0))
            + d * ((h_lo + h_hi) / 2.0)
        )
        return {
            "axes": np.asarray(axes_w, dtype=np.float64),
            "extents": np.asarray(extents, dtype=np.float64),
            "centre": mid,
            "n_points": int(pts.shape[0]),
            "gravity_aligned": True,
        }
    c = pts.mean(axis=0)
    cov = (pts - c).T @ (pts - c)
    evecs = np.linalg.eigh(cov)[1][:, ::-1].T  # rows: major .. minor
    proj = (pts - c) @ evecs.T
    spans = [_trimmed_span(proj[:, k], BOX_TRIM) for k in range(3)]
    extents = np.array([hi - lo for lo, hi in spans])
    order = np.argsort(-extents)
    mid = c + sum(evecs[k] * ((spans[k][0] + spans[k][1]) / 2.0) for k in range(3))
    return {
        "axes": evecs[order],
        "extents": extents[order],
        "centre": mid,
        "n_points": int(pts.shape[0]),
        "gravity_aligned": False,
    }


def _subsample(pts: np.ndarray, max_pts: int) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] > max_pts:
        pts = pts[:: pts.shape[0] // max_pts + 1]
    return pts


def cloud_extremes(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Closest and farthest approach between two clouds, outlier-tolerant.

    Per point of A its nearest and farthest distance to B (and vice versa).
    closest = the LARGER of the two clouds' trimmed-low nearest distances,
    farthest = the SMALLER of the two trimmed-high farthest distances: a few
    stray points on either side (a mask bleeding toward the other object, a
    flying pixel far behind it) are the extreme entries of exactly one of
    the two per-cloud arrays, and the max/min picks the array the strays are
    not in. Raw min/max are returned alongside for reference.
    """
    a = _subsample(a, EXTREMES_MAX_PTS)
    b = _subsample(b, EXTREMES_MAX_PTS)
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise SceneError("an instance without 3D points cannot be measured")
    min_a = np.full(a.shape[0], np.inf)
    max_a = np.zeros(a.shape[0])
    min_b = np.full(b.shape[0], np.inf)
    max_b = np.zeros(b.shape[0])
    step = max(1, 2_000_000 // max(b.shape[0], 1))
    for i in range(0, a.shape[0], step):
        d = np.sqrt(((a[i : i + step, None, :] - b[None, :, :]) ** 2).sum(-1))
        min_a[i : i + step] = d.min(axis=1)
        max_a[i : i + step] = d.max(axis=1)
        min_b = np.minimum(min_b, d.min(axis=0))
        max_b = np.maximum(max_b, d.max(axis=0))
    lo_q = 100.0 * CLOUD_TRIM
    hi_q = 100.0 * (1.0 - CLOUD_TRIM)
    return {
        "closest": max(
            float(np.percentile(min_a, lo_q)), float(np.percentile(min_b, lo_q))
        ),
        "farthest": min(
            float(np.percentile(max_a, hi_q)), float(np.percentile(max_b, hi_q))
        ),
        "raw_min": float(min_a.min()),
        "raw_max": float(max_a.max()),
        "n_a": int(a.shape[0]),
        "n_b": int(b.shape[0]),
    }
