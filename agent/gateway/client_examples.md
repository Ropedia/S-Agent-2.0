# Perception API (curl reference)

A 3D perception service reachable over plain HTTP. It gives you metric
reconstruction, open-vocabulary segmentation, numeric 3D measurement, and PNG
rendering. No GPU, no Python libraries, and no image uploads are needed on your
side: you pass **file paths**, and you get **file paths** back.

Base URL: `http://host.docker.internal:8090` (from inside a container)
Shared work dir: `/work` (host: `/scratch/ywxzml3j/wry/action/agent/work`)

Every image you reference and every artifact you get back lives in the shared
work dir. Use either `/work/...` or the host path; responses come back in the
same namespace you used.

## The frame-identity contract (read this)

Every array-producing response includes `frame_indices`: the **absolute** frame
numbers that array describes. Each saved `.npy` also has a `<file>.npy.json`
sidecar recording the same thing.

When you combine arrays (e.g. measure a mask against a reconstruction), the
server checks that the frame you are measuring exists in **all** inputs. If it
does not, you get **HTTP 409** instead of a plausible-looking wrong number.
Fix a 409 by re-running detection on a frame the reconstruction actually covers.

## Reading the status code

| Status | Meaning | What to do |
|---|---|---|
| 200 `"found": true` | Worked, results attached | Use them |
| 200 `"found": false` | Ran fine, nothing matched | Reword the prompt or try another frame. This is a real observation, not an error |
| 400 | Your request was malformed | Fix the arguments |
| 409 | Frame mismatch | Align the frames |
| 502 / 503 | The model or server failed | Retry once, then work without the tool |

An empty detection is **never** an error. Treat `found: false` as evidence.

---

## GET /health

```bash
curl -s http://host.docker.internal:8090/health
```

```json
{"ok":true,"upstream":"http://127.0.0.1:8078","tools":["reconstruct","detect","render","measure"]}
```

## POST /reconstruct

Metric 3D reconstruction over one or more images.

```bash
curl -s -X POST http://host.docker.internal:8090/reconstruct \
  -H 'Content-Type: application/json' \
  -d '{"images":["/work/f0.png","/work/f1.png"],"frame_indices":[0,1]}'
```

Returns `metric_scale`, a 4x4 camera-to-world pose and 3x3 intrinsic matrix per
frame (keyed by absolute frame index), and paths to `.npy` arrays:

```json
{
  "found": true,
  "frame_indices": [0, 1],
  "metric_scale": 0.9073,
  "image_size": {"height": 180, "width": 320},
  "arrays": {
    "points": "/work/artifacts/points_01e4a1c418b9.npy",
    "depth": "/work/artifacts/depth_2f1c9ab7de40.npy",
    "confidence": "/work/artifacts/confidence_88b0e1f4a2c3.npy"
  },
  "camera_poses": {"0": [[...]], "1": [[...]]},
  "intrinsics": {"0": [[...]], "1": [[...]]},
  "depth_stats": {"finite_fraction": 1.0, "min": 0.1597, "max": 1.9141, "mean": 0.6012}
}
```

* `points` is `(N, H, W, 3)` world-space XYZ in **metres**
* `depth` is `(N, H, W)` camera-space z in **metres**
* Arrays are resampled to your input image size, so pixel `(y, x)` in a mask
  indexes the same pixel in the point map.

`frame_indices` is optional; it defaults to `[0, 1, 2, ...]`. Use real absolute
frame numbers if your images come from a longer sequence.

## POST /detect

Open-vocabulary detection and segmentation on a single image.

```bash
curl -s -X POST http://host.docker.internal:8090/detect \
  -H 'Content-Type: application/json' \
  -d '{"image":"/work/f0.png","prompt":"sink","frame_index":0}'
```

```json
{
  "found": true,
  "n": 1,
  "frame_indices": [0],
  "mask_size": {"height": 180, "width": 320},
  "arrays": {"masks": "/work/artifacts/masks_bc02594729a3.npy"},
  "box_format": "xyxy pixels, derived from the mask extent",
  "instances": [
    {"index": 0, "label": "sink", "score": 0.9766,
     "box_xyxy": [68.0, 38.0, 147.0, 78.0],
     "mask_pixels": 2335, "mask_coverage": 0.0405}
  ]
}
```

`masks` is `(n_instances, H, W)` boolean. Boxes are `xyxy` pixels derived from
the mask itself.

A prompt that matches nothing returns **HTTP 200**:

```bash
curl -s -X POST http://host.docker.internal:8090/detect \
  -H 'Content-Type: application/json' \
  -d '{"image":"/work/f0.png","prompt":"spaceship"}'
```

```json
{"found": false, "n": 0, "frame_indices": [0], "message": "The detector ran successfully and found no match ..."}
```

Point prompts (normalized 0-1, `1` = foreground) and box prompts
(normalized `xywh`) also work:

```bash
curl -s -X POST http://host.docker.internal:8090/detect \
  -H 'Content-Type: application/json' \
  -d '{"image":"/work/f0.png","points":[[0.334,0.322]],"point_labels":[1],"frame_index":0}'
```

## POST /segment

Prompted segmentation of **named objects** — `prompts` is required: list the
objects the question mentions, in the stem AND the answer options. One
open-vocabulary detection per prompt (all instances of each concept), merged
across prompts and mask-IoU-deduped, so two prompts naming the same object
yield one instance. One call per frame. (The former no-prompt automatic mode
is gone: blanket segment-everything filled `/bev` with anonymous "region"
labels.)

```bash
curl -s -X POST http://host.docker.internal:8090/segment \
  -H 'Content-Type: application/json' \
  -d '{"image":"/work/f0.png","frame_index":0,"prompts":["black car","traffic light"]}'
```

```json
{
  "found": true,
  "n": 3,
  "frame_indices": [0],
  "mask_size": {"height": 1200, "width": 1600},
  "arrays": {"masks": "/work/artifacts/segmasks_c906ead6c2a8.npy"},
  "overlay": "/work/renders/segment_overlay_0bfd3cc6ae05.png",
  "box_format": "xyxy pixels, derived from the mask extent",
  "prompts": ["black car", "traffic light"],
  "id_note": "The number stamped on each colored region of the overlay is that instance's 0-based index in the masks array (usable as /measure's instance); its 'label' is the prompt that produced it.",
  "instances": [
    {"id": 0, "label": "black car", "box_xyxy": [455.0, 448.0, 1509.0, 887.0],
     "centroid_xy": [974, 660], "mask_pixels": 307274,
     "mask_coverage": 0.16, "score": 0.7070}
  ]
}
```

`overlay` is a PNG of the image with every instance color-filled and stamped
with its id — view it to align what you see with the ids. The `masks` envelope
is the same format `/detect` produces, so an id feeds `/measure` as `instance`
directly; the sidecar carries per-instance `labels` that `/bev` uses to name
map objects. A prompt that matches nothing contributes nothing (`found: false`
when none match — an observation, not a failure). Optional knobs: `max_masks`
(default 30), `nms_iou` (default 0.7).

## POST /drop

Permanently remove wrong or duplicate instances from a frame's saved
masks — for when the question implies ONE such object ("the desk") but
`/segment` returned several. Pure CPU.

```bash
curl -s -X POST http://host.docker.internal:8090/drop \
  -H 'Content-Type: application/json' \
  -d '{"frame":2,"instances":[1],"image":"frame_2.png"}'
```

Unlike `/bev`'s per-call `erase` (view-level, for the motion split, must be
repeated on every call), `/drop` rewrites the masks artifact itself: the
instance's mask is zeroed IN PLACE — the row and every surviving id stay
stable — and saved as a NEW masks artifact that auto-discovery serves to
every later `/bev`, compass grounding, and `/measure`. The sidecar records
the accumulated `dropped_ids`; `/bev` skips them outright (neither labeled
nor listed as unlabeled). `instances` takes overlay ids only (labels
cannot pick among same-label duplicates — that ambiguity is the reason the
endpoint exists); ids out of range are HTTP 400 listing what the frame
has, re-dropping an empty instance is a no-op with a note. With `image`,
the reply carries a re-rendered `overlay` (surviving ids unchanged) to
verify against; `remaining` lists what is left either way.

## POST /match

LoMa-B keypoint matches for **every unordered frame pair**, saved to disk (one
`(n, 4)` float32 array `[xA, yA, xB, yB]` per pair, pixel coordinates on each
frame's original grid, frame-identity sidecar naming the pair). At most 12
images per call. Optional: `num_keypoints` (64..8192, default 2048 upstream).

```bash
curl -s -X POST http://host.docker.internal:8090/match \
  -H 'Content-Type: application/json' \
  -d '{"images":["/work/f0.png","/work/f1.png"],"frame_indices":[0,1]}'
```

```json
{
  "ok": true,
  "matcher": "loma_B",
  "n_pairs": 1,
  "pairs": [
    {"frames": [0, 1], "n_matches": 133,
     "matches": "/work/artifacts/matches_9e038f8701a7.npy"}
  ]
}
```

## POST /egomotion

Camera rotation and translation **direction** between one frame pair, from the
matched keypoints on the STATIC scene: matches landing inside a moving
instance mask (ids from `/segment`'s masks array) in either frame are excluded,
the fundamental matrix is USAC-MAGSAC-estimated from the rest, upgraded to an
essential matrix with an assumed focal prior (1.2 × max(W, H), principal point
centered), and decomposed. Pure CPU — no GPU call. `matches`/`masks_a`/
`masks_b` paths are optional: the newest saved artifact for the pair (by
frame-identity sidecar) is auto-discovered; pass them explicitly to override.
Optional knobs: `ransac_thresh_px` (default 0.75), `moving_margin_px`
(default 3, dilation of the moving masks before filtering).

```bash
curl -s -X POST http://host.docker.internal:8090/egomotion \
  -H 'Content-Type: application/json' \
  -d '{"pair":[0,1],"moving_a":[2],"moving_b":[]}'
```

```json
{
  "ok": true,
  "pair": [0, 1],
  "n_matches": 187, "n_excluded_moving": 12, "n_static": 175,
  "n_inliers": 80, "inlier_ratio": 0.4571, "n_cheirality": 74,
  "rotation": {"matrix_b_in_a": [["..."]],
               "yaw_deg": -83.4, "pitch_deg": 2.1, "roll_deg": -0.8},
  "translation": {"direction_xyz_in_a": [0.91, -0.02, 0.41],
                  "note": "unit direction ... magnitude is unrecoverable"},
  "summary": "camera B relative to camera A: turned left by ~83°, moved right and forward.",
  "axes_note": "+x right, +y down, +z forward (OpenCV convention); yaw>0 = turned right, pitch>0 = tilted up, roll>0 = clockwise"
}
```

Fewer than 8 static correspondences, or a degenerate configuration, returns
`ok: false` with a `reason` (HTTP 200 — an honest negative, not a tool error).
Under 15 RANSAC inliers the reply carries `low_evidence: true`; treat the
summary as weak evidence. Sign conventions are pinned by
`tests/test_gateway_match_egomotion.py` against synthetic ground truth.

## POST /gravity

GeoCalib-estimated gravity direction relative to one frame's camera: the
gateway sends the frame to the GPU server's `spatial_Gravity` tool
([GeoCalib](https://github.com/cvg/GeoCalib), single-image geometric
calibration), which returns the unit DOWN direction in camera coords
(+x right, +y down, +z forward; an upright level camera gives (0, 1, 0)),
saved as a frame-identity artifact for `/bev` to discover.
`vertical_lean_deg` / `camera_pitch_deg` are derived from the vector for
reporting (same parameterization the earlier VLM estimator used);
`confidence` buckets GeoCalib's `gravity_uncertainty_deg` (<=2° high,
<=6° medium, else low).

```bash
curl -s -X POST http://host.docker.internal:8090/gravity \
  -H 'Content-Type: application/json' \
  -d '{"image":"/work/f0.png","frame_index":0}'
```

```json
{
  "ok": true, "frame_index": 0,
  "gravity_xyz": [0.0682, 0.9758, -0.2079],
  "vertical_lean_deg": -4.0, "camera_pitch_deg": 12.0,
  "confidence": "medium", "method": "geocalib",
  "gravity_uncertainty_deg": 2.36,
  "artifact": "/work/artifacts/gravity_c5204a2f877e.npy"
}
```

Sign conventions are pinned by `tests/test_gateway_gravity_bev.py` (gateway
side) and `gpu_models/geocalib_model.py`'s docstring (GeoCalib's `vec3d` is
world-UP; the tool negates it). One GPU round trip, ~1 s.

## POST /bev

Gravity-aligned bird's-eye-view map **fused from every reconstructed
frame**, composed from saved artifacts (`/segment` masks + labels per frame,
`/reconstruct` points + poses, `/gravity` vector — all auto-discovered;
explicit paths override, `masks` for the reference frame only). Pure CPU.
All frames share one world coordinate system, so every frame's points
project into the reference camera's ground plane — the map covers what any
view saw. The reference camera sits at (0, 0), up the map is its
ground-projected forward, the grid is 1 m. Every reconstructed camera is
marked `f<N>` with a wedge showing the direction it faces; `heading_deg`
reports that facing clockwise from map-up (0 = same way the reference camera
faces, 90 = its right; null when a camera looks nearly straight up/down).
Objects come from EVERY frame's masks — `/segment`
is prompted, so those are exactly the objects the question names. Each is
drawn as a dot at its ground-plane centroid, named `fN:id <label>`;
same-label instances within `merge_radius_m` (default 0.75) merge into one
entry whose `instances` lists each view. The flattened 2D layout is also
saved as artifacts: `bev_cameras` (n,3 — right_m, forward_m, heading_deg,
NaN when a camera looks nearly straight up/down) and `bev_objects` (k,2 —
centroid right_m, forward_m; per-row `labels` in the sidecar). Masks
detected on the original grid are offset via the points sidecar's
`crop_box` when DA3 center-cropped.

```bash
curl -s -X POST http://host.docker.internal:8090/bev \
  -H 'Content-Type: application/json' \
  -d '{"frame":0}'
```

```json
{
  "ok": true, "frame": 0,
  "bev": "/work/renders/bev_aac22ab4c49c.png",
  "extent_m": {"right": [-2.8, 1.9], "forward": [-0.5, 5.4]},
  "fused_frames": [0, 1], "frames_with_masks": [0, 1],
  "objects": [
    {"label": "bookshelf", "shown_as": "f0:0", "instances": ["f0:0", "f1:2"],
     "n_views": 2, "right_m": 0.79, "forward_m": 2.59,
     "height_rel_camera_m": 0.4, "distance_m": 2.71}
  ],
  "cameras": [{"frame": 0, "right_m": 0.0, "forward_m": 0.0, "up_m": 0.0, "heading_deg": 0.0},
              {"frame": 1, "right_m": -0.03, "forward_m": 0.01, "up_m": 0.4, "heading_deg": 24.5}],
  "axes_note": "map coordinates are metres in the reference camera's ground frame ..."
}
```

Optional knobs: `max_labels` (default 30), `min_label_coverage` (default 0 —
every question object stays on the map), `range_m`, `ceiling_margin_m`
(default 1.0 — points more than this far above the camera are treated as
ceiling and left off the map). Fewer than 100 usable 3D points returns
`ok: false` with a `reason` (HTTP 200).

`erase: ["f1:pan"]` (frame + that frame's instance label or overlay id)
removes those instances' points from the fusion before mapping; the PNG
carries a red `ERASED: ...` line and the reply an `erased` list. For a
moving object seen in frames i and j, the pair `erase fj:X` / `erase fi:X`
yields two maps showing X where frame i / frame j saw it — the dot's
displacement between them is the motion a single flattened map cannot
express. A selector that matches nothing is HTTP 400 listing that frame's
available instances. `up_m` per camera is the true-vertical offset vs the
reference camera (the map flattens height; vertical questions read `up_m` +
`height_rel_camera_m`, not the picture).

`compass` anchors compass directions on the map from ONE orientation fact
(nothing in a reconstruction knows north): the declared direction from
`start` to `end` IS `orientation`, which pins the whole rose.

```bash
curl -s -X POST http://host.docker.internal:8090/bev \
  -H 'Content-Type: application/json' \
  -d '{"frame":0,"compass":{"start":"scene","end":"f0:bed",
       "orientation":"southwest"}}'
```

`start`/`end` each name a point or a plane: `"f<N>:<label-or-id>"` a
segmented instance (ground centroid; `start_plane`/`end_plane: true` fit
the ground line through its points instead — the object's vertical plane),
`"scene"` the fused footprint's centre (midpoint of the robust 1/99
percentile extent), `"camera:f<N>"` that camera's position,
`"camplane:f<N>"` that camera's image plane (always a plane). At most one
side may be a plane; a point-to-plane arrow runs along the perpendicular.
Orientations: the 8 compass names or N/NE/E/SE/S/SW/W/NW.

`compass.queries` (list, max 8, same anchor grammar, optional
`start_plane`/`end_plane`) reads directions OUT of the compass: each query
is answered with its `bearing_deg` plus `nearest` — the two closest
compass directions with angular distances — and `margin_deg`
(< 10 adds an `ambiguous_note`); its arrow is drawn in blue, labeled with
the winner. Queries run against the north declared in the same call, or,
without anchor fields, against the saved `bev_compass` for the frame
(HTTP 400 naming the fix when none exists). An empty compass object is
HTTP 400.

The PNG gains the anchor arrow (planes drawn as dashed lines, the scene
centre as a cross), a derived rose top-right, and a `COMPASS: north = ...`
header line. The reply gains a `compass` block — `north_bearing_deg`,
`directions_deg` per orientation (clockwise from map-up, same convention
as `heading_deg`), and the grounded `anchor` — and the mapping is saved as
a `bev_compass` artifact ((8,2) unit ground directions, rows follow the
sidecar's `orientations`). A later `/bev` on the SAME reference frame
rediscovers the artifact and redraws the rose (`compass.restored_from_
artifact: true`), so split/erase maps keep their orientation. Coincident
anchors, two-plane requests, unknown instances, and non-compass
orientation words are HTTP 400 with the reason.

## POST /measure

Numeric answers over saved arrays. Use this when you cannot look at images.

Centroid of a segmented object in 3D:

```bash
curl -s -X POST http://host.docker.internal:8090/measure \
  -H 'Content-Type: application/json' \
  -d '{"op":"centroid",
       "points":"/work/artifacts/points_01e4a1c418b9.npy",
       "masks":"/work/artifacts/masks_bc02594729a3.npy",
       "frame_index":0,"instance":0}'
```

```json
{"found": true, "op": "centroid", "centroid_xyz_m": [-0.2889, -0.1821, 0.7758],
 "n_points": 2335, "frame_indices": [0], "units": "metres"}
```

3D bounding box (physical size of an object):

```bash
curl -s -X POST http://host.docker.internal:8090/measure \
  -H 'Content-Type: application/json' \
  -d '{"op":"bbox3d",
       "points":"/work/artifacts/points_01e4a1c418b9.npy",
       "masks":"/work/artifacts/masks_bc02594729a3.npy",
       "frame_index":0}'
```

```json
{"found": true, "op": "bbox3d",
 "min_xyz_m": [-0.4448, -0.3229, 0.6029], "max_xyz_m": [-0.0699, -0.0517, 0.8565],
 "extent_xyz_m": [0.375, 0.2713, 0.2536], "volume_m3": 0.0258,
 "n_points": 2335, "frame_indices": [0], "units": "metres"}
```

Distance between two detected objects (both measured on the same frame):

```bash
curl -s -X POST http://host.docker.internal:8090/measure \
  -H 'Content-Type: application/json' \
  -d '{"op":"distance",
       "a":{"points":"/work/artifacts/points_01e4a1c418b9.npy",
            "masks":"/work/artifacts/masks_bc02594729a3.npy","frame_index":0},
       "b":{"points":"/work/artifacts/points_01e4a1c418b9.npy",
            "masks":"/work/artifacts/masks_8ac7639d6d80.npy","frame_index":0}}'
```

```json
{"found": true, "op": "distance", "distance_m": 0.2141,
 "delta_xyz_m": [0.1512, 0.1515, -0.0044], "n_points_a": 2335, "n_points_b": 350,
 "frame_indices": [0, 0], "units": "metres"}
```

Distance between two literal 3D points:

```bash
curl -s -X POST http://host.docker.internal:8090/measure \
  -H 'Content-Type: application/json' \
  -d '{"op":"distance","point_a":[0,0,0],"point_b":[0.3,0.4,0]}'
```

Optional per-region knobs: `instance` (which detected object, default `0`),
`confidence` (a reconstruction confidence `.npy`) with `min_confidence` to drop
low-confidence points, and `allow_cross_frame: true` on a `distance` call to
compare regions measured on different frames of a static scene.

An empty mask returns `200` with `"found": false` and `n_points: 0`.

## POST /render

Write a PNG you can open, if your tooling can read images.

```bash
# false-colour depth for one frame (warm = near, cool = far)
curl -s -X POST http://host.docker.internal:8090/render \
  -H 'Content-Type: application/json' \
  -d '{"kind":"depth","array":"/work/artifacts/depth_2f1c9ab7de40.npy","frame_index":0}'

# a mask on its own, white on black
curl -s -X POST http://host.docker.internal:8090/render \
  -H 'Content-Type: application/json' \
  -d '{"kind":"mask","array":"/work/artifacts/masks_bc02594729a3.npy","instance":0}'

# the mask tinted over the original image
curl -s -X POST http://host.docker.internal:8090/render \
  -H 'Content-Type: application/json' \
  -d '{"kind":"overlay","array":"/work/artifacts/masks_bc02594729a3.npy",
       "instance":0,"image":"/work/f0.png","alpha":0.5}'
```

```json
{"found": true, "kind": "overlay", "path": "/work/renders/render_overlay_c5f40b7917ea.png",
 "frame_indices": [0], "size": {"height": 180, "width": 320}}
```

---

## A complete worked example

Measure how far a faucet is from a sink, in metres, from two images:

```bash
G=http://host.docker.internal:8090

# 1. reconstruct the scene
PTS=$(curl -s -X POST $G/reconstruct -H 'Content-Type: application/json' \
  -d '{"images":["/work/f0.png","/work/f1.png"],"frame_indices":[0,1]}' \
  | jq -r '.arrays.points')

# 2. segment each object on frame 0
SINK=$(curl -s -X POST $G/detect -H 'Content-Type: application/json' \
  -d '{"image":"/work/f0.png","prompt":"sink","frame_index":0}' | jq -r '.arrays.masks')
FAUCET=$(curl -s -X POST $G/detect -H 'Content-Type: application/json' \
  -d '{"image":"/work/f0.png","prompt":"faucet","frame_index":0}' | jq -r '.arrays.masks')

# 3. measure between them
curl -s -X POST $G/measure -H 'Content-Type: application/json' \
  -d "{\"op\":\"distance\",
       \"a\":{\"points\":\"$PTS\",\"masks\":\"$SINK\",\"frame_index\":0},
       \"b\":{\"points\":\"$PTS\",\"masks\":\"$FAUCET\",\"frame_index\":0}}" \
  | jq -r '.distance_m'
# 0.214089
```

Note that both detections use `frame_index: 0`, the same frame the
reconstruction covers. Detecting on frame 0 and then measuring against frame 1
returns a 409 rather than a wrong distance.

## Practical notes

* Reconstruction needs at least one image; two or more give you camera motion
  and a better-conditioned scene. Up to 32 images per call.
* Depth and coordinates are metric (metres), but they come from a monocular
  model: treat them as good relative estimates, not survey measurements.
* If a prompt returns nothing, try a broader word ("cup" instead of "mug"), a
  different frame, or a point prompt at a pixel you trust.
* Request bodies are capped at 32 MB. You pass paths, so bodies stay tiny.
