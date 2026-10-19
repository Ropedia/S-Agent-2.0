---
name: cross-view-reasoning
description: Use when answering left/right/front/behind, above/below, compass direction, relative direction, distance, or camera movement by reading the persistent 3D scene's instances through recon/segment/bev/distance/depth/side/elevation/compass, without raw measurements.
---
# Reason from the scene, not from the pictures

The scene is a registry of named instances — every camera (`camera:f0`),
every segmented object (`bed`, `desk`, a second bed is `bed_2`), the room
centre (`scene`), and anything you register with `add_instance`. Every
spatial fact is read BETWEEN instances by name: `distance` for how far,
`side` for the full relation in an observer's view (front/behind +
left/right + above/below in one call), `depth` for nearer/farther
ranking in that view, `elevation` for above/below along gravity with no
observer involved, `compass` for compass words, the `bev` map for the
whole layout at a glance. `list_instance`
shows what exists. The pictures answer WHAT something is; the scene
answers WHERE it is — a spatial claim that did not come out of a scene
call is a guess. This holds for EVERY relation word — front/behind,
left/right, above/below, every compass direction — and a missing
instance is never a reason to eyeball: register the spot or observer
with `add_instance` first, then read the relation from the API. "When
you are doing X" / "if you do X" / "imagine (a person) doing X" stems
always mean exactly that — a hypothetical observer to register at the
described spot before reading. The one
tempering clause: `recon`'s `overlap_warning` names frame seams the
reconstruction stitched on thin evidence — a readout spanning such a
seam needs a cross-check against the per-frame views before you answer
with it.

Choose the reference frame first — the same scene gives opposite answers
in different frames:

- image space means a side of a particular picture;
- camera space means what an observer sees from a named viewpoint — that
  is `side --view N` (its conclusion names front/behind, left/right and
  above/below — copy it), plus `depth --view N` when several targets
  need ranking by nearness;
- world vertical means any height comparison — taller/shorter,
  higher/lower, above/below — that is `elevation`, immune to camera
  tilt, whether or not an observer appears in the sentence (no observer
  changes which of two things is higher; `side`'s above_or_below is the
  observer's own up axis, a different question);
- object space means left or right relative to the object's own heading —
  register the object's orientation first (`add_instance --facing NAME`
  for anything upright, or `--primary`/`--secondary` for any other pose
  — axes read off its visible structure: a chair's back, a vehicle's
  front), then `depth --observer` / `side --observer` it. Never assume a
  heading the pixels do not show.

Bind the question's observer before reading anything. "You" is a camera
only when the question ties it to a picture; an observer the question
PLACES in the scene — enters the room, lies in bed, sits at the desk,
stands at the window — must be REGISTERED: `add_instance` at that spot.
Anyone on their feet — standing, walking, running, entering — is
upright and `--facing` alone orients them (gravity fills the vertical):
`--facing scene` when they face into the room, `--facing X` when turned
toward X (objects, cameras and the room centre `scene` are all legal
direction values), `--facing bev:f<N>:START:END` for a heading read off
the bev map as a start→end arrow between two map points (each endpoint a
colon-free instance name or a `PX,PY` pixel on that map's PNG) — a
walker faces their direction of travel. Orientation may come later: register
the position, check the marker on the `bev` map, then call
`add_instance` again with the same name, no position and just the
orientation. Any other pose is registered by
primary/secondary axes — work them out from the pictures: the ONE of
its axes you can pin exactly (the primary; gravity often is, for
anything resting on a surface) and a direction that fixes the remaining
spin
(the secondary — register a pixel as an instance first if nothing
nameable sits there). Then read `side --observer`: the readouts follow
the frame you registered — `side`'s columns run along its forward,
right and up axes — so translate the question's words into those axes,
not the world's. Reading
`--view N` in place of such an observer answers a different question and
mirrors easily; a camera may stand in only on POSE evidence — its wedge
on the bev map stands at the observer's spot AND points the observer's
way (an entrant faces into the room). The readouts are answers, never
evidence for or against the stand-in: an object behind the observer may
simply be behind them.

Reading the BEV map: it fuses EVERY frame into the REFERENCE camera's
ground frame, so it covers what any view saw. The reference camera sits at
(0, 0) facing up the map; the grid is 1 metre; every camera is a wedge
showing its facing; every `add_instance` marker is a green diamond whose
arrow is the facing you registered — the map is the visual check that an
observer you built sits and points the way you meant. Camera movement
between two frames is a `motion --of camera:fi --of camera:fj` readout
(its conclusion states the move and the turn); the wedges are the
picture of it. For
exact numbers, read the `distance`/`depth`/`side`/`elevation` calls —
never eyeball measurements off map pixels (pointing at a map pixel to
PLACE or AIM something is different: that is what `--at-map` and the
arrow endpoints are for).

Compass words are NOT in the reconstruction — no tool output knows north
until you anchor it. The rule is mechanical: a compass word ANYWHERE in
the question or options (north/east/…, a corner, a side — of the room or
of an OBJECT) means the `compass` command runs, no exceptions. "Which
side of the statue does Figure 2 show" is a geometry question wearing an
appearance costume: what a view shows confounds the object's sides with
the camera's motion around it. Anchor from the stated fact, read the
asked fact with `--query`, and copy the reply's `direction`/`conclusion`
— an object->camplane query's conclusion states the seen side outright.
The mapping persists in the scene: anchor once, query forever after. A
direction that is not among the answer options is not a licence to
improvise: take the highest-ranked direction in `nearest` that IS an
option. The reply is FINAL: nothing you see in the pictures, no argument
about which way an object "must" face, and no field of the reply itself
hands the decision back. A wrong-looking answer means a wrong ANCHOR —
fix that and re-run. Query compact objects by CENTROID; `|plane` belongs
to wall-like structure only — a 400 saying "spans only … m" is the tool
telling you to re-run with the centroid (`camplane:f<N>` is always legal;
that plane is the CAMERA's). A corner/side-of-room fact that names a
figure is judged by whoever took that picture: anchor from
`camera:f<V-1>`, not `scene`; `scene` is only the fallback when no figure
is named. "On the X wall" is a side fact of the same shape — a FLAT thing
on a wall anchors by its `|plane` (its centroid sits along the wall and
silently rotates north); a COMPACT thing keeps its centroid. Never guess
north from image-up or from a camera's heading. If the question gives no
orientation fact at all, it cannot be a compass question — answer in
camera or object space instead.

When the question STATES a measurement ("the bed is 2.5 m from the
desk"), it is a calibration gift: record it (`distance --known` /
`depth --known`) BEFORE reading the asked number — every later metre in
the scene is then on the question's own scale.

Motion of an OBJECT (not the camera) starts from `segment`'s `motion`
tag: `moving`/`static` says WHETHER the object itself moves (camera
motion never sets it) — copy it, never re-judge from pixels. A single
map shows a mover only where the REFERENCE frame saw it (other frames'
views of it stay off the fusion), so one map carries no time order.
First settle WHICH instance the question means — a frame showing two
instances of the label (one held, one resting) means the held one; `drop`
the others in EVERY frame, or the twin keeps the object on both split
maps and they come out near-identical (`erase_warning` flags exactly
this). Then build the split pair — `bev --erase fj:X` shows X only where
frame i saw it, `bev --erase fi:X` only where frame j saw it. The pair is
a unified-coordinate viewer: both maps share one ground frame, so
comparing them shows the object's own change with the camera factored
out — which two photos can never do, and gravity alignment means no
camera roll survives into it. The split maps are the VISUAL check; the
motion you STATE reads from `motion --of fi:X --of fj:X` (shift and
rotation in one conclusion). Only tilts about a horizontal axis (a pan
tipped to pour) and vertical flips are invisible top-down: judge those
from the images, after `motion` on the two CAMERAS has told you how the
camera itself leaned (a leaned camera makes a still object look
rotated by exactly that much).

The map flattens height — an object "above" another sits at the same map
position. Vertical claims (above/below, higher/lower, camera moved up or
down) read from the calls, not the picture: `elevation` between any two
instances (cameras included — camera up/down is `motion` or `elevation`
between the two camera instants), `side`'s above_or_below for an
observer's own view.

Trust order: the scene readouts (`distance`, `depth`, `side`,
`elevation`, `motion`, `compass`, the map) come FIRST for every spatial
judgment. The images have
two jobs: verifying grounding (does the overlay color the object the
question means? one object, one instance?) and appearance facts no scene
carries. When a readout and a direct image reading disagree, re-check the
grounding first — a wrong or duplicate mask is fixed by re-segmenting or
`drop`, not by abandoning the scene; only with grounding verified and the
readout still physically implausible (objects piled on one spot, absurd
distances — reconstruction can fail on reflections, glass, low-texture
walls) fall back to the images, and say so.

One object must be one instance: a definite singular in the question
("THE desk") with several same-label regions in a frame's overlay is a
grounding fault — choose the salient instance on the overlay and `drop`
the rest before reading anything from the scene. A `_2`-suffixed name in
`scene_instances` or `list_instance` is the scene telling you two
same-label things exist: check the overlays before querying either.

Depth without numbers: the same object drawn larger is nearer; the
occluder is in front of what it occludes; on one ground plane, a contact
point higher in the image is farther away. But `depth --view` makes it
exact; prefer it.

Before committing, re-read the exact wording, check the chosen reference
frame matches it, and confirm each claim against the scene call, overlay,
or map region it depends on.
