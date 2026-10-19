## Perception API — a live 3D scene

These CLI calls maintain ONE persistent 3D scene of the pictured space.
Almost everything becomes a named INSTANCE in that scene — every camera,
every object you segment, any point you register yourself — and every
spatial fact (distance, depth, direction, front/behind/left/right) is READ
OUT of the scene by instance name. The model's job is what the scene
cannot do: recognizing what things are, choosing what to segment, and
translating the question's wording into scene calls.

**Build the scene** — ALWAYS the first perception call:

```bash
python3 /app/perception.py recon
```

Reconstructs every `/app/frame_N.png` into one metric 3D scene (DA3), and
initializes the persistent scene state: each camera pose becomes instance
`camera:f<N>` (position AND orientation), the room centre becomes `scene`.
The reply is the initial instance list — the geometry itself stays inside
the scene, and you read it back through the calls below by instance name.
NEVER do coordinate math yourself: any number you feel like computing has
a call that states it.

In a TWO-view scene, an `overlap_warning` in the reply means the two
frames share almost no covisible view (the camera jumped — through a
doorway, or a cut): the reconstruction stitched those moments together
on THIN evidence, and it CANNOT tell you which side is placed wrong. Treat every
readout that SPANS a named seam — side/distance/elevation/motion between
things seen on opposite sides of it, the fused map across it — as
suspect: prefer relations among things seen within the same span, and
double-check anything that crosses the seam against the per-frame views
before answering with it.

**Segment the question's objects** — each becomes a named instance:

```bash
python3 /app/perception.py segment \
  --prompt "bed@f0" --prompt "shelf@f4" --prompt "desk@f2"
```

Prompts are the objects the question mentions — in the question text and
in every answer option — and nothing else. EVERY "object in Figure N"
pairing gets its own `@f<N-1>` suffix (Figure N is zero-based
`frame_{N-1}.png`); only an object named with no figure runs unsuffixed on
every frame. A figure that names the VIEWPOINT ("relative to you in the
last photo") is not an object binding — that object runs unsuffixed, and
the observer becomes `--view N` at readout time; the fused scene answers
even when that view cannot see the object (it may be behind the camera). One ID-stamped overlay PNG per queried frame (`view` each);
ids are per-frame, and each region's `label` names the prompt that
produced it. A prompt that matches nothing in a frame contributes no
region there — an observation, not an error.

The reply's `scene_instances` lists what was registered: each label is now
an instance named by that label (`bed`, `desk`), the same physical object
seen from several frames merged into one instance; a second, separate
object with the same label becomes `<label>_2`. These names are the
handles every readout below takes.

Each instance row also carries a `motion` tag: `moving` means the OBJECT
ITSELF moves between frames (camera motion never sets it), `static` that
it holds still. Trust it both ways. For "which object moved?" the tags
ARE the answer — copy them instead of comparing pixels across photos.
And a `moving` instance's fused cross-frame position is unreliable (each
frame saw it somewhere else): reference it per frame (`f<N>:<label>`)
and read HOW it moved with `motion` (below).

**Duplicate cleanup** — when the question implies ONE such object in a
frame ("THE desk in Figure 3") but the overlay shows several:

```bash
python3 /app/perception.py drop --instance f2:1
```

Pick the intended instance from the overlay (larger, more central,
matching the question's wording) and drop the others by overlay id
(`--instance` repeatable; ids only). A drop is PERMANENT and the scene
follows it: the reply's `scene_instances` shows the registry after the
drop. A duplicate left in place corrupts every later readout that
references the object by name.

**Bird's-eye-view map** — after recon and segment:

```bash
python3 /app/perception.py bev --frame 1
python3 /app/perception.py bev --frame 1 --erase f0:pan   # split map
```

Returns a top-down map PNG (`view` it) fusing every frame's STATIC
points — plus everything the reference frame itself saw — into the
reference camera's ground frame. A moving object therefore shows up
where the REFERENCE frame saw it: build the map on the frame whose
moment you are asking about, and switch `--frame` to see the mover at
another moment. Gravity is estimated automatically
(GeoCalib over the frames, best-confidence views fused) — there is no
separate gravity step. Cameras are wedges showing which way each faces,
the grid is 1 metre, question objects are labeled dots, and anything you
registered with `add_instance` is a GREEN DIAMOND (its arrow = the
facing you gave it) — after registering an observer, `view` the map and
check the diamond sits where you meant and its arrow points the way you
meant, BEFORE reading side from it. The map is for SEEING the layout;
the reply carries names, not coordinates — every relation you go on to
state reads from the calls below (`side`/`distance`/`elevation`/
`motion`/`compass`), and the map flattens height entirely.

`--erase f<N>:<label-or-id>` (repeatable) removes that frame's view of an
object for one call. For a moving object seen in frames i and j, the pair
`--erase fj:X` / `--erase fi:X` is a UNIFIED-COORDINATE viewer: both maps
share one ground frame, so they show the object at its two moments with
the camera factored out — which photos never do, and immune to camera
roll by construction. They are the VISUAL check; the motion you STATE
reads from `motion --of fi:X --of fj:X` (below). Settle WHICH instance
the question means first (`drop` the twins in EVERY frame —
`erase_warning` in the reply means one is still there).

**List the scene** — what exists, under what name:

```bash
python3 /app/perception.py list_instance
```

Every instance: its name, where it came from, and whether it has an
orientation. Run it whenever unsure what a readout call can reference.

**Register your own instance** — for a thing the scene doesn't hold yet
(a person, a door named only in an option, a spot on the floor). A
question stem of the form "when you are doing X", "if you do X",
"imagine you ...", "imagine a person doing X" IS this call's trigger:
the hypothetical person it places means register an instance at the
described spot (position + facing) and read the asked relation from it:

```bash
python3 /app/perception.py add_instance --name doorway \
  --source "the doorway named in option C, picked off frame_2" \
  --view 2 --pixel 411,236
```

Three position sources: `--at-map f<N>:PX,PY` places it at a PIXEL of
the frame-N bev map PNG — point at the map exactly the way you point at
any image (height fills in from the reconstruction at that spot); the
MOST PRECISE source, and it requires that map to have been built and
viewed first. `--view N --pixel x,y` lifts an ORIGINAL-image pixel of
that frame to its 3D point (wide-angle picks land badly — prefer the
map once it exists); `--at-camera N` uses that camera's position (e.g.
"the photographer").
`--name` is the unique handle later calls use; `--source` says in one
clause how it entered the scene. Optional orientation makes `side
--observer` work from it.

The orientation can come LATER: register the position alone, run `bev`
to SEE the marker on the map, then call `add_instance` again with the
same `--name`, NO position flags and just the orientation — it orients
the existing instance in place (a later call may correct it the same
way). This pairs naturally with the map arrow: place the spot, look at
the map, then give the facing as two points read off it.

An orientation is a PRIMARY axis (taken exactly) plus a SECONDARY
direction (adjusted minimally to sit orthogonal to it). Axes are the
INSTANCE'S OWN, named by either member of each pair — forward/back,
right/left, top/bottom (for a person, forward is the way the face
points and top is toward the head; an object's axes read off its
structure — a chair's front, its top) — and direction values are
existing instances of any kind (objects, cameras `camera:f1`, the room
centre `scene`), the literal `gravity` (the downward direction), or
`bev:f<N>:START:END` — an ARROW between two points on the bev map
rendered from frame N (direction = start → end). Each endpoint is a
colon-free instance name, or a `PX,PY` pixel on that map's PNG:
`bev:f0:bed:door` points the way bed→door lies on the map. Use an
arrow when the direction is plain on the map but no single instance
sits along it.
Forms:

- `--facing VALUE` — upright, facing that way: `--facing scene`
  reads "standing at this spot, facing into the room" — the usual pick
  for someone at a door or window (`--facing-camera N` = `--facing
  camera:fN`), and `--facing bev:f0:bed:door` faces the way that map
  arrow points.
  Anyone on their feet — standing, walking, running, entering — is
  upright, so `--facing` alone is the whole orientation: gravity fills
  in the vertical, you only pick the heading (a walker faces their
  direction of travel). A single bare axis flag is the same upright
  sugar with that axis as the heading: back to the door is `--back
  door`, the window on their right is `--right window`.
- `--primary AXIS=VALUE --secondary AXIS=VALUE` — any pose. The primary
  is the ONE of its axes you can pin exactly and to what; the secondary
  is any direction that fixes the remaining spin. Find both yourself
  from the pictures: gravity is often the exact one for anything
  resting on a surface, and the secondary target must already be an
  instance (register a pixel as one first if nothing nameable sits
  there).
  `--primary forward=door` with no secondary means EXACTLY toward the
  door (tilting if the door sits higher or lower), upright filled in
  as the default secondary.
- legacy: three bare axis flags, one per pair (`--forward desk --right
  window --top ceiling_lamp`) — rarely needed; two directions already
  determine every pose.

**Distance** — between any two instances, by name:

```bash
python3 /app/perception.py distance --a bed --b desk
python3 /app/perception.py distance --a bed --b desk --known 2.5
```

Straight-line metres between the two (plus horizontal/vertical
components). `--known` records a distance the QUESTION states between
exactly these two as the scene's scale calibration: from then on every
distance, depth, and map metre is calibrated. When the question gives a
measurement, calibrate FIRST, then read the asked one.

**Side** — the FULL relation in an observer's view, in one call:

```bash
python3 /app/perception.py side --view 1 --of bed
python3 /app/perception.py side --observer sitting_person --of doorway
```

`--view N` reads camera f<N>'s view; `--observer NAME` reads any oriented
instance's view. Each target row decomposes the offset along the
observer's FORWARD, RIGHT and UP axes (`forward_m`/`right_m`/`up_m`,
positive = in front of it / on its right / above it), states each axis's
verdict (`front_or_behind`, `left_or_right`, `above_or_below`) plus the
combined eight-way `direction` (front-left, rear-right, ...) — when the
options speak that language, it IS the answer — and the `conclusion`
sentence names it all outright; copy it, do not re-derive a side from
the pictures. For a CAMERA observer the frame is its gravity-aligned
ground heading (the same frame the bev map draws), so a pitched camera
cannot misread "behind on the floor plan" as "in front along its
optical axis"; a registered observer keeps its own registered axes.
Every "in X's view" fact has this call as its source.

**Depth** — nearness ranking and depth calibration:

```bash
python3 /app/perception.py depth --view 1 --of bed --of desk
python3 /app/perception.py depth --observer sitting_person --of doorway --known 3
```

Same observer grammar as `side`. Each target row carries `depth` along
the observer's forward axis (positive = in front, negative = behind); the
`order` line ranks many targets nearest to farthest — use it for "which
is closer" comparisons. `--known` (one `--of`) calibrates scale from a
depth the question states.

**Elevation** — MANDATORY for every height comparison (taller/shorter,
higher/lower, above/below, "which is highest"):

```bash
python3 /app/perception.py elevation --of table --of lamp --of shelf
```

The FIRST `--of` is the reference; every later row reads that target's
signed height against it (`up_m` positive = above it, `above_or_below`,
`conclusion` — copy it), and three or more targets add a lowest-to-highest
`order` line. Heights follow the scene's gravity, so camera tilt cannot
skew them — and no observer changes which of two things is higher, so
this is the source whether or not an observer appears in the sentence.
`side`'s above/below is a different, egocentric question (offset along
the observer's own up axis, which tilts with the observer) — use it only
when the question asks that. The bev map is a floor plan and carries NO
height information — never answer a height question from it.

**Motion** — MANDATORY for "did it move/turn, and how" questions:

```bash
python3 /app/perception.py motion --of camera:f0 --of camera:f2
python3 /app/perception.py motion --of f1:pan --of f3:pan
```

Exactly two `--of`. Two CAMERAS give the relative pose between those
instants: the `conclusion` states how the camera moved (forward/right/up
in the first camera's ground frame, gravity-referenced), turned
(clockwise/counterclockwise), and tilted its view (UP/DOWN) — the answer
to every "how did the camera move between the shots" question, and the
thing to run BEFORE comparing angles between two photos (a lean change
makes still objects look rotated). The camera-pair reply also carries
`camera_poses` — the raw 4x4 camera-to-world extrinsics artifact. That
path is the ONE sanctioned exception to "never do coordinate math":
when the question itself sets up a frame to compute in (its own axes, a
rotation order to name — "rotate about the Y-axis then the X-axis"),
read the extrinsics and compute in THAT frame; for every plain
direction question the conclusion IS the answer. The SAME OBJECT in two
frames (`f<i>:X`, `f<j>:X`) gives its shift — with its direction as
seen from the from-frame camera (RIGHT/LEFT, AWAY/TOWARD) — and
rotation from a rigid fit of its reconstructed points; the
split-pair bev maps are the visual check. Two rotation readings come
back and they answer DIFFERENT questions: `rotation` is about the
vertical axis seen from above (map-style "did it turn left/right"),
while the "in the image plane" line (`image_rotation`) is the turn as
the from-frame camera sees it — THE reading for "rotated clockwise or
counterclockwise (in the picture)" questions, and it already accounts
for camera roll, which raw photo comparison does not. `segment`'s
`motion` tag names WHICH instances move; this call states HOW. Copy the
conclusion — do not re-derive motion from the pictures.

**Compass** — MANDATORY whenever a compass word appears ANYWHERE in the
question or options (north/south/east/west or any diagonal, a "X corner",
a "X side" — of the room OR of an object; "which side of the statue do
you see" is a compass question):

```bash
python3 /app/perception.py compass \
  --start statue --end camplane:f0 --orientation north \
  --query "statue->camplane:f1"
```

Nothing in the reconstruction knows north — the ONE orientation fact the
question states pins the whole compass: the direction from `--start` to
`--end` IS `--orientation`. The mapping is PERSISTED in the scene: any
later compass call queries directly without re-anchoring, and later `bev`
maps draw the rose from it. `--query "START->END"` (repeatable) reads the
asked fact back: the reply's `direction`/`conclusion` ARE the tool's
answer — copy that direction; do not re-derive one from where a camera
stands, from an object's supposed facing, or from the pictures. When the
concluded direction is not among the answer options, take the
highest-ranked direction in `nearest` that IS an option — the ranking is
the whole answer space. For an object->camplane query the conclusion also
states the side-seen fact outright ("view fN sees the object's X side");
that sentence IS the answer to "which side do you see". The reply is
FINAL: no reading of the pictures, no argument about which way an object
"must" face, and no field of the reply reopens the question. A
wrong-looking answer means a wrong ANCHOR — fix the anchor and re-run;
that is the only way to change the answer.

Anchor/query references: any instance name (`bed`, `camera:f0`, `scene`,
your `add_instance` names), a per-frame region `f<N>:<label-or-id>`, or
`camplane:f<N>` (that camera's image plane — always a plane). Append
`|plane` to a segmented instance to use the plane fitted through it
instead of its centroid.

**Centroid or plane?** Decide by what the question asks about:

- CENTROID (the default) answers WHERE an object is: "where is X", "which
  corner", "X relative to Y". Anything that makes sense as a dot is a
  point.
- PLANE belongs ONLY to extended flat structure whose surface IS the
  fact — a wall, a shelf wall, a long counter — and `camplane:f<N>`
  (always legal: that plane is the CAMERA's). A point-to-plane arrow runs
  along the PERPENDICULAR: the direction to the SURFACE.
- Never put a plane through a compact object (a telephone, a lamp, a
  chair): its fitted direction is reconstruction noise. The tool rejects
  a plane on under 0.5 m of ground trace — that 400 is the correction:
  re-run with the centroid.

Translate the question's wording mechanically — a STATED fact becomes the
anchor, the ASKED fact becomes a `--query` of the same shape:

- "in Figure V you see the X side of object O" —
  `--start O --end camplane:f<V-1> --orientation X`,
  or asked: `--query "O->camplane:f<V-1>"`
- "a is X of b" — `--start b --end a --orientation X`,
  or asked: `--query "b->a"`
- "O (in Figure V) is in the X corner / on the X side of the room" —
  corner and side facts are judged by whoever TOOK the picture:
  `--start camera:f<V-1> --end O --orientation X`; only a fact naming NO
  figure falls back to `--start scene`
- "O is on the X wall" — the camera that framed O was looking X. A FLAT
  thing on a wall (a book storage area, a counter run) states a fact
  about its SURFACE: `--start camera:f<V-1> --end O --end-plane
  --orientation X` (centroids of flat things sit tens of degrees off the
  perpendicular and silently rotate north). A COMPACT thing on a wall
  keeps its centroid.

One call usually does the whole question: the stated fact as the anchor,
the asked fact as the query.

Use no other perception command: this task is answered from the views,
the depth maps, the map(s), and the scene readouts — not from raw
measurements. Do not call `detect`, `measure`, `match`, `egomotion`,
`gravity`, `render`, or the Python `look()` API.

{skill_index}
{skill_bodies}
