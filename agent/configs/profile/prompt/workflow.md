## How to work this problem (mandatory protocol)

Follow these steps in order. Steps 1-4 come before any answer reasoning.
The rule of the protocol: EVERY spatial-relation claim in your answer —
distance, depth, front/behind, left/right, above/below, and every
compass direction (N/S/E/W and the diagonals) — must be READ from a
scene call (`bev`, `distance`, `depth`, `side`, `elevation`, `motion`,
`compass`), never eyeballed off the pictures. The images are for recognizing and
verifying WHAT things are; the scene is for WHERE they are. When the
relation you need involves a position or viewpoint the scene has no
instance for — an observer the question places, a spot `segment` cannot
name — that is never a reason to eyeball: think about where it is and
which way it faces, register it with `add_instance`, then read the
relation from the API like any other.

1. **Build the scene FIRST.** Whenever the input has multiple views, the
   very first perception call is:

   ```bash
   python3 /app/perception.py recon
   ```

   It reconstructs all frames into one metric 3D scene and registers every
   camera as instance `camera:f<N>`. Do NOT page through the frames after
   it — the scene, not your eyes, answers WHERE things are, and the
   frames that matter reach you anyway: each segment overlay (step 2)
   shows its frame with the masks on it, and the bev map (step 3) shows
   the whole layout. `view` a specific frame only when you need a
   WHAT-fact from it: confirming what an object is, its appearance, or
   picking a pixel for `add_instance`. If the reply carries an
   `overlap_warning`, note the named seams NOW: any readout you later
   take between things on opposite sides of a seam is on thin evidence —
   cross-check it against the per-frame views before answering with it.
2. **Segment the question's objects.** List every object the question
   mentions — in the question text AND in every answer option (a chair
   named only by option C still counts). Bind each object to its figure:
   EVERY "object in Figure N" pairing gets an `@f<N-1>` suffix (`Figure N`
   is zero-based `frame_{N-1}.png`) and is segmented only there; only an
   object named with NO figure runs unsuffixed. ⚠️ A figure that sets the
   VIEWPOINT is not an object binding: "where is X relative to you in
   the last photo" binds the OBSERVER (`side --view N`), NOT X — X runs
   UNSUFFIXED so the scene registers it from whichever frames saw it
   best. The fused scene then reads X's position from view N even when
   that view cannot see X at all (a "rear left/rear right" option is
   the giveaway that X may sit BEHIND that camera — only a cross-frame
   segment can answer it). "the bed in Figure 1 …
   the desk in Figure 3 … the shelf in Figure 5" is:

   ```bash
   python3 /app/perception.py segment \
     --prompt "bed@f0" --prompt "desk@f2" --prompt "shelf@f4"
   ```

   Segment ONLY the question's objects — no scenery. `view` every
   returned overlay: each colored region carries that frame's id, and the
   reply's `scene_instances` names the instances now registered in the
   scene. Then clean up duplicates: when the question implies exactly ONE
   such object in a frame ("THE desk in Figure 3") but that frame's
   overlay shows several, decide from the overlay which one the question
   means — the salient one — and permanently remove the others:

   ```bash
   python3 /app/perception.py drop --instance f2:1
   ```

   A duplicate left in place corrupts every later readout that references
   the object by name (and a stray `bed_2` in `scene_instances` is the
   scene telling you a duplicate or a second object exists — check the
   overlays).
3. **Build the bird's-eye-view map.** Pick the reference frame — the
   viewpoint the question is asked from (`Figure N` is zero-based
   `frame_{N-1}.png`; "the last picture" means the highest frame) — and
   run:

   ```bash
   python3 /app/perception.py bev --frame 1
   ```

   then `view` the returned PNG. Gravity is estimated automatically; the
   map fuses EVERY frame into the reference camera's ground frame (the
   reference camera at (0, 0) facing up the map, 1-metre grid, cameras as
   wedges, question objects as named dots, your `add_instance` markers as
   green diamonds with a facing arrow — re-run `bev` after registering an
   observer to eyeball that it sits and points the way you meant).
4. **Calibrate if the question states a measurement.** "The bed is 2.5 m
   from the desk" / "the door is 3 m in front of the camera" is not
   flavor text — record it BEFORE reading any other number:

   ```bash
   python3 /app/perception.py distance --a bed --b desk --known 2.5
   ```

   (or `depth --view N --of door --known 3`). Every later metre — maps
   included — is then calibrated.
5. **Decide what the question actually needs, then read it from the
   scene:**
   - **Distances** ("how far", "closer/farther", "nearest"): `distance
     --a X --b Y` per pair. Never subtract map pixels by eye.
   - **What an observer sees**: `side` reads the target's FULL relation
     in one call — front/behind, left/right, above/below and the
     combined eight-way `direction` (rear-left, front-right, ... — when
     the options use those words, it IS the answer), with a `conclusion`
     sentence naming everything (copy it). For a camera observer the
     frame is its gravity-aligned ground heading, same as the bev map —
     camera tilt cannot skew the verdicts. `depth` is the specialist for
     nearer/farther ranking across targets (`order`) and `--known` depth
     calibration. Both take `--view N` for a camera or `--observer NAME`
     for anything with an orientation:

     ```bash
     python3 /app/perception.py side --view 1 --of bed
     python3 /app/perception.py depth --view 1 --of bed --of desk
     ```
   - **Any height comparison** — taller/shorter, higher/lower, "is X
     above/below Y", "which is highest": `elevation --of X --of Y` —
     heights along gravity, first `--of` is the reference, camera tilt
     cannot skew it. This holds WHETHER OR NOT an observer appears in
     the sentence: height runs along gravity, and no observer changes
     which of two things is higher. `side`'s above_or_below answers a
     different, egocentric question — offset along the OBSERVER'S OWN
     up axis (someone lying down has a horizontal "up") — use it only
     when the question asks that.

     **Bind the observer BEFORE reading.** The question's observer is a
     camera ONLY when it points at a picture ("in the last picture",
     "the photographer", "you" at the moment a frame was taken). ANY
     other observer — someone who enters, stands, sits or lies somewhere
     ("when you enter the bedroom", "lying in bed", "standing at the
     window") — does not exist in the scene until YOU register it.
     These stems ARE the signal, every time they appear: "when you are
     doing X", "if you do X", "imagine you ...", "imagine a person
     doing X" — a hypothetical person placed somewhere means REGISTER an
     instance at the described spot with `add_instance` (position +
     facing), then read the asked relation from it. The phrasing is the
     instruction; never answer such a question without the registered
     observer:

     ```bash
     python3 /app/perception.py add_instance --name entrant \
       --source "observer entering through the doorway (question stem)" \
       --at-map f0:310,545 --facing scene
     python3 /app/perception.py side --observer entrant --of lamp
     ```

     Position = the spot the question puts them. Once the bev map
     exists, the MOST PRECISE source is to point AT the map:
     `--at-map f<N>:PX,PY` — pixel coordinates on that frame's bev map
     PNG, exactly like pointing at any image (height fills in from the
     reconstruction there; requires having built and viewed that
     frame's map — picks on wide-angle photos land badly, so prefer the
     map). Otherwise a photo pixel (`--view N --pixel x,y`) or
     `--at-camera N` for "the photographer". Orientation = a
     PRIMARY axis (taken exactly) + a SECONDARY direction (projected
     orthogonal to it); values are instances (cameras and `scene`
     included), the literal `gravity`, or `bev:f<N>:START:END` — an
     arrow between two points on the bev map: each endpoint is a
     colon-free instance name (`bev:f0:bed:door` = the way bed→door
     lies on the map) or a `PX,PY` pixel on that map's PNG; use it when
     the heading is plain on the map but nothing nameable sits along
     it. Anyone ON THEIR FEET — standing, walking,
     running, entering — is upright: `--facing X` is the whole
     orientation (`--facing scene` = facing into the room; a walker
     faces their direction of travel), gravity fills in the vertical by
     default. When the facing is not obvious yet, register the POSITION
     alone, run `bev` to see the marker, then call `add_instance` again
     with the same name, no position and just the orientation — often a
     map arrow between two points you now see — and it orients the
     instance in place.
     For ANY other pose, do the work of finding the two axes yourself
     from the pictures BEFORE reading: the ONE of its axes you can pin
     exactly and to what (the primary — gravity often is, for anything
     resting on a surface), and ONE direction that fixes the remaining
     spin (the secondary — a nameable instance one of its axes points
     toward; register a pixel as an instance first if nothing nameable
     sits there), then `--primary AXIS=VALUE --secondary AXIS=VALUE`.
     The readouts follow the frame YOU registered — `side`'s columns
     run along its forward, right and up axes (`depth` along forward
     only) — so translate the question's words into those axes, not the
     world's. Substituting a camera for a non-camera observer is the
     most common way to get a MIRRORED answer. It is allowed only with
     scene evidence about the camera's POSE — on the bev map its wedge
     stands at the observer's spot AND points the observer's way (an
     entrant faces INTO the room; a wedge pointing out through the door
     is not them). The readouts themselves are answers, never evidence:
     an object reading as behind the observer may simply BE behind them
     — do not use a readout's sign to accept or reject the stand-in.
   - **Scene layout relative to the reference camera**: look at the
     step-3 map (1 m grid, reference camera at the bottom facing up the
     map); any relation you go on to STATE still reads from `side
     --view N` / `distance` — the map is for seeing, the calls are for
     saying.
   - **Compass directions — MANDATORY whenever a compass word appears
     ANYWHERE in the question or options** (north/…, a "X corner", a "X
     side" — of the room OR of an object; "which side of the statue do
     you see" is a compass question, not an appearance question). Anchor
     from the ONE stated orientation fact and read the asked fact back in
     the same call (translation rules in the API section):

     ```bash
     python3 /app/perception.py compass \
       --start camera:f0 --end bed --orientation southwest \
       --query "desk->shelf"
     ```

     The mapping persists in the scene — later compass calls query
     directly, and later maps draw the rose. The query reply's
     `direction`/`conclusion` ARE the answer — copy the direction; if it
     is not among the options, take the highest-ranked direction in
     `nearest` that IS an option. The reply is FINAL. If you believe it
     is wrong, the anchor is what to fix; re-run, do not overrule.
   - **Object motion / rotation — MANDATORY whenever the question asks
     whether or how an object moved, rotated, or turned.** `segment`
     already tagged every instance `moving` or `static` — that tag
     settles WHETHER and WHICH (copy it; never re-judge from pixels);
     what remains is HOW. FIRST settle
     which instance the question means (`drop` the twins in EVERY frame —
     two pans, one held one resting, means the held one). Then build the
     split pair for X seen in frames i and j:

     ```bash
     python3 /app/perception.py bev --frame 1 --erase fj:X
     python3 /app/perception.py bev --frame 1 --erase fi:X
     ```

     `view` BOTH maps — they are the VISUAL check: same ground frame,
     camera factored out, immune to camera roll (rotation shows in the
     two bottom-left insets, handle/spur direction). The readout you
     STATE comes from `motion`:

     ```bash
     python3 /app/perception.py motion --of fi:X --of fj:X
     ```

     Its `conclusion` says the shift and rotation outright — copy it. A
     reply with `erase_warning` means a twin is still mapped — go back,
     `drop`, re-run. Only horizontal-axis tilts and vertical flips are
     invisible top-down: judge those from the images, and run `motion`
     on the two CAMERAS first — its conclusion states how the camera
     leaned between the shots, which otherwise masquerades as object
     motion.
   - **Camera movement** ("did the camera move/turn between the
     shots?"): `motion --of camera:f0 --of camera:f2` — the conclusion
     states how it moved (forward/right/up in the first camera's ground
     frame) and turned (clockwise/counterclockwise), gravity-referenced.
     The map's wedges are the visual check.
   - **Object orientation** ("which way does X face?"): if X is static
     relative to the scene (furniture almost always is), apparent
     rotation across frames is CAMERA motion — read `motion` on the two
     cameras (the wedges on the map are the picture of it). For X's own
     facing, identify its visible front in one view, then register it
     (`add_instance --facing NAME` toward whatever its front points at,
     or `--primary`/`--secondary` for a non-upright pose) and read
     `side --observer X`.
6. **Answer from the scene readouts FIRST.** The scene calls and the map
   are the primary basis for every spatial judgment. The frames serve two
   roles: verifying the grounding (does the overlay color the object the
   question means?) and appearance facts no scene carries (what something
   is, its color, its visible side). When a scene readout and a direct
   image reading disagree, FIRST re-check the grounding on the overlay —
   a wrong or duplicate mask is fixed by re-segmenting or `drop`, not by
   abandoning the scene. Only with the grounding verified and the readout
   still physically implausible (objects piled on one spot, absurd
   distances) fall back to the images, and say so explicitly. `run
   list_instance` whenever unsure what the scene holds. Do not call
   `detect`, `measure`, `match`, `egomotion`, `gravity`, `render`, or the
   Python `look()` API.
7. Keep the answer file current while investigating. Re-read the question
   and the answer file before finishing.
