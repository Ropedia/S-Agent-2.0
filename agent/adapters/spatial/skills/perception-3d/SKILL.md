---
name: perception-3d
description: Use whenever the answer turns on a number you would otherwise guess from the picture -- a distance, a height, a size, how far apart two things are, which of two is nearer, or how the camera moved. Returns metric values in metres from a 3D reconstruction. Reach for this any time you catch yourself about to estimate a quantity by eye, because that is the most common way these questions are answered wrongly.
---
# Measuring instead of estimating

Depth is what a single image genuinely does not tell you: two objects that look
adjacent can be metres apart, and apparent size confounds "big" with "close".
If the question asks how far, how big, how tall, or which is nearer, measure it.

```python
from perception import look

scene = look("frame_0.png", "frame_1.png")   # all frames in ONE call
stove, sink = scene.find("stove"), scene.find("sink")

print(stove.distance_to(sink))   # 1.83   metres
print(stove.where())             # (0.41, -0.12, 2.05)  metres, world frame
```

That world frame is the **first camera's** frame, not a gravity-aligned one, so
its `+Y` means "up in the first photo" — measured 35.8 degrees off true vertical
on a real pair. See the height question below before using `y` for anything.

**Pass every frame the question shows to one `look()` call.** Frames measured
together share one coordinate system, which is what makes multi-view questions
answerable; measuring them separately produces numbers that cannot be compared.
Nothing needs carrying between steps — no arrays, no paths, no grid alignment.

## Three question shapes this settles

**"How far apart / how big?"** `distance_to` directly. For a size, measure
across the object — find two parts of it and take the distance, or use `.box`
if the question is really about extent in the image.

**"How tall / which is higher / is it above?"** Do **not** read the `y` from
`.where()` — measure along the real vertical, which is different in every scene:

```python
print(scene.height_of(stove))   # metres above the fitted ground plane
print(scene.up_vector())        # the unit vector it used, if you want to see it
```

On one object, raw `y` read **0.084 m** against **0.330 m** along true up — a 4x
error, and the sign flips outright on a tilted camera. Compare `height_of(a)`
with `height_of(b)` for "is A above B". Where no ground can be fitted,
`height_of` raises and `up_vector()` returns `None` rather than falling back to
`+Y`: that means *no reliable height here*, so answer from a second line of
evidence, not from `y`.

**"Which is closer to X?"** Measure both and compare rather than trusting one
absolute number: a systematic scale error affects both equally and cancels.

**"How did the camera move?"** Camera objects support `.where()`,
`.distance_to()`, `.direction_from()` — same API as Thing objects:

```python
cam_a, cam_b = scene.camera(0), scene.camera(1)
print(cam_a.where())               # (0.0, 0.0, 0.0)
print(cam_a.distance_to(cam_b))   # 0.24 m
print(cam_a.direction_from(cam_b)) # "left and behind"
```

For frame-of-reference questions, see **spatial-frames**.

## Turning a number into an answer

The options are usually coarse, which works in your favour: 1.83 m against
`A: 0.5m  B: 2m  C: 5m  D: 10m` is unambiguous. Pick the nearest rather than
expecting an exact match — the reconstruction has real error, and a value
between two options means double-check rather than round blindly.

**When two objects' distances differ by a few centimetres, do not answer from
that gap.** That is inside the noise. Find a second line of evidence
(occlusion, relative size, what the camera motion implies) and prefer the
conclusion two independent signals agree on.

## When not to trust the number

The first `.where()` builds the reconstruction and later calls reuse it, so
measurements within one scene are mutually consistent. But:

- **A confident number on the wrong object is the main failure mode.** The
  measurement is only as good as the detection under it. Run `.show()` on
  whatever your answer rests on and look at the mask.
- **Reflective, transparent and very distant surfaces** have no reliable depth;
  asking for a position there raises an error, which is honest rather than a
  malfunction. Measure a nearby solid object instead.
- **Sanity-check the magnitude.** Indoor furniture metres apart is plausible;
  0.02 m or 400 m means something upstream is wrong. See **evidence-check**.
