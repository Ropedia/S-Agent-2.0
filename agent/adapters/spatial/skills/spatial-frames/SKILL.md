---
name: spatial-frames
description: Use for any question about left/right, front/behind, above/below, a direction, or where one thing is relative to another or to the viewer. Read this BEFORE computing anything, because these questions are ambiguous by default -- the same scene gives opposite answers depending on whose viewpoint the question means, and picking the wrong one makes a perfect measurement wrong. Also covers camera motion between frames.
---
# Whose left is it?

Almost every spatial question assumes a frame of reference and almost none says
which. **Settling that is the first job, before any measurement.** Getting it
wrong is not a small error: left and right swap outright.

Four frames, and questions really do use all of them:

- **Image space** — where things sit in the picture. Says nothing reliable about
  the world: an object drifting rightward may have moved right, or the camera may
  have panned left. **Never answer a real-world direction from 2D appearance.**
- **Camera space** — left/right/front/behind *as seen from one camera at one
  frame*. Changes every frame, so always name the frame.
- **World space** — fixed coordinates shared by all frames, which is what makes
  multi-frame questions answerable at all.
- **Object space** — "to the car's left" is not your left: an object facing you
  has its left/right **mirrored**. Work out which way it faces, then define
  left/right from that heading.

## Reading the question

Match the wording to the frame before touching a tool:

| the question says | frame it means |
|---|---|
| "in Photo 2, which side of the frame…" | image space |
| "standing at the doorway looking in…" | camera space, that viewpoint |
| "from the fridge's point of view…" | object space — mirrored |
| "which is closer to the window" | world space |
| "how did the camera move between…" | world space, compare two cameras |

## Computing it, not eyeballing it

```python
from perception import look

scene = look("frame_0.png", "frame_1.png")
stove = scene.find("stove")
print(stove.direction_from(scene.camera(0)))   # 'front-right'
```

`direction_from` answers in the **camera's own frame**, which is what a
"standing here, looking that way" question asks for. Give it the camera of the
frame the question names — `scene.camera(1)` for Photo 2. Mixing frames is the
most common way to get a direction question backwards.

**Object space needs one more step.** `direction_from` has no notion of which
way the *object* faces, so for "to the car's left" find what the car faces
(its windshield, its front wheels), get both positions with `.where()`, and
reason about the heading yourself. Then remember the mirror: if the car faces
the camera, its left is your right.

**Above and below are the exception.** Up comes from gravity, not from anyone's
viewpoint, so no choice of frame settles it — and the `y` of `.where()` is the
first camera's axis, not the real vertical. Use `scene.height_of()`; the
mechanics and the failure case are in **perception-3d**.

## Camera motion between frames

"Did I move forward? Turn left?" is about two cameras, not any object:

```python
a, b = scene.camera(0), scene.camera(1)
print(a.position(), b.position())   # world metres; the difference is the motion
```

Compare componentwise against the first camera's axes: a change along its
forward axis is approach or retreat, along its right axis a sideways move. If
the positions are nearly identical but the view changed a lot, the camera
rotated rather than translated.

## Before you commit

`direction_from` returning `'front'` for something visibly off to one side is a
signal worth checking, not a result to accept: the object may be near the
viewing axis, or the detection may have landed on the wrong thing. Run
`.show()` on it and look. A direction built on a misplaced mask is confidently
wrong, and nothing in the number tells you so — see **evidence-check**.
