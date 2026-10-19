---
name: perception-detect
description: Use when the question turns on WHICH thing you are looking at -- is it present, how many are there, which of several similar objects does the question mean, or exactly where is it in the frame. Also the first step of nearly every measurement, because the 3D tools need to know which object you mean. Returns objects you can then measure, compare or look at.
---
# Finding the object the question is about

Almost every spatial question names things (`the stove`, `the second chair`) and
you have to decide which pixels that is before anything else is meaningful. A
measurement of the wrong object is worse than no measurement -- it comes back
confident.

```python
from perception import look

scene = look("frame_0.png", "frame_1.png")   # every frame the question shows
stove = scene.find("stove")                  # -> a Thing (or Things, if several)
print(stove)                                 # <Thing 'stove' frame=0 score=0.91>
```

`find` returns everything it matched, ordered by confidence, and using it as a
single object means "the best match". When the question is about a particular
one of several, index it: `chairs = scene.find("chair"); chairs[1]`.

**Nothing found is a real answer, not an error.** `scene.find("bicycle")`
printing `<no 'bicycle' found>` is evidence the bicycle is absent -- which for
a presence question *is* the answer. Try a different word once before believing
it; `couch` and `sofa` are not the same query.

## Three things this settles on its own

**How many.** `len(scene.find("chair"))` answers counting questions directly.
Look at the scores before trusting a large count: several low-confidence hits on
the same object inflate it.

**Which one.** When two candidates are plausible, `.show()` is the deciding
move -- it draws the mask over the frame so you can see what got selected:

```python
chairs = scene.find("chair")
print([c.score for c in chairs])   # [0.93, 0.88, 0.41]  <- the third is doubtful
print(chairs[0].show())            # /perception/renders/overlay_a1.png
```

Then `view` that path. **Do this whenever your answer rests on one detection.**
A mask covering the wrong object is the single most common way these questions
go wrong, and it is invisible unless you look.

**Where in the frame.** `.box` gives pixel corners `(x1, y1, x2, y2)`, enough
for "which side of the image" without any 3D work.

## Feeding a measurement

For anything metric -- distance, size, which is nearer, what direction -- pass
the object straight to the measuring methods. The alignment between a detection
and the 3D reconstruction is handled inside the library, so there is nothing to
convert and no array paths to carry around:

```python
stove, sink = scene.find("stove"), scene.find("sink")
print(stove.distance_to(sink))                 # metres
print(stove.direction_from(scene.camera(0)))   # 'front-right'
```

See **perception-3d** for what those numbers mean and when to distrust them,
and **spatial-frames** before answering any left/right/front/behind question --
the answer depends on whose viewpoint the question assumes.
