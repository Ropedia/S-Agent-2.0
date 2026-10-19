---
name: compass-calibration
description: Use when the question or the options are in compass terms — north, south, east, west, or a bearing in degrees. The reconstruction's axes are the FIRST CAMERA's axes, not the world's, so nothing in the 3D data says which way is north. Read this before mapping any axis to a compass direction; getting the mapping wrong puts the answer exactly 90 or 180 degrees out.
---
# Which way is north?

**Nothing in the reconstruction knows.** `look()` puts the first camera at the
origin looking along its own axis, so `+X` is that camera's right and `+Z` its
forward. Neither has any relation to north. A guess here does not make the
answer slightly wrong — it makes it exactly 90 or 180 degrees wrong, which
usually lands on a different option.

So: **north has to come from the question text, or you do not have it.**

## When the question names a direction

Most compass questions state one, e.g. "the book storage area is on the west
wall", "Figure 1 shows the north side of the statue". That sentence is your
only calibration. Use it as a measurement, not as a hint:

```python
scene = look("frame_0.png", "frame_1.png")
book = scene.find("book storage")          # the thing the question anchors
# Its bearing from the room centre tells you which axis direction is "west".
print(book.where())
```

If the anchored object sits toward `-X`, then `-X` is west, so `+X` is east,
`+Z`/`-Z` are north/south — work out which by the right-hand rule for the
scene, then **verify against a second known fact before trusting it**.

## Verify, because the derivation is easy to get backwards

The common failure is a chain like "the camera faces the north wall, so `+Z` is
north, so `+X` is east" — plausible, and wrong half the time, because whether
the camera's right is east or west depends on whether it is facing the wall or
standing against it.

**Check your mapping against something else the frames show.** If the question
says the blackboard is on the north wall and you also see a window the scene
describes as east-facing, your mapping must put both where they belong. If it
cannot, flip the axis and check again. Two agreeing facts is the difference
between a calibration and a guess.

State the mapping explicitly before you use it — "+Z = north, +X = east,
verified against the window" — so that a wrong answer is traceable to the
mapping rather than buried in arithmetic.

## When the question names no direction

Some compass questions give no anchor at all. Then the reconstruction cannot
answer them, and neither can you by measurement. Say so in your reasoning and
fall back on what the scene itself implies — sun direction, a labelled map, the
convention that building fronts face streets — and treat the answer as the
weaker kind of evidence it is. **Do not silently assume `+X` is east**: that is
the step that produces confidently wrong answers.

## Relative directions do not need any of this

"Left of", "in front of", "closer to the camera" are answerable directly in the
camera's own axes, with no compass involved:

```python
thing.direction_from(scene.camera(0))   # front-right, behind-left, ...
```

If the options are relative rather than compass terms, skip this skill and see
**spatial-frames**.
