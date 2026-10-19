## Perception API

A perception gateway runs **outside** this container, fronting GPU models:
Depth Anything 3 for metric depth and 3D structure, SAM 3 for text-prompted
detection and masks. Use it when a measurement would settle the question.

Reach it from Python, in `/app`, with `/usr/local/bin/python3`:

```python
from perception import look
scene = look("frame_0.png", "frame_1.png")   # every frame the question shows
```

**It takes no setup** — it finds the gateway itself, stages this task's frames,
and keeps detections and the 3D reconstruction on one grid, so there are no
array paths to carry between steps.

**What you can ask a scene, and when each is the right move, is in the skills
below — read the relevant one before you call.** They also cover how to turn
what comes back into an answer, and when a returned number should not be
trusted, which is the part that decides whether a measurement helps you.

A CLI form exists too (`/app/perception.py detect|reconstruct|measure|render`),
but it needs more steps and can get the grids wrong.

{skill_bodies}
