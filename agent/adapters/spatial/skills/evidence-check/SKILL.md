---
name: evidence-check
description: Use before committing to an answer that rests on a perception result, and whenever two pieces of evidence disagree. Covers what each kind of evidence is actually good for, the one check that catches most wrong answers, and how to resolve a disagreement instead of picking a side. Read this when you have a number and are about to trust it.
---
# Before you trust it

The dangerous failure is not a tool that errors — you can see those. It is a
tool that succeeds on the wrong input and hands back a plausible number.

| Source | Good for | Fails at |
|---|---|---|
| Looking at a frame | what is present, roughly where, identity | anything metric; 2D motion is not camera motion |
| `find` | which pixels an object covers | confidently returns the **wrong object** on an ambiguous word |
| `where` / `distance_to` | metres, positions, comparisons | only as good as the detection under it |
| A rendered overlay | seeing whether a mask is sane | a projection — shows placement, hides precision |

## The one check worth doing every time

**Look at the mask your answer depends on.**

```python
stove = scene.find("stove")
print(stove.distance_to(scene.find("sink")))   # 1.83
print(stove.show())                            # then `view` that path
```

If the mask is on the counter beside the stove, that 1.83 was never a
stove-to-sink distance — and nothing in the number said so. This single step
catches more wrong answers than any other check, because a misplaced mask is
both the most common fault and the most confidently reported.

## When two sources disagree

Do not pick the one you prefer. The disagreement has a cause, and finding it
usually hands you the answer. In order:

1. **Wrong object.** Does the mask cover what the question asks about?
   `.show()` and look. Try a different word — `couch` and `sofa` do not match
   the same things.
2. **Wrong frame.** Is the camera the one the question means? A question about
   Photo 2 needs `scene.camera(1)`. Mixing frames is how directions come out
   backwards.
3. **Wrong reference.** Camera-relative and object-relative left/right are
   different answers, and an object facing you has its left/right mirrored.
   See **spatial-frames**.
4. **Below the noise.** Two distances differing by centimetres are not
   distinguishable. If your conclusion depends on that gap, you do not have a
   conclusion yet.

## Failure modes actually seen here

- **Reading coordinates off a rendered image by eye.** A percentage guessed from
  a picture is a guess wearing a number. Measure it.
- **Re-issuing a failed call unchanged.** The error text names the bad argument;
  read it and change that argument.
- **Treating "not found" as a broken tool.** `<no 'bicycle' found>` is evidence
  — either it is absent, or your wording missed it. For a presence question that
  *is* the answer.
- **Concluding from one measurement with nothing agreeing with it.** Prefer a
  conclusion two independent signals support: a measured distance plus what
  occlusion implies, or a direction plus what the camera motion implies.
- **Answering the more familiar question.** "Which is nearer the window" is not
  "which is nearer the camera". Re-read the question against your answer before
  writing it down.

## Committing

Write the letter to `/app/answer.txt` and say it in your reply. **An answer
stated only in prose scores zero** — the file is what gets read. If you are
unsure, still commit your single best letter; an empty file scores the same as a
wrong one, so a guess strictly dominates silence.
