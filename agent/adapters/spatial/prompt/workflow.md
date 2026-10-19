## How to work this problem

**1. Settle what is being asked.** Before anything else, name the quantity and
the frame of reference. A spatial question almost never states which frame it
assumes — "to the left of" can mean left in the image, left from the camera at
one particular frame, or left from the object's own facing direction, and those
are different answers. Get this wrong and no amount of correct computation
saves you.

**2. Write down what would make an answer defensible.** Put it in a file so it
survives a long investigation:

```bash
cat > /app/checklist.md <<'EOF'
- which object must be correctly identified
- which quantity must be measured, and in which frame
- what result would change my mind
EOF
```

Fill it in with the specifics of *this* question, then re-read it before you
commit. Do not guess an answer yet — you have no evidence.

**3. Decide what looking cannot settle.** Your eyes are reliable for what is in
the scene and roughly where. They are unreliable for anything metric, and for
anything that depends on which viewpoint the question means. Reach for the
perception tools whenever the answer turns on:

- a distance, a size, a height, a depth, or "which is closer / larger"
- which object is which, or exactly which pixels an object covers
- left / right / in front / behind — these depend on *whose* frame of
  reference, and the image alone cannot tell you
- how the camera moved between frames
- **a compass direction — north, south, east, west, or a bearing.** The 3D data
  does not know which way is north; that has to come from the question text or
  you do not have it
- anything you would otherwise answer by eyeballing a number

And once you have a result, before it carries your answer:

- **is this the right object?** Render it and look — a mask on the wrong thing
  produces a confident wrong number, and nothing in the number says so
- **do two sources agree?** When they disagree, the disagreement has a cause
  worth finding rather than a side worth picking

Reading a quantity off an image is the most common way to get these wrong.
If the question asks for a quantity, measure it.

**4. Read the skill before you act, every time.** Each skill carries both how a
capability is invoked *and* how to reason with it — which frame its output is
in, how to read what comes back, which spellings get rejected. `invoke_skill`
first, then act. Working from memory is what produces a rejected call, or worse
a number that looks right and is not. This is not optional overhead: the skills
are where the method lives, not just the syntax.

**5. Verify before you commit.** A detection can return a confident mask of the
wrong object; a reconstruction can return plausible depth that is wrong.
Neither announces it. Before you commit to the one result your answer really
rests on, `invoke_skill` **evidence-check** — it lists what each kind of
evidence is good and bad at, and the specific mistakes that have produced
confident wrong answers here. Then look at that result yourself, and prefer a
conclusion two independent sources agree on over one resting on a single
number.
