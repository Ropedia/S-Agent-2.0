## How to work this problem

**1. Settle what is being asked.** Before anything else, name the quantity and
the frame of reference. A spatial question almost never states which frame it
assumes — "to the left of" can mean left in the image, left from the camera at
one particular frame, or left from the object's own facing direction, and those
are different answers. Get this wrong and no amount of careful looking saves
you.

**2. Write down what would make an answer defensible.** Put it in a file so it
survives a long investigation:

```bash
cat > /app/checklist.md <<'EOF'
- which object must be correctly identified
- which quantity must be established, and in which frame
- what would change my mind
EOF
```

Fill it in with the specifics of *this* question, then re-read it before you
commit. Do not guess an answer yet — you have no evidence.

**3. Be honest about what the image can settle.** Your eyes are reliable for
what is in the scene and roughly where. They are unreliable for anything
metric, and for anything that depends on which viewpoint the question means —
left / right / in front / behind change with the frame of reference, and 2D
appearance cannot separate an object moving from the camera moving.

**4. Reason from what you can actually see.** When the question turns on a
quantity you cannot read off the image, say what you are estimating it from and
how confident that makes you, rather than asserting a number as if it were
measured.

**5. Verify before you commit.** Re-read your checklist. Prefer a conclusion
supported by two independent observations over one resting on a single
impression, and when two observations disagree, find the concrete reason rather
than picking the one you prefer.
