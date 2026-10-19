# Spatial reasoning task

Answer one question about the image(s) below. It asks how the camera moved
between two views, as a set of named quantities.

## Question

{question_block}

## Images

{image_block}

{vision_block}

{perception_block}

{workflow_block}

## Answering

Write the answer to `/app/answer.txt` in **exactly the format the question
specifies** — the comma-separated `name:value` list, nothing else:

```bash
echo -n "move_left:2.6,move_down:0.1,move_forward:0.2,rotate_up:10,rotate_left:0" > /app/answer.txt
```

Use the direction words the question offers (a value is never negative — say
`move_left` rather than `move_right:-2.6`), keep translations in metres and
rotations in degrees, and give every one of the five components even when a
component is `0`.

Write your current best answer early and replace it if your conclusion changes.
Before finishing, read the file back:

```bash
cat /app/answer.txt
```

The file must contain exactly that one line. A missing file, an empty file, or
an essay scores zero. Each of the five components is scored on how *close* it
is, so an estimate on every axis is worth far more than a refusal.
