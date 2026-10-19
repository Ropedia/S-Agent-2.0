# Spatial reasoning task

Answer one question about the image(s) below. This is an embodied / spatial
reasoning question, and it asks how the camera moved between two views, as a
set of named quantities.

## Question

{question_block}

## Images

{image_block}

{vision_block}

{perception_block}

{workflow_block}

## Answering

When your checklist is satisfied and not before, submit your answer with this
one command — it writes the line, reads it back so you can see it landed, and
puts the same line in your output as a backstop:

```bash
echo -n "move_left:2.6,move_down:0.1,move_forward:0.2,rotate_up:10,rotate_left:0" > /app/answer.txt && \
  cat /app/answer.txt && echo "ANSWER: $(cat /app/answer.txt)"
```

Requirements:

- `/app/answer.txt` must hold just that one comma-separated `name:value` line,
  in exactly the format the question specifies. No explanation, no units
  spelled out, no reasoning transcript.
- Use the direction words the question offers; a value is never negative
  (`move_left:2.6`, not `move_right:-2.6`).
- Translations in metres, rotations in degrees, and all five components
  present even when one of them is `0`.
- If you are unsure, still commit to your best estimate on every axis. Each
  component is scored on how *close* it is, so an estimate earns partial credit
  while a missing or empty file scores zero.
