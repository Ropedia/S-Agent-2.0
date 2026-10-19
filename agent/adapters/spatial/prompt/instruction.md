# Spatial reasoning task

Answer one multiple-choice question about the image(s) below. This is an
embodied / spatial reasoning question: it may ask about 3D layout, distances,
object state, camera viewpoint changes, or where a robot gripper trajectory
ends up.

## Question

{question_block}

## Images

{image_block}

{vision_block}

{perception_block}
{workflow_block}
## Answering

When your checklist is satisfied and not before, submit your answer with this
one command — it writes the letter, reads it back so you can see it landed, and
puts the same letter in your output as a backstop:

```bash
echo -n "{example_letter}" > /app/answer.txt && \
  cat /app/answer.txt && echo "ANSWER: $(cat /app/answer.txt)"
```

One letter — {choice_letters}.

Requirements:

- `/app/answer.txt` must hold just the letter. No explanation, no restated
  question, no reasoning transcript.
- A file containing an essay scores zero even if the essay reaches the right
  conclusion.
- If you are unsure, still commit to your single best letter. A missing or
  empty file scores zero.
