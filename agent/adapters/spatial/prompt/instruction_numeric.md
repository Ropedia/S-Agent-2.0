# Spatial reasoning task

Answer one question about the image(s) below. This is an embodied / spatial
reasoning question, and it asks for a **quantity**: how many of something there
are, how far apart two things are, or how large one is.

## Question

{question_block}

## Images

{image_block}

{vision_block}

{perception_block}

{workflow_block}

## Answering

When your checklist is satisfied and not before, submit your answer with this
one command — it writes the number, reads it back so you can see it landed, and
puts the same number in your output as a backstop:

```bash
echo -n "3.5" > /app/answer.txt && \
  cat /app/answer.txt && echo "ANSWER: $(cat /app/answer.txt)"
```

One number, in the unit the question names — it says *in meters* or *in
centimeters*; a count has no unit and is a whole number.

Requirements:

- `/app/answer.txt` must hold just the number. No units, no explanation, no
  restated question, no reasoning transcript.
- Not a range and not an inequality: `2.4`, never `2.4 m`, `~2.4` or `2-3`.
- A file containing an essay scores zero even if the essay reaches the right
  conclusion.
- If you are unsure, still commit to your single best estimate. This question is
  scored on how *close* the number is, so a rough measurement earns partial
  credit while a missing or empty file scores zero.
