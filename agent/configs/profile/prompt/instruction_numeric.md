# Spatial reasoning task

Answer one question about the image(s) below. It asks for a **quantity** — how
many of something there are, how far apart two things are, or how large one is.

## Question

{question_block}

## Images

{image_block}

{vision_block}

{perception_block}

{workflow_block}

## Answering

Write a single number to `/app/answer.txt` — digits only, no units, no words:

```bash
echo -n "3.5" > /app/answer.txt
```

Answer in the unit the question names — it says *in meters* or *in
centimeters*; a count has no unit and is a whole number. Do not write a range,
an inequality, or a number with a unit attached — `2.4`, not `2.4 m`, `~2.4`,
or `2-3`.

Write your current best answer early and replace it if your conclusion changes.
Before finishing, read the file back:

```bash
cat /app/answer.txt
```

The file must contain exactly one number. A missing file, an empty file, or an
essay scores zero. An estimate is worth far more than nothing: this question is
scored on how *close* the number is, so a rough measurement still earns partial
credit while a refusal earns none.
