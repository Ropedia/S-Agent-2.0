# Spatial reasoning task

Answer one multiple-choice question about the image(s) below. It may ask about
3D layout, distance, object identity, camera movement, or a robot trajectory.

## Question

{question_block}

## Images

{image_block}

{vision_block}

{perception_block}

{workflow_block}

## Answering

Write one letter — {choice_letters} — to `/app/answer.txt`:

```bash
echo -n "{example_letter}" > /app/answer.txt
```

Write your current best answer early and replace it if your conclusion changes.
Before finishing, read the file back:

```bash
cat /app/answer.txt
```

The file must contain exactly one choice letter. A missing file, an empty file,
or an essay scores zero. If uncertain, commit to your best letter.
