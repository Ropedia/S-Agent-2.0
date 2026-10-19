# Profile

The prompt and skill files that define S-Agent 2.0's task instruction.

- `prompt/instruction.md` — the task instruction template (multiple-choice items)
- `prompt/instruction_numeric.md`, `prompt/instruction_vci.md` — the numeric and
  view-change answer contracts used by some ReVSI / SPAR items
- `prompt/perception_api.md` — the toolkit tutorial: every command, its reply
  shape and how to read it
- `prompt/workflow.md` — the suggested order of operations; `workflow_nopercept.md`
  is the no-tools control
- `skills/cross-view-reasoning/SKILL.md` — the cross-view reasoning card

`scripts/run.sh` points the task generator at this directory.
