# Claude Skills for AutoRound

This directory contains Claude Code skills maintained for the `auto-round`
repository. These skills capture repeatable workflows for common contributor
tasks such as adding quantization data types, export formats, VLM model support,
inference backends, and pull request review.

## Directory Structure

Each skill lives in its own directory under `.claude/skills/`. A skill may
include:

- `SKILL.md`: the main workflow and operating instructions
- `references/`: focused reference material used by the skill

## Available Skills

- `add-quantization-datatype`: guides integration of a new quantization data
  type (e.g., INT, FP8, MXFP, NVFP) into the `auto_round/data_type/` registry
- `add-export-format`: covers addition of a new model export format (e.g.,
  auto_round, auto_gptq, auto_awq, gguf, llm_compressor)
- `add-vlm-model`: walks through adding support for a new Vision-Language Model,
  including template, calibration dataset, and block handler registration
- `add-inference-backend`: guides integration of a new hardware inference backend
  (e.g., CUDA, HPU, Triton)
- `review-pr`: provides a structured workflow for reviewing pull requests,
  including Chinese translation verification

## Maintenance Guidelines

- Keep skill names short and task-oriented.
- Prefer repository-local paths, commands, and examples.
- Avoid hardcoding fast-changing support matrices unless the skill is actively
  maintained alongside those changes.
- Treat skills as contributor tooling: optimize for clarity, actionability, and
  low maintenance overhead.
