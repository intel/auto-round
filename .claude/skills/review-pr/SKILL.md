---
name: review-pr
description: "Review a pull request for the AutoRound repository with a structured checklist covering code quality, test coverage, documentation, Chinese translations, and quantization-specific concerns. Use when reviewing or preparing to submit a PR."
---

# Pull Request Review Workflow for AutoRound

## Overview

This skill provides a structured workflow for reviewing pull requests in the
AutoRound repository. It covers code quality, testing, documentation, and
project-specific requirements like Chinese translation parity.

## Review Checklist

### 1. Code Quality

- [ ] Code follows existing patterns in the codebase (decorator registration,
      factory patterns, etc.)
- [ ] No hardcoded paths or credentials
- [ ] Proper error handling at system boundaries
- [ ] No unnecessary abstractions or over-engineering
- [ ] Import organization follows existing conventions
- [ ] Apache 2.0 license header present on new files:
  ```python
  # Copyright (c) 2025 Intel Corporation
  #
  # Licensed under the Apache License, Version 2.0 (the "License");
  # ...
  ```

### 2. Quantization-Specific Concerns

- [ ] Numerical stability: scale computation avoids division by zero
- [ ] Gradient flow: uses `round_ste()` or equivalent STE for differentiable
      rounding
- [ ] Tensor shapes: group_size reshaping handles padding correctly
- [ ] dtype consistency: scale_dtype, compute_dtype used properly
- [ ] Memory efficiency: no unnecessary tensor copies on GPU
- [ ] Device handling: tensors moved to correct device before operations

### 3. Registration Points

When the PR adds new functionality, verify all registration points are updated:

| Feature | Registration Location |
|---------|----------------------|
| Data type | `auto_round/data_type/__init__.py` import + `@register_dtype` |
| Export format | `auto_round/formats.py` `@OutputFormat.register()` |
| VLM model | `special_model_handler.py` `SPECIAL_MULTIMODAL_BLOCK` + lists |
| Backend | `auto_round/inference/backend.py` `BackendInfos` dict |
| Dataset | `auto_round/calib_dataset.py` `@register_dataset` |
| Scheme preset | `auto_round/schemes.py` `PRESET_SCHEMES` dict |

### 4. Test Coverage

- [ ] New functionality has corresponding tests
- [ ] Tests use existing fixtures (`tiny_opt_model_path`, `dataloader`, etc.)
- [ ] Tests are placed in the correct backend directory (`test_cpu/`, `test_cuda/`, etc.)
- [ ] Tests use minimal iterations (`iters=2, nsamples=2`) for speed
- [ ] No flaky assertions (avoid exact float comparisons)

### 5. Documentation

- [ ] README.md updated if user-facing features change
- [ ] **Chinese translation updated**: Any changes to `*.md` files must have
      corresponding updates in their `*_CN.md` counterparts:
  - `README.md` → `README_CN.md`
  - `docs/step_by_step.md` → `docs/step_by_step_CN.md`
  - `docs/environments.md` → `docs/environments_CN.md`
- [ ] Translation maintains equivalent content and structure (not just copied
      English text)
- [ ] Docstrings added for new public APIs

### 6. Contributing Requirements

- [ ] Commits are signed off (`git commit -s`) per DCO
- [ ] No unrelated changes mixed in
- [ ] PR description clearly explains the motivation and changes
- [ ] Breaking changes are called out explicitly

## Chinese Translation Verification

This is a **hard requirement** for the AutoRound project. Use this procedure:

1. **Identify modified markdown files**:
   ```bash
   git diff --name-only HEAD~1 -- '*.md'
   ```

2. **Check for corresponding CN files**:
   For each modified `.md` file, verify a `_CN.md` counterpart exists and is
   also modified:
   - `README.md` → `README_CN.md`
   - `docs/step_by_step.md` → `docs/step_by_step_CN.md`
   - `docs/environments.md` → `docs/environments_CN.md`

3. **Compare structure**:
   - Same number of sections/headings
   - Same tables, code blocks, and links
   - Equivalent content (not machine-translated gibberish)

4. **Files that do NOT need CN translation** (no `_CN` counterpart exists):
   - `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`
   - `test/README.md`
   - `docs/publication_list.md`, `docs/tips_and_tricks.md`, accuracy result docs

## Common Issues to Watch For

### Quantization Bugs

- **Scale overflow**: Large models with small group_size can produce FP16 overflow
  in scales. Check for `torch.clamp` or `torch.finfo` guards.
- **Asymmetric zero-point drift**: Zero-points must be integer-rounded for INT
  quantization.
- **GGUF super-block alignment**: GGUF formats require specific block sizes
  (typically 256 elements). Verify padding/alignment logic.

### Export Compatibility

- **Format detection**: Verify `quantize_config.json` or equivalent metadata is
  saved correctly for the target framework to detect.
- **Weight name mapping**: Ensure packed weight names match what the inference
  framework expects.
- **Mixed-precision layers**: Layers excluded from quantization (e.g., `lm_head`)
  must be saved in their original format.

### Backend Selection

- **Priority conflicts**: New backends should not override existing backends unless
  intentional. Check `priority` values.
- **Feature checker coverage**: Ensure checkers don't silently reject valid layers
  (test with real model shapes).
