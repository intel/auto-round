---
name: verify
description: Run lint, format checks, and tests to verify changes before committing. Use after implementing a feature or fix to catch issues early.
---

# Verify Changes

Run these checks in sequence. Stop at first failure and fix before proceeding.

## Step 1: Ruff Lint

```bash
ruff check auto_round/ auto_round_extension/ --no-cache
```

Fix any issues with `ruff check --fix`.

## Step 2: Black Format Check

```bash
black --check --line-length 120 auto_round/ auto_round_extension/
```

Fix with `black --line-length 120 <files>`.

## Step 3: isort Import Order

```bash
isort --check --profile black -l 120 --known-first-party auto_round,auto_round_extension auto_round/ auto_round_extension/
```

Fix with `isort --profile black -l 120 --known-first-party auto_round,auto_round_extension <files>`.

## Step 4: CPU Tests

```bash
pytest test/test_cpu/ -x -q
```

If specific files were changed, scope tests:
```bash
pytest test/test_cpu/ -x -q -k "relevant_test_name"
```

## Step 5: Codespell (optional, for docs/strings changes)

```bash
codespell auto_round/ auto_round_extension/ --skip="*.safetensors,*.bin,*.pt"
```

## Reporting

After all checks pass, report a one-line summary: "All checks passed: ruff ✓, black ✓, isort ✓, tests ✓"

If any check fails, show the error and fix it before proceeding to the next check.
