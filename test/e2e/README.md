# End-to-End Tests

E2E tests verify the complete quantization workflow from model loading
through quantization, serialization, and deployment to a real inference
engine (vLLM, SGLang, ...). They use **real, full-size models** on a
**real GPU** and are intentionally slow; they are designed to run on a
weekly CI schedule (or on a self-hosted GPU runner), not on every PR.

## Running E2E Tests

```bash
# All e2e tests on the default (small) matrix
pytest test/e2e/ -v

# A specific sub-suite
pytest test/e2e/test_cuda/ -v

# Only the vLLM throughput suite
pytest test/e2e/test_cuda/test_vllm_throughput.py -v -s

# Larger / more demanding model matrix (needs >=24 GiB GPU)
pytest test/e2e/test_cuda/ --e2e-model-preset=large -v -s

# Everything (small + large)
pytest test/e2e/test_cuda/ --e2e-model-preset=all -v -s
```

## CI Schedule

The CUDA e2e tests run in the **weekly** CUDA pipeline
(`.azure-pipelines/weekly-test-cuda.yml`) on a self-hosted / RunPod GPU agent.
The CPU e2e tests run in the **nightly** pipeline
(`.azure-pipelines/nightly-test.yml`).

## Layout

| Path | What it tests |
|------|---------------|
| `test/e2e/test_cuda/conftest.py` | Shared fixtures, env gates, model matrix, benchmark helpers |
| `test/e2e/test_cuda/test_vllm_throughput.py` | Quantize + serve with vLLM; measure tokens/s and TTFT |
| `test/e2e/test_cuda/test_sglang_throughput.py` | Quantize + serve with SGLang; measure tokens/s and TTFT |

## Skipping Cases

Each test that requires a real GPU **auto-skips** when:

- CUDA is not available (`pytest.skip("CUDA is not available ...")`).
- The GPU is SM 12.x (Blackwell) with CUDA < 12.9 — vLLM's `gptq_marlin`
  JIT kernel cannot compile. This is the same constraint as the
  existing integration suite.
- The GPU has less free memory than the case requires (e.g. the 7B
  cases need ≥18 GiB). The case skips with a clear message.

So running the file on a CPU-only host or a small GPU will not fail —
it will simply print `SKIPPED`.

## Benchmark Output

Each throughput test appends a JSON line to
`test/output/{vllm,sglang}_throughput.jsonl` with the following
fields:

```json
{
  "engine": "vllm",
  "model": "saved_w4_a16",
  "fmt": "auto_round",
  "bits": 4,
  "group_size": 128,
  "num_prompts": 4,
  "max_new_tokens": 64,
  "total_time_s": 12.34,
  "output_tokens_per_s": 25.7,
  "gen_tokens_per_s": 28.4,
  "ttft_s": 0.05,
  "sample_output": " Paris."
}
```

This file is intended for trend tracking in CI; the tests themselves
only assert that the engine produced non-empty, non-garbage output at
**≥1 tok/s** (a deliberately loose regression bound, not a perf gate).

## Adding New Cases

1. Add a new `ModelCase` to `DEFAULT_MODEL_CASES` or `LARGE_MODEL_CASES`
   in `conftest.py`.
2. If the new model is gated on hardware (e.g. only Ampere+), set
   `min_gpu_gib` so the fixture auto-skips on smaller cards.
3. Re-run the relevant test class locally and confirm the throughput
   number is recorded in the JSONL file.

## Local Debugging

```bash
# Run a single case, with stdout/stderr on
pytest test/e2e/test_cuda/test_vllm_throughput.py::TestVllmThroughput::test_quantize_and_serve[default-qwen3-1.7b-w4a16-auto_round] -v -s

# Override the quantize-and-save knobs to make a single run cheaper
# (fewer iters, fewer calibration samples) when iterating
QUICK=1 pytest test/e2e/test_cuda/test_vllm_throughput.py -v -s
```
