# Integration Tests

Integration tests verify that AutoRound works correctly with external frameworks like vLLM, SGLang, and HuggingFace.

## Running Integration Tests

```bash
# Run all integration tests
pytest test/integration/ -v

# Run specific integration tests
pytest test/integration/test_cpu/ -v
pytest test/integration/test_cuda/ -v
```

## CI Schedule

These tests run in the **nightly** CI pipelines:

- CPU integration → `.azure-pipelines/nightly-test.yml`
- XPU integration → `.azure-pipelines/nightly-test-xpu.yml`
- CUDA integration (vLLM / SGLang / LLMCompressor) → `.azure-pipelines/weekly-test-cuda.yml`
