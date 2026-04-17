# FP4 packing test memory gate plan

## Request

Update `test/test_cuda/quantization/test_packing.py` so `test_packing_fp4` only exercises `torch.Size([151936, 2048])` when the CUDA node has more than 40 GB of memory, then verify, commit, and open a PR.

## Finishing criteria

1. Only the `torch.Size([151936, 2048])` parameter is conditionally skipped.
2. The skip condition is based on available CUDA device memory and has a clear reason.
3. The remaining shapes continue to run unchanged.
4. The targeted test file is verified as far as the current environment allows.
5. The final git diff is committed without touching unrelated worktree changes.
6. A PR is opened if git remote configuration and authentication allow it.

## Approach

1. Inspect the test file and nearby test conventions for resource-gated CUDA tests.
2. Add a small helper that checks whether the active CUDA device has more than 40 GiB of memory.
3. Convert the large shape entry into a `pytest.param(...)` with a `skipif` mark using that helper.
4. Run targeted pytest coverage for `test_packing_fp4`.
5. Commit the targeted file plus this plan note.
6. Push a branch and create a PR.
