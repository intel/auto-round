# Remove Threaded Packing From Exporters

## Summary
- Replace worker-thread packing with a plain per-layer loop in every exporter that used the shared `ThreadPoolExecutor` packing pattern.
- Keep model formats, saved configs, meta-device checks, and progress reporting behavior unchanged.
- Cover the regression with tests that fail if packing tries to construct a thread pool or call `threadpoolctl` again.

## Implementation
- Update these exporters to pack layers serially:
  - `auto_round/export/export_to_autogptq/export.py`
  - `auto_round/export/export_to_autoround/export.py`
  - `auto_round/export/export_to_awq/export.py`
  - `auto_round/export/export_to_llmcompressor/export_to_fp.py`
  - `auto_round/export/export_to_llmcompressor/export_to_static_fp.py`
  - `auto_round/export/export_to_autoround/export_to_fp8.py`
  - `auto_round/export/export_to_autoround/export_to_nvfp_mx.py`
- Remove the now-unused `ThreadPoolExecutor` and `threadpoolctl` imports from those files.
- Leave the rest of each export flow untouched, including `pack_layer(...)` behavior and output serialization.

## Tests
- Add CPU export regression tests for:
  - `auto_gptq`
  - `auto_awq`
  - `auto_round`
  - `auto_round:auto_awq`
  - FP8 `auto_round`
- Add CPU llm-compressor regression tests for:
  - static FP8 export
  - MXFP8 export
- Extend NVFP4 CPU save-quantized coverage so it fails if the autoround NVFP/MX export path tries to use a thread pool.

## Assumptions
- The target repo is `auto-round`.
- The intended fix is to remove threaded packing entirely rather than keep a partial hardware-specific workaround.
- No public API or config schema changes are required.
