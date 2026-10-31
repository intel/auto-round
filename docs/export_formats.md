# Output formats and export implementation

All output formats are implemented under `auto_round/export/formats/`. The former
`export_to_*` and `formats/backends` directories have been removed, without import
compatibility facades. Update direct imports of those internal modules.
The high-level `AutoRound.save_quantized` API and supported format strings remain.

## Layout

```text
auto_round/export/
  base.py                       OutputFormat and capability checks
  registry.py                   registration, aliases and public names
  resolver.py                   format selection and scheme resolution
  common/
    packing.py                  model preparation, traversal, packing utilities
    artifacts.py                tokenizer and processor saving
    serialization.py            checkpoint/shard writing and output layout
  formats/
    gptq/                       format.py, packing.py, config.py
    awq/                        format.py, packing.py, config.py, utils.py
    autoround/                  format.py, integer.py, fp8.py, float.py, qlinear modules
    compressed_tensors/         format.py, integer.py, float.py, static_fp.py, config.py
    gguf/                       format.py, writer.py, conversion and GGUF helpers
    mlx/                        format.py, packing.py, config.py
    svdquant/                   format.py, packing/export helpers and model adapters
    fake.py
    fp8.py
```

A format owns its compatibility checks, packing implementation, checkpoint
metadata and saving behavior. Complex formats use a package; a small format can
use one module. Shared tools implement actual common operations rather than a
second dispatch system.

## Adding a format

1. Create a format module or package under `formats/`. Subclass `OutputFormat`
   and register it with `@register_format("my_format")`.
2. Implement `pack_layer` and `save_quantized` in that format. Any
   format-specific packing/configuration helpers belong in the same package.
3. Import the module in `registry.load_builtin_formats` so its registration runs.

No exporter module, hard-coded supported-name list, resolver edit or compressor
edit is required for an ordinary format. Builtins load lazily to avoid cycles
when utility code imports lightweight GGUF configuration. Keep optional runtime
imports local to the functions that use them.

```python
from auto_round.export import OutputFormat, register_format


@register_format("my_format")
class MyFormat(OutputFormat):
    def pack_layer(self, layer_name, model, device=None, **kwargs):
        # Replace the selected module with this format's packed representation.
        ...

    def save_quantized(
        self,
        output_dir,
        model=None,
        tokenizer=None,
        layer_config=None,
        inplace=True,
        device="cpu",
        serialization_dict=None,
        **kwargs,
    ):
        # Build format-specific metadata and write this format's checkpoint.
        ...
        return model
```

`pack_layer` mutates the selected module; its return value is ignored. The save
method returns the exported model. `layer_config` describes per-layer settings;
`serialization_dict` contains export metadata. The compressor passes tokenizer,
device and other options as keywords. The instance method is also the packing
entry point used during whole-model saving where the format supports it.

Override `check_scheme_args` to validate resolved scheme values; a
`support_schemes` list alone does not validate a scheme instance. Override immediate
packing/saving capabilities for whole-model formats such as SVDQuant. GGUF retains
its streaming writer, converter lifecycle, and additional packing context through
`immediate_pack`; it does not use the Hugging Face checkpoint writer.

## Shared operations and format differences

- `prepare_model(model, inplace, device=...)` copies before moving when
  `inplace=False`, preserving the caller's model.
- `pack_layers(model, names, pack_layer, device=...)` snapshots names and skips
  unavailable meta tensors, reusing the format's packing callback.
- `save_artifacts(...)` writes supported tokenizer/processor artifacts only when
  an output directory is supplied.
- `save_model(...)` retains checkpoint sharding, immediate-saving and pipeline
  handling. Format-specific dtype and config decisions remain explicit.

AutoRound and compressed-tensors resolve a packing/writer pair together rather
than maintaining independent pack/save dispatch trees. Native AutoRound can reuse
GPTQ/AWQ packing while keeping its own metadata. Requested name (`auto_round`),
resolved backend (`auto_round:auto_gptq`) and checkpoint `packing_format` have
distinct roles; do not derive checkpoint metadata from the requested name alone.

`SUPPORTED_FORMATS` reads registered public names plus `GGUF_CONFIG` aliases.
Internal aliases can use `public=False`. GGUF+fake exports retain fake-first order.
Existing checkpoint layouts and metadata remain format-owned. Two inconsistencies
are corrected with regression tests: copying precedes moving under `inplace=False`,
and native E5M3 whole-model packing uses the same packer as immediate packing.
AWQ no-output saving now returns its packed copy when `inplace=False`.

## Validation

The common export tests cover registration, aliases, resolution, numerical packing
helpers, conversion helpers, and the absence of legacy exporter imports. Tiny CPU
tests exercise real integer/FP8 payloads, metadata, checkpoint saving, no-output
paths and model-copy isolation. `test_custom_format_save.py` constructs a local tiny
OPT model and tokenizer, performs real RTN quantization and saves through a custom
format implementing only the two common methods. No network download is needed.

Verification in the current development environment: the main common/CPU export
and CLI group passed (520 tests; 42 compressed-tensors tests skipped because that optional
dependency is unavailable); 18 additional fake/GGUF lifecycle CPU tests passed.
A wheel was built and checked for canonical packages, the GGUF commit marker,
and absence of the removed exporter paths. Pre-commit passed, with the black
hook run separately per file because its multi-file process pool hangs in the
sandbox.

The full GGUF conversion suite is not green in this environment: the installed
GGUF package lacks `MODEL_ARCH.DFLASH`, and transformers lacks `PrefixChange`.
Those failures were observed before migration. External inference-engine and
hardware-dependent tests are not certified by these CPU results.
