# Self-Contained Nunchaku Diffusion Pipeline Design

## Goal

Export a portable Diffusers pipeline whose non-transformer components remain BF16
and whose transformer is an AutoRound SVDQuant MXFP4 Nunchaku onefile. The output
must load through `FluxPipeline.from_pretrained(output_dir)` without separately
loading a BF16 base model or manually injecting a transformer.

## Output Contract

The output preserves the standard Diffusers component layout:

```text
output/
  model_index.json
  scheduler/
  tokenizer/
  tokenizer_2/
  text_encoder/
  text_encoder_2/
  vae/
  transformer/
    config.json
    diffusion_pytorch_model.safetensors
```

`transformer/diffusion_pytorch_model.safetensors` is the packed Nunchaku onefile. No BF16
transformer weights are written. Root-level convenience duplicates from the source
repository are not copied.

## AutoRound Responsibilities

- Reuse the existing diffusion model detection and loading path from the complete
  pipeline directory.
- Quantize only `pipe.transformer`, as the diffusion backend already does.
- Save all other registered pipeline components through their existing
  `save_pretrained` methods.
- Save the original transformer config next to the packed onefile.
- Read `model_class` from the onefile safetensors metadata and rewrite only the
  transformer entry in `model_index.json` to `["nunchaku", model_class]`.
- Do not import Nunchaku or DeepCompressor.

The custom standalone `model_loader` route is removed. The CLI receives the full
Diffusers pipeline path directly.

## Nunchaku Responsibilities

When a local directory is passed to a Nunchaku transformer `from_pretrained`, look
for `diffusion_pytorch_model.safetensors` first. If present, load it through the existing onefile path,
including metadata-based precision and hardware checks. If absent, preserve the
legacy split-directory behavior.

## Loading Contract

`model_index.json` identifies the transformer as:

```json
"transformer": ["nunchaku", "NunchakuFluxTransformer2dModel"]
```

Diffusers imports the installed `nunchaku` package, recognizes the class as a
`ModelMixin`, and calls its `from_pretrained` method with the transformer directory.

## Validation

1. Unit tests verify onefile metadata rewrites the pipeline component entry.
2. Unit tests verify Nunchaku directory onefile discovery and legacy fallback.
3. A lightweight Diffusers fixture loads through `DiffusionPipeline.from_pretrained`
   without a manually supplied transformer.
4. The full FLUX output contains BF16 auxiliary components and no BF16 transformer.
5. The output loads after the original BF16 source path is made unavailable.
6. A full image generation smoke test produces a coherent image.
