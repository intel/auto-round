# AutoRound for Diffusion Models (Experimental)

This feature is experimental and may be subject to changes, including potential bug fixes, API modifications, or adjustments to default parameters.

AutoRound uses the new compressor and calibration architecture for diffusion quantization. Diffusion models are routed by `auto_round/compressors/entry.py`, diffusion-specific compressor behavior lives in `auto_round/compressors/diffusion_mixin.py`, and calibration is handled by `auto_round/calibration/diffusion.py`.

## Quantization

Quantization for diffusion models is limited:

1. Only the transformer module of diffusion models is quantized.
2. Loading quantized diffusion models is not supported yet, so use `fake` format for quantization.
3. Calibration dataset currently supports `coco2014` and user customized `.tsv` files.

### API Usage (CPU/GPU) Recommended

```python
import torch
from auto_round import AutoRound

# Quantize the model
autoround = AutoRound(
    "black-forest-labs/FLUX.1-dev",
    scheme="MXFP8",
    dataset="coco2014",
    num_inference_steps=10,
    guidance_scale=7.5,
    generator_seed=None,
    batch_size=1,
)

# Save the quantized model
output_dir = "./tmp_autoround"
# Loading quantized diffusion models is not supported yet, so use fake format.
autoround.quantize_and_save(output_dir, format="fake", inplace=True)
```

- `dataset`: the dataset for quantization training. Currently supports `coco2014` and user customized `.tsv` files.
- `num_inference_steps`: the reference number of denoising steps.
- `guidance_scale`: controls how much the image generation process follows the text prompt.
- `generator_seed`: a seed that controls the initial noise from which an image is generated.

For more hyperparameters, refer to [Homepage Detailed Hyperparameters](../../../README.md#quantization-scheme--configuration).

### CLI Usage

A user guide detailing the full list of supported arguments is provided by calling `auto-round -h` on the terminal.

```bash
auto-round \
    --model black-forest-labs/FLUX.1-dev \
    --scheme MXFP8 \
    --format fake \
    --batch_size 1 \
    --dataset coco2014 \
    --output_dir ./tmp_autoround
```

### Diffusion Support Matrix

For diffusion models, currently we validate quantization on the following models, which involves quantizing the transformer component of the pipeline.

| Model | calibration dataset | Model Link |
| --- | --- | --- |
| black-forest-labs/FLUX.1-dev | COCO2014 | - |
| Tongyi-MAI/Z-Image | COCO2014 | - |
| Tongyi-MAI/Z-Image-Turb | COCO2014 | - |
| stepfun-ai/NextStep-1.1 | COCO2014 | - |
| AIDC-AI/Ovis-Image-7B | COCO2014 | - |

<details>
<summary style="font-size:17px;">Calibration Dataset</summary>

For diffusion models, we use [**coco2014**](https://github.com/mlcommons/inference/raw/refs/heads/master/text_to_image/coco2014/captions/captions_source.tsv) calibration dataset as the default.

To use a custom dataset, build a `.tsv` file with the following structure and pass it through `--dataset`:

```text
id      caption
0       YOUR_PROMPT
1       YOUR_PROMPT
...     ...
```

- `id`: the id used to map generated images and prompts.
- `caption`: the text prompt used to generate the images.

</details>