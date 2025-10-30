# AutoRound for Diffusion Models (Experimental)

This feature is experimental and may be subject to changes, including potential bug fixes, API modifications, or adjustments to default parameters.

## Quantization

### API Usage (CPU/GPU) Recommended

By default, AutoRoundDiffusion only quantizes the transformer module of diffusion models and uses `COCO2014 captions` for calibration.

```python
import torch
from auto_round import AutoRound
from diffusers import AutoPipelineForText2Image

# Load the model
model_name = "black-forest-labs/FLUX.1-dev"
pipe = AutoPipelineForText2Image.from_pretrained(model_name, dtype=torch.bfloat16)

# Quantize the model
autoround = AutoRound(
    pipe,
    scheme="MXFP8",
    dataset="coco2014",
    num_inference_steps=10,
    guidance_scale=7.5,
    generator_seed=None,
    batch_size=1,
)
autoround.quantize()

# Save the quantized model
output_dir = "./tmp_autoround"
# Currently loading the quantized diffusion model is not supported, so use fake format
autoround.save_quantized(output_dir, format="fake", inplace=True)
```

- `dataset`: the dataset for quantization training. Currently only support coco2014 and user customized .tsv file.

- `num_inference_steps`: The reference number of denoising steps.

- `guidance_scale`: Control how much the image generation process follows the text prompt. The more it is, the more closely it follows the prompt.

- `generator_seed`: A seed that controls the initial noise from which an image is generated.

for more hyperparameters introduction, please refer [Homepage Detailed Hyperparameters](../../README.md#api-usage-gaudi2cpugpu)

### CLI Usage

A user guide detailing the full list of supported arguments is provided by calling ```auto-round -h``` on the
terminal.

```bash
auto-round \
    --model black-forest-labs/FLUX.1-dev \
    --scheme MXFP8 \
    --format fake \
    --batch_size 1 \
    --output_dir ./tmp_autoround
```

### Diffusion Support Matrix

For diffusion models, currently we only validate quantizaion on the FLUX.1-dev, which involves quantizing the transformer component of the pipeline.

| Model     | calibration dataset |
|--------------|--------------|
| black-forest-labs/FLUX.1-dev | COCO2014      |



<details>
<summary style="font-size:17px;">Calibration Dataset</summary>

For diffusion models, we used [**coco2014**]("https://github.com/mlcommons/inference/raw/refs/heads/master/text_to_image/coco2014/captions/captions_source.tsv") calibration dataset as our default.

If users want to use their own dataset, please build the dataset file in ".tsv" format following below structure and use it through argument --dataset (tsv file):
```
id      caption
0       YOUR_PROMPT
1       YOUR_PROMPT
...     ...
```
- `id`: The id used to map generated images and prompts.
- `caption`: The text prompt used to generate the images.


</details>
