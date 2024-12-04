# AutoRound for MLLMs

This feature is experimental and may be subject to changes, including potential bug fixes, API modifications, or
adjustments to default parameters

## Quantization

### API Usage (Gaudi2/CPU/GPU) Recommended

By default, AutoRoundMLLM only quantizes the text module of VLMs and uses `NeelNanda/pile-10k` for calibration. To
quantize the entire model, you can enable `quant_nontext_module` by setting it to True, though support for this feature
is limited.

```python
from auto_round import AutoRoundMLLM
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer

## load the model
model_name = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

## quantize the model
bits, group_size, sym = 4, 128, True
autoround = AutoRoundMLLM(model, tokenizer, processor, bits=bits, group_size=group_size, sym=sym)
autoround.quantize()

# save the quantized model, set format='auto_gptq' to use AutoGPTQ format
output_dir = "./tmp_autoround"
autoround.save_quantized(output_dir, format='auto_round', inplace=True)
```

- `dataset`: the dataset for quantization training. Currently only support NeelNanda/pile-10k, llava_conv_58k,
  llava_instruct_80k and llava_instruct_150k. Please note that the feasibility of the Llava calibration dataset has only been validated on five models so far.

- `quant_nontext_module`: whether to quantize non-text module, e.g. vision component.

for more hyperparameters introduction, please
refer [Homepage Detailed Hyperparameters](../../README.md#api-usage-gaudi2cpugpu)

### Basic Usage

A user guide detailing the full list of supported arguments is provided by calling ```auto-round-mllm -h``` on the
terminal. Set the format you want in `format` and
multiple formats exporting has been supported. **Only five model families are supported now.

```bash
auto-round-mllm \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --bits 4 \
    --group_size 128 \
    --format "auto_round" \
    --output_dir ./tmp_autoround
```

<details>
<summary style="font-size:17px;">Calibration Dataset</summary>

For mllm, we used **text-only** calibration dataset (NeelNanda/pile-10k) as our default. If the model type does not
support plain text calibration(e.g. Llama-3.2-vision), it will also automatically switch to llava dataset and adjust the
hyperparameters.

Through argument --dataset(text file), user can use other datasets such as "liuhaotian/llava_conv_58k" "
liuhaotian/llava_instruct_80k", "liuhaotian/llava_instruct_150k" or a file path to use local file.

</details>



<details>
<summary style="font-size:17px;">Nontext Module Quantization</summary>

### Support Matrix

For typical VLLMs, we assume that the default quantization, which excludes quantizing the visual component, is
supported. The design of vision components in MLLM model APIs is not standardized, and some models do not support the
quantization of non-text modules.

Currently, the quantization of vision components is supported for Llama-3.2-11B-Vision, Phi-3.5-Vision-Instruct, and
Llava-v1.5-7B.

| Model        | Eval Lib  | calibration dataset | quant nontext module |
|--------------|-----------|---------------------|----------------------|
| Qwen2-VL     | vlmeval   | pile/llava          | -                    |
| Llama-Vision | lmms_eval | llava               | ✔                    |
| Phi3-Vision  | vlmeval   | pile/llava          | ✔                    |
| Llava-v1.5   | lmms_eval | pile/llava          | -                    |
| CogVLM2      | lmms_eval | pile/llava          | ✔                    |

### New Models Support

#### Template

For autoround MLLMs, using Template to customize different operations for different models. User can add a custom chat
template through json file as below.

```json
{
  "model_type": "qwen2_vl",
  "format_user": "<|im_start|>user\n{{content}}<|im_end|>\n",
  "format_assistant": "<|im_start|>assistant\n{{content}}<|im_end|>\n",
  "format_system": "<|im_start|>system\n{{content}}<|im_end|>\n",
  "format_observation": "<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n",
  "format_separator": "\n",
  "default_system": "You are a helpful assistant.",
  "replace_tokens": [
    "<image>",
    "<|vision_start|><|image_pad|><|vision_end|>"
  ],
  "extra_encode": "True",
  "processor": "qwen2_vl"
}
```

The special token ```{{content}}``` is a placeholder to tell the preprocessor where to fill in the corresponding
dialogue content.

```format_*```: Add specific token to chat content depends on different role names.

For example, the input conversations:<br>
```[{'role': 'user', 'value': '<image>\nWhat are the colors of the bus in the image?'}, {'role': 'assistant', 'value': 'The bus in the image is white and red.'}]```

Using the above template, the input will be converted to the specified format required by Qwen2-vl as below: <br>
```'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat are the colors of the bus in the image?<|im_end|>\n<|im_start|>assistant\nThe bus in the image is white and red.<|im_end|>\n<|im_start|>user\nWhat feature can be seen on the back of the bus?<|im_end|>\n<|im_start|>assistant\nThe back of the bus features an advertisement.<|im_end|>\n<|im_start|>user\nIs the bus driving down the street or pulled off to the side?<|im_end|>\n<|im_start|>assistant\nThe bus is driving down the street, which is crowded with people and other vehicles.<|im_end|>\n'```.

#### Processor

Processor is callback interface for calling different processors, such as texts or images processors, for MLLMs. User
can define own processor and use registration function to declare. For more information, please refer to the relevant
code in ```auto_round/mllm/processor.py```.

</details>

## Inference

For the AutoRound format, please add the following code at the beginning of the original model's inference code.

```python
from auto_round import AutoRoundConfig  ## must import for auto-round format
```

For more details on quantization, inference, evaluation, and environment, see the following recipe:

- [Qwen2-VL-7B-Instruct](../../docs/Qwen2-VL-7B-Instruct-sym.md)
- [Llama-3.2-11B-Vision](../../docs/Llama-3.2-11B-Vision-Instruct-sym.md)
- [Phi-3.5-vision-instruct](../../docs/Phi-3.5-vision-instruct-sym.md)
- [llava-v1.5-7b](../../docs/llava-v1.5-7b-sym.md)
- [cogvlm2-llama3-chat-19B](../../docs/cogvlm2-llama3-chat-19B-sym.md)




