# AutoRound for MLLMs

## Basic Usage (Gaudi2/CPU/GPU)

A user guide detailing the full list of supported arguments is provided by calling ```auto-round-mllm -h``` on the
terminal.Alternatively, you can use ```auto_round_mllm``` instead of ```auto-round-mllm```. Set the format you want
in `format` and
multiple formats exporting has been supported.

```bash
# experimental feature, default hyperparameters may be changed later
auto—round-mllm \
    --model Qwen/Qwen2-VL-2B-Instruct\
    --bits 4 \
    --batch_size 1 \
    --gradient_accumulate_steps 4 \
    --group_size 128 \
    --format "auto_round" \
    --output_dir ./tmp_autoround
```

## API Usage (Gaudi2/CPU/GPU)

```python
from auto_round import AutoRoundMLLM
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer

model_name = "Qwen/Qwen2-VL-2B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, trust_remote_code=True)
dataset = "/path/to/llava.json"
extra_data_dir = "/path/to/images/dir"

bits, group_size = 4, 128
autoround = AutoRoundMLLM(model, tokenizer, processor=processor, bits=bits, group_size=group_size,
                          dataset=dataset, extra_data_dir=extra_data_dir)

autoround.quantize()
output_dir = "./tmp_autoround"
autoround.save_quantized(output_dir, format='auto_round', inplace=True)
```

### Dataset

For mllm, we used liuhaotian/llava_conv_58k as our default calib datasets. Through argument ```--dataset```, user can
use other datasets such as "liuhaotian/llava_instruct_80k", "liuhaotian/llava_instruct_150k" or a file path to use local
file.

### Support Matrix

So far, auto-round for mllm supports five model families, include Qwen2-VL, Llama-Vision, Phi3-Vision, Llava-v1.5 and
CogVLM2.

| Model        | Eval Lib  | calibration dataset | quant nontext module |
|--------------|-----------|---------------------|----------------------|
| Qwen2-VL     | vlmeval   | pile/llava          | -                    |
| Llama-Vision | lmms_eval | llava               | ✔                    |
| Phi3-Vision  | vlmeval   | pile/llava          | ✔                    |
| Llava-v1.5   | lmms_eval | pile/llava          | -                    |
| CogVLM2      | lmms_eval | pile/llava          | ✔                    |

## New Models Support

### Template

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

### Processor

Processor is callback interface for calling different processors, such as texts or images processors, for MLLMs. User
can define own processor and use registration function to declare. For more information, please refer to the relevant
code in ```auto_round/mllm/processor.py```.