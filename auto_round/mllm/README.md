# AutoRound for MLLMs

## Quantization

### API Usage (Gaudi2/CPU/GPU) Recommended
AutoRound uses the text module of MLLM (LLM component) as the main quantization target. with NeelNanda/pile-10k as the default calibration dataset.

```python
    from auto_round import AutoRoundMLLM
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
    ## load the model
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, trust_remote_code=True)
        
    ## quantize the model
    bits, group_size = 4, 128
    autoround = AutoRoundMLLM(model, tokenizer, processor, bits=bits, group_size=group_size)
    autoround.quantize()

    # save the quantized model, set format='auto_gptq' to use AutoGPTQ format
    output_dir = "./tmp_autoround"
    autoround.save_quantized(output_dir, format='auto_round', inplace=True)
```

<details>
<summary style="font-size:17px;">Basic Usage (Gaudi2/CPU/GPU)</summary>
    A user guide detailing the full list of supported arguments is provided by calling ```auto-round-mllm -h``` on the terminal. Alternatively, you can use ```auto_round_mllm``` instead of ```auto-round-mllm```. Set the format you want in `format` and
    multiple formats exporting has been supported.

```bash
    # experimental feature, default hyperparameters may be changed later
    auto—round-mllm \
        --model Qwen/Qwen2-VL-2B-Instruct \
        --bits 4 \
        --group_size 128 \
        --format "auto_round" \
        --output_dir ./tmp_autoround
```

- `dataset`: the dataset for quantization training. current support NeelNanda/pile-10k,llava_conv_58k,llava_instruct_80k. It can be a custom one.

- `quant_nontext_module`: whether to quantize non-text module, e.g. vision component. 

- `extra_data_dir`:dataset dir for storing images/audio/videos, default to None. Can be a dir path or multiple dir path with format as 'image=path_to_image,video=path_to_video,audio=path_to_audio' By default, it will search in the relative path, and if not find, will automatic download.

</details>


<details>
<summary style="font-size:17px;">Calibration Dataset</summary>

For mllm, we used **text-only** calibration dataset (NeelNanda/pile-10k) as our default. If the model type does not support plain text calibration(e.g. Llama-3.2-vision), it will also automatically switch to llava dataset and adjust the hyperparameters.

Through argument --dataset(text file), user can use other datasets such as "liuhaotian/llava_conv_58k" "liuhaotian/llava_instruct_80k", "liuhaotian/llava_instruct_150k" or a file path to use local file.


### Support List

The llava calibration dataset supports the five existing MLLMs. 

|Model          |Eval Lib   |calibration dataset|Feasibility of quantification|
|---------------|-----------|-------------------|--------------------|
|Qwen/Qwen2-VL-Instruct            |vlmeval    |llava         |✔                   |
|meta-llama/Llama-3.2-11B-Vision   |vlmeval/lmms_eval  |llava              |✔                   |
|microsoft/Phi-3.5-vision-instruct |vlmeval    |llava         |✔                   |
|liuhaotian/llava-v1.5-7b          |lmms_eval  |llava         |✔                   |
|THUDM/cogvlm2-llama3-chat-19B     |lmms_eval  |llava         |✔                   |

</details>



<details>
<summary style="font-size:17px;">Nontext Module Quantization</summary>

### Support Matrix

The design of the MLLM model API is not uniform, and some models do not support the quantization nontext module. Quantization of the vision components of Llama-3.2-11B-Vision, Phi-3.5-vision-instruct and llava-v1.5-7b is currently supported.

|Model          |Eval Lib   |quant nontext module|
|---------------|-----------|-------------------|
|Qwen/Qwen2-VL-Instruct            |vlmeval    |-                    |
|meta-llama/Llama-3.2-11B-Vision   |lmms_eval  |✔                   |
|microsoft/Phi-3.5-vision-instruct |vlmeval    |✔                   |
|liuhaotian/llava-v1.5-7b          |lmms_eval  |-                    |
|THUDM/cogvlm2-llama3-chat-19B     |lmms_eval  |✔                   |



### New Models Support
#### Template
For autoround MLLMs, using Template to customize different operations for different models. User can add a custom chat template through json file as below.
```json
{
    "model_type": "qwen2_vl",
    "format_user": "<|im_start|>user\n{{content}}<|im_end|>\n",
    "format_assistant": "<|im_start|>assistant\n{{content}}<|im_end|>\n",
    "format_system": "<|im_start|>system\n{{content}}<|im_end|>\n",
    "format_observation": "<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n",
    "format_separator": "\n",
    "default_system": "You are a helpful assistant.",
    "replace_tokens": ["<image>", "<|vision_start|><|image_pad|><|vision_end|>"],
    "extra_encode": "True",
    "processor": "qwen2_vl" 
}
```
The special token ```{{content}}``` is a placeholder to tell the preprocessor where to fill in the corresponding dialogue content.

```format_*```: Add specific token to chat content depends on different role names.

For example, the input conversations:<br>
 ```[{'role': 'user', 'value': '<image>\nWhat are the colors of the bus in the image?'}, {'role': 'assistant', 'value': 'The bus in the image is white and red.'}]```

Using the above template, the input will be converted to the specified format required by Qwen2-vl as below: <br>
 ```'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nWhat are the colors of the bus in the image?<|im_end|>\n<|im_start|>assistant\nThe bus in the image is white and red.<|im_end|>\n<|im_start|>user\nWhat feature can be seen on the back of the bus?<|im_end|>\n<|im_start|>assistant\nThe back of the bus features an advertisement.<|im_end|>\n<|im_start|>user\nIs the bus driving down the street or pulled off to the side?<|im_end|>\n<|im_start|>assistant\nThe bus is driving down the street, which is crowded with people and other vehicles.<|im_end|>\n'```.

#### Processor
Processor is callback interface for calling different processors, such as texts or images processors, for MLLMs. User can define own processor and use registration function to declare. For more information, please refer to the relevant code in ```auto_round/mllm/processor.py```.

</details>



## Inference for Models
For the AutoRound format, please add the following code at the beginning of the original model's inference code.

```python
from auto_round import AutoRoundConfig ## must import for auto-round format
```

For more details on quantization, inference, evaluation, and environment, see the following recipe:

- [Qwen2-VL-Instruct](../../docs/Qwen2-VL-7B-Instruct_sym.md)
- [Llama-3.2-11B-Vision](../../docs/Llama-3.2-11B-Vision-Instruct_sym.md) 
- [Phi-3.5-vision-instruct](../../docs/Phi-3.5-vision-instruct_sym.md)
- [llava-v1.5-7b](../../docs/llava-v1.5-7b_sym.md)



