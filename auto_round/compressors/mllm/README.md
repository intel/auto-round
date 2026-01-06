# AutoRound for MLLMs

This feature is experimental and may be subject to changes, including potential bug fixes, API modifications, or
adjustments to default parameters

## Quantization

### API Usage (Gaudi2/CPU/GPU) Recommended

By default, AutoRound only quantizes the text module of VLMs and uses `NeelNanda/pile-10k` for calibration. To
quantize the entire model, you can enable `quant_nontext_module` by setting it to True, though support for this feature
is limited.

```python
from auto_round import AutoRound  # same as llm, AutoRound can determine mllm automatically

model_name = "Qwen/Qwen2-VL-2B-Instruct"

## quantize the model
autoround = AutoRound(model_name, scheme="W4A16", dataset="NeelNanda/pile-10k", quant_nontext_module=False)
output_dir = "./tmp_autoround"
autoround.quantize_and_save(output_dir, format="auto_round")
```

- `dataset`: the dataset for quantization training. Currently only support NeelNanda/pile-10k, llava_conv_58k,
  llava_instruct_80k and llava_instruct_150k. Please note that the feasibility of the Llava calibration dataset has only been validated on five models so far.

- `quant_nontext_module`: whether to quantize non-text module, e.g. vision component.

for more hyperparameters introduction, please
refer [Homepage Detailed Hyperparameters](../../README.md#api-usage-gaudi2cpugpu)

### Basic Usage

A user guide detailing the full list of supported arguments is provided by calling ```auto-round -h``` on the
terminal. Set the format you want in `format` and
multiple formats exporting has been supported. **Only five model families are supported now.

```bash
auto-round \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --scheme w4a16 \
    --format "auto_round" \
    --output_dir ./tmp_autoround
```

### VLM Support Matrix

For most VLMs, we typically support the default quantization configuration, which involves quantizing only the language
component while excluding the visual component. Besides, we also support quantizing non-text modules of models that
follow the Hugging Face standard, i.e., those with a typical processor, though inference may have some issues due to
model architecture or kernel limitations.

| Model                                         | calibration dataset | quant nontext module | Quantized Model Link                                                                                                                                                                                                                                                                                                                                                                                                                                             | 
|-----------------------------------------------|---------------------|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| allenai/Molmo                                 | pile                | X                    | [Molmo-7B-D-0924-int4-sym](https://huggingface.co/OPEA/Molmo-7B-D-0924-int4-sym-inc), [Molmo-72B-0924-int4-sym-gptq](https://huggingface.co/OPEA/Molmo-72B-0924-int4-sym-gptq-inc), [Molmo-72B-0924-int4-sym](https://huggingface.co/OPEA/Molmo-72B-0924-int4-sym-inc)                                                                                                                                                                                           |
| deepseek-ai/deepseek-vl2                      | pile/llava          | √                    | [deepseek-vl2-int4-sym-gptq](https://huggingface.co/OPEA/deepseek-vl2-int4-sym-gptq-inc)                                                                                                                                                                                                                                                                                                                                                                         |
| fancyfeast/llama-joycaption-beta-one-hf-llava | pile                | X                    | [NeoChen1024-int4-gptq](https://huggingface.co/NeoChen1024/llama-joycaption-beta-one-hf-llava-GPTQ-4bit-sym-autoround)                                                                                                                                                                                                                                                                                                                                           
| google/gemma-3                                | pile/llava          | √                    | [gemma-3-12b-it-AutoRound-gguf-q4-0](https://huggingface.co/OPEA/gemma-3-12b-it-AutoRound-gguf-q4-0), [gemma-3-27b-it-AutoRound-gguf-q4-0](https://huggingface.co/OPEA/gemma-3-27b-it-AutoRound-gguf-q4-0), [gemma-3-12b-it-int4-AutoRound](https://huggingface.co/OPEA/gemma-3-12b-it-int4-AutoRound), [gemma-3-27b-it-int4-AutoRound](https://huggingface.co/OPEA/gemma-3-27b-it-int4-AutoRound)                                                               |
| HuggingFaceTB/SmolVLM                         | pile/llava          | √                    | [SmolVLM-Instruct-int4-sym](https://huggingface.co/OPEA/SmolVLM-Instruct-int4-sym-inc)                                                                                                                                                                                                                                                                                                                                                                           |
| ibm-granite/granite-vision-3.2                | pile/llava          | -                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| liuhaotian/Llava-v1.5                         | pile/llava          | X                    | [llava-v1.5-7b-int4-sym](https://huggingface.co/OPEA/llava-v1.5-7b-int4-sym-inc)                                                                                                                                                                                                                                                                                                                                                                                 |
| meta-llama/Llama-3.2-Vision                   | llava               | √                    | [Llama-3.2V-11B-cot-int4-sym](https://huggingface.co/OPEA/Llama-3.2V-11B-cot-int4-sym-inc), [Llama-3.2-11B-Vision-Instruct-qvision-int4-sym](https://huggingface.co/OPEA/Llama-3.2-11B-Vision-Instruct-qvision-int4-sym-inc), [Llama-3.2-90B-Vision-Instruct-int4-sym](https://huggingface.co/OPEA/Llama-3.2-90B-Vision-Instruct-int4-sym-inc), [Llama-3.2-11B-Vision-Instruct-int4-sym](https://huggingface.co/OPEA/Llama-3.2-11B-Vision-Instruct-int4-sym-inc) |
| microsoft/Phi3.5-Vision                       | pile/llava          | √                    | [Phi-3.5-vision-instruct-int4-sym](https://huggingface.co/OPEA/Phi-3.5-vision-instruct-int4-sym-inc), [Phi-3.5-vision-instruct-qvision-int4-sym](https://huggingface.co/OPEA/Phi-3.5-vision-instruct-qvision-int4-sym-inc)                                                                                                                                                                                                                                       |
| mistralai/Mistral-Small-3.1                   | pile/llava          | X                    | [Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-gptq-sym](https://huggingface.co/OPEA/Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-gptq-sym), [Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym](https://huggingface.co/OPEA/Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym)                                                                                                                                                     |
| moonshotai/Kimi-VL                            | pile/llava          | √                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Qwen/Qwen2-VL                                 | pile/llava          | -                    | [Qwen2-VL-7B-Instruct-int4-sym](https://huggingface.co/OPEA/Qwen2-VL-7B-Instruct-int4-sym-inc), [Qwen2-VL-72B-Instruct-int4-sym](https://huggingface.co/OPEA/Qwen2-VL-72B-Instruct-int4-sym-inc), [Qwen2-VL-72B-Instruct-int2-sym](https://huggingface.co/OPEA/Qwen2-VL-72B-Instruct-int2-sym-inc)                                                                                                                                                               |
| rhymes-ai/Aria                                | pile/llava          | √                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| THUDM/CogVLM2                                 | pile/llava          | √                    | [cogvlm2-llama3-chat-19B-int4-sym](https://huggingface.co/OPEA/cogvlm2-llama3-chat-19B-int4-sym-inc), [cogvlm2-llama3-chat-19B-qvision-int4-sym](https://huggingface.co/OPEA/cogvlm2-llama3-chat-19B-qvision-int4-sym-inc)                                                                                                                                                                                                                                       |
| THUDM/glm-4v                                  | pile                | X                    | [glm-4v-9b-int4-sym](https://huggingface.co/OPEA/glm-4v-9b-int4-sym-inc)                                                                                                                                                                                                                                                                                                                                                                                         |

√ means support, - means support to export but cannot infer, X means not support.



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

### New Models Support

#### Template

For autoround MLLMs, using Template to customize different operations for different models. User can use template to support new model which not in support list.
```python
from auto_round.mllm.template import _register_template

model_type = model.config.model_type
_register_template(model_type=model_type, default_dataset="NeelNanda/pile-10k", processor=PROCESSORS["hf"])
```

#### Processor

Processor is callback interface for calling different processors, such as texts or images processors, for MLLMs. User
can define own processor and use registration function to declare. For more information, please refer to the relevant
code in ```auto_round/mllm/processor.py```.

</details>








