---
name: run-quant
description: Run auto-round quantization on a model. User-triggered only — use /run-quant <model_name> to start.
disable-model-invocation: true
---

# Run Quantization

Run auto-round quantization with the provided arguments.

## Usage

`/run-quant $ARGUMENTS`

Parse `$ARGUMENTS` for the model name/path. Ask the user for any missing required parameters:

- **model**: HuggingFace model name or local path (required — from $ARGUMENTS)
- **bits**: Quantization bits (default: 4)
- **group_size**: Group size (default: 128)
- **format**: Export format — one of: `auto_round`, `auto_gptq`, `auto_awq`, `gguf` (default: `auto_round`)
- **device**: Target device — `cpu`, `cuda`, `hpu`, `xpu` (default: `cuda` if available, else `cpu`)
- **output_dir**: Output directory (default: `./tmp_autoround/<model_basename>-w<bits>g<group_size>`)

## Execution

```bash
auto-round \
  --model "$MODEL" \
  --bits $BITS \
  --group_size $GROUP_SIZE \
  --format "$FORMAT" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE"
```

## Common Variations

For VLMs (vision-language models), use `auto-round-mllm` instead:
```bash
auto-round-mllm --model "$MODEL" --bits $BITS --group_size $GROUP_SIZE --format "$FORMAT" --output_dir "$OUTPUT_DIR"
```

For quick/light quantization presets:
```bash
auto-round-fast --model "$MODEL" --bits $BITS    # faster, slightly lower quality
auto-round-light --model "$MODEL" --bits $BITS   # lightest calibration
auto-round-best --model "$MODEL" --bits $BITS    # highest quality, slower
```

## After Quantization

Report the output directory and model size. Suggest running a quick inference test:
```python
from auto_round import AutoRoundConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

quantized_model_path = "$OUTPUT_DIR"
model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
output = model.generate(**tokenizer("Hello, ", return_tensors="pt").to(model.device), max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
