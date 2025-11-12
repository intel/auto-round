
## Support Matrix

| Model Family            | MXFP4 | MXFP8 |
| ----------------------- | ----- | ----- |
| Qwen/Qwen3-235B-A22B    | ✅     | ✅     |
| deepseek-ai/DeepSeek-R1 | ✅     | ✅     |

### Quantize Model

- MXFP8
```bash
export QWEN_MODEL=Qwen/Qwen3-235B-A22B
export DS_MODEL=deepseek-ai/DeepSeek-R1

python quantize.py --model $QWEN_MODEL -t qwen_mxfp8 --use_autoround_format  
python quantize.py --model $DS_MODEL -t ds_mxfp8 --use_autoround_format  
```

- MXFP4
```bash
export QWEN_MODEL=Qwen/Qwen3-235B-A22B
export DS_MODEL=deepseek-ai/DeepSeek-R1
python quantize.py --model $QWEN_MODEL -t qwen_mxfp4 --use_autoround_format 
python quantize.py --model $DS_MODEL -t qwen_mxfp4 --use_autoround_format 
```


### Prompt Tests

Usage: 
```bash
bash ./run_generate.sh -s [mxfp4|mxfp8] -m [model_path] -tp [tensor_parallel_size]
```

- MXFP8
```bash
bash ./run_generate.sh -s mxfp8 -m /path/to/qwen_mxfp8 -tp 4 
bash ./run_generate.sh -s mxfp8 -m /path/to/ds_mxfp8 -tp 8
```
- MXFP4
```bash
bash ./run_generate.sh -s mxfp4 -m /path/to/qwen_mxfp4 -tp 4 
bash ./run_generate.sh -s mxfp4 -m /path/to/ds_mxfp4 -tp 8 
```
### Evaluation Tests

WIP



