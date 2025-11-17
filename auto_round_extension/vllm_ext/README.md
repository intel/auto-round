- vllm https://github.com/yiliu30/vllm-fork/tree/fused-moe-ar
```
VLLM_USE_PRECOMPILED=1 pip install --editable . -vvv
```
- Allow python patches vLLM with vLLM-Ext
```
cd auto-round/auto_round_extension/vllm_ext
source apply_ext.sh
```

- Enable vLLM-Ext
```bash
VLLM_ENABLE_AR_EXT=1 vllm serve ...
```