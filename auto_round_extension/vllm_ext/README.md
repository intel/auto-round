-  Build and Install vLLM 

```
https://github.com/yiliu30/vllm-fork/tree/fused-moe-ar
VLLM_USE_PRECOMPILED=1 pip install --editable . -vvv
```
- Apply vLLM-Ext Patches(allow python recognize them)
```
cd auto-round/auto_round_extension/vllm_ext
source apply_ext.sh
```

- Enable vLLM-Ext at Runtime
```bash
VLLM_ENABLE_AR_EXT=1 vllm serve ...
```