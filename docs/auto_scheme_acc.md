### The GGUF results

We tested with a fake model because the main branch currently has layer name mismatches between Transformers and GGUF.

~~~bash
python3 -m auto_round Qwen/Qwen3-8B     --options "gguf:q2_k_s,gguf:q4_k_s"     --target_bits 3.5     --ignore_scale_zp_bits     --iters 0     --format fake     --output_dir "./test_gguf"
~~~


eval

~~~
# Start vLLM serve
vllm serve ./test_gguf/Qwen3-8B-w2g16/ --port 8000 --max-model-len 8192 --host 127.0.0.1 --served-model-name qwen3-8b

# Perform five repeated evaluations on math_500 and gpqa_diamond.
evalscope eval --model qwen3-8b --api-url http://127.0.0.1:8000/v1   --api-key EMPTY   --datasets math_500 gpqa_diamond  --eval-batch-size 64  --generation-config "{\"n\":5}"

# Evaluate mmlu_pro
evalscope eval --model qwen3-8b --api-url http://127.0.0.1:8001/v1   --api-key EMPTY   --datasets mmlu_pro  --eval-batch-size 128
~~~
| evalscope, options q2ks,q4ks avgbits 3.5, ignore_scale_zp | math_500 (repeat=5) | gpqa_diamond (repeat=5) | mmlu_pro |
|-----------------------------------------------------------|---------------------|-------------------------|----------|
| qwen3-8b: BF16                                            | 0.8083              | 0.4586                  | 0.6934   |
| qwen3-8b: Fake quantized model                            | 0.7924              | 0.4313                  | 0.6751   |

| evalscope, options q2ks,q4ks avgbits 3.5, ignore_scale_zp | math_500 (repeat=5) | gpqa_diamond (repeat=5) | mmlu_pro |
|-----------------------------------------------------------|---------------------|-------------------------|----------|
| qwen3.5-4b: BF16                                          | 0.5365              | 0.3263                  | 0.5891   |
| qwen3.5-4b: Fake quantized model                          | 0.505               | 0.3172                  | 0.5948   |


### GGUF Format results
**using lm_eval to eval**
```bash
lm_eval --model hf --model_args pretrained=test_gguf/,gguf_file=Qwen3-8B-Q4_K_M.gguf --device cuda:0 --tasks arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,mmlu,openbookqa,piqa,truthfulqa_mc1,winogrande
```
| Qwen3-8B | Quant Type | Actual Bits | Avg Acc | arc_challenge | arc_easy | boolq | hellaswag | lambada_openai | openbookqa | piqa | truthfulqa_mc1 | winogrande | mmlu |

|--------|------------|-------------|---------|---------------|----------|-------|-----------|----------------|------------|------|----------------|------------|------|

| f16 | f16 | 16 | | | | | | | | | | | |

| unsloth | Q2_K | 3.20 | 0.6075 | 0.5145 | 0.8051 | 0.8526 | 0.5309 | 0.6255 | 0.3166 | 0.7530 | 0.3452 | 0.6590 | 0.6729 |

| unsloth | Q2_K_L | 3.34 | 0.6078 | 0.5128 | 0.8068 | 0.8529 | 0.5295 | 0.6276 | 0.3189 | 0.7546 | 0.3427 | 0.6567 | 0.6750 |

| bartowski | Q2_K | 3.20 | 0.6071 | 0.5128 | 0.8018 | 0.8511 | 0.5312 | 0.6257 | 0.3100 | 0.7568 | 0.3366 | 0.6677 | 0.6772 |

| bartowski | Q2_K_L | 3.79 | 0.5761 | 0.5162 | 0.8018 | 0.5306 | 0.5306 | 0.6266 | 0.3120 | 0.7606 | 0.3305 | 0.6717 | 0.6806 |

| auto_round | 2.5b mixed[q2,q3,q4] | 3.23 | 0.6153 | 0.5239 | 0.8102 | 0.8615 | 0.5242 | 0.6598 | 0.3140 | 0.7503 | 0.3390 | 0.6756 | 0.6943 |

| auto_round | 2.5b mixed[q2,q3,q4] lm_head Q6K | 3.25 | 0.6093 | 0.5060 | 0.8093 | 0.8538 | 0.5119 | 0.6598 | 0.3040 | 0.7459 | 0.3256 | 0.6835 | 0.6929 |

| auto_round | 2.5b mixed[q2,q3,q4,q6] | 3.24 | 0.6074 | 0.5034 | 0.8131 | 0.8508 | 0.5109 | 0.6571 | 0.3020 | 0.7470 | 0.3317 | 0.6685 | 0.6897 |

| auto_round | 2.5b mixed[q2,q3,q4,q6] iters 200 | 3.25 | 0.6046 | 0.5060 | 0.8136 | 0.8541 | 0.5234 | 0.6565 | 0.3040 | 0.7606 | 0.3403 | 0.6827 | 0.6919 |

| unsloth | Q3_K_M | 4.02 | 0.6260 | 0.5503 | 0.8270 | 0.8596 | 0.5589 | 0.6301 | 0.3120 | 0.7655 | 0.3770 | 0.6677 | 0.7115 |

| bartowski | Q3_K_M | 4.02 | 0.6246 | 0.5478 | 0.8304 | 0.8606 | 0.5554 | 0.6321 | 0.3200 | 0.7661 | 0.3647 | 0.6575 | 0.7118 |

| auto_round | 3.5b mixed[q2,q3,q4] | 4.17 | 0.6284 | 0.5478 | 0.8300 | 0.8648 | 0.5651 | 0.6429 | 0.3100 | 0.7612 | 0.3574 | 0.6819 | 0.7225 |

| auto_round | 3.5b mixed[q2,q3,q4] lm_head Q6K | 4.19 | 0.6251 | 0.5384 | 0.8338 | 0.8609 | 0.5581 | 0.6354 | 0.3160 | 0.7612 | 0.3574 | 0.6725 | 0.7175 |

| auto_round | 3.5b mixed[q2,q3,q4,q6] | 4.19 | 0.6246 | 0.5401 | 0.8259 | 0.8624 | 0.5582 | 0.6387 | 0.3080 | 0.7628 | 0.3586 | 0.6725 | 0.7191 |

| auto_round | 3.5b mixed[q2,q3,q4,q6] iters 200 | 4.19 | 0.6318 | 0.5520 | 0.8346 | 0.8676 | 0.5624 | 0.6404 | 0.3160 | 0.7682 | 0.3696 | 0.6890 | 0.7183 |

| unsloth | Q4_K_M | 4.90 | 0.6308 | 0.5520 | 0.8329 | 0.8691 | 0.5685 | 0.6437 | 0.3180 | 0.7568 | 0.3623 | 0.6803 | 0.7248 |

| bartowski | Q4_K_M | 4.90 | 0.6301 | 0.5546 | 0.8359 | 0.8654 | 0.5655 | 0.6443 | 0.3120 | 0.7612 | 0.3599 | 0.6748 | 0.7269 |

**using llama-server and evalscope to eval**
```bash
llama-server \
    --model test_gguf/Qwen3-8B-Q2_K_S.gguf \
    --host 127.0.0.1 \
    --port 8001 \
    --alias qwen \
    --ctx-size 65536 \
    --gpu-layers all \
    --parallel 16 \
    --predict 2048 \
    --reasoning off \
    --no-webui \
```
```bash
evalscope eval \
    --model qwen \
    --api-url http://127.0.0.1:8001/v1 \
    --api-key EMPTY \
    --datasets math_500 \
    --eval-batch-size 16
```

| Qwen3-8B | Quant Type | Actual Bits | gpqa_diamond | math_500 |

|--------|------------|-------------|--------------|----------|

| f16 | f16 | 16 | 0.4596 | 0.798 |

| unsloth | Q2_K | 3.20 | 0.3131 | 0.726 |

| unsloth | Q2_K_L | 3.34 | 0.3788 | 0.732 |

| auto_round | 2.5b mixed[q2,q3,q4] | 3.23 | 0.3636 | 0.706 |

| auto_round | 2.5b mixed[q2,q3,q4] lm_head Q6K | 3.25 | 0.3485 | 0.714 |

| auto_round | 2.5b mixed[q2,q3,q4,q6] | 3.24 | 0.3687 | 0.698 |

| auto_round | 2.5b mixed[q2,q3,q4,q6] iters 200 | 3.25 | 0.3990 | 0.700 |

| unsloth | Q3_K_M | 4.02 | 0.4192 | 0.778 |

| auto_round | 3.5b mixed[q2,q3,q4] | 4.17 | 0.4495 | 0.816 |

| auto_round | 3.5b mixed[q2,q3,q4] lm_head Q6K | 4.19 | 0.4495 | 0.788 |

| auto_round | 3.5b mixed[q2,q3,q4,q6] | 4.19 | 0.4444 | 0.814 |

| auto_round | 3.5b mixed[q2,q3,q4,q6] iters 200 | 4.19 | 0.5051 | 0.788 |

| unsloth | Q4_K_M | 4.90 | 0.4545 | 0.792 |

### Other results

We use **lm-eval** for evaluation. For LLaMA, we enabled `add_bos_token` and
`removed @use_kernel_forward_from_hub("RMSNorm")`
in [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L52C1-L52C40)
to stabilize accuracy during evaluation. All other settings follow the default configurations of AutoRound and lm-eval.

We ignore the scale and zp bits in the tables below. The accuracy may change a little as we modified a little of the
implementation. We will rerun all the experiments.

For mxfp experiment, we use fake model while for weight only model we use real model. **No tuning is applied unless
explicitly stated.**

*Average accuracy across `lambada_openai`, `hellaswag`, `piqa`, `winogrande`, and `mmlu`.*

### Table 1 MXFP4/8 mixed accuracy.

| Average bits     | Llama3.1-8B-I  |  Qwen2.5-7B-I  |    Qwen3-8B    |   Qwen3-32B    |
|:-----------------|:--------------:|:--------------:|:--------------:|:--------------:|
| **BF16**         | 0.7076 (100%)  | 0.7075 (100%)  | 0.6764 (100%)  | 0.7321 (100%)  |
| **Pure 4-bit**   | 0.6626 (93.6%) | 0.6550 (92.6%) | 0.6316 (93.4%) | 0.6901 (94.3%) |
| **Ours 4.5-bit** | 0.6808 (96.2%) | 0.6776 (95.8%) | 0.6550 (96.8%) | 0.7176 (98.0%) |
| **Ours 5-bit**   | 0.6857 (96.9%) | 0.6823 (96.4%) | 0.6594 (97.5%) | 0.7201 (98.3%) |
| **Ours 6-bit**   | 0.6975 (98.6%) | 0.6970 (98.5%) | 0.6716 (99.3%) | 0.7303 (99.8%) |

We compare the proposed method against naive layer-wise bit allocation strategies, such as assigning higher
precision to the network’s head((near lm-head) or tailad(close to embedding)) layers, to demonstrate its relative
performance advantages.

### Table 2  Comparison with other recipes at an average of 5 bits of mxfp datatype

| Avg. bits = 5         |   Llama3.1-8B-I    |    Qwen2.5-7B-I    |      Qwen3-8B      |
|:----------------------|:------------------:|:------------------:|:------------------:|
| **Tail layers 8-bit** |   0.6671 (94.3%)   |   0.6616 (93.5%)   |   0.6410 (94.8%)   |
| **Head layers 8-bit** |   0.6657 (94.1%)   |   0.6686 (94.5%)   |   0.6356 (94.0%)   |
| **Ours**              | **0.6857 (96.9%)** | **0.6823 (96.4%)** | **0.6594 (97.5%)** |

### Table 3  Comparison with other recipes at an average of 4.5 bits of mxfp datatype

| Avg. bits = 4.5       |   Llama3.1-8B-I    |    Qwen2.5-7B-I    |      Qwen3-8B      |
|:----------------------|:------------------:|:------------------:|:------------------:|
| **Tail layers 8-bit** |   0.6614 (93.5%)   |   0.6535 (92.4%)   |   0.6373 (94.2%)   |
| **Head layers 8-bit** |   0.6568 (92.8%)   |   0.6642 (93.9%)   |   0.6305 (93.2%)   |
| **Ours**              | **0.6808 (96.2%)** | **0.6776 (95.5%)** | **0.6550 (95.8%)** |

### Table4   Comparison with other recipes at an average of 3 bits of W2G128 and W4G128

| Avg. bits = 4.5       | Llama3.1-8B-I | Qwen2.5-7B-I | Qwen3-8B |
|:----------------------|:-------------:|:------------:|:--------:|
| **Tail layers 4-bit** |    0.6058     |    0.3798    |  0.4536  |
| **Head layers 4-bit** |    0.3198     |    0.3270    |  0.3196  |
| **Ours**              |    0.6148     |    0.4058    |  0.4862  |
