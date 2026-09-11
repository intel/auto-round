torch 2.14 is used (<2.14 will cause torch compile exception), H200 with 140G or A100 with 80g
x denote exceptions

| Model                                             | AR Version | Device   | ram   | vram      | time cost | comment                                                     |
|---------------------------------------------------|------------|----------|-------|-----------|-----------|-------------------------------------------------------------|
| Qwen/Qwen3.6-35B-A3B                              | 0.15       | A100     | 25GB  | 25GB      | ~200m     |                                                             |
| Qwen/Qwen3.6-35B-A3B                              | 0.16       | A100     | 30G   | 28G       | ~60m      |                                                             |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | 0.15       | A100     | >124G | ❌        | ❌        | '_ExpertContainer' object has no attribute 'gate_proj'      |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | 0.16       | A100     | 21G   | 40G       | ~50m      | vllm inference issue, probally not related to AutoRound     |
| Qwen3.8-Flash-Next 180B                           | 0.15       | H200     | >256G | ❌        | ❌        | group_size issue                                            |
| Qwen3.8-Flash-Next 180B                           | 0.16       | H200     | ~120G | ~100G     | 9h        | 95GB ngrams are on cpu, full attention layer is much slower |
| zai-org/GLM-5.3-Flash-BF16 320B                   | 0.15       | H200 X 2 | 50    | (100,120) | ~8H       |                                                             |
| zai-org/GLM-5.3-Flash-BF16 320B                   | 0.16       | H200 X 2 | 40    | (110,120) | ~2.5h     | vllm inference issue, probally not related to AutoRound     |
