torch 2.14 is used (<2.14 will cause torch compile exception), H200 with 140G or A100 with 80g
x denote exceptions

| Model                    | AR Version | Device | ram   | vram  | time cost | comment                |   |   |   |   |
|--------------------------|------------|--------|-------|-------|-----------|------------------------|---|---|---|---|
| Qwen/Qwen3.6-35B-A3B     | 0.15       | A100   |       |       | ~200m     |                        |   |   |   |   |
| Qwen/Qwen3.6-35B-A3B     | 0.16       | A100   | 30G   | 28G   | ~50m      |                        |   |   |   |   |
| Qwen3.8-Flash-Next 180B  | 0.15       | H200   | -     | ❌    | ❌        | group_size issue       |   |   |   |   |
| Qwen3.8-Flash-Next  180B | 0.16       | H200   | ~120G | ~100G |           | 95GB ngrams are on cpu |   |   |   |   |
