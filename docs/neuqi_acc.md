### NeUQI Accuracy Validation

**NeUQI** ([arXiv 2505.17595](https://arxiv.org/abs/2505.17595)) is a calibration-free grid search for the optimized-RTN path, enabled with `--enable_neuqi`: asymmetric layers run a joint (scale, integer zero-point) search, symmetric layers a two-stage signed scale search; on the zero-shot path both are weighted by the activation imatrix (collected automatically), while with `iters > 0` the anchor runs unweighted. With `iters > 0`, the search result anchors the SignRound tuning grid (frozen init).

#### Protocol

KL divergence between the BF16 base model and the dequantized W4A16 checkpoint over the **full vocabulary**, computed in **fp32** on openwebtext (8 sequences × 8192 tokens, 65,528 positions). Both passes run through the same evaluation stack, so the only difference is the quantization; **lower is better**. `top1` is the next-token agreement rate, `top5` the Jaccard similarity of the top-5 sets. Wall times are single-GPU runs on one RTX 3090 (24 GB) for Qwen3.8-27B (W4A16, group size 128).

#### Results — iters=0 (calibration-free zero-shot)

| recipe | mean | median | p90 | p99 | top1 | top5 | wall |
|:--|--:|--:|--:|--:|--:|--:|--:|
| asym, NeUQI | **0.02579** | 0.01361 | 0.05046 | 0.21753 | 0.9234 | 0.8514 | 11m |
| sym, NeUQI | **0.02734** | 0.01468 | 0.05358 | 0.22014 | 0.9213 | 0.8463 | **9m** |
| sym, incumbent OptRTN search | 0.02852 | 0.01533 | 0.05678 | 0.23453 | 0.9191 | 0.8429 | 13m19s |

- The symmetric two-stage search **strictly dominates the incumbent search**: −4.1% mean KL, −32% wall time, and better agreement at every quantile.
- There is no incumbent asymmetric search — `--enable_neuqi` adds the optimized-init capability for asymmetric layers; the quality ladder is flat from 64/32 to 256/64, so the [backend-aware default](./environments.md) narrows the eager sweep to 64/32 at no measured cost.

#### Results — iters > 0 (SignRoundV2 tuning, `enable_alg_ext`)

| recipe | mean | median | p90 | p99 | top1 | top5 | wall |
|:--|--:|--:|--:|--:|--:|--:|--:|
| asym, SignRoundV2 + NeUQI, iters=200 (64/32 grid) | **0.01921** | 0.00905 | 0.03591 | 0.17297 | 0.9370 | 0.8727 | 2h57m21s |
| asym, SignRoundV2 + NeUQI | **0.02106** | 0.01050 | 0.04071 | 0.18038 | 0.9317 | 0.8661 | 49m07s |
| sym, SignRoundV2 + NeUQI | **0.02191** | 0.01116 | 0.04237 | 0.18473 | 0.9304 | 0.8627 | 56m52s |
| sym, SignRoundV2 without NeUQI | 0.02203 | 0.01112 | 0.04307 | 0.18239 | 0.9295 | 0.8630 | 55m40s |

- At `iters=50` the NeUQI anchor is **quality-neutral on sym** (0.02191 vs 0.02203 without NeUQI): SignRoundV2's own tuning dominates once it runs, so the anchor's role is a no-cost initialization, not an accuracy lever. (An earlier no-NeUQI baseline of 0.02626 was not SignRoundV2-equivalent — most of its gap was the quantizer version, not NeUQI.)
- Asymmetric tuning lands at 0.02106, on par with the best prior asymmetric recipe family at the same iteration budget; the dose-response continues to 0.01921 at 200 iterations. (No SignRoundV2 asym no-NeUQI baseline has been measured; the asym anchor claim is parity-with-prior, not an A/B.)
