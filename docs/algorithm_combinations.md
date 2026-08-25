# Algorithm Combinations

AutoRound can be combined with several algorithms
before (or during) quantization. This page summarizes each combination and rates
it along two dimensions:

- **Accuracy Gain** — does the transform improve the accuracy of the quantized
  model compared with plain AutoRound?
- **Deployment** — can the resulting model actually be deployed/served today
  (kernel support, export path, real inference engine)?

## Legend

| Light | Meaning                                                         |
|:-----:|:----------------------------------------------------------------|
|  🟢   | Good — clear benefit / ready to deploy                          |
|  🟡   | Partial — conditional benefit / limited or experimental support |
|  🔴   | Poor — no measurable benefit / not deployable yet               |

> **Note:** A Partial/Poor **Accuracy Gain** rating may stem from two factors:
> (1) limitations in our current implementation, and (2) our own internal,
> subjective evaluation. Both are subject to change as the implementation matures
> and more benchmarks become available.

## Matrix

| Combination                                            | Accuracy Gain | Deployment | Details                                 | CLI Usage                        | Comments                                                                                | Reference                                            |
|:-------------------------------------------------------|:-------------:|:----------:|:----------------------------------------|:---------------------------------|:----------------------------------------------------------------------------------------|:-----------------------------------------------------|
| AutoRound + AWQ (activation-aware scaling)             |      🟢       |     🟢     | [awq_details](awq_details.md)           | `--algorithm awq,signround`      | Recommended when activations are quantized (e.g., W4A4).                                | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| AutoRound + Hadamard rotation                          |      🟢       |     🔴     | [rotation_details](rotation_details.md) | `--algorithm hadamard,signround` | Especially helpful for INT4 (W4A4) and some MXFP4 scenarios.   no production kernel.    | [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) |
| AutoRound + SpinQuant                                  |      🟡       |     🔴     | [rotation_details](rotation_details.md) | Python API only                  | Learns rotation matrices; higher accuracy at extra training cost. no production kernel. | [arXiv:2405.16406](https://arxiv.org/abs/2405.16406) |
| AutoRound + LFQ (logit-aware final-block quantization) |      🔴       |     🟢     | [lfq_acc](lfq_acc.md)                   | `--enable_lfq`                   | Refines the final block to lift low-bit generation quality                              | [arXiv:2605.29756](https://arxiv.org/abs/2605.29756) |
| AutoRound + MX Attention (mxfp4 varaint)               |      🟡       |     🔴     | [mxnv_acc](mxnv_acc.md)                 | `--data_type mx_fp4_rceil_v2`    | Adopt 7.25 as the denominator for scale calculation                                     | [arXiv:2607.24377](https://arxiv.org/abs/2607.24377) |
| AutoRound + SVDQuant (low-rank outlier absorption)     |      🟡       |     🟡     | [svdquant_details](svdquant_details.md) | `--algorithm svdquant,signround` | Recommended for diffusion models; currently only FLUX is supported.                     | [arXiv:2411.05007](https://arxiv.org/abs/2411.05007) |

