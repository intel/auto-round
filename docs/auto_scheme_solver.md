# AutoScheme Bit-Allocation Solver

AutoScheme assigns a per-layer quantization scheme by solving a knapsack problem: every
layer has a set of candidate options, each with a *bit cost* and a *predicted loss cost*
(the Delta-Loss score), and the solver picks one option per layer so that the total loss
is minimised subject to an average-bits budget.

Two solvers are available, selected with `AutoScheme.solver` (or `--auto_scheme_solver`
on the CLI):

| Solver | Description |
|:---|:---|
| `dp` (default) | Knapsack dynamic program. Exact on a discretised bit grid, but the state space grows with the bit budget. |
| `lagrangian` | Solves the same knapsack through its Lagrangian dual. For a price `lam` (loss per bit) every layer independently picks `argmin_s loss_s + lam * bits_s`; total bits decrease monotonically in `lam`, so a bisection drives the solution onto the budget. Hits a *fractional* avg_bits target exactly and needs no discretised state space. |

Because the dual only lands on the convex hull of each layer's (bits, loss) curve, two
primal repair passes close the integrality gap: a greedy repair that spends leftover
budget, and a pairwise swap local search where one downgrade funds one upgrade. In
practice the repaired dual solution is budget-tight and matches the DP allocation.

## Usage

```python
from auto_round import AutoRound
from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

scheme = AutoScheme(
    avg_bits=4.5,
    options="MXFP4,MXFP8",
    solver="lagrangian",  # default is "dp"
)
ar = AutoRound(model_name, scheme=scheme)
model, layer_config = ar.quantize()
```

CLI:

```bash
auto_round Qwen/Qwen3-8B --avg_bits 4.5 --options "MXFP4,MXFP8" --solver lagrangian
```

## Accuracy

All runs use RTN (`iters=0`), identical calibration data (128 samples, seqlen 512),
identical options and identical `avg_bits`. Only the solver differs, so any delta is
attributable to the allocation itself. `lm_head` is not quantized.

Evaluated with lm-eval on `lambada_openai`, `hellaswag`, `piqa`, `winogrande`,
`truthfulqa_mc1`, `openbookqa`, `boolq`, `arc_easy`, `arc_challenge`, `mmlu`.
The reported **Avg** is the mean over all evaluated tasks *including* the MMLU sub-tasks,
so it does not equal the mean of the ten columns shown.

### Summary

| Experiment | Model | `dp` | `lagrangian` | Delta |
|:---|:---|:---:|:---:|:---:|
| INT, avg_bits 3.5 | Qwen3-8B | 0.6990 | **0.7045** | **+0.55pp** |
| INT, avg_bits 3.5 | Llama-3.1-8B-Instruct | 0.6386 | **0.6388** | +0.02pp |
| INT, avg_bits 3.0 | Qwen3-8B | 0.4643 | 0.4643 | 0.00pp |
| INT, avg_bits 3.0 | Llama-3.1-8B-Instruct | 0.5794 | **0.5813** | **+0.19pp** |
| MXFP4/8, avg_bits 4.5 | Qwen3-8B | 0.6942 | 0.6942 | 0.00pp |
| MXFP4/8, avg_bits 4.5 | Llama-3.1-8B-Instruct | 0.6221 | 0.6221 | 0.00pp |

The Lagrangian solver never lost on any tested configuration: it matched the DP in 3 of
6 cases and beat it in 3.

### Table 1 — INT W2A16/W4A16/W8A16, avg_bits 3.5

**Qwen3-8B**

| Solver | Bit histogram | lambada | hellaswag | piqa | winogrande | truthfulqa | openbookqa | boolq | arc_easy | arc_chal | mmlu | Avg |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dp` | int2 65, int4 187 | 0.3276 | 0.6763 | 0.7144 | 0.6409 | 0.3599 | 0.3880 | 0.8425 | 0.6515 | 0.4343 | 0.6954 | 0.6990 |
| `lagrangian` | int2 63, int4 189 | 0.3088 | 0.6783 | 0.7111 | 0.6527 | 0.3660 | 0.3740 | 0.8410 | 0.6515 | 0.4292 | 0.7008 | **0.7045** |

**Llama-3.1-8B-Instruct**

| Solver | Bit histogram | lambada | hellaswag | piqa | winogrande | truthfulqa | openbookqa | boolq | arc_easy | arc_chal | mmlu | Avg |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dp` | int2 55, int4 169 | 0.6103 | 0.7549 | 0.7797 | 0.7261 | 0.3378 | 0.4200 | 0.8343 | 0.7433 | 0.4949 | 0.6281 | 0.6386 |
| `lagrangian` | int2 52, int4 172 | 0.5973 | 0.7547 | 0.7840 | 0.7214 | 0.3366 | 0.4180 | 0.8336 | 0.7483 | 0.5051 | 0.6248 | **0.6388** |

### Table 2 — INT W2A16/W4A16/W8A16, avg_bits 3.0

**Qwen3-8B** — both solvers produce a *bit-identical* allocation, hence identical scores.

| Solver | Bit histogram | lambada | hellaswag | piqa | winogrande | truthfulqa | openbookqa | boolq | arc_easy | arc_chal | mmlu | Avg |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dp` | int2 129, int4 123 | 0.1091 | 0.5721 | 0.6730 | 0.5572 | 0.3305 | 0.3320 | 0.5636 | 0.5215 | 0.3456 | 0.4466 | 0.4643 |
| `lagrangian` | int2 129, int4 123 | 0.1091 | 0.5721 | 0.6730 | 0.5572 | 0.3305 | 0.3320 | 0.5636 | 0.5215 | 0.3456 | 0.4466 | 0.4643 |

**Llama-3.1-8B-Instruct**

| Solver | Bit histogram | lambada | hellaswag | piqa | winogrande | truthfulqa | openbookqa | boolq | arc_easy | arc_chal | mmlu | Avg |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dp` | int2 109, int4 115 | 0.5700 | 0.5870 | 0.7013 | 0.6338 | 0.3219 | 0.3880 | 0.6939 | 0.6077 | 0.4232 | 0.5649 | 0.5794 |
| `lagrangian` | int2 106, int4 118 | 0.5750 | 0.5875 | 0.7002 | 0.6511 | 0.3182 | 0.3800 | 0.6985 | 0.6149 | 0.4258 | 0.5669 | **0.5813** |

### Table 3 — MXFP4/MXFP8, avg_bits 4.5

Both models produce a *bit-identical* allocation under either solver, so every per-task
score matches exactly.

**Qwen3-8B**

| Solver | Bit histogram | lambada | hellaswag | piqa | winogrande | truthfulqa | openbookqa | boolq | arc_easy | arc_chal | mmlu | Avg |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dp` | mx_fp4 173, mx_fp8 79 | 0.6072 | 0.6943 | 0.7519 | 0.6551 | 0.3317 | 0.4060 | 0.8627 | 0.7597 | 0.5188 | 0.6795 | 0.6942 |
| `lagrangian` | mx_fp4 173, mx_fp8 79 | 0.6072 | 0.6943 | 0.7519 | 0.6551 | 0.3317 | 0.4060 | 0.8627 | 0.7597 | 0.5188 | 0.6795 | 0.6942 |

**Llama-3.1-8B-Instruct**

| Solver | Bit histogram | lambada | hellaswag | piqa | winogrande | truthfulqa | openbookqa | boolq | arc_easy | arc_chal | mmlu | Avg |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dp` | mx_fp4 159, mx_fp8 65 | 0.6375 | 0.7557 | 0.7884 | 0.7040 | 0.3097 | 0.4060 | 0.8156 | 0.7525 | 0.5000 | 0.6122 | 0.6221 |
| `lagrangian` | mx_fp4 159, mx_fp8 65 | 0.6375 | 0.7557 | 0.7884 | 0.7040 | 0.3097 | 0.4060 | 0.8156 | 0.7525 | 0.5000 | 0.6122 | 0.6221 |

### Interpretation

The two solvers optimise the same objective over the same score table, so where the
problem has a clear optimum they converge to the *same* allocation — that is the case for
MXFP4/8 at 4.5 bits and for INT at 3.0 bits on Qwen3-8B, where the bit histograms and all
per-task scores are identical.

Differences appear only where the knapsack has near-ties: the DP works on a discretised
bit grid, while the Lagrangian dual handles a fractional budget exactly, so it can settle
on a slightly different trade-off (e.g. `int2 63 / int4 189` instead of
`int2 65 / int4 187`). In every such case the Lagrangian choice was at least as good.

Note that individual tasks move in both directions even when the average improves —
on Qwen3-8B at 3.5 bits the Lagrangian allocation loses 1.9pp on `lambada_openai` but
gains on `winogrande`, `truthfulqa_mc1`, `hellaswag` and `mmlu`. Judge the allocation on
the aggregate, not on any single task.

### Reproducing

```bash
sh run_autoscheme_staged_ab.sh mxfp4
```

See `autoscheme_staged_ab.py` for the A/B driver and `run_autoscheme_staged_ab.sh` for
the per-experiment settings.

