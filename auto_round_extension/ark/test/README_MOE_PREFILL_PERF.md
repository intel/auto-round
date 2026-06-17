# MoE Prefill Performance Test

## Overview

The `test_moe_prefill_perf.py` file provides comprehensive performance benchmarks for MoE (Mixture of Experts) prefill operations with TFLOPS (Tera Floating Point Operations Per Second) calculations.

## What is MoE Prefill?

**Prefill** is the phase during LLM inference where many tokens (e.g., the entire prompt or a batch of sequences) are processed simultaneously. In MoE models, tokens are routed to different experts, and each expert may receive multiple tokens. This is different from **decode** (token generation), where typically only one token per expert is processed at a time.

## Features

### 1. **Comprehensive Data Type Support**
- FP16 (float16)
- BF16 (bfloat16)
- INT8 (symmetric and asymmetric quantization)
- INT4 (symmetric and asymmetric quantization)
- INT2 (symmetric and asymmetric quantization)
- FP8 (float8_e4m3fn and float8_e5m2)

### 2. **TFLOPS Calculation**
The test calculates TFLOPS for each configuration using the formula:
```
FLOPs = total_tokens × K × N × 2
TFLOPS = FLOPs / (time_in_seconds) / 1e12
```

Where:
- `total_tokens`: Total number of tokens across all experts
- `K`: Input feature dimension
- `N`: Output feature dimension
- `×2`: Each multiply-add operation counts as 2 FLOPs

### 3. **Various MoE Configurations**
The test covers multiple realistic MoE scenarios:
- **Small models** (8 experts, Mixtral-style): 4096×4096, 4096×14336, 14336×4096
- **Medium models** (8 experts): Various token distributions
- **Large models** (16, 32, 64 experts, DeepSeek-style): 2048×2048
- **Uneven distributions**: Simulates real-world routing patterns

### 4. **Baseline Comparison**
Each test compares the ARK MoE kernel against a baseline PyTorch implementation:
- **Baseline**: Per-expert matrix multiplication using `torch.matmul`
- **ARK Kernel**: Optimized `ark.moe_gemm` with fused operations
- **Speedup**: Reports speedup ratio (baseline_time / ark_time)

## How to Run

### Run all tests:
```bash
cd /path/to/auto_round_extension/ark/test
pytest -v -s test_moe_prefill_perf.py
```

### Run specific data type:
```bash
# FP16 tests only
pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_fp

# INT4 tests only
pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_int4

# INT8 symmetric quantization with bfloat16 activations
pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_int8 -k "bfloat16 and not asym"
```

**Note**: The `-s` flag is required to see the printed timing tables and TFLOPS output.

## Output Format

The test prints formatted tables with the following columns:

```
shape          E      N      K  tokens    baseline(ms)        ark(ms)     speedup    TFLOPS
small  E=8     8   4096   4096     252         12.3456        4.5678       2.70x       45.2
medium E=8     8   4096  14336     528         23.4567        8.9012       2.64x       78.9
...
```

Where:
- **shape**: Configuration label
- **E**: Number of experts
- **N**: Output feature dimension
- **K**: Input feature dimension
- **tokens**: Total tokens across all experts
- **baseline(ms)**: PyTorch baseline latency (milliseconds)
- **ark(ms)**: ARK kernel latency (milliseconds)
- **speedup**: Performance improvement ratio
- **TFLOPS**: Throughput in tera floating point operations per second

## Requirements

- Intel XPU (Arc GPU) with PyTorch XPU support
- `auto_round_kernel` built with `ARK_SYCL_TLA=ON`
- Test dependencies from `test_moe.py` (pack/dequant helpers)

## Architecture

```
test_moe_prefill_perf.py
├── Timing utilities (_xpu_time_ms)
│   └── Uses XPU events for accurate GPU timing
├── FLOPS calculation (_compute_moe_flops)
│   └── Computes theoretical FLOPs for TFLOPS metric
├── Baseline implementation (_default_moe_prefill)
│   └── Per-expert PyTorch matmul for comparison
├── Test shapes (PREFILL_SHAPES)
│   └── Various realistic MoE configurations
└── Test cases (TestMoEGemmPrefillPerf)
    ├── test_perf_fp (FP16/BF16)
    ├── test_perf_int4 (INT4 sym/asym)
    ├── test_perf_int8 (INT8 sym/asym)
    ├── test_perf_int2 (INT2 sym/asym)
    └── test_perf_fp8 (FP8 e4m3fn/e5m2)
```

## Example Output

```
==================================================================
FP weights (float16)  -- ark.moe_gemm (prefill) vs per-expert A @ W.T
==================================================================
shape              E      N      K  tokens    baseline(ms)        ark(ms)     speedup    TFLOPS
------------------------------------------------------------------
small  E=8         8   4096   4096     252         12.3456        4.5678       2.70x       45.2
medium E=8         8   4096  14336     528         23.4567        8.9012       2.64x       78.9
medium E=8         8  14336   4096     528         25.6789        9.1234       2.82x       76.5
large  E=16       16   2048   2048     256          5.6789        2.3456       2.42x       91.2
large  E=32       32   2048   2048     256          5.7890        2.4567       2.36x       87.3
large  E=64       64   2048   2048     256          5.8901        2.5678       2.29x       83.5
uneven E=8         8   4096   4096     610         28.9012       10.1234       2.86x       52.1
```

## Key Metrics

1. **TFLOPS**: Higher is better - indicates compute throughput
2. **Speedup**: Higher is better - shows performance gain over baseline
3. **Latency (ms)**: Lower is better - actual kernel execution time

## Integration with CI/CD

This test can be integrated into performance regression testing:
- Set minimum TFLOPS thresholds for each configuration
- Track speedup ratios over time
- Alert on performance degradation

## Related Files

- `test_moe.py`: Correctness tests for MoE GEMM
- `test_moe_decode_perf.py`: Performance tests for MoE decode (single token per expert)
- `test_bench_bmg.py`: SDPA performance benchmarks with TFLOPS
