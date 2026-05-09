## What is AutoRound Kernel?
AutoRound Kernel is a low-bit acceleration library for Intel platform. 

The kernels are optimized for the following CPUs:
* Intel Xeon Scalable processor (formerly Sapphire Rapids, and Emerald Rapids)
* Intel Xeon 6 processors (formerly Sierra Forest and Granite Rapids)

The kernels are optimized for the following GPUs:
* Intel Arc B-Series Graphics and Intel Arc Pro B-Series Graphics
  (formerly Battlemage)

## Key Features
AutoRound Kernel provides weight-only linear computational capabilities for LLM inference. Specifically, the weight-only-quantization configs we support are given in the table below:
### CPU 
| Weight dtype     |          Compute dtype           |    Scale dtype    | Algorithm<sup>[1]</sup> |
|------------------|:--------------------------------:|:-----------------:|:-----------------------:|
| INT8             | INT8<sup>[2]</sup> / BF16 / FP32 |    BF16 / FP32    |       sym / asym        |
| INT4             |        INT8 / BF16 / FP32        |    BF16 / FP32    |       sym / asym        |
| INT3             |        INT8 / BF16 / FP32        |    BF16 / FP32    |       sym / asym        |
| INT2             |        INT8 / BF16 / FP32        |    BF16 / FP32    |       sym / asym        |
| INT5             |        INT8 / BF16 / FP32        |    BF16 / FP32    |       sym / asym        |
| INT6             |        INT8 / BF16 / FP32        |    BF16 / FP32    |       sym / asym        |
| INT7             |        INT8 / BF16 / FP32        |    BF16 / FP32    |       sym / asym        |
| INT1             |        INT8 / BF16 / FP32        |    BF16 / FP32    |       sym / asym        |
| FP8 (E4M3, E5M2) |           BF16 / FP32            | FP32 / FP8 (E8M0) |           NA            |
| FP4 (E2M1)       |           BF16 / FP32            |    BF16 / FP32    |           NA            |

### XPU 
| Weight dtype     |  Compute dtype |     Scale dtype   |  Algorithm |
|------------------|:--------------:|:-----------------:|:----------:|
| INT8             |  INT8 / FP16   |       FP16        |    sym     |
| INT4             |  INT8 / FP16   |       FP16        |    sym     |
| FP8 (E4M3, E5M2) |      FP16      | FP16 / FP8 (E8M0) |     NA     |

<sup>[1]</sup>: Quantization algorithms for integer types: symmetric or asymmetric.  
<sup>[2]</sup>: Includes dynamic activation quantization; results are dequantized to floating-point formats.  
  

## Installation
### 1. Install via pip
```bash
pip install auto-round-lib
```

### 2. Install from Source
```bash
python setup.py bdist_wheel;pip install dist/*
```

### Validated Hardware Environment
#### CPU based on [Intel 64 architecture or compatible processors](https://en.wikipedia.org/wiki/X86-64):
* Intel Xeon Scalable processor (Granite Rapids)
#### GPU built on Intel's Xe architecture:
* Intel Arc B-Series Graphics (Battlemage)

### Resources

#### QuantLinear API
  ARK exposes a unified weight-only linear interface through QuantLinear, QuantLinearGPTQ, QuantLinearAWQ, and QuantLinearFP8. Please refer to the [QLinear](auto_round_kernel/qlinear.py) for more integration details.

  The expected lifecycle is: create the module, load quantized tensors from the checkpoint, call post_init() once to repack weights into the ARK-friendly layout, and then call forward() during inference.

   Minimal usage:
```python
from auto_round_kernel.qlinear import QuantLinear

qlinear = QuantLinear(
    bits=4,
    group_size=128,
    sym=True,
    in_features=in_features,
    out_features=out_features,
    bias=bias is not None,
    weight_dtype=weight_dtype,
)
# Load qweight, qzeros, scales, and bias from checkpoint.
qlinear.post_init()

# Run inference
y = qlinear(x)
```

#### A Weight-Only Example
  A runnable end-to-end example is available in [test_weightonly.py](test/test_weightonly.py). It demonstrates how to prepare quantized weights and scales, call repack_quantized_weight to build ARK-packed weights, verify correctness with unpack_weight, and run woqgemm on CPU and XPU.
