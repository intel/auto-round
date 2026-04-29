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
### Install via pip
```bash
# Install the latest auto-round kernel which may upgrade your PyTorch version automatically
pip install auto-round-lib
```

<details>
<summary>Other Installation Methods</summary>

### Install via Script
```bash
curl -fsSL https://raw.githubusercontent.com/intel/auto-round/main/auto_round_extension/ark/install_kernel.py
python3 install_kernel.py
```
**Notes:**  
This installation script will detect the current environment and install the auto-round-lib.

### Install via auto_round
```bash
pip install auto-round
auto-round-lib-install
```

</details>

### Validated Hardware Environment
#### CPU based on [Intel 64 architecture or compatible processors](https://en.wikipedia.org/wiki/X86-64):
* Intel Xeon Scalable processor (Granite Rapids)
#### GPU built on Intel's Xe architecture:
* Intel Arc B-Series Graphics (Battlemage)