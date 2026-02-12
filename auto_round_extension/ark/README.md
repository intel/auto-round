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
# Install auto-round kernel with respective to specific PyTorch version (e.g., v2.8.x)
pip install auto-round-lib torch~=2.8.0 
```

<details>
<summary>Other Installation Methods</summary>

### Install via Script
```bash
curl -fsSL https://raw.githubusercontent.com/intel/auto-round/main/auto_round_extension/ark/install_kernel.py
python3 install_kernel.py
```
**Notes:**  
Recommend to use this method if you want to keep your current PyTorch and auto-round versions.  
This installation script will detect the current environment and install the corresponding auto-round-lib version. 

### Install via auto_round
```bash
pip install auto-round
auto-round-lib-install
```

</details>

### Versioning Scheme
The version number of auto-round-lib follows the format:  
`{auto-round major version}.{auto-round minor version}.{oneAPI version}.{lib version}`   

**For example: v0.10.1.1**  
- The first two digits (0.10) correspond to the major and minor version of the auto_round framework.
- The third component (1) represents the major version of Intel oneAPI: `1` indicates support for oneAPI 2025.1 (typically Torch 2.8), `2` indicates support for oneAPI 2025.2 (typically Torch 2.9), and `3` indicates support for oneAPI 2025.3 (typically Torch 2.10).
- The final digit (1) is the patch version of auto-round-lib, reflecting updates, bug fixes, or improvements to the kernel package itself.

**Version mapping table**

| auto-round-lib Version | auto-round Version | oneAPI Version | Typical PyTorch Version |
|:----------------------:|:------------------:|:--------------:|:-----------------------:|
|        0.10.1.x        |       0.10.x       |     2025.1     |          2.8.x          |
|        0.10.2.x        |       0.10.x       |     2025.2     |          2.9.x          |
|        0.10.3.x        |       0.10.x       |     2025.3     |         2.10.x          |

**Notes:** oneAPI version is aligned with PyTorch version during auto-round-lib binary build, but oneAPI toolkit is not required in runtime. 

### Validated Hardware Environment
#### CPU based on [Intel 64 architecture or compatible processors](https://en.wikipedia.org/wiki/X86-64):
* Intel Xeon Scalable processor (Granite Rapids)
#### GPU built on Intel's Xe architecture:
* Intel Arc B-Series Graphics (Battlemage)