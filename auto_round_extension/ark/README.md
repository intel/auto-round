## What is AutoRound Kernel?
[TODO]

## Key Features
[TODO]

## Installation
### Recommended to Install via Script

```bash
curl -fsSL https://raw.githubusercontent.com/intel/auto-round/main/auto_round_extension/ark/install_kernel.py
python3 install_kernel.py
```

<details>
<summary>Other Installation Methods</summary>

### Install via pip
```bash
# install the latest auto-round-kernel version and this cmd will update your local pytorch version if needed
pip install auto-round-kernel 
# or install together with your pytorch version, e.g., for torch 2.8.x
pip install torch~=2.8.0 auto-round-kernel
```
### Install via auto_round
```bash
pip install auto-round
kernel-install
```

</details>

### Versioning Scheme
The version number of auto-round-kernel follows the format:  
`{auto-round major version}.{auto-round minor version}.{oneAPI version}.{kernel version}`   

**For example: v0.9.1.1**  
- The first two digits (0.9) correspond to the major and minor version of the auto_round framework
- The third digit (1) represents the major version of Intel oneAPI. This digit is also aligned with the supported PyTorch version: `1` indicate support for oneAPI 2025.1 and torch 2.8, `2` indicate support for oneAPI 2025.2 and torch 2.9
- The final digit (1) is the patch version of auto-round-kernel, reflecting updates, bug fixes, or improvements to the kernel package itself


