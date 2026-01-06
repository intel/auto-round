## What is AutoRound Kernel?
[TODO]

## Key Features
[TODO]

## Installation
### Install via pip
```bash
# install the latest auto-round-kernel version and this cmd will update your local pytorch version if needed
pip install auto-round-kernel 
# or install together with a specific pytorch version to install the corresponding auto-round-kernel version, e.g., for torch 2.8.x
pip install auto-round-kernel torch~=2.8.0 
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
This installation script will detect the current environment and install the corresponding auto-round-kernel version. 

### Install via auto_round
```bash
pip install auto-round
auto-round-kernel-install
```

</details>

### Versioning Scheme
The version number of auto-round-kernel follows the format:  
`{auto-round major version}.{auto-round minor version}.{oneAPI version}.{kernel version}`   

**For example: v0.9.1.1**  
- The first two digits (0.9) correspond to the major and minor version of the auto_round framework.
- The third digit (1) represents the major version of Intel oneAPI and PyTorch version: `1` indicate support for oneAPI 2025.1 and torch 2.8, `2` indicate support for oneAPI 2025.2 and torch 2.9.
- The final digit (1) is the patch version of auto-round-kernel, reflecting updates, bug fixes, or improvements to the kernel package itself.

**Version mapping table**

| auto-round-kernel Version | auto-round Version | oneAPI Version | Supported PyTorch Version |
|:-------------------------:|:------------------:|:--------------:|:-------------------------:|
|          0.9.1.x          |       0.9.x        |     2025.1     |           2.8.x           |
|          0.9.2.x          |       0.9.x        |     2025.2     |           2.9.x           |

**Notes:** oneAPI version is aligned with PyTorch version during auto-round-kernel binary build, but it is not required in runtime. 

### Validated Hardware Environment
#### CPU based on [Intel 64 architecture or compatible processors](https://en.wikipedia.org/wiki/X86-64):
* Intel Xeon Scalable processor (Granite Rapids)
#### GPU built on Intel's Xe architecture:
* Intel Arc B-Series Graphics (Battlemage)