import re
from io import open
import os
from setuptools import find_packages, setup
import sys
from functools import lru_cache
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"
try:
    filepath = "./auto_round/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)

version = __version__


# All BUILD_* flags are initially set to `False`` and
# will be updated to `True` if the corresponding environment check passes.
BUILD_CUDA_EXT = int(os.environ.get("BUILD_CUDA_EXT", "0")) == 1
PYPI_RELEASE = os.environ.get("PYPI_RELEASE", None)
BUILD_HPU_ONLY = os.environ.get("BUILD_HPU_ONLY", "0") == "1"


def is_cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception as e:
        print(f"Checking CUDA availability failed: {e}")
        return False


if is_cuda_available():
    # When CUDA is available, we build CUDA extension by default
    BUILD_CUDA_EXT = True


@lru_cache(None)
def is_habana_framework_installed():
    """Check if Habana framework is installed.
    Only check for the habana_frameworks package without importing it to avoid
    initializing lazy-mode-related components.
    """
    from importlib.util import find_spec

    package_spec = find_spec("habana_frameworks")
    return package_spec is not None


@lru_cache(None)
def is_hpu_available():
    try:
        import habana_frameworks.torch.core as htcore  # pylint: disable=E0401
        return True
    except ImportError:
        return False

if is_hpu_available() or is_habana_framework_installed():
    # When HPU is available, we build HPU only by default
    BUILD_HPU_ONLY = True

def is_cpu_env():
    try:
        import torch
    except Exception as e:
        print(
            f"Building extension requires PyTorch being installed, please install PyTorch first: {e}.\n NOTE: This issue may be raised due to pip build isolation system (ignoring local packages). Please use `--no-build-isolation` when installing with pip, and refer to https://github.com/intel/auto-round for more details.")
        sys.exit(1)
    if torch.cuda.is_available():
        return False
    try:
        import habana_frameworks.torch.core as htcore 
        return False
    except:
        return True


def fetch_requirements(path):
    requirements = []
    with open(path, "r") as fd:
        requirements = [r.strip() for r in fd.readlines()]
    return requirements

def detect_local_sm_architectures():
    """
    Detect compute capabilities of one machine's GPUs as PyTorch does.

    Copied from https://github.com/pytorch/pytorch/blob/v2.2.2/torch/utils/cpp_extension.py#L1962-L1976
    """
    arch_list = []

    for i in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(i)
        supported_sm = [int(arch.split('_')[1])
                        for arch in torch.cuda.get_arch_list() if 'sm_' in arch]
        max_supported_sm = max((sm // 10, sm % 10) for sm in supported_sm)
        # Capability of the device may be higher than what's supported by the user's
        # NVCC, causing compilation error. User's NVCC is expected to match the one
        # used to build pytorch, so we use the maximum supported capability of pytorch
        # to clamp the capability.
        capability = min(max_supported_sm, capability)
        arch = f'{capability[0]}.{capability[1]}'
        if arch not in arch_list:
            arch_list.append(arch)

    arch_list = sorted(arch_list)
    arch_list[-1] += '+PTX'
    return arch_list


UNSUPPORTED_COMPUTE_CAPABILITIES = ['3.5', '3.7', '5.0', '5.2', '5.3']

if BUILD_CUDA_EXT:
    try:
        import torch
    except Exception as e:
        print(
            f"Building PyTorch CUDA extension requires PyTorch being installed, please install PyTorch first: {e}.\n NOTE: This issue may be raised due to pip build isolation system (ignoring local packages). Please use `--no-build-isolation` when installing with pip, and refer to https://github.com/intel/auto-round for more details.")
        sys.exit(1)
    if not torch.cuda.is_available():
        print(
            f"set BUILD_CUDA_EXT to False as no cuda device is available")
        BUILD_CUDA_EXT = False


if BUILD_CUDA_EXT:
    CUDA_VERSION = None
    ROCM_VERSION = os.environ.get('ROCM_VERSION', None)
    if ROCM_VERSION and not torch.version.hip:
        print(
            f"Trying to compile auto-round for ROCm, but PyTorch {torch.__version__} "
            "is installed without ROCm support."
        )
        sys.exit(1)

    if not ROCM_VERSION:
        default_cuda_version = torch.version.cuda
        CUDA_VERSION = "".join(os.environ.get("CUDA_VERSION", default_cuda_version).split("."))

    if ROCM_VERSION:
        version += f"+rocm{ROCM_VERSION}"
    else:
        if not CUDA_VERSION:
            print(
                f"Trying to compile auto-round for CUDA, but Pytorch {torch.__version__} "
                "is installed without CUDA support."
            )
            sys.exit(1)

        torch_cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
        if torch_cuda_arch_list is not None:
            torch_cuda_arch_list = torch_cuda_arch_list.replace(' ', ';')
            archs = torch_cuda_arch_list.split(';')

            requested_but_unsupported_archs = {arch for arch in archs if arch in UNSUPPORTED_COMPUTE_CAPABILITIES}
            if len(requested_but_unsupported_archs) > 0:
                raise ValueError(
                    f"Trying to compile AutoRound for CUDA compute capabilities {torch_cuda_arch_list}, but AutoRound does not support the compute capabilities {requested_but_unsupported_archs} (AutoRound requires Pascal or higher). Please fix your environment variable TORCH_CUDA_ARCH_LIST (Reference: https://github.com/pytorch/pytorch/blob/v2.2.2/setup.py#L135-L139).")
        else:
            local_arch_list = detect_local_sm_architectures()
            local_but_unsupported_archs = {arch for arch in local_arch_list if arch in UNSUPPORTED_COMPUTE_CAPABILITIES}
            if len(local_but_unsupported_archs) > 0:
                raise ValueError(
                    f"PyTorch detected the compute capabilities {local_arch_list} for the NVIDIA GPUs on the current machine, but AutoRound can not be built for compute capabilities {local_but_unsupported_archs} (AutoRound requires Pascal or higher). Please set the environment variable TORCH_CUDA_ARCH_LIST (Reference: https://github.com/pytorch/pytorch/blob/v2.2.2/setup.py#L135-L139) with your necessary architectures.")

        # For the PyPI release, the version is simply x.x.x to comply with PEP 440.
        if not PYPI_RELEASE:
            version += f"+cu{CUDA_VERSION}"

additional_setup_kwargs = {}
include_dirs = ["autoround_cuda"]
if BUILD_CUDA_EXT:
    from torch.utils import cpp_extension

    if not ROCM_VERSION:
        from distutils.sysconfig import get_python_lib

        conda_cuda_include_dir = os.path.join(get_python_lib(), "nvidia/cuda_runtime/include")

        print("conda_cuda_include_dir", conda_cuda_include_dir)
        if os.path.isdir(conda_cuda_include_dir):
            include_dirs.append(conda_cuda_include_dir)
            print(f"appending conda cuda include dir {conda_cuda_include_dir}")
    if os.name == "nt":
        # On Windows, fix an error LNK2001: unresolved external symbol cublasHgemm bug in the compilation
        cuda_path = os.environ.get("CUDA_PATH", None)
        if cuda_path is None:
            raise ValueError(
                "The environment variable CUDA_PATH must be set to the path to the CUDA install when installing from source on Windows systems.")
        extra_link_args = ["-L", f"{cuda_path}/lib/x64/cublas.lib"]
    else:
        extra_link_args = []
    extensions = []
    extensions.append(
        cpp_extension.CUDAExtension(
            "autoround_exllamav2_kernels",
            [
                "auto_round_extension/cuda/exllamav2/ext.cpp",
                "auto_round_extension/cuda/exllamav2/cuda/q_matrix.cu",
                "auto_round_extension/cuda/exllamav2/cuda/q_gemm.cu",
            ],
            extra_link_args=extra_link_args
        )
    )
    additional_setup_kwargs = {
        "ext_modules": extensions,
        "cmdclass": {'build_ext': cpp_extension.BuildExtension}
    }

PKG_INSTALL_CFG = {
    "include_packages": find_packages(
        include=[
            "auto_round",
            "auto_round.*",
            "auto_round_extension",
            "auto_round_extension.*",
        ],
    ),
    "install_requires": fetch_requirements("requirements.txt"),
    "extras_require": {
        "hpu": fetch_requirements("requirements-hpu.txt"),
        "cpu": fetch_requirements("requirements-cpu.txt"),
    },
}

if __name__ == "__main__":
    # There are two ways to install hpu-only package:
    # 1. pip install -vvv --no-build-isolation -e .[hpu]
    # 2. Within the gaudi docker where the HPU is available, we install the hpu package by default.

    include_packages = PKG_INSTALL_CFG.get("include_packages", {})
    install_requires = PKG_INSTALL_CFG.get("install_requires", [])
    extras_require = PKG_INSTALL_CFG.get("extras_require", {})

    setup(
        name="auto_round",
        author="Intel AIPT Team",
        version=version,
        author_email="wenhua.cheng@intel.com, weiwei1.zhang@intel.com",
        description="Repository of AutoRound: Advanced Weight-Only Quantization Algorithm for LLMs",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords="quantization,auto-around,LLM,SignRound",
        license="Apache 2.0",
        url="https://github.com/intel/auto-round",
        packages=include_packages,
        include_dirs=include_dirs,
        ##include_package_data=False,
        install_requires=install_requires,
        extras_require=extras_require,
        python_requires=">=3.7.0",
        classifiers=[
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: Apache Software License",
        ],
        include_package_data=True,
        package_data={"": ["mllm/templates/*.json"]},
        **additional_setup_kwargs,
    )
