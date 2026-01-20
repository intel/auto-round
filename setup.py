import os
import re
import subprocess
import sys
from functools import lru_cache
from io import open

from setuptools import find_packages, setup

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"
try:
    filepath = "./auto_round/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)

# All BUILD_* flags are initially set to `False`` and
# will be updated to `True` if the corresponding environment check passes.
PYPI_RELEASE = os.environ.get("PYPI_RELEASE", None)
BUILD_HPU_ONLY = os.environ.get("BUILD_HPU_ONLY", "0") == "1"


def is_commit_on_tag():
    try:
        result = subprocess.run(
            ["git", "describe", "--exact-match", "--tags"], capture_output=True, text=True, check=True
        )
        tag_name = result.stdout.strip()
        return tag_name
    except subprocess.CalledProcessError:
        return False


def get_build_version():
    if is_commit_on_tag():
        return __version__
    try:
        result = subprocess.run(["git", "describe", "--tags"], capture_output=True, text=True, check=True)
        distance = result.stdout.strip().split("-")[-2]
        commit = result.stdout.strip().split("-")[-1]
        return f"{__version__}.dev{distance}+{commit}"
    except subprocess.CalledProcessError:
        return __version__


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
            f"Building extension requires PyTorch being installed, please install PyTorch first: {e}.\n NOTE: This issue may be raised due to pip build isolation system (ignoring local packages). Please use `--no-build-isolation` when installing with pip, and refer to https://github.com/intel/auto-round for more details."
        )
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
        requirements = [r.strip() for r in fd]
    return requirements


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
    # auto-round[cpu] is deprecated, will be removed from v1.0.0
    "extras_require": {"cpu": fetch_requirements("requirements-cpu.txt"), "kernel": ["auto-round-kernel"]},
}

###############################################################################
# Configuration for auto_round_lib
# From pip:
# pip install auto-round-lib
# From source:
# python setup.py lib install
###############################################################################


LIB_REQUIREMENTS_FILE = "requirements-lib.txt"
LIB_INSTALL_CFG = {
    "include_packages": find_packages(
        include=[
            "auto_round",
            "auto_round.*",
            "auto_round_extension",
            "auto_round_extension.*",
        ],
    ),
    "install_requires": fetch_requirements(LIB_REQUIREMENTS_FILE),
}

if __name__ == "__main__":

    package_name = "auto_round"
    # There are two ways to install hpu-only package:
    # 1. python setup.py lib install
    # 2. Within the gaudi docker where the HPU is available, we install the "auto_round_lib" by default.
    # 3. This package is deprecated and will be removed from v1.0.0 release, please replace with auto_round_hpu.
    is_user_requesting_library_build = "lib" in sys.argv
    if is_user_requesting_library_build:
        sys.argv.remove("lib")
    should_build_library = is_user_requesting_library_build or BUILD_HPU_ONLY
    if should_build_library:
        package_name = "auto_round_lib"

    # From v0.9.3, auto-round-hpu will be published to replace auto-round-lib.
    hpu_build = "hpu" in sys.argv
    if hpu_build:
        sys.argv.remove("hpu")
        package_name = "auto_round_hpu"

    if should_build_library or hpu_build:
        INSTALL_CFG = LIB_INSTALL_CFG
    else:
        INSTALL_CFG = PKG_INSTALL_CFG

    include_packages = INSTALL_CFG.get("include_packages", {})
    install_requires = INSTALL_CFG.get("install_requires", [])
    extras_require = INSTALL_CFG.get("extras_require", {})

    setup(
        name=package_name,
        author="Intel AIPT Team",
        version=get_build_version(),
        author_email="wenhua.cheng@intel.com, weiwei1.zhang@intel.com, heng.guo@intel.com",
        description="Repository of AutoRound: Advanced Weight-Only Quantization Algorithm for LLMs",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords="quantization,auto-around,LLM,SignRound",
        license="Apache 2.0",
        url="https://github.com/intel/auto-round",
        packages=include_packages,
        install_requires=install_requires,
        extras_require=extras_require,
        python_requires=">=3.10.0",
        classifiers=[
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: Apache Software License",
        ],
        include_package_data=True,
        package_data={"": ["mllm/templates/*.json"]},
    )
