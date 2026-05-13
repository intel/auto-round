# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import ctypes
import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

build_mode = os.environ.get("BUILD_MODE", "dev").lower()
try:
    file_path = "./auto_round_kernel/version.py"
    with open(file_path) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, f"Failed to read version from {file_path}: {error}"


def get_build_version():
    if os.path.exists("PKG-INFO"):
        with open("PKG-INFO", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()

    if build_mode == "release":
        return __version__
    try:
        result = subprocess.run(["git", "describe", "--tags"], capture_output=True, text=True, check=True)
        distance = result.stdout.strip().split("-")[-2]
        commit = result.stdout.strip().split("-")[-1]
        return f"{__version__}.dev{distance}+{commit}"
    except subprocess.CalledProcessError:
        return __version__


def fetch_requirements(path):
    requirements = []
    with open(path, "r") as fd:
        requirements = [r.strip() for r in fd]
    return requirements


def parse_major_minor(version_str):
    major, minor = version_str.split(".")[:2]
    return int(major), int(minor)


def detect_oneapi_version():
    """
    Auto-detect the sourced oneAPI version using the icx compiler or environment variables.
    Returns a string like '2025.3' or None if detection fails.
    """
    try:
        result = subprocess.run(
            ["icx", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        match = re.search(r"Compiler\s+(\d{4}\.\d+)", result.stdout)
        if match:
            return match.group(1)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    cmplr_root = os.environ.get("CMPLR_ROOT", "")
    match = re.search(r"compiler[/\\](\d{4}\.\d+)", cmplr_root, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


oneapi_version = detect_oneapi_version()
print(f"oneapi_version detected: {oneapi_version}")

if not oneapi_version:
    raise RuntimeError(
        "Failed to auto-detect oneAPI version. "
        "Please ensure you have sourced the oneAPI environment (e.g., source /opt/intel/oneapi/setvars.sh) "
        "and that the 'icx' compiler is in your PATH."
    )

requirements = fetch_requirements("requirements.txt")
enable_sycl_tla = parse_major_minor(oneapi_version) >= (2025, 3)


def get_system_memory_gb():
    if hasattr(os, "sysconf"):
        page_size_names = ("SC_PAGE_SIZE", "SC_PAGESIZE")
        page_size = None
        for name in page_size_names:
            if name in os.sysconf_names:
                page_size = os.sysconf(name)
                break
        if page_size is not None and "SC_PHYS_PAGES" in os.sysconf_names:
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            if phys_pages > 0:
                return (page_size * phys_pages) / (1024**3)

    if sys.platform == "win32":

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        memory_status = MEMORYSTATUSEX()
        memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
            return memory_status.ullTotalPhys / (1024**3)

    return 64


def get_sycl_tla_job_count(cpu_job_count):
    memory_gb = get_system_memory_gb()
    memory_based_jobs = max(
        1, int(memory_gb // 16)
    )  # about 5GB/job for SYCL TLA build, use at most 5/16 of total memory to avoid OOM
    return min(cpu_job_count, memory_based_jobs)


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "auto_round_kernel"
BUILD_DIR = ROOT / "build"
XBUILD_DIR = ROOT / "xbuild"


class CMakeBuild(build_ext):
    def run(self):
        cmake_cmd = [
            "cmake",
            "-S",
            str(SRC_DIR),
            "-B",
            str(BUILD_DIR),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        if sys.platform == "win32":
            cmake_cmd.append("-GNinja")
        subprocess.check_call(cmake_cmd)

        n_job = os.cpu_count() or 2
        n_job = n_job // 2
        subprocess.check_call(["cmake", "--build", str(BUILD_DIR), "-j", str(n_job)])

        ext = "pyd" if sys.platform == "win32" else "so"

        so_files = list(BUILD_DIR.rglob(f"auto_round_kernel*.{ext}"))
        if not so_files:
            raise RuntimeError(f"Can't find auto_round_kernel*.{ext} in '{BUILD_DIR}'")

        target = Path(self.build_lib) / "auto_round_kernel"
        target.mkdir(parents=True, exist_ok=True)
        for so in so_files:
            self.copy_file(str(so), str(target / so.name))

        cmake_cmd = [
            "cmake",
            "-S",
            str(SRC_DIR),
            "-B",
            str(XBUILD_DIR),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_COMPILER=icx",
            "-DARK_XPU=ON",
            f"-DARK_SYCL_TLA={'ON' if enable_sycl_tla else 'OFF'}",
        ]
        if sys.platform == "win32":
            cmake_cmd.append("-GNinja")
        xpu_n_job = get_sycl_tla_job_count(n_job) if enable_sycl_tla else n_job
        subprocess.check_call(cmake_cmd)
        subprocess.check_call(["cmake", "--build", str(XBUILD_DIR), "-j", str(xpu_n_job)])

        so_files = list(XBUILD_DIR.rglob(f"auto_round_kernel*.{ext}"))
        if not so_files:
            raise RuntimeError(f"Can't find auto_round_kernel*.{ext} in '{XBUILD_DIR}'")

        for so in so_files:
            self.copy_file(str(so), str(target / so.name))


class BuildPyThenCMake(build_py):
    def run(self):
        self.run_command("build_ext")
        super().run()


ext_modules = [Extension("auto_round_kernel.auto_round_kernel", sources=[])]

setup(
    name="auto-round-lib",
    version=get_build_version(),
    description="Auto Round Kernel binary package",
    author_email="yu.luo@intel.com",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="quantization,auto-around,LLM,kernel",
    license="Apache 2.0",
    url="https://github.com/intel/auto-round/auto_round_extension/ark",
    packages=["auto_round_kernel"],
    include_package_data=False,
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": CMakeBuild,
        "build_py": BuildPyThenCMake,
    },
    install_requires=requirements,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
    ],
)
