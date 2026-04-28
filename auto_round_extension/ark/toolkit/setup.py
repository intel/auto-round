# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import ctypes
import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

build_mapping = {
    "2025.3": {
        "deps": ["torch>=2.10.0", "dpcpp-cpp-rt~=2025.3.0", "onednn~=2025.3.0; sys_platform=='linux'"],
        "default_build_version": "0.10.3.2",
    },
    "2025.2": {
        "deps": ["torch~=2.9.0", "dpcpp-cpp-rt~=2025.2.0", "onednn~=2025.2.0; sys_platform=='linux'"],
        "default_build_version": "0.10.2.2",
    },
    "2025.1": {
        "deps": ["torch~=2.8.0", "dpcpp-cpp-rt~=2025.1.0", "onednn~=2025.1.0; sys_platform=='linux'"],
        "default_build_version": "0.10.1.2",
    },
}


def parse_major_minor(version_str):
    major, minor = version_str.split(".")[:2]
    return int(major), int(minor)


oneapi_version = os.environ.get("ONEAPI_VERSION")
if oneapi_version:
    oneapi_version = ".".join(oneapi_version.split(".")[:2])
else:
    raise RuntimeError(
        "Please set ONEAPI_VERSION environment variable to match your sourced oneAPI version, e.g., 2025.3"
    )

build_config = build_mapping.get(oneapi_version)
if build_config is None:
    raise RuntimeError(f"Unsupported ONEAPI_VERSION: {oneapi_version}")

requirements = build_config["deps"]
version = os.environ.get("RELEASE_VERSION") or build_config["default_build_version"]
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


class CMakeBuild(build_ext):
    def run(self):
        # Step 1: cmake configure and build for default settings
        cmake_cmd = ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"]
        if sys.platform == "win32":
            cmake_cmd.append("-GNinja")
        subprocess.check_call(cmake_cmd)
        n_job = os.cpu_count() or 2
        n_job = n_job // 2  # use half of available cores for the build to avoid OOM on CI machines
        subprocess.check_call(["cmake", "--build", "build", "-j", str(n_job)])

        if sys.platform == "win32":
            ext = "pyd"
        else:
            ext = "so"

        # Step 2: copy .so files from the first build
        so_files = list(Path("build").rglob(f"auto_round_kernel*.{ext}"))
        if not so_files:
            raise RuntimeError("Can't find auto_round_kernel*.so in 'build', please check cmake outputs！")

        target = Path(self.build_lib) / "auto_round_kernel"
        target.mkdir(parents=True, exist_ok=True)
        for so in so_files:
            print(f"Copying {so} → {target}")
            self.copy_file(str(so), str(target / so.name))

        cmake_cmd = [
            "cmake",
            "-B",
            "xbuild",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_COMPILER=icx",
            "-DARK_XPU=ON",
            f"-DARK_SYCL_TLA={'ON' if enable_sycl_tla else 'OFF'}",
        ]
        if sys.platform == "win32":
            cmake_cmd.append("-GNinja")
        # Step 3: cmake configure and build for XPU settings
        xpu_n_job = get_sycl_tla_job_count(n_job) if enable_sycl_tla else n_job
        subprocess.check_call(cmake_cmd)
        subprocess.check_call(["cmake", "--build", "xbuild", "-j", str(xpu_n_job)])

        # Step 4: copy .so files from the second build
        so_files = list(Path("xbuild").rglob(f"auto_round_kernel*.{ext}"))
        if not so_files:
            raise RuntimeError("Can't find auto_round_kernel*.so in 'xbuild', please check cmake outputs！")

        for so in so_files:
            print(f"Copying {so} → {target}")
            self.copy_file(str(so), str(target / so.name))


class BuildPyThenCMake(build_py):
    def run(self):
        self.run_command("build_ext")
        super().run()


ext_modules = [Extension("auto_round_kernel.auto_round_kernel", sources=[])]

setup(
    name="auto-round-lib",
    version=version,
    description="Auto Round Kernel binary package",
    author_email="yu.luo@intel.com",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="quantization,auto-around,LLM,kernel",
    license="Apache 2.0",
    url="https://github.com/intel/auto-round",
    packages=find_packages(include=["auto_round_kernel", "auto_round_kernel.*"]),
    include_package_data=True,
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
