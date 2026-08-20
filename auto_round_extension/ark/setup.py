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
    print(f"Getting build version for build mode: {build_mode}")
    if build_mode == "release":
        return __version__
    if os.path.exists("PKG-INFO"):
        with open("PKG-INFO", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Version:"):
                    version = line.split(":", 1)[1].strip()
                    print(f"Read version from PKG-INFO: {version}")
                    return version
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cd", "--date=format:%Y%m%d%H%M"], capture_output=True, text=True, check=True
        )
        date_str = result.stdout.strip()
        version = f"{__version__}.dev{date_str}"
        print(f"Git commit date: {date_str}, version: {version}")
        return version
    except subprocess.CalledProcessError as e:
        print(f"Warning: git command failed: {e}. Falling back to version {__version__}")
        return __version__


def fetch_requirements(path):
    requirements = []
    with open(path, "r") as fd:
        for r in fd:
            line = r.strip()
            if line and not line.startswith("-"):
                requirements.append(line)
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


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "on", "true", "yes")


requirements = fetch_requirements("requirements.txt")
enable_sycl_tla = parse_major_minor(oneapi_version) >= (2025, 3)
enable_dnnl = env_flag("ARK_DNNL", default=not enable_sycl_tla)
enable_joint_matrix = env_flag("ARK_JOINT_MATRIX", default=False)


def detect_sycl_target():
    def _read_command_output(cmd):
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
            return result.stdout or "", None
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            return "", str(error)

    def _map_detected_device_to_sycl_target(device_text):
        text = device_text.lower()

        if "b70" in text or "0xe223" in text or "g31" in text:
            return "intel_gpu_bmg_g31"

        if "b60" in text or "b580" in text or "g21" in text:
            return "intel_gpu_bmg_g21"

        if "pvc" in text or "max 1550" in text or "max 1100" in text:
            return "intel_gpu_pvc"

        return None

    override = os.environ.get("ARK_SYCL_TARGET")
    if override:
        print(f"Using ARK_SYCL_TARGET override: {override}")
        return override

    detection_failures = []

    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            props = torch.xpu.get_device_properties(0)
            detected = _map_detected_device_to_sycl_target(f"{props.name} {hex(props.device_id)}")
            if detected:
                print(f"Detected SYCL target from torch.xpu: {detected}")
                return detected
            detection_failures.append("torch.xpu returned an unrecognized XPU device")
        else:
            detection_failures.append("torch.xpu is unavailable")
    except Exception as error:
        detection_failures.append(f"torch.xpu detection failed: {error}")

    for cmd in (["xpu-smi", "discovery", "-d", "0"], ["sycl-ls"]):
        output, error = _read_command_output(cmd)
        detected = _map_detected_device_to_sycl_target(output)
        if detected:
            print(f"Detected SYCL target from {' '.join(cmd)}: {detected}")
            return detected

        if error:
            detection_failures.append(f"{' '.join(cmd)} failed: {error}")
        elif output:
            detection_failures.append(f"{' '.join(cmd)} returned an unrecognized device")

    print("Warning: unable to determine exact SYCL target.")
    for failure in detection_failures:
        print(f"Warning: {failure}")
    print("Warning: falling back to bmg")

    return "bmg"


sycl_target = detect_sycl_target() if enable_sycl_tla else None


def get_system_memory_gb():
    cgroup_memory_paths = (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    for memory_path in cgroup_memory_paths:
        try:
            memory_limit = Path(memory_path).read_text().strip()
        except OSError:
            continue
        if memory_limit == "max":
            continue
        try:
            memory_bytes = int(memory_limit)
        except ValueError:
            continue
        if 0 < memory_bytes < (1 << 60):
            memory_gb = memory_bytes / (1024**3)
            print(f"System memory from {memory_path}: {memory_gb:.2f} GiB ({memory_bytes} bytes)")
            return memory_gb

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
                memory_gb = (page_size * phys_pages) / (1024**3)
                print(f"System memory from sysconf: {memory_gb:.2f} GiB")
                return memory_gb

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
            memory_gb = memory_status.ullTotalPhys / (1024**3)
            print(f"System memory from Windows: {memory_gb:.2f} GiB")
            return memory_gb

    print("System memory detection failed; using fallback: 64 GiB")
    return 64


def get_cpu_count():
    if hasattr(os, "sched_getaffinity"):
        affinity = os.sched_getaffinity(0)
        cpu_count = max(1, len(affinity))
        print(f"CPU count from sched_getaffinity(0): {cpu_count}; allowed CPUs: {sorted(affinity)}")
        return cpu_count
    cpu_count = os.cpu_count() or 1
    print(f"CPU count from os.cpu_count(): {cpu_count}")
    return cpu_count


def get_sycl_tla_job_count(cpu_job_count):
    override = os.environ.get("ARK_SYCL_TLA_JOBS")
    if override is not None:
        try:
            jobs = int(override)
        except ValueError as error:
            raise ValueError("ARK_SYCL_TLA_JOBS must be a positive integer") from error
        if jobs < 1:
            raise ValueError("ARK_SYCL_TLA_JOBS must be a positive integer")
        final_jobs = min(cpu_job_count, jobs)
        print(f"SYCL TLA jobs: min(cpu_job_count={cpu_job_count}, override={jobs}) = {final_jobs}")
        return final_jobs

    memory_gb = get_system_memory_gb()
    memory_based_jobs = max(1, int(memory_gb // 3))  # reserve about 3GB per SYCL TLA compiler job
    final_jobs = min(cpu_job_count, memory_based_jobs)
    print(
        f"SYCL TLA jobs: memory={memory_gb:.2f} GiB, memory_based_jobs={memory_based_jobs}, "
        f"cpu_job_count={cpu_job_count}, final={final_jobs}"
    )
    return final_jobs


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

        cpu_count = get_cpu_count()
        n_job = max(1, cpu_count // 2)
        print(f"CPU build jobs: max(1, {cpu_count} // 2) = {n_job}")
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
            "-DARK_RESCALE=ON",
            f"-DARK_DNNL={'ON' if enable_dnnl else 'OFF'}",
            f"-DARK_JOINT_MATRIX={'ON' if enable_joint_matrix else 'OFF'}",
            f"-DARK_SYCL_TLA={'ON' if enable_sycl_tla else 'OFF'}",
        ]
        if sycl_target:
            cmake_cmd.append(f"-DDPCPP_SYCL_TARGET={sycl_target}")
        if sys.platform == "win32":
            cmake_cmd.append("-GNinja")
        xpu_n_job = get_sycl_tla_job_count(n_job) if enable_sycl_tla else n_job
        print(f"Building XPU extension with {xpu_n_job} parallel job(s)")
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
