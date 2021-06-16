"""Utilities for system operations."""

import math
import os
import re
import subprocess
import sys


def bash_command(cmd: str) -> int:
    process = subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output, errors = process.communicate()
    if output:
        sys.stdout.write(str(output))
    if errors:
        sys.stdout.write(str(errors))
    return process.returncode


def sh_command(cmd: str) -> int:
    return subprocess.call(cmd, shell=True)


# System information


def cpu_count() -> int:
    """Fail-safe method to get cpu count. Also respects docker/cgroup limitations."""
    try:
        import psutil

        cpu_count = psutil.cpu_count()
    except Exception:
        # psutil is probably not installed
        cpu_count = os.cpu_count()

    try:
        # Try to read out docker cpu quota if it exists
        quota_file = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
        if os.path.isfile(quota_file):
            cpu_quota = math.ceil(
                int(os.popen("cat " + quota_file).read().replace("\n", "")) / 100000
            )
            if 0 < cpu_quota < cpu_count:
                cpu_count = cpu_quota
    except Exception:
        # Do nothing
        pass

    return cpu_count


def total_memory():
    """Fail-safe method to get total memory. Also respects docker/cgroup limitations."""
    import psutil

    memory = psutil.virtual_memory().total
    try:
        if os.path.isfile("/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as file:
                mem_limit = file.read().replace("\n", "").strip()
                if mem_limit and 0 < int(mem_limit) < int(memory):
                    # if mem limit from cgroup bigger than total memory -> use total memory
                    memory = int(mem_limit)
    except Exception:
        # Do nothing
        pass

    return memory


def cpu_info():
    import platform

    if platform.system() == "Windows":
        return _get_cpu_by_pycpuinfo()
    try:
        if platform.system() == "Darwin":
            return _get_cpu_by_sysctl()
        elif platform.system() == "Linux":
            return _get_cpu_by_proc_cpuinfo()
    except Exception:
        # Use pycpuinfo only if other ways fail, since it takes about 1 sec
        return _get_cpu_by_pycpuinfo()


def _get_cpu_by_sysctl():
    os.environ["PATH"] += ":/usr/sbin"
    command = ["sysctl", "-n", "machdep.cpu.brand_string"]
    return subprocess.check_output(command).decode().strip()


def _get_cpu_by_proc_cpuinfo():
    command = ["cat", "/proc/cpuinfo"]
    all_info = subprocess.check_output(command).decode()
    model_pattern = re.compile("^\s*model name\s*:")
    for line in all_info.split("\n"):
        if model_pattern.match(line):
            return model_pattern.sub("", line, 1).strip()


def _get_cpu_by_pycpuinfo():
    try:
        import cpuinfo

        return cpuinfo.get_cpu_info()["brand"]
    except Exception:
        # if cpuinfo not installed
        import platform

        return platform.processor()


def set_num_threads(num_threads: int):
    try:
        import mkl

        mkl.set_num_threads(num_threads)
    except Exception:
        pass

    os.environ["OMP_NUM_THREADS"] = str(num_threads)  # OpenMP
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)  # OpenBLAS
    os.environ["MKL_NUM_THREADS"] = str(num_threads)  # MKL
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)  # Accelerate
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)  # Numexpr
    os.environ["NUMEXPR_MAX_THREADS"] = str(num_threads)  # Numexpr - maximum
    os.environ["NUMBA_NUM_THREADS"] = str(num_threads)  # Numba
    os.environ["SPARK_WORKER_CORES"] = str(num_threads)  # Spark Worker
