"""
Hardware / environment / git provenance capture
===================================================

Implements Computational_Package_1_v2.md Section 2's "Common Protocol"
requirement: every timed experiment records ``hardware.json`` and
``environment.txt`` so a later reader can determine exactly what
produced a given number without investigative archaeology. Most of the
manuscript-vs-repository discrepancies chased down earlier in this
project (DECISIONS.md, Codebase_v3_Proposal.md S2.5) came down to "which
run, which code version, which machine state produced this number" --
questions that took real work to answer after the fact only because
nothing recorded them at the time. This module exists to retire that
class of question going forward.

Every capture function is defensive: a missing tool (git, nvidia-smi) or
an unreadable field degrades to ``None``/``"available": False`` rather
than raising. Provenance capture must never be the reason a multi-hour
experiment run crashes.
"""

import contextlib
import io
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scipy
import torch


# BLAS-relevant thread-count environment variables. The computational
# package spec requires these to be *set*, not just recorded -- BLAS
# libraries read them at load time, so they must be set in the
# environment before numpy/scipy is first imported (e.g. at the top of an
# experiment script's entry point, or in the submitting shell / SLURM job
# script). This module can only report what's currently in effect; it
# cannot retroactively fix an unset value in a process where numpy has
# already loaded.
_BLAS_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _run(args: list, cwd: Optional[Path] = None, timeout: float = 10.0) -> Optional[str]:
    """Run a subprocess, returning stripped stdout or None on any failure.

    Never raises: a missing executable, non-repo directory, or timeout
    all degrade to None rather than propagating.
    """
    try:
        result = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


BUNDLE_PROVENANCE_FILE = "PROVENANCE.json"


def capture_git_info(repo_root: Optional[Path] = None) -> dict:
    """Commit hash, branch, and any uncommitted diff.

    Defaults to the repository containing this file. Outside a git
    checkout (e.g. the source bundle uploaded to a cluster, which ships
    without ``.git``), falls back to the ``PROVENANCE.json`` that
    ``scripts/make_hprc_bundle.py`` writes into the bundle, marked
    ``"source": "bundle"``. Returns ``{"available": False}`` if neither
    exists -- callers should treat that as "provenance unknown", not an
    error.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if commit is None:
        bundle_file = Path(repo_root) / BUNDLE_PROVENANCE_FILE
        try:
            with open(bundle_file) as f:
                bundled = json.load(f)
        except (OSError, ValueError):
            return {"available": False}
        return {"available": True, "source": "bundle", **bundled}

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    diff = _run(["git", "diff", "HEAD"], cwd=repo_root)
    return {
        "available": True,
        "source": "git",
        "commit": commit,
        "branch": branch,
        "dirty": bool(diff),
        # Full diff can be large; still worth having in the JSON verbatim
        # since a dirty tree is exactly the "which code produced this"
        # ambiguity this module exists to eliminate.
        "diff": diff or None,
    }


def capture_blas_thread_env() -> dict:
    """Currently-set values of the BLAS thread-count environment variables."""
    return {k: os.environ.get(k) for k in _BLAS_THREAD_ENV_VARS}


def _cpu_model_name() -> Optional[str]:
    """Best-effort human-readable CPU model string, cross-platform."""
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    proc = platform.processor()
    return proc or None


def capture_cpu_info() -> dict:
    """``logical_cores`` is the whole machine; ``usable_cores`` is what this
    process may actually run on -- they differ inside a scheduler
    allocation (e.g. 64 vs the job's own cores on a cluster node)."""
    usable = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {
        "model": _cpu_model_name(),
        "machine": platform.machine(),
        "logical_cores": os.cpu_count(),
        "usable_cores": usable,
        "hostname": platform.node(),
        "system": f"{platform.system()} {platform.release()}",
    }


_SLURM_ENV_VARS = (
    "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
    "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_NTASKS",
    "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE",
    "CUDA_VISIBLE_DEVICES", "SLURMD_NODENAME", "SLURM_JOB_ACCOUNT",
)


def _scontrol_fields(kind: str, name: Optional[str], keys) -> Optional[dict]:
    """Selected ``Key=Value`` fields of ``scontrol show <kind> <name>``."""
    if not name:
        return None
    out = _run(["scontrol", "show", kind, str(name)])
    if out is None:
        return None
    fields = {}
    for token in out.split():
        if "=" in token:
            k, v = token.split("=", 1)
            if k in keys:
                fields[k] = v
    return fields


def capture_scheduler_info() -> dict:
    """The SLURM allocation this run executed under (``{"slurm": False}``
    on a workstation) -- what a timing on a shared cluster node actually
    had to work with, which the node's hardware alone doesn't say. Adds the
    node's type (its features, GPUs, cores, memory) and whether the job held
    the node exclusively (``OverSubscribe=NO``), from ``scontrol``."""
    if "SLURM_JOB_ID" not in os.environ:
        return {"slurm": False}
    info = {"slurm": True, **{k: os.environ.get(k) for k in _SLURM_ENV_VARS}}
    info["node"] = _scontrol_fields("node", os.environ.get("SLURMD_NODENAME"),
                                    ("NodeName", "AvailableFeatures", "ActiveFeatures", "Gres",
                                     "CPUTot", "RealMemory", "Partitions"))
    job = _scontrol_fields("job", os.environ.get("SLURM_JOB_ID"),
                           ("OverSubscribe", "Account", "Partition", "TRES", "NumCPUs"))
    info["job"] = job
    info["exclusive"] = (job or {}).get("OverSubscribe") == "NO" if job else None
    return info


def capture_thread_info() -> dict:
    """Threads actually in effect: PyTorch's intra- and inter-op pools
    (BLAS's are in ``blas_thread_env``)."""
    return {"torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads()}


def capture_gpu_info() -> dict:
    """GPU name, driver version, and total memory via nvidia-smi.

    Returns ``{"available": False}`` on a CPU-only machine or any
    non-NVIDIA setup -- nvidia-smi absent/failing is expected there, not
    an error.
    """
    output = _run([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ])
    if output is None:
        return {"available": False}

    gpus = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append({"name": parts[0], "driver_version": parts[1], "memory_total": parts[2]})
    return {
        "available": bool(gpus),
        "gpus": gpus,
        "cuda_available_to_torch": torch.cuda.is_available(),
        "torch_cuda_build_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }


def capture_package_versions() -> dict:
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
    }


def capture_hardware_json(repo_root: Optional[Path] = None) -> dict:
    """Everything Section 2 of the package spec asks for in ``hardware.json``,
    except the BLAS/LAPACK build details (those go verbatim in
    ``environment.txt`` via :func:`capture_environment_text` -- they're
    multi-line structured text, not a clean JSON field).
    """
    return {
        "cpu": capture_cpu_info(),
        "gpu": capture_gpu_info(),
        "scheduler": capture_scheduler_info(),
        "blas_thread_env": capture_blas_thread_env(),
        "threads": capture_thread_info(),
        "packages": capture_package_versions(),
        "git": capture_git_info(repo_root),
    }


def capture_environment_text() -> str:
    """Verbatim ``numpy.show_config()`` and ``scipy.show_config()`` output
    -- the BLAS/LAPACK backend detail the package spec asks for, which
    doesn't reduce cleanly to JSON -- and PyTorch's version and
    ``torch.__config__.show()`` (Addendum v2.2 Section 2.4).
    """
    parts = []
    for label, module in (("numpy.show_config()", np), ("scipy.show_config()", scipy)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            module.show_config()
        parts.append(f"{'=' * 20} {label} {'=' * 20}\n{buf.getvalue()}")
    parts.append(f"{'=' * 20} torch {torch.__version__} (from {torch.__file__}) {'=' * 20}\n"
                 f"{torch.__config__.show()}")
    return "\n\n".join(parts)


# Addendum v2.2 Section 2.4: the L-BFGS evaluation counts and the 16x cap
# (lilq.solvers) assume PyTorch 2.10, whose strong-Wolfe line search is
# bounded by the step's remaining max_eval; 2.9.1's is not (checked on Grace).
REQUIRED_TORCH_VERSION = "2.10.0"


def assert_torch_version(required: str = REQUIRED_TORCH_VERSION) -> str:
    """Raise unless the imported torch is ``required`` (a local build suffix
    such as ``+cu126`` is allowed); returns the version."""
    version = torch.__version__
    if not (version == required or version.startswith(required + "+")):
        raise RuntimeError(f"torch {version} imported from {torch.__file__}; {required} is required "
                           f"(Addendum v2.2 Section 2.4). Is the venv first on PYTHONPATH?")
    return version


def save_provenance(output_dir: Path, repo_root: Optional[Path] = None) -> None:
    """Write ``hardware.json`` and ``environment.txt`` into ``output_dir``.

    Intended to be called once per experiment run, alongside whatever
    other results that run produces -- see any ``experiments/run_*.py``
    script for the call site.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hardware = capture_hardware_json(repo_root)
    with open(output_dir / "hardware.json", "w") as f:
        json.dump(hardware, f, indent=2)

    with open(output_dir / "environment.txt", "w") as f:
        f.write(capture_environment_text())
