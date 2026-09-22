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


def capture_git_info(repo_root: Optional[Path] = None) -> dict:
    """Commit hash, branch, and any uncommitted diff.

    Defaults to the repository containing this file. Returns
    ``{"available": False}`` if git isn't on PATH or this isn't a git
    checkout (e.g. a bare source copy) -- callers should treat that as
    "provenance unknown", not an error.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if commit is None:
        return {"available": False}

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    diff = _run(["git", "diff", "HEAD"], cwd=repo_root)
    return {
        "available": True,
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
    return {
        "model": _cpu_model_name(),
        "machine": platform.machine(),
        "logical_cores": os.cpu_count(),
        "system": f"{platform.system()} {platform.release()}",
    }


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
        "blas_thread_env": capture_blas_thread_env(),
        "packages": capture_package_versions(),
        "git": capture_git_info(repo_root),
    }


def capture_environment_text() -> str:
    """Verbatim ``numpy.show_config()`` and ``scipy.show_config()`` output
    -- the BLAS/LAPACK backend detail the package spec asks for, which
    doesn't reduce cleanly to JSON.
    """
    parts = []
    for label, module in (("numpy.show_config()", np), ("scipy.show_config()", scipy)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            module.show_config()
        parts.append(f"{'=' * 20} {label} {'=' * 20}\n{buf.getvalue()}")
    return "\n\n".join(parts)


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
