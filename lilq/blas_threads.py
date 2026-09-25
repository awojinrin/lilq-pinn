"""
BLAS thread-count configuration
==================================

Computational_Package_1_v2.md Section 2 requires timed runs to explicitly
*set* -- not just record (see ``lilq.provenance``) -- ``OMP_NUM_THREADS``,
``OPENBLAS_NUM_THREADS``, and ``MKL_NUM_THREADS``. An unpinned BLAS
library will happily grab every core it sees, which is a real source of
run-to-run timing noise, especially on a shared HPC node fighting other
users' jobs or your own ``--ntasks`` request (Q5's finding: no
thread-pinning existed anywhere in this codebase, historically -- see
DECISIONS.md).

BLAS reads these environment variables **at library load time**, not
dynamically, so they must be set before numpy/scipy is first imported --
even transitively. This means importing this module must be the very
first import in an experiment script's entry point, before
``import numpy`` and before any ``from lilq...``/``from problems...``
import that would pull numpy in on your behalf. It intentionally has no
other dependency (not even ``lilq.utils``) so it can safely be imported
first.

Usage, at the very top of an experiment script, right after the
``sys.path`` setup and before any other import::

    import lilq.blas_threads  # noqa: F401  (must import before numpy)
    import numpy as np
    ...

Configuring happens automatically as this module's own import-time side
effect (the one place in this codebase where that pattern is actually the
correct choice, not the antipattern fixed elsewhere in DECISIONS.md --
there is no non-environment-variable way to configure a native BLAS
library's thread pool before it loads). It never overrides a value your
environment already set (``os.environ.setdefault``, not assignment) --
your shell, a SLURM job script's own thread pinning, or a batch
scheduler's cgroup limits always win.
"""

import os

_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _usable_cpu_count() -> int:
    """Cores this process may actually run on. On Linux this respects the
    cpuset a scheduler (SLURM cgroups, taskset) confines the job to --
    ``os.cpu_count()`` reports the whole node (64 on a FASTER node)
    regardless of how few cores the job was given."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def _default_thread_count() -> int:
    """Never more threads than cores this job owns. In order:
    ``SLURM_CPUS_PER_TASK`` (set by ``--cpus-per-task``);
    ``SLURM_CPUS_ON_NODE`` (the job's cores on this node, set under
    ``--ntasks``-style requests, where ``SLURM_CPUS_PER_TASK`` is absent);
    otherwise the cores the process is allowed to run on (every logical
    core on an unscheduled workstation).
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(var)
        if value and value.isdigit():
            return int(value)
    return _usable_cpu_count()


def configure(n_threads: int = None) -> int:
    """Set the BLAS thread-count environment variables if not already set.

    Parameters
    ----------
    n_threads : int, optional
        Value to use for any of the three variables not already present
        in the environment. Defaults to :func:`_default_thread_count`.

    Returns
    -------
    int
        The value actually in effect for ``OMP_NUM_THREADS`` after this
        call -- either what was already set, or what this call just set.

    Note: calling this *after* numpy/scipy has already been imported in
    this process has no effect on the already-loaded BLAS library's
    thread pool. There is no supported way to reconfigure that
    post-import; the only fix is importing this module earlier.
    """
    value = str(n_threads if n_threads is not None else _default_thread_count())
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, value)
    return int(os.environ[_THREAD_ENV_VARS[0]])


def pin_torch() -> int:
    """Set PyTorch's intra-op thread count to the same allocation as BLAS
    (``OMP_NUM_THREADS``; Addendum v2.1 Section 2). Imports torch, so it is a
    function, never an import-time effect of this module; ``lilq.utils`` and
    the baseline drivers call it right after importing torch."""
    import torch
    n = int(os.environ.get(_THREAD_ENV_VARS[0]) or configure())
    torch.set_num_threads(n)
    return n


configure()
