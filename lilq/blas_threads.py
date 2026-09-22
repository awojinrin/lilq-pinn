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


def _default_thread_count() -> int:
    """SLURM's own allocation if running under sbatch/salloc (correctly
    matches --cpus-per-task rather than the whole node's core count);
    otherwise every logical core on the machine.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        return int(slurm_cpus)
    return os.cpu_count() or 1


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


configure()
