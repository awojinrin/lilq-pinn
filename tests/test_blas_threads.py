"""Tests for lilq.blas_threads -- the BLAS thread-count env-var
configuration implementing Computational_Package_1_v2.md Section 2's
"set (not just record) OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS"
requirement.

Import-order-dependent behavior (the whole point of this module) can't be
observed within this test process, since something will already have
imported it by the time any test runs -- verified via fresh subprocesses
instead, each with a controlled starting environment.
"""

import os
import subprocess
import sys

from lilq import blas_threads

_CHECK_CODE = (
    "import os\n"
    "import lilq.blas_threads\n"
    "for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):\n"
    "    print(v, '=', os.environ.get(v))\n"
)


def _run_fresh(env_overrides: dict) -> dict:
    """Run _CHECK_CODE in a subprocess with a controlled environment
    (inheriting the current one, plus/minus the given overrides), and
    parse the printed var=value lines into a dict."""
    env = os.environ.copy()
    for k, v in env_overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v

    result = subprocess.run(
        [sys.executable, "-c", _CHECK_CODE],
        capture_output=True, text=True, timeout=60, env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0, result.stderr
    parsed = {}
    for line in result.stdout.strip().splitlines():
        key, _, value = line.partition(" = ")
        parsed[key] = value
    return parsed


def test_import_sets_all_three_vars_when_unset():
    result = _run_fresh({
        "OMP_NUM_THREADS": None, "OPENBLAS_NUM_THREADS": None,
        "MKL_NUM_THREADS": None, "SLURM_CPUS_PER_TASK": None,
    })
    expected = str(os.cpu_count())
    assert result["OMP_NUM_THREADS"] == expected
    assert result["OPENBLAS_NUM_THREADS"] == expected
    assert result["MKL_NUM_THREADS"] == expected


def test_import_respects_slurm_cpus_per_task():
    result = _run_fresh({
        "OMP_NUM_THREADS": None, "OPENBLAS_NUM_THREADS": None,
        "MKL_NUM_THREADS": None, "SLURM_CPUS_PER_TASK": "8",
    })
    assert result["OMP_NUM_THREADS"] == "8"
    assert result["OPENBLAS_NUM_THREADS"] == "8"
    assert result["MKL_NUM_THREADS"] == "8"


def test_import_never_overrides_a_preexisting_explicit_value():
    result = _run_fresh({
        "OMP_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": None,
        "MKL_NUM_THREADS": None, "SLURM_CPUS_PER_TASK": "8",
    })
    # The caller's own OMP_NUM_THREADS=3 must win over both the SLURM
    # value and the module's own default -- only the two unset vars get
    # SLURM's value.
    assert result["OMP_NUM_THREADS"] == "3"
    assert result["OPENBLAS_NUM_THREADS"] == "8"
    assert result["MKL_NUM_THREADS"] == "8"


def test_configure_return_value_reflects_effective_setting(monkeypatch):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)

    effective = blas_threads.configure(n_threads=4)

    assert effective == 4
    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4"
    assert os.environ["MKL_NUM_THREADS"] == "4"
