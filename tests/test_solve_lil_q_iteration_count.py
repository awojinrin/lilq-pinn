"""Regression guard for the LiL-Q outer-iteration count.

Context (Codebase_v3_Proposal.md S1.1): the pre-GitHub Buckley-Leverett
implementation reported ``total_iterations`` as ``len(metrics.data[...])``,
which included a pre-loop "initial state" snapshot recorded before any solve
ran -- every reported count was one higher than the number of quasilinear
solves actually performed. The shared ``lilq.solvers.solve_lil_q`` introduced
during the June 2026 consolidation counts solves directly (``n_quasi_iters =
quasi_iter + 1``, set inside the loop) and does not have this bug -- verified
by reading the source, and locked in here so a future refactor can't
reintroduce it silently the way the original bug went unnoticed.
"""

import numpy as np

from lilq.solvers import solve_lil_q


def _make_trivial_problem(n_coefs: int = 4):
    """A deterministic, arbitrarily-cheap linearized system for exercising
    the outer quasilinear loop without any real physics -- solve_lil_q only
    needs ``assemble_system_fn`` and ``compute_nonlinear_loss_fn`` to have
    the right call signature.
    """
    rng = np.random.default_rng(0)
    A_fixed = rng.standard_normal((2 * n_coefs, n_coefs))
    b_fixed = rng.standard_normal(2 * n_coefs)

    def assemble_system_fn(beta):
        # Same well-posed linear system every call: the outer loop's
        # iteration count is what's under test, not convergence dynamics.
        return A_fixed, b_fixed

    return assemble_system_fn


def test_iteration_count_equals_max_when_never_converging():
    """With convergence unreachable, total_iterations must equal
    max_quasi_iters exactly -- not max_quasi_iters + 1."""
    assemble_system_fn = _make_trivial_problem()

    def compute_nonlinear_loss_fn(beta):
        # Never below R_tol, so the loop always runs to max_quasi_iters.
        return 1.0, 1.0, 0.0, 0.0

    init_coeffs = np.zeros(4)
    max_iters = 5

    _, _, summary = solve_lil_q(
        assemble_system_fn,
        compute_nonlinear_loss_fn,
        init_coeffs,
        max_quasi_iters=max_iters,
        R_tol=1e-4,
        verbose=False,
    )

    assert summary["total_iterations"] == max_iters
    assert summary["converged"] is False


def test_iteration_count_equals_one_on_immediate_convergence():
    """If the very first solve already satisfies R_tol, exactly one solve
    was performed -- the pre-loop initial-state record must not be counted
    as an iteration."""
    assemble_system_fn = _make_trivial_problem()
    call_count = {"n": 0}

    def compute_nonlinear_loss_fn(beta):
        call_count["n"] += 1
        # First call is the pre-loop initial-state check (loss above R_tol);
        # every call after the first solve reports convergence.
        loss = 1.0 if call_count["n"] == 1 else 1e-6
        return loss, loss, 0.0, 0.0

    init_coeffs = np.zeros(4)

    _, _, summary = solve_lil_q(
        assemble_system_fn,
        compute_nonlinear_loss_fn,
        init_coeffs,
        max_quasi_iters=100,
        R_tol=1e-4,
        verbose=False,
    )

    assert summary["total_iterations"] == 1
    assert summary["converged"] is True


import pytest


@pytest.mark.parametrize("k", [2, 3, 7])
def test_run_that_stops_after_k_solves_reports_k(k):
    """Addendum v2.1 fix 3.1: a run that stops after k solves reports k."""
    solves = {"n": 0}
    A_fixed, b_fixed = np.eye(4), np.ones(4)

    def assemble_system_fn(beta):
        solves["n"] += 1
        return A_fixed, b_fixed

    def compute_nonlinear_loss_fn(beta):
        # Above R_tol until the k-th solve has been performed.
        loss = 1e-6 if solves["n"] >= k else 1.0
        return loss, loss, 0.0, 0.0

    _, _, summary = solve_lil_q(assemble_system_fn, compute_nonlinear_loss_fn, np.zeros(4),
                                max_quasi_iters=50, R_tol=1e-4, verbose=False)
    assert solves["n"] == k
    assert summary["total_iterations"] == k
    assert summary["converged"] is True


def test_buckley_leverett_lil_q_reports_the_solves_it_performs(monkeypatch):
    """Fix 3.1 on the real problem: count the outer least-squares solves
    independently (the pretraining fit's lstsq call passes no ``cond`` and
    is not counted) and compare with what the run reports."""
    import scipy.linalg
    from problems.buckley_leverett import BLConfig, BLOptConfig, run_lil_q

    real_lstsq, solves = scipy.linalg.lstsq, {"n": 0}

    def counting_lstsq(*args, **kwargs):
        if "cond" in kwargs:
            solves["n"] += 1
        return real_lstsq(*args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "lstsq", counting_lstsq)
    summary = run_lil_q(BLConfig(N_x=8, N_t=8), BLOptConfig(R_tol=8.5e-2), verbose=False)[-1]
    assert summary["converged"] is True
    assert 1 < solves["n"] < BLOptConfig().max_quasi_iters_lil
    assert summary["total_iterations"] == solves["n"]
