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
