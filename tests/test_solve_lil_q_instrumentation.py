"""End-to-end integration test: solve_lil_q's full Section 3.1
instrumentation wiring (iteration_logger + compute_residual_vector_fn),
run through the real solver -- not the tracker in isolation
(tests/test_lil_q_diagnostics_tracker.py) or the solver without
instrumentation (tests/test_solve_lil_q_iteration_count.py).

The synthetic problem is Newton's method for sqrt(c), expressed as a
1-coefficient LiL-Q system: quasilinearizing x^2 - c = 0 around x_k gives
2*x_k*(x_{k+1}) = x_k^2 + c, i.e. assemble_system_fn returns
A=[[2*x_k]], b=[x_k^2+c] -- solving that linear system for x_{k+1} is
exactly Newton's update x_{k+1} = (x_k^2+c)/(2*x_k), which has
textbook-known quadratic convergence. That gives this test a real,
independently-verifiable expectation (order_obs -> 2) rather than just
"the solver ran and produced some numbers."
"""

import math

import numpy as np
import pytest

from lilq.iteration_log import IterationLogger
from lilq.solvers import solve_lil_q


def _newton_sqrt_problem(c: float):
    def assemble_system_fn(beta):
        x_k = beta[0]
        A = np.array([[2.0 * x_k]])
        b = np.array([x_k ** 2 + c])
        return A, b

    def compute_nonlinear_loss_fn(beta):
        x = beta[0]
        residual = x ** 2 - c
        total = residual ** 2
        return total, total, 0.0, 0.0

    def compute_residual_vector_fn(beta):
        x = beta[0]
        return np.array([x ** 2 - c])

    return assemble_system_fn, compute_nonlinear_loss_fn, compute_residual_vector_fn


def test_full_instrumentation_end_to_end():
    c = 2.0  # solving for sqrt(2)
    assemble_fn, loss_fn, residual_fn = _newton_sqrt_problem(c)
    logger = IterationLogger()

    coefficients, metrics, summary = solve_lil_q(
        assemble_fn, loss_fn, init_coeffs=np.array([1.0]),
        max_quasi_iters=10, R_tol=1e-14, verbose=False,
        iteration_logger=logger,
        compute_residual_vector_fn=residual_fn,
    )

    # Converged to the real answer -- confirms the instrumentation wiring
    # didn't perturb the actual solve.
    assert coefficients[0] == pytest.approx(math.sqrt(c), abs=1e-6)

    rows = logger.rows
    assert len(rows) == summary["total_iterations"]
    assert len(rows) >= 3  # Newton from x0=1 needs a few steps at this tolerance

    # k values are 1..N in order, matching the outer iteration count.
    assert [row["k"] for row in rows] == list(range(1, len(rows) + 1))

    # norm_R_h must be monotonically decreasing -- Newton's method on a
    # convex scalar problem from a sane starting point doesn't oscillate.
    norms = [row["norm_R_h"] for row in rows]
    assert all(norms[i] > norms[i + 1] for i in range(len(norms) - 1))

    # order_obs is undefined (NaN) for the first row, defined after.
    assert math.isnan(rows[0]["order_obs"])
    for row in rows[2:]:
        assert not math.isnan(row["order_obs"])

    # The actual point of the synthetic problem: observed order should
    # approach 2 (quadratic convergence) as Newton's method settles in --
    # check the last well-defined value, not an early transient one.
    late_orders = [row["order_obs"] for row in rows[2:] if not math.isnan(row["order_obs"])]
    assert late_orders[-1] == pytest.approx(2.0, abs=0.3)

    # chi is NaN throughout for this specific problem, and correctly so:
    # it's a 1-equation-1-unknown system, exactly solvable every
    # iteration, so R_lin_k = A@beta_new - b is exactly zero every time
    # -- phase_indicator's documented degenerate case (zero denominator).
    # A real value for chi needs an overdetermined system with genuine
    # residual after the linear solve, which
    # test_lil_q_diagnostics_tracker.py::test_chi_is_real_with_residual_vector_callback
    # already covers directly; this end-to-end test's job is order_obs
    # (via Newton's known quadratic convergence), not chi.
    assert all(math.isnan(row["chi"]) for row in rows)

    # Small P (=1) -- every row should use the SVD conditioning path.
    assert all(row["kappa_method"] == "svd" for row in rows)
    assert all(row["num_rank_svd"] == 1 for row in rows)

    # rcond/solver_path constants, present on every row.
    assert all(row["solver_path"] == "cpu_gelsy" for row in rows)

    # Timing columns are populated and non-negative.
    assert all(row["t_assemble_s"] >= 0.0 for row in rows)
    assert all(row["t_solve_s"] >= 0.0 for row in rows)
    # t_cum_s must be non-decreasing across iterations.
    cum = [row["t_cum_s"] for row in rows]
    assert all(cum[i] <= cum[i + 1] for i in range(len(cum) - 1))


def test_instrumentation_is_fully_optional_and_backward_compatible():
    """Omitting iteration_logger/compute_residual_vector_fn must give
    numerically identical results to before this sub-batch existed --
    same coefficients, same iteration count, same summary."""
    c = 2.0
    assemble_fn, loss_fn, _residual_fn = _newton_sqrt_problem(c)

    coeffs_plain, _metrics_plain, summary_plain = solve_lil_q(
        assemble_fn, loss_fn, init_coeffs=np.array([1.0]),
        max_quasi_iters=10, R_tol=1e-14, verbose=False,
    )

    logger = IterationLogger()
    coeffs_logged, _metrics_logged, summary_logged = solve_lil_q(
        assemble_fn, loss_fn, init_coeffs=np.array([1.0]),
        max_quasi_iters=10, R_tol=1e-14, verbose=False,
        iteration_logger=logger,  # no compute_residual_vector_fn
    )

    assert coeffs_plain[0] == pytest.approx(coeffs_logged[0], abs=0.0)
    assert summary_plain["total_iterations"] == summary_logged["total_iterations"]
    # chi/stall_flag degrade gracefully without the residual-vector callback.
    assert all(math.isnan(row["chi"]) for row in logger.rows)
    assert all(row["stall_flag"] is False for row in logger.rows)


def test_csv_round_trip_for_the_real_solve(tmp_path):
    c = 2.0
    assemble_fn, loss_fn, residual_fn = _newton_sqrt_problem(c)
    logger = IterationLogger()

    solve_lil_q(
        assemble_fn, loss_fn, init_coeffs=np.array([1.0]),
        max_quasi_iters=10, R_tol=1e-14, verbose=False,
        iteration_logger=logger, compute_residual_vector_fn=residual_fn,
    )

    out_path = tmp_path / "iterations.csv"
    logger.to_csv(out_path)

    import csv
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(logger.rows)
    assert float(rows[-1]["norm_R_h"]) < 1e-6  # converged, per R_tol=1e-14 on the squared residual
