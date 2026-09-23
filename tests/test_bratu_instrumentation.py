"""End-to-end integration test: Bratu's ``run_lil_q`` wired to the shared
Section 3.1 instrumentation (``lilq.iteration_log``) -- sub-batch 3's
"proof of concept" checkpoint. Unlike ``tests/test_solve_lil_q_instrumentation.py``
(synthetic 1-coefficient Newton problem exercising ``solve_lil_q`` directly),
this test runs a real, small Bratu problem through ``problems.bratu.run_lil_q``
end-to-end: basis, pretraining, collocation, the actual weighted least-squares
system -- and checks the produced ``iterations.csv`` rows against an
independent recomputation of the nonlinear residual (check B2).
"""

import math

import numpy as np
import pytest

from lilq.iteration_log import IterationLogger, last_solve_row, solve_rows
from lilq.basis import create_basis_2d
from lilq.collocation import generate_collocation_points_2d
from problems.bratu import BratuConfig, BratuOptConfig, run_lil_q


def _small_config():
    # Small basis (P = N_x*N_y = 9) with enough oversampling (k_ratio=5)
    # to keep the PDE block genuinely overdetermined -- real least-squares
    # residual every iteration, not an exactly-solvable degenerate system.
    # Same scale as experiments/run_all_dry.py's smoke-test config.
    config = BratuConfig(N_x=3, N_y=3, k_ratio=5)
    opt = BratuOptConfig()
    return config, opt


def test_run_lil_q_with_logger_is_bit_identical_to_without():
    """Adding iteration_logger must not change the actual solve -- same
    backward-compatibility discipline as solve_lil_q itself."""
    config, opt = _small_config()

    basis_plain, coeffs_plain, _metrics_plain, summary_plain = run_lil_q(
        config, opt, verbose=False,
    )

    logger = IterationLogger()
    basis_logged, coeffs_logged, _metrics_logged, summary_logged = run_lil_q(
        config, opt, verbose=False, iteration_logger=logger,
    )

    assert np.array_equal(coeffs_plain, coeffs_logged)
    assert summary_plain["total_iterations"] == summary_logged["total_iterations"]
    assert summary_plain["final_loss"] == summary_logged["final_loss"]
    assert len(solve_rows(logger.rows)) == summary_logged["total_iterations"]


def test_run_lil_q_produces_full_iterations_csv(tmp_path):
    config, opt = _small_config()
    logger = IterationLogger()

    basis, coefficients, metrics, summary = run_lil_q(
        config, opt, verbose=False, iteration_logger=logger,
    )

    rows = logger.rows
    assert len(solve_rows(rows)) == summary["total_iterations"]
    assert [row["k"] for row in rows] == list(range(len(rows)))

    # n_interior_rows/interior_weight were supplied -- interior norms must
    # be real numbers, not the NaN "not configured" default.
    assert all(not math.isnan(row["norm_Rlin_interior"]) for row in solve_rows(rows))
    assert all(not math.isnan(row["norm_R_interior"]) for row in solve_rows(rows))

    # compute_residual_vector_fn was supplied -- chi should be a real
    # number at least once the system has a genuine post-solve residual
    # (this problem's PDE block is overdetermined by construction, so
    # R_lin_k is not exactly zero the way the 1x1 Newton synthetic
    # problem in test_solve_lil_q_instrumentation.py is).
    assert any(not math.isnan(row["chi"]) for row in solve_rows(rows))

    # P = 9 <= DEFAULT_SVD_CONDITIONING_THRESHOLD -- every row uses the
    # SVD conditioning path (no pivoted-QR fallback needed at this size).
    assert all(row["kappa_method"] == "svd" for row in solve_rows(rows))
    assert all(row["num_rank_svd"] == 9 for row in solve_rows(rows))

    out_path = tmp_path / "iterations.csv"
    logger.to_csv(out_path)
    import csv
    with open(out_path, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(rows)


def test_check_b2_residual_identity_against_direct_evaluation():
    """Check B2 (Computational_Package_1_v2.md Section 3.1): the logged
    ``norm_R_h`` at the final iterate must match the nonlinear residual
    evaluated directly at the collocation points, independent of the
    logger/tracker/solver's own internal computation of that quantity.

    This reconstructs A_u/A_uxx/A_uyy/A_bc from scratch using only the
    public basis/collocation API (create_basis_2d,
    generate_collocation_points_2d, basis.evaluate/derivative) -- the same
    calls run_lil_q itself makes, with the same config/seed, so it lands
    on the same collocation points -- but does not import or reuse any of
    problems.bratu's private system/loss/residual helper functions.
    """
    config, opt = _small_config()
    logger = IterationLogger()

    basis, coefficients, metrics, summary = run_lil_q(
        config, opt, verbose=False, iteration_logger=logger,
    )

    # Independent reconstruction of the collocation points and basis
    # matrices, matching run_lil_q's own setup exactly (same public calls,
    # same config/seed -> deterministic, same points).
    basis_check = create_basis_2d(
        config.basis_type, config.N_x, config.N_y,
        config.x_domain, config.y_domain,
    )
    pts = generate_collocation_points_2d(
        config.x_domain, config.y_domain, config.N_x, config.N_y,
        k_ratio=config.k_ratio, collocation_ratios=(0.85, 0.15),
        has_initial_condition=False, seed=config.seed,
        sampling=config.sampling,
    )
    x_pde, y_pde = pts['x_pde'], pts['y_pde']
    A_u = basis_check.evaluate(x_pde, y_pde)
    A_uxx = basis_check.derivative(x_pde, y_pde, dx=2, dy=0)
    A_uyy = basis_check.derivative(x_pde, y_pde, dx=0, dy=2)

    x_bc = np.concatenate([pts['x_bc_left'], pts['x_bc_right'],
                            pts['x_bc_bottom'], pts['x_bc_top']])
    y_bc = np.concatenate([pts['y_bc_left'], pts['y_bc_right'],
                            pts['y_bc_bottom'], pts['y_bc_top']])
    A_bc = basis_check.evaluate(x_bc, y_bc)

    n_pde = pts['n_pde']
    n_bc = len(x_bc)

    # Direct nonlinear operator evaluation at the final coefficients.
    u = A_u @ coefficients
    pde_res = (A_uxx @ coefficients) + (A_uyy @ coefficients) + config.lambda_ * np.exp(u)
    bc_res = A_bc @ coefficients

    w_pde = np.sqrt(opt.lambda_pde / n_pde)
    w_bc = np.sqrt(opt.lambda_bc / n_bc)
    R = np.concatenate([w_pde * pde_res, w_bc * bc_res])
    norm_R_direct = float(np.linalg.norm(R))

    norm_R_logged = logger.rows[-1]["norm_R_h"]

    rel_err = abs(norm_R_direct - norm_R_logged) / (abs(norm_R_direct) + 1e-30)
    assert rel_err < 1e-10

    # And the final logged norm must match the coefficients solve_lil_q
    # actually returned (the tracker's last row and the returned
    # coefficients must describe the same solve state).
    assert norm_R_logged == pytest.approx(float(np.sqrt(summary["final_loss"])), rel=1e-10)


def test_run_json_requires_iteration_logger():
    config, opt = _small_config()
    with pytest.raises(ValueError):
        run_lil_q(config, opt, verbose=False, run_json_path="unused.json")


def test_run_json_written_and_self_consistent(tmp_path):
    config, opt = _small_config()
    logger = IterationLogger()
    out_path = tmp_path / "run.json"

    basis, coefficients, metrics, summary = run_lil_q(
        config, opt, verbose=False, iteration_logger=logger, run_json_path=out_path,
    )

    import json
    with open(out_path) as f:
        meta = json.load(f)

    assert meta["N_total"] == sum(meta["N_composition"].values())
    assert meta["P_total"] == len(coefficients)
    assert meta["K_max"] == opt.max_quasi_iters_lil
    assert meta["solver_driver"] == "gelsy"
    assert meta["device"] == "cpu"
    assert meta["initial_coefficients"] == "zero"
    assert meta["stopping_reason"] in ("target", "iteration_cap")
    from lilq.run_metadata import first_stall_iteration
    assert meta["first_stall_iteration"] == first_stall_iteration(logger.rows)
    # Check B2, measured on every iteration of this real solve.
    assert meta["b2_check"]["rel_err"] < 1e-10
