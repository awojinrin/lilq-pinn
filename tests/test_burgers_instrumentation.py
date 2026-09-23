"""End-to-end integration test: Burgers' ``run_lil_q`` wired to the shared
Section 3.1 instrumentation (``lilq.iteration_log``) -- sub-batch 4's
rollout of the pattern proven on Bratu in
``tests/test_bratu_instrumentation.py``.
"""

import math

import numpy as np
import pytest

from lilq.iteration_log import IterationLogger
from lilq.basis import create_basis_2d
from lilq.collocation import generate_collocation_points_2d
from problems.burgers import BurgersConfig, BurgersOptConfig, BurgersPhysics, run_lil_q


def _small_config():
    # Small basis (P = N_x*N_t = 9) with enough oversampling (k_ratio=5)
    # to keep the PDE block genuinely overdetermined. Same scale as
    # tests/test_bratu_instrumentation.py's config.
    config = BurgersConfig(N_x=3, N_t=3, k_ratio=5)
    opt = BurgersOptConfig()
    return config, opt


def test_run_lil_q_with_logger_is_bit_identical_to_without():
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
    assert len(logger) == summary_logged["total_iterations"]


def test_run_lil_q_produces_full_iterations_csv(tmp_path):
    config, opt = _small_config()
    logger = IterationLogger()

    basis, coefficients, metrics, summary = run_lil_q(
        config, opt, verbose=False, iteration_logger=logger,
    )

    rows = logger.rows
    assert len(rows) == summary["total_iterations"]
    assert [row["k"] for row in rows] == list(range(1, len(rows) + 1))

    assert all(not math.isnan(row["norm_Rlin_interior"]) for row in rows)
    assert all(not math.isnan(row["norm_R_interior"]) for row in rows)
    assert any(not math.isnan(row["chi"]) for row in rows)

    assert all(row["kappa_method"] == "svd" for row in rows)
    assert all(row["num_rank_svd"] == 9 for row in rows)

    out_path = tmp_path / "iterations.csv"
    logger.to_csv(out_path)
    import csv
    with open(out_path, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(rows)


def test_check_b2_residual_identity_against_direct_evaluation():
    """Check B2: the logged norm_R_h at the final iterate must match the
    nonlinear residual evaluated directly at the collocation points,
    reconstructed from scratch via the public basis/collocation/physics
    API (same config/seed -> same points as run_lil_q's own setup), not
    via any of problems.burgers's private helper functions.
    """
    config, opt = _small_config()
    logger = IterationLogger()

    basis, coefficients, metrics, summary = run_lil_q(
        config, opt, verbose=False, iteration_logger=logger,
    )

    physics = BurgersPhysics(config)
    basis_check = create_basis_2d(
        config.basis_type, config.N_x, config.N_t,
        config.x_domain, (0, config.T_final),
    )
    pts = generate_collocation_points_2d(
        config.x_domain, (0, config.T_final), config.N_x, config.N_t,
        k_ratio=config.k_ratio, collocation_ratios=(0.85, 0.05, 0.10),
        has_initial_condition=True, seed=config.seed,
        sampling=config.sampling,
    )

    x_pde, t_pde = pts['x_pde'], pts['y_pde']
    A_u = basis_check.evaluate(x_pde, t_pde)
    A_ux = basis_check.derivative(x_pde, t_pde, dx=1, dy=0)
    A_ut = basis_check.derivative(x_pde, t_pde, dx=0, dy=1)
    A_uxx = basis_check.derivative(x_pde, t_pde, dx=2, dy=0)

    A_ic = basis_check.evaluate(pts['x_ic'], pts['y_ic'])
    ic_target = physics.initial_condition(pts['x_ic']).astype(np.float64)

    A_bc_left = basis_check.evaluate(pts['x_bc_left'], pts['y_bc_left'])
    A_bc_right = basis_check.evaluate(pts['x_bc_right'], pts['y_bc_right'])

    n_pde, n_ic = pts['n_pde'], pts['n_ic']
    n_bc_l, n_bc_r = len(pts['x_bc_left']), len(pts['x_bc_right'])

    u = A_u @ coefficients
    u_x = A_ux @ coefficients
    u_t = A_ut @ coefficients
    u_xx = A_uxx @ coefficients
    pde_res = u_t + u * u_x - config.viscosity * u_xx
    ic_res = A_ic @ coefficients - ic_target
    bl_res = A_bc_left @ coefficients
    br_res = A_bc_right @ coefficients

    w_pde = np.sqrt(opt.lambda_pde / n_pde)
    w_ic = np.sqrt(opt.lambda_ic / n_ic)
    w_bl = np.sqrt(opt.lambda_bc / n_bc_l)
    w_br = np.sqrt(opt.lambda_bc / n_bc_r)
    R = np.concatenate([w_pde * pde_res, w_ic * ic_res, w_bl * bl_res, w_br * br_res])
    norm_R_direct = float(np.linalg.norm(R))

    norm_R_logged = logger.rows[-1]["norm_R_h"]

    rel_err = abs(norm_R_direct - norm_R_logged) / (abs(norm_R_direct) + 1e-30)
    assert rel_err < 1e-10
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
