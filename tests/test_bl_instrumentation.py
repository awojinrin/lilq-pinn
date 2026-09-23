"""End-to-end integration test: Buckley-Leverett's ``run_lil_q`` wired to
the shared Section 3.1 instrumentation (``lilq.iteration_log``) --
sub-batch 4's rollout of the pattern proven on Bratu in
``tests/test_bratu_instrumentation.py``. Covers both BL configurations
(viscous and gravity) since both share the same ``run_lil_q``, gated only
by ``physics.flux``/``flux_derivative`` internally dispatching on
``config.N_g``.
"""

import dataclasses
import math

import numpy as np
import pytest
import torch

from lilq.iteration_log import IterationLogger, last_solve_row, solve_rows
from lilq.basis import create_basis_2d
from lilq.collocation import generate_collocation_points_2d
from problems.buckley_leverett import BLConfig, BLOptConfig, BLPhysics, run_lil_q


def _small_config(gravity: bool):
    base = BLConfig.with_gravity() if gravity else BLConfig()
    # Small basis (P = N_x*N_t = 9) with enough oversampling (k_ratio=5)
    # to keep the PDE block genuinely overdetermined.
    config = dataclasses.replace(base, N_x=3, N_t=3, k_ratio=5)
    opt = BLOptConfig()
    return config, opt


def _run_and_check_b2(config, opt):
    logger = IterationLogger()
    basis, coefficients, metrics, summary = run_lil_q(
        config, opt, verbose=False, iteration_logger=logger,
    )

    rows = logger.rows
    assert len(solve_rows(rows)) == summary["total_iterations"]
    assert [row["k"] for row in rows] == list(range(len(rows)))
    assert all(not math.isnan(row["norm_Rlin_interior"]) for row in solve_rows(rows))
    assert all(not math.isnan(row["norm_R_interior"]) for row in solve_rows(rows))

    physics = BLPhysics(config)
    basis_check = create_basis_2d(
        config.basis_type, config.N_x, config.N_t,
        config.x_domain, (0, config.T_final),
    )
    pts = generate_collocation_points_2d(
        config.x_domain, (0, config.T_final), config.N_x, config.N_t,
        k_ratio=config.k_ratio, collocation_ratios=(0.9, 0.05, 0.05),
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
    bc_l_target = physics.bc_left(pts['y_bc_left']).astype(np.float64)
    bc_r_target = physics.bc_right(pts['y_bc_right']).astype(np.float64)

    n_pde, n_ic = pts['n_pde'], pts['n_ic']
    n_bc_l, n_bc_r = len(pts['x_bc_left']), len(pts['x_bc_right'])

    S = A_u @ coefficients
    S_x = A_ux @ coefficients
    S_t = A_ut @ coefficients
    S_xx = A_uxx @ coefficients

    S_tensor = torch.tensor(S, dtype=torch.float64)
    f_p = physics.flux_derivative(S_tensor).numpy()
    f_x = f_p * S_x

    pde_res = S_t + f_x + physics.D * S_xx
    ic_res = A_ic @ coefficients - ic_target
    bl_res = A_bc_left @ coefficients - bc_l_target
    br_res = A_bc_right @ coefficients - bc_r_target

    w_pde = np.sqrt(opt.lambda_pde / n_pde)
    w_ic = np.sqrt(opt.lambda_ic / n_ic)
    w_bl = np.sqrt(opt.lambda_bc / n_bc_l)
    w_br = np.sqrt(opt.lambda_bc / n_bc_r)
    R = np.concatenate([w_pde * pde_res, w_ic * ic_res, w_bl * bl_res, w_br * br_res])
    norm_R_direct = float(np.linalg.norm(R))

    norm_R_logged = rows[-1]["norm_R_h"]
    rel_err = abs(norm_R_direct - norm_R_logged) / (abs(norm_R_direct) + 1e-30)
    assert rel_err < 1e-10
    assert norm_R_logged == pytest.approx(float(np.sqrt(summary["final_loss"])), rel=1e-10)

    return logger, summary


def test_viscous_with_logger_is_bit_identical_to_without():
    config, opt = _small_config(gravity=False)

    basis_plain, coeffs_plain, _mp, summary_plain = run_lil_q(config, opt, verbose=False)
    logger = IterationLogger()
    basis_logged, coeffs_logged, _ml, summary_logged = run_lil_q(
        config, opt, verbose=False, iteration_logger=logger,
    )

    assert np.array_equal(coeffs_plain, coeffs_logged)
    assert summary_plain["total_iterations"] == summary_logged["total_iterations"]
    assert summary_plain["final_loss"] == summary_logged["final_loss"]


def test_gravity_with_logger_is_bit_identical_to_without():
    config, opt = _small_config(gravity=True)

    basis_plain, coeffs_plain, _mp, summary_plain = run_lil_q(config, opt, verbose=False)
    logger = IterationLogger()
    basis_logged, coeffs_logged, _ml, summary_logged = run_lil_q(
        config, opt, verbose=False, iteration_logger=logger,
    )

    assert np.array_equal(coeffs_plain, coeffs_logged)
    assert summary_plain["total_iterations"] == summary_logged["total_iterations"]
    assert summary_plain["final_loss"] == summary_logged["final_loss"]


def test_check_b2_viscous():
    config, opt = _small_config(gravity=False)
    _run_and_check_b2(config, opt)


def test_check_b2_gravity():
    config, opt = _small_config(gravity=True)
    _run_and_check_b2(config, opt)


def test_csv_round_trip(tmp_path):
    config, opt = _small_config(gravity=False)
    logger, summary = _run_and_check_b2(config, opt)

    out_path = tmp_path / "iterations.csv"
    logger.to_csv(out_path)
    import csv
    with open(out_path, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(logger.rows)
    assert len(csv_rows) == summary["total_iterations"] + 1  # plus the terminal row


def test_run_json_requires_iteration_logger():
    config, opt = _small_config(gravity=False)
    with pytest.raises(ValueError):
        run_lil_q(config, opt, verbose=False, run_json_path="unused.json")


def test_run_json_initial_coefficients_distinguishes_viscous_and_gravity(tmp_path):
    import json
    from lilq.run_metadata import first_stall_iteration

    for gravity, expected_substring in [(False, 'fitted'), (True, 'zero')]:
        config, opt = _small_config(gravity=gravity)
        logger = IterationLogger()
        out_path = tmp_path / f"run_{gravity}.json"

        basis, coefficients, metrics, summary = run_lil_q(
            config, opt, verbose=False, iteration_logger=logger, run_json_path=out_path,
        )

        with open(out_path) as f:
            meta = json.load(f)

        assert meta["N_total"] == sum(meta["N_composition"].values())
        assert meta["P_total"] == len(coefficients)
        assert meta["K_max"] == opt.max_quasi_iters_lil
        assert meta["solver_driver"] == "gelsy"
        assert expected_substring in meta["initial_coefficients"]
        assert meta["first_stall_iteration"] == first_stall_iteration(logger.rows)
        # Check B2, measured on every iteration of this real solve.
        assert meta["b2_check"]["rel_err"] < 1e-10
