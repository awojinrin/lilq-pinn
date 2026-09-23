"""End-to-end integration test: Darcy's ``solve_lilq_darcy`` wired to the
shared Section 3.1 instrumentation (``lilq.iteration_log``) -- the Darcy
half of the Beltrami/Darcy sub-batch following sub-batch 5's Kovasznay
precedent.

Unlike every other problem in this codebase, Darcy's LiL-Q system is
**linear** (no Bellman-Kalaba quasilinearization loop -- see the module
docstring in ``problems/darcy.py``), so this produces exactly one logged
row (k=1) per solve rather than driving an outer iteration loop.
"""

import math

import numpy as np
import pytest

from lilq.iteration_log import IterationLogger
from problems.darcy import (
    DarcyConfig, DarcyPhysics, solve_lilq_darcy,
    _create_basis_h_tilde, _create_basis_u, _create_basis_v,
)


def _small_config(**overrides):
    # NX_CELLS/NY_CELLS are fixed by the permeability data file's shape
    # (60x220) -- only the basis orders are varied for test speed.
    kwargs = dict(ORDER_H=6, ORDER_U=6, ORDER_V=6)
    kwargs.update(overrides)
    return DarcyConfig(**kwargs)


def _independent_residual_norm(config, physics, coeffs):
    """Rebuild A/b from scratch via the module's own basis-construction
    helpers and the public basis evaluate/derivative API (not
    solve_lilq_darcy's own precomputed A_Dx/A_Dy/A_CE objects), and
    evaluate the residual at the given coefficients, for check B2."""
    basis_h_tilde = _create_basis_h_tilde(config.ORDER_H)
    basis_u = _create_basis_u(config.ORDER_U)
    basis_v = _create_basis_v(config.ORDER_V)

    x_c = (np.arange(config.NX_CELLS) + 0.5) / config.NX_CELLS
    y_c = (np.arange(config.NY_CELLS) + 0.5) / config.NY_CELLS
    Xg, Yg = np.meshgrid(x_c, y_c, indexing='ij')
    x_pde, y_pde = Xg.ravel(), Yg.ravel()
    n_pde = len(x_pde)
    sqrt_K = np.sqrt(physics.K_star.ravel())
    R = physics.R

    dh_dx = basis_h_tilde.derivative(x_pde, y_pde, dx=1, dy=0)
    dh_dy = basis_h_tilde.derivative(x_pde, y_pde, dx=0, dy=1)
    Phi_u = basis_u.evaluate(x_pde, y_pde)
    du_dx = basis_u.derivative(x_pde, y_pde, dx=1, dy=0)
    Phi_v = basis_v.evaluate(x_pde, y_pde)
    dv_dy = basis_v.derivative(x_pde, y_pde, dx=0, dy=1)

    n_h, n_u, n_v = basis_h_tilde.n_basis, basis_u.n_basis, basis_v.n_basis
    A_Dx = np.hstack([(sqrt_K * R)[:, None] * dh_dx, Phi_u / sqrt_K[:, None], np.zeros((n_pde, n_v))])
    b_Dx = np.zeros(n_pde)
    A_Dy = np.hstack([sqrt_K[:, None] * dh_dy, np.zeros((n_pde, n_u)), Phi_v / sqrt_K[:, None]])
    b_Dy = -sqrt_K
    A_CE = np.hstack([np.zeros((n_pde, n_h)), R * du_dx, dv_dy])
    b_CE = np.zeros(n_pde)

    A = np.vstack([A_Dx, A_Dy, A_CE])
    b = np.concatenate([b_Dx, b_Dy, b_CE])
    return float(np.linalg.norm(A @ coeffs - b))


def test_with_logger_is_bit_identical_to_without():
    config = _small_config()
    physics = DarcyPhysics(config, verbose=False)

    result_plain = solve_lilq_darcy(config, physics, verbose=False)
    logger = IterationLogger()
    result_logged = solve_lilq_darcy(config, physics, verbose=False, iteration_logger=logger)

    assert np.array_equal(result_plain['c_h_tilde'], result_logged['c_h_tilde'])
    assert np.array_equal(result_plain['c_u'], result_logged['c_u'])
    assert np.array_equal(result_plain['c_v'], result_logged['c_v'])
    assert result_plain['metrics']['fvm_rel_L2'] == result_logged['metrics']['fvm_rel_L2']
    assert len(logger) == 1


def test_lstsq_solver_method_guard_raises_when_logger_given():
    config = _small_config(solver_method='lstsq')
    physics = DarcyPhysics(config, verbose=False)
    with pytest.raises(NotImplementedError):
        solve_lilq_darcy(config, physics, verbose=False, iteration_logger=IterationLogger())


def test_single_row_reflects_linear_system_structure():
    """Darcy's LiL-Q system is linear and has no separate BC row block
    (BCs are satisfied exactly by basis construction) -- this pins down
    the resulting degenerate-but-correct values rather than leaving them
    as untested assumptions."""
    config = _small_config()
    physics = DarcyPhysics(config, verbose=False)
    logger = IterationLogger()

    solve_lilq_darcy(config, physics, verbose=False, iteration_logger=logger)

    assert len(logger) == 1
    row = logger.rows[0]
    assert row["k"] == 1
    # No k-1/k-2 history exists for a single-shot solve.
    assert math.isnan(row["order_obs"])
    assert row["stall_flag"] is False
    # The system is linear: the nonlinear residual and the linearized
    # residual coincide exactly (both are literally A@beta - b), so chi
    # is exactly 0 -- not NaN, and not some other placeholder.
    assert row["chi"] == 0.0
    # No separate BC block exists -- every row is "interior", so the
    # interior-only norm equals the full norm exactly.
    assert row["norm_R_interior"] == pytest.approx(row["norm_R_h"], rel=1e-12)
    assert row["norm_Rlin_interior"] == pytest.approx(row["norm_Rlin_h"], rel=1e-12)
    assert row["num_rank_svd"] == row["num_rank_gelsy"]
    assert row["solver_path"] == "cpu_gelsy"


def test_check_b2_residual_identity_against_direct_evaluation():
    config = _small_config()
    physics = DarcyPhysics(config, verbose=False)
    logger = IterationLogger()

    result = solve_lilq_darcy(config, physics, verbose=False, iteration_logger=logger)

    coeffs = np.concatenate([result['c_h_tilde'], result['c_u'], result['c_v']])
    norm_R_direct = _independent_residual_norm(config, physics, coeffs)
    norm_R_logged = logger.rows[0]["norm_R_h"]

    rel_err = abs(norm_R_direct - norm_R_logged) / (abs(norm_R_direct) + 1e-30)
    assert rel_err < 1e-10


def test_csv_round_trip(tmp_path):
    config = _small_config()
    physics = DarcyPhysics(config, verbose=False)
    logger = IterationLogger()

    solve_lilq_darcy(config, physics, verbose=False, iteration_logger=logger)

    out_path = tmp_path / "iterations.csv"
    logger.to_csv(out_path)
    import csv
    with open(out_path, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 1


def test_run_json_requires_iteration_logger():
    config = _small_config()
    physics = DarcyPhysics(config, verbose=False)
    with pytest.raises(ValueError):
        solve_lilq_darcy(config, physics, verbose=False, run_json_path="unused.json")


def test_run_json_written_and_self_consistent(tmp_path):
    config = _small_config()
    physics = DarcyPhysics(config, verbose=False)
    logger = IterationLogger()
    out_path = tmp_path / "run.json"

    result = solve_lilq_darcy(config, physics, verbose=False, iteration_logger=logger,
                              run_json_path=out_path)

    import json
    with open(out_path) as f:
        meta = json.load(f)

    assert meta["N_total"] == sum(meta["N_composition"].values())
    assert meta["P_total"] == sum(meta["P_composition"].values())
    assert meta["P_total"] == result['c_h_tilde'].size + result['c_u'].size + result['c_v'].size
    assert meta["K_max"] == 1
    assert meta["solver_driver"] == "gelsy"
    assert meta["device"] == "cpu"
    assert meta["stopping_reason"] == "direct_solve"
    # Darcy's single row (k=1) never stalls -- no k-1 history.
    assert meta["first_stall_iteration"] is None
