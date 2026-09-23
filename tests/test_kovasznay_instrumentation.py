"""End-to-end integration test: Kovasznay's ``solve_kovasznay`` wired to
the shared Section 3.1 instrumentation (``lilq.iteration_log``) -- the
Kovasznay-specific follow-up to sub-batch 4 (Bratu/Burgers/BL), flagged
separately in DECISIONS.md because Kovasznay has its own self-contained
quasilinearization loop rather than calling ``lilq.solvers.solve_lil_q``.

Unlike Bratu/Burgers/BL, Kovasznay has no pre-existing
``_make_lil_nonlinear_loss_fn``/system-assembly helpers to extend --
``_make_kovasznay_nonlinear_loss_fn``/``_make_kovasznay_residual_vector_fn``
are new, generalizing the same weighted-block convention to Kovasznay's
momentum/continuity/BC/pin structure (12 row-blocks instead of the
scalar problems' 2-4).
"""

import dataclasses
import math

import numpy as np
import pytest

from lilq.iteration_log import IterationLogger
from problems.kovasznay import (
    KovasznayConfig, KovasznayPhysics, solve_kovasznay,
    _generate_collocation, _make_kovasznay_residual_vector_fn,
)
from lilq.basis import create_basis_2d


def _small_config(**overrides):
    # Small basis (chebyshev, N_x=N_y=4 -> Pu=Pv=Pp=16, P_total=48) with
    # k_ratio=5 for a genuinely overdetermined PDE block, few iterations
    # for speed.
    kwargs = dict(N_x=4, N_y=4, k_ratio=5, max_iter=10)
    kwargs.update(overrides)
    return KovasznayConfig(**kwargs)


def _independent_residual_norm(config, coefficients):
    """Rebuild the weighted residual vector from scratch via the public
    basis/collocation/physics API (not solve_kovasznay's own internals)
    and evaluate it at the given [theta_u; theta_v; theta_p], for check
    B2. Mirrors solve_kovasznay's own basis/collocation setup exactly
    (same config/seed -> same points, deterministic)."""
    physics = KovasznayPhysics(config)
    nu = physics.nu

    basis_u = create_basis_2d(config.basis_type, config.N_x, config.N_y,
                               config.x_domain, config.y_domain)
    basis_v = create_basis_2d(config.basis_type, config.N_x, config.N_y,
                               config.x_domain, config.y_domain)
    basis_p = create_basis_2d(config.basis_type, config.N_x, config.N_y,
                               config.x_domain, config.y_domain)
    Pu, Pv, Pp = basis_u.n_basis, basis_v.n_basis, basis_p.n_basis
    P_total = Pu + Pv + Pp

    pts = _generate_collocation(config, P_total)
    xp, yp = pts['x_pde'], pts['y_pde']
    n_pde = pts['n_pde']

    Phi_u = basis_u.evaluate(xp, yp)
    Phi_u_x = basis_u.derivative(xp, yp, dx=1, dy=0)
    Phi_u_y = basis_u.derivative(xp, yp, dx=0, dy=1)
    Phi_u_xx = basis_u.derivative(xp, yp, dx=2, dy=0)
    Phi_u_yy = basis_u.derivative(xp, yp, dx=0, dy=2)
    Phi_v = basis_v.evaluate(xp, yp)
    Phi_v_x = basis_v.derivative(xp, yp, dx=1, dy=0)
    Phi_v_y = basis_v.derivative(xp, yp, dx=0, dy=1)
    Phi_v_xx = basis_v.derivative(xp, yp, dx=2, dy=0)
    Phi_v_yy = basis_v.derivative(xp, yp, dx=0, dy=2)
    Phi_p_x = basis_p.derivative(xp, yp, dx=1, dy=0)
    Phi_p_y = basis_p.derivative(xp, yp, dx=0, dy=1)

    bc_blocks = {}
    for edge in ['bot', 'top', 'left', 'right']:
        xe, ye = pts[f'x_{edge}'], pts[f'y_{edge}']
        bc_blocks[edge] = {
            'Phi_u': basis_u.evaluate(xe, ye),
            'Phi_v': basis_v.evaluate(xe, ye),
            'u_exact': physics.exact_u(xe, ye),
            'v_exact': physics.exact_v(xe, ye),
            'n': len(xe),
        }

    x_pin = np.array([config.x_domain[0]], dtype=np.float64)
    y_pin = np.array([config.y_domain[0]], dtype=np.float64)
    Phi_p_pin = basis_p.evaluate(x_pin, y_pin)
    p_pin_val = physics.exact_p(x_pin[0], y_pin[0])

    residual_vector_fn = _make_kovasznay_residual_vector_fn(
        Phi_u, Phi_u_x, Phi_u_y, Phi_u_xx, Phi_u_yy,
        Phi_v, Phi_v_x, Phi_v_y, Phi_v_xx, Phi_v_yy,
        Phi_p_x, Phi_p_y, bc_blocks, Phi_p_pin, p_pin_val,
        nu, Pu, Pv, Pp, n_pde, config.lambda_mom, config.lambda_cont, config.lambda_bc,
    )
    return float(np.linalg.norm(residual_vector_fn(coefficients)))


def test_with_logger_is_bit_identical_to_without():
    config = _small_config()

    result_plain = solve_kovasznay(config, verbose=False)
    logger = IterationLogger()
    result_logged = solve_kovasznay(config, verbose=False, iteration_logger=logger)

    assert np.array_equal(result_plain['theta_u'], result_logged['theta_u'])
    assert np.array_equal(result_plain['theta_v'], result_logged['theta_v'])
    assert np.array_equal(result_plain['theta_p'], result_logged['theta_p'])
    assert result_plain['n_outer_iters'] == result_logged['n_outer_iters']
    # Deterministic parts of history unchanged (solve_time is real
    # wall-clock and will legitimately differ run-to-run).
    for key in ('iteration', 'coeff_change', 'pde_residual', 'continuity_residual', 'cond_number'):
        assert result_plain['history'][key] == result_logged['history'][key]
    assert len(logger) == result_logged['n_outer_iters']


def test_produces_full_iterations_csv(tmp_path):
    config = _small_config()
    logger = IterationLogger()

    result = solve_kovasznay(config, verbose=False, iteration_logger=logger)

    rows = logger.rows
    assert len(rows) == result['n_outer_iters']
    assert [row["k"] for row in rows] == list(range(1, len(rows) + 1))

    # lambda_mom == lambda_cont by default -- interior norms should be real.
    assert config.lambda_mom == config.lambda_cont
    assert all(not math.isnan(row["norm_Rlin_interior"]) for row in rows)
    assert all(not math.isnan(row["norm_R_interior"]) for row in rows)

    # compute_residual_vector_fn is always supplied -- chi should be real
    # at least once (this system is overdetermined, not exactly solvable).
    assert any(not math.isnan(row["chi"]) for row in rows)

    P_total = 3 * 16  # Pu=Pv=Pp=16 for N_x=N_y=4 chebyshev
    assert all(row["kappa_method"] == "svd" for row in rows)
    assert all(row["num_rank_svd"] == P_total for row in rows)
    assert all(row["num_rank_gelsy"] == P_total for row in rows)
    assert all(row["rcond"] > 0 for row in rows)
    assert all(row["t_assemble_s"] >= 0.0 for row in rows)
    assert all(row["t_solve_s"] >= 0.0 for row in rows)

    out_path = tmp_path / "iterations.csv"
    logger.to_csv(out_path)
    import csv
    with open(out_path, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(rows)


def test_check_b2_residual_identity_against_direct_evaluation():
    config = _small_config(max_iter=15, tol=1e-14)
    logger = IterationLogger()

    result = solve_kovasznay(config, verbose=False, iteration_logger=logger)

    coefficients = np.concatenate([result['theta_u'], result['theta_v'], result['theta_p']])
    norm_R_direct = _independent_residual_norm(config, coefficients)
    norm_R_logged = logger.rows[-1]["norm_R_h"]

    rel_err = abs(norm_R_direct - norm_R_logged) / (abs(norm_R_direct) + 1e-30)
    assert rel_err < 1e-10


def test_interior_norms_nan_when_lambda_mom_and_cont_differ():
    """The tracker's interior-row unweighting assumes a single scalar
    weight across the leading rows -- momentum (w_mom) and continuity
    (w_cont) only collapse to one scalar when lambda_mom == lambda_cont.
    When they differ, solve_kovasznay must correctly fall back to NOT
    populating n_interior_rows/interior_weight (NaN), rather than
    silently reporting a wrong number."""
    config = _small_config(max_iter=3)
    config = dataclasses.replace(config, lambda_mom=1.0, lambda_cont=2.0)
    logger = IterationLogger()

    solve_kovasznay(config, verbose=False, iteration_logger=logger)

    assert all(math.isnan(row["norm_Rlin_interior"]) for row in logger.rows)
    assert all(math.isnan(row["norm_R_interior"]) for row in logger.rows)
