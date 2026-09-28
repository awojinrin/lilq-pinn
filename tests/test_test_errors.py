"""Per-iterate test errors (Computational_Package_1_v2.md Section 3.1 item
10). The fast tensor-grid evaluator must equal full basis evaluation, and
every problem's terminal-row errors must equal an independent
computation of the same quantity at the returned solution -- the
problem's own final-error code where it uses the same grid.
"""

import numpy as np
import pytest
import torch

from lilq.basis import create_basis_2d, create_basis_nd
from lilq.iteration_log import IterationLogger
from lilq.test_errors import max_abs, rel_l2, tensor_grid_values


@pytest.mark.parametrize("basis_type", ["chebyshev", "fourier", "cos_fourier", "sin_fourier",
                                        "cos_sin", "sin_cheb"])
@pytest.mark.parametrize("orders", [(0, 0), (2, 0), (0, 2), (1, 1)])
def test_tensor_grid_values_matches_full_evaluation_2d(basis_type, orders):
    rng = np.random.default_rng(0)
    basis = create_basis_2d(basis_type, 5, 6, (-0.5, 1.0), (0.0, 2.0))
    c = rng.standard_normal(basis.n_basis)
    xs, ys = np.linspace(-0.5, 1.0, 7), np.linspace(0.0, 2.0, 9)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    if orders == (0, 0):
        full = basis.evaluate(X.ravel(), Y.ravel()) @ c
    else:
        full = basis.derivative(X.ravel(), Y.ravel(), dx=orders[0], dy=orders[1]) @ c
    grid = tensor_grid_values(basis, c, [xs, ys], orders)
    assert grid.shape == (7, 9)
    np.testing.assert_allclose(grid.ravel(), full, rtol=1e-12, atol=1e-12)


def test_tensor_grid_values_matches_full_evaluation_4d():
    rng = np.random.default_rng(1)
    basis = create_basis_nd('chebyshev', 3, [(-1, 1)] * 3 + [(0, 1)])
    c = rng.standard_normal(basis.n_basis)
    axes = [np.linspace(-1, 1, 4), np.linspace(-1, 1, 5), np.linspace(-1, 1, 3), np.linspace(0, 1, 2)]
    G = np.meshgrid(*axes, indexing='ij')
    full = basis.derivative(*[g.ravel() for g in G], orders=[1, 0, 2, 1]) @ c
    np.testing.assert_allclose(tensor_grid_values(basis, c, axes, [1, 0, 2, 1]).ravel(), full,
                               rtol=1e-12, atol=1e-12)


def test_rel_l2_and_max_abs():
    exact = np.array([3.0, 4.0])
    assert rel_l2(np.array([3.0, 5.0]), exact) == pytest.approx(1 / 5)
    assert max_abs(np.array([2.0, 4.5]), exact) == pytest.approx(1.0)


def test_rows_carry_errors_of_their_own_iterate():
    """Row k's errors are at beta^(k): for a zero start, row 0 of a
    problem with an exact solution is a 100% error."""
    from problems.kovasznay import KovasznayConfig, solve_kovasznay
    logger = IterationLogger()
    solve_kovasznay(KovasznayConfig(N_x=5, N_y=5, max_iter=3), verbose=False, iteration_logger=logger)
    assert logger.rows[0]["eps_u"] == pytest.approx(1.0)
    assert logger.rows[-1]["eps_u"] < 1.0


def test_kovasznay_terminal_errors_match_full_evaluation_on_the_test_grid():
    from problems.kovasznay import KovasznayConfig, KovasznayPhysics, TEST_GRID, solve_kovasznay
    config = KovasznayConfig(N_x=8, N_y=8, k_ratio=4, tol=1e-9, max_iter=20)
    logger = IterationLogger()
    r = solve_kovasznay(config, verbose=False, iteration_logger=logger)
    phys = KovasznayPhysics(config)
    X, Y = np.meshgrid(np.linspace(*config.x_domain, TEST_GRID[0]),
                       np.linspace(*config.y_domain, TEST_GRID[1]), indexing='ij')
    x, y = X.ravel(), Y.ravel()
    u = r['basis_u'].evaluate(x, y) @ r['theta_u']
    p = r['basis_p'].evaluate(x, y) @ r['theta_p']
    ue, pe = phys.exact_u(x, y), phys.exact_p(x, y)
    last = logger.rows[-1]
    assert last["eps_u"] == pytest.approx(np.linalg.norm(u - ue) / np.linalg.norm(ue), rel=1e-10)
    assert last["eps_p"] == pytest.approx(np.linalg.norm(p - pe) / np.linalg.norm(pe), rel=1e-10)
    pm, pem = p - p.mean(), pe - pe.mean()
    assert last["eps_p_meanfree"] == pytest.approx(np.linalg.norm(pm - pem) / np.linalg.norm(pem), rel=1e-10)
    assert last["maxerr_u"] == pytest.approx(np.max(np.abs(u - ue)), rel=1e-10)


def test_beltrami_terminal_errors_equal_the_papers_compute_errors():
    from problems.beltrami import BeltramiConfig, solve_beltrami
    config = BeltramiConfig(N_vel=3, N_p=3, N_x=4, N_y=4, N_z=4, N_t=4,
                            N_bc=3, N_t_bc=3, N_ic=3, max_iter=4)
    logger = IterationLogger()
    r = solve_beltrami(config, verbose=False, iteration_logger=logger)
    last = logger.rows[-1]
    assert last["eps_u"] == pytest.approx(r["rel_l2_u"], rel=1e-10)
    assert last["eps_v"] == pytest.approx(r["rel_l2_v"], rel=1e-10)
    assert last["eps_p"] == pytest.approx(r["rel_l2_p"], rel=1e-10)


def test_darcy_terminal_errors_equal_its_fvm_comparison():
    from problems.darcy import DarcyConfig, DarcyPhysics, solve_lilq_darcy
    config = DarcyConfig(ORDER_H=6, ORDER_U=6, ORDER_V=6)
    logger = IterationLogger()
    r = solve_lilq_darcy(config, DarcyPhysics(config, verbose=False), verbose=False,
                         iteration_logger=logger)
    last = logger.rows[-1]
    assert last["eps_p"] == pytest.approx(r["metrics"]["fvm_rel_L2"], rel=1e-10)
    assert last["maxerr_p"] == pytest.approx(r["metrics"]["fvm_max_err_psi"], rel=1e-10)


def test_elasticity_terminal_errors_equal_its_own_final_errors():
    from problems.elasticity import ElasticityConfig, solve_elasticity
    logger = IterationLogger()
    r = solve_elasticity(ElasticityConfig(N_x=6, N_y=6), verbose=False, iteration_logger=logger)
    last = logger.rows[-1]
    assert last["eps_u"] == pytest.approx(r["rel_l2_ux"], rel=1e-10)
    assert last["eps_v"] == pytest.approx(r["rel_l2_uy"], rel=1e-10)


def _residual_ms_on_grid(basis, c, xs, ts, residual):
    X, T = np.meshgrid(xs, ts, indexing='ij')
    x, t = X.ravel(), T.ravel()
    u = basis.evaluate(x, t) @ c
    ux = basis.derivative(x, t, dx=1, dy=0) @ c
    ut = basis.derivative(x, t, dx=0, dy=1) @ c
    uxx = basis.derivative(x, t, dx=2, dy=0) @ c
    uyy = basis.derivative(x, t, dx=0, dy=2) @ c
    return float(np.mean(residual(u, ux, ut, uxx, uyy) ** 2))


def test_bratu_terminal_residual_matches_full_evaluation():
    from problems.bratu import BratuConfig, BratuOptConfig, TEST_GRID, run_lil_q
    config = BratuConfig(N_x=6, N_y=6, lambda_=6.2)
    logger = IterationLogger()
    basis, c, _, _ = run_lil_q(config, BratuOptConfig(), verbose=False, iteration_logger=logger)
    xs = np.linspace(*config.x_domain, TEST_GRID[0])
    ys = np.linspace(*config.y_domain, TEST_GRID[1])
    expected = _residual_ms_on_grid(basis, c, xs, ys,
                                    lambda u, ux, ut, uxx, uyy: uxx + uyy + config.lambda_ * np.exp(u))
    assert logger.rows[-1]["eps_u"] == pytest.approx(expected, rel=1e-10)


def test_burgers_terminal_residual_matches_full_evaluation():
    from problems.burgers import BurgersConfig, BurgersOptConfig, TEST_GRID, run_lil_q
    config = BurgersConfig(N_x=6, N_t=6)
    logger = IterationLogger()
    basis, c, _, _ = run_lil_q(config, BurgersOptConfig(), verbose=False, iteration_logger=logger)
    xs = np.linspace(*config.x_domain, TEST_GRID[0])
    ts = np.linspace(0.0, config.T_final, TEST_GRID[1])
    expected = _residual_ms_on_grid(basis, c, xs, ts,
                                    lambda u, ux, ut, uxx, uyy: ut + u * ux - config.viscosity * uxx)
    assert logger.rows[-1]["eps_u"] == pytest.approx(expected, rel=1e-10)


def test_bl_terminal_error_matches_full_evaluation():
    from problems.buckley_leverett import BLConfig, BLOptConfig, TEST_GRID, reference_solution, run_lil_q
    config = BLConfig(N_x=6, N_t=6)
    logger = IterationLogger()
    basis, c, _, _ = run_lil_q(config, BLOptConfig(max_quasi_iters_lil=5), verbose=False,
                               iteration_logger=logger)
    xs = np.linspace(*config.x_domain, TEST_GRID[0])
    ts = np.linspace(0.0, config.T_final, TEST_GRID[1])
    X, T = np.meshgrid(xs, ts, indexing="ij")
    S = (basis.evaluate(X.ravel(), T.ravel()) @ c).reshape(X.shape)
    S_ref = reference_solution(config)
    assert logger.rows[-1]["eps_u"] == pytest.approx(np.linalg.norm(S - S_ref) / np.linalg.norm(S_ref), rel=1e-10)
    assert logger.rows[-1]["maxerr_u"] == pytest.approx(np.abs(S - S_ref).max(), rel=1e-10)
