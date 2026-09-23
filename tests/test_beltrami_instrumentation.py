"""End-to-end integration test: Beltrami's ``solve_beltrami`` wired to
the shared Section 3.1 instrumentation (``lilq.iteration_log``) -- the
Beltrami sub-batch (with Darcy) following sub-batch 5's Kovasznay
precedent. Like Kovasznay, Beltrami has its own self-contained
quasilinearization loop (no ``solve_lil_q`` call), so this manually
drives a ``LilQDiagnosticsTracker`` inside it, generalizing the
one-scalar-per-block weighting convention to Beltrami's 26 row-blocks
(momentum x3, continuity, BC-u/v/w x6 faces, IC-u/v/w, pressure pin).
"""

import dataclasses
import math

import numpy as np
import pytest

from lilq.iteration_log import IterationLogger
from problems.beltrami import (
    BeltramiConfig, BeltramiPhysics, solve_beltrami,
    _generate_collocation, _make_beltrami_residual_vector_fn,
)
from lilq.basis import create_basis_nd


def _small_config(**overrides):
    # Small 4D tensor-product bases (N_vel=N_p=3) with a small collocation
    # grid, for speed -- big enough that the system is genuinely
    # overdetermined.
    kwargs = dict(
        N_vel=3, N_p=3, N_x=4, N_y=4, N_z=4, N_t=4,
        N_bc=3, N_t_bc=3, N_ic=3, max_iter=8,
    )
    kwargs.update(overrides)
    return BeltramiConfig(**kwargs)


def _rebuild_matrices(config):
    """Independent reconstruction of Beltrami's basis/collocation
    matrices via the public API (create_basis_nd, _generate_collocation,
    basis.evaluate/derivative) -- same config/seed as solve_beltrami's
    own setup (deterministic, same points), not reusing any of
    solve_beltrami's own internals."""
    physics = BeltramiPhysics(config)
    nu = physics.nu
    domains = [config.x_domain, config.y_domain, config.z_domain, config.t_domain]
    basis_u = create_basis_nd(config.basis_type, config.N_vel, domains)
    basis_v = create_basis_nd(config.basis_type, config.N_vel, domains)
    basis_w = create_basis_nd(config.basis_type, config.N_vel, domains)
    basis_p = create_basis_nd(config.basis_type, config.N_p, domains)
    Pu, Pv, Pw, Pp = basis_u.n_basis, basis_v.n_basis, basis_w.n_basis, basis_p.n_basis
    P_total = Pu + Pv + Pw + Pp

    pts = _generate_collocation(config, physics, P_total)
    xp, yp, zp, tp = pts['x_pde'], pts['y_pde'], pts['z_pde'], pts['t_pde']
    n_pde = pts['n_pde']

    def _mats(bas, x, y, z, t):
        ev = lambda *o: bas.derivative(x, y, z, t, orders=list(o))
        return {'val': bas.evaluate(x, y, z, t),
                'dx': ev(1, 0, 0, 0), 'dy': ev(0, 1, 0, 0), 'dz': ev(0, 0, 1, 0), 'dt': ev(0, 0, 0, 1),
                'dxx': ev(2, 0, 0, 0), 'dyy': ev(0, 2, 0, 0), 'dzz': ev(0, 0, 2, 0)}

    Mu = _mats(basis_u, xp, yp, zp, tp)
    Mv = _mats(basis_v, xp, yp, zp, tp)
    Mw = _mats(basis_w, xp, yp, zp, tp)
    Mp = {'val': basis_p.evaluate(xp, yp, zp, tp),
          'dx': basis_p.derivative(xp, yp, zp, tp, orders=[1, 0, 0, 0]),
          'dy': basis_p.derivative(xp, yp, zp, tp, orders=[0, 1, 0, 0]),
          'dz': basis_p.derivative(xp, yp, zp, tp, orders=[0, 0, 1, 0])}

    bc_blocks = {}
    for fname, (xb, yb, zb, tb) in pts['bc'].items():
        bc_blocks[fname] = {
            'Phi_u': basis_u.evaluate(xb, yb, zb, tb),
            'Phi_v': basis_v.evaluate(xb, yb, zb, tb),
            'Phi_w': basis_w.evaluate(xb, yb, zb, tb),
            'u_ex': physics.exact_u(xb, yb, zb, tb),
            'v_ex': physics.exact_v(xb, yb, zb, tb),
            'w_ex': physics.exact_w(xb, yb, zb, tb),
            'n': len(xb),
        }

    xi, yi, zi, ti = pts['x_ic'], pts['y_ic'], pts['z_ic'], pts['t_ic']
    ic_block = {
        'Phi_u': basis_u.evaluate(xi, yi, zi, ti),
        'Phi_v': basis_v.evaluate(xi, yi, zi, ti),
        'Phi_w': basis_w.evaluate(xi, yi, zi, ti),
        'u_ex': physics.exact_u(xi, yi, zi, ti),
        'v_ex': physics.exact_v(xi, yi, zi, ti),
        'w_ex': physics.exact_w(xi, yi, zi, ti),
        'n': len(xi),
    }

    x0 = np.array([physics.x_domain[0]]); y0 = np.array([physics.y_domain[0]])
    z0 = np.array([physics.z_domain[0]]); t0_ = np.array([physics.t_domain[0]])
    Phi_p_pin = basis_p.evaluate(x0, y0, z0, t0_)
    p_pin_val = physics.exact_p(x0[0], y0[0], z0[0], t0_[0])

    return dict(Mu=Mu, Mv=Mv, Mw=Mw, Mp=Mp, bc_blocks=bc_blocks, ic_block=ic_block,
                Phi_p_pin=Phi_p_pin, p_pin_val=p_pin_val, nu=nu,
                Pu=Pu, Pv=Pv, Pw=Pw, Pp=Pp, n_pde=n_pde)


def test_with_logger_is_bit_identical_to_without():
    config = _small_config()

    result_plain = solve_beltrami(config, verbose=False)
    logger = IterationLogger()
    result_logged = solve_beltrami(config, verbose=False, iteration_logger=logger)

    assert np.array_equal(result_plain['theta_u'], result_logged['theta_u'])
    assert np.array_equal(result_plain['theta_v'], result_logged['theta_v'])
    assert np.array_equal(result_plain['theta_w'], result_logged['theta_w'])
    assert np.array_equal(result_plain['theta_p'], result_logged['theta_p'])
    assert result_plain['n_outer_iters'] == result_logged['n_outer_iters']
    for key in ('iteration', 'coeff_change', 'pde_residual', 'continuity_residual'):
        assert result_plain['history'][key] == result_logged['history'][key]
    assert len(logger) == result_logged['n_outer_iters']


def test_gpu_guard_raises_when_logger_given():
    config = _small_config(max_iter=2, use_gpu=True)
    with pytest.raises(NotImplementedError):
        solve_beltrami(config, verbose=False, iteration_logger=IterationLogger())


def test_produces_full_iterations_csv(tmp_path):
    config = _small_config()
    logger = IterationLogger()

    result = solve_beltrami(config, verbose=False, iteration_logger=logger)

    rows = logger.rows
    assert len(rows) == result['n_outer_iters']
    assert [row["k"] for row in rows] == list(range(1, len(rows) + 1))

    assert config.lambda_mom == config.lambda_cont
    assert all(not math.isnan(row["norm_Rlin_interior"]) for row in rows)
    assert all(not math.isnan(row["norm_R_interior"]) for row in rows)
    assert any(not math.isnan(row["chi"]) for row in rows)

    # This small config's system is mildly rank-deficient (a smaller-scale
    # analogue of the null space the manuscript documents for the full
    # config -- Section 3.7) -- assert the two rank estimates agree with
    # each other rather than assuming full rank.
    assert result['n_params'] == 3 * 81 + 81  # Pu=Pv=Pw=Pp=3^4=81
    assert all(row["num_rank_svd"] == row["num_rank_gelsy"] for row in rows)
    assert all(row["num_rank_svd"] <= result['n_params'] for row in rows)
    assert all(row["t_assemble_s"] >= 0.0 for row in rows)
    assert all(row["t_solve_s"] >= 0.0 for row in rows)

    out_path = tmp_path / "iterations.csv"
    logger.to_csv(out_path)
    import csv
    with open(out_path, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(rows)


def test_large_P_uses_pivoted_qr_at_final_iterate_only():
    """Section 3.1 item 8: for Beltrami specifically, conditioning above
    the SVD threshold uses pivoted QR at the final iterate only, not a
    full per-iteration SVD. P_total=7984 here matches the manuscript's
    own Beltrami configuration exactly."""
    config = _small_config(N_vel=6, N_p=8, N_x=5, N_y=5, N_z=5, N_t=5,
                            N_bc=4, N_t_bc=4, N_ic=4, max_iter=4)
    assert 3 * 6 ** 4 + 8 ** 4 == 7984
    logger = IterationLogger()

    solve_beltrami(config, verbose=False, iteration_logger=logger)

    rows = logger.rows
    assert all(row["kappa_method"] is None for row in rows[:-1])
    assert all(math.isnan(row["kappa"]) for row in rows[:-1])
    assert rows[-1]["kappa_method"] == "qr_pivoted"
    assert not math.isnan(rows[-1]["kappa"])


def test_check_b2_residual_identity_against_direct_evaluation():
    config = _small_config(max_iter=15, tol=1e-13)
    logger = IterationLogger()

    result = solve_beltrami(config, verbose=False, iteration_logger=logger)

    mats = _rebuild_matrices(config)
    residual_vector_fn = _make_beltrami_residual_vector_fn(
        mats['Mu'], mats['Mv'], mats['Mw'], mats['Mp'],
        mats['bc_blocks'], mats['ic_block'], mats['Phi_p_pin'], mats['p_pin_val'],
        mats['nu'], mats['Pu'], mats['Pv'], mats['Pw'], mats['Pp'], mats['n_pde'],
        config.lambda_mom, config.lambda_cont, config.lambda_bc, config.lambda_ic,
    )

    theta_final = np.concatenate([result['theta_u'], result['theta_v'],
                                   result['theta_w'], result['theta_p']])
    norm_R_direct = float(np.linalg.norm(residual_vector_fn(theta_final)))
    norm_R_logged = logger.rows[-1]["norm_R_h"]

    rel_err = abs(norm_R_direct - norm_R_logged) / (abs(norm_R_direct) + 1e-30)
    assert rel_err < 1e-10


def test_interior_norms_nan_when_lambda_mom_and_cont_differ():
    config = _small_config(max_iter=3)
    config = dataclasses.replace(config, lambda_mom=1.0, lambda_cont=2.0)
    logger = IterationLogger()

    solve_beltrami(config, verbose=False, iteration_logger=logger)

    assert all(math.isnan(row["norm_Rlin_interior"]) for row in logger.rows)
    assert all(math.isnan(row["norm_R_interior"]) for row in logger.rows)


def test_run_json_requires_iteration_logger():
    config = _small_config(max_iter=2)
    with pytest.raises(ValueError):
        solve_beltrami(config, verbose=False, run_json_path="unused.json")


def test_run_json_written_and_self_consistent(tmp_path):
    config = _small_config(max_iter=8)
    logger = IterationLogger()
    out_path = tmp_path / "run.json"

    result = solve_beltrami(config, verbose=False, iteration_logger=logger,
                             run_json_path=out_path)

    import json
    with open(out_path) as f:
        meta = json.load(f)

    assert meta["N_total"] == sum(meta["N_composition"].values())
    assert meta["P_total"] == result["n_params"]
    assert meta["P_total"] == sum(meta["P_composition"].values())
    assert meta["K_max"] == config.max_iter
    assert meta["solver_driver"] == "gelsy"
    assert meta["device"] == "cpu"
    assert meta["stopping_reason"] in ("target", "iteration_cap")
    # first_stall_iteration must match an independent scan of the logger.
    from lilq.run_metadata import first_stall_iteration
    assert meta["first_stall_iteration"] == first_stall_iteration(logger.rows)
