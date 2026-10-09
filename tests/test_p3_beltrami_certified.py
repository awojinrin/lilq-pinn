"""Package 3, item 1: Beltrami on CC-CGL grids (``experiments/p3_1_beltrami_certified.py``)."""

import dataclasses
import json

import numpy as np
import pytest

import experiments.p3_1_beltrami_certified as m
from lilq import certified as cf
from problems.beltrami import BeltramiConfig, BeltramiPhysics, _generate_collocation, solve_beltrami

TINY = (2, 3, (6, 7))                                     # N_vel, N_p, M levels: a stand-in for B1


def _paper_blocks(cfg):
    """The paper's equispaced points with its weights (lambda / n per block), as System blocks."""
    pts = _generate_collocation(cfg, BeltramiPhysics(cfg), None)
    n = pts['n_pde']
    blocks = {'interior': {'points': np.column_stack([pts['x_pde'], pts['y_pde'], pts['z_pde'], pts['t_pde']]),
                           'w': np.full(n, 1.0 / n), 'share': 1.0}}
    for name, (xb, yb, zb, tb) in pts['bc'].items():
        blocks[name] = {'points': np.column_stack([xb, yb, zb, tb]), 'w': np.full(len(xb), 1.0 / len(xb)), 'share': 1.0}
    ni = pts['n_ic']
    blocks['initial'] = {'points': np.column_stack([pts['x_ic'], pts['y_ic'], pts['z_ic'], pts['t_ic']]),
                         'w': np.full(ni, 1.0 / ni), 'share': 1.0}
    return blocks


def test_assembly_matches_the_papers_solver():
    """Given the paper's points and weights, the new assembly is the paper's
    linearized system (``solve_beltrami``'s own A, b) at k = 0 and k = 1."""
    cfg = BeltramiConfig(N_vel=3, N_p=4, N_x=5, N_y=5, N_z=5, N_t=5, N_bc=4, N_t_bc=4, N_ic=5,
                         n_pressure_pin_levels=4, tol=0.0, conditioning_every_iteration=False)
    sys_ = m.System(cfg, _paper_blocks(cfg))
    r1 = solve_beltrami(dataclasses.replace(cfg, max_iter=1), verbose=False, return_final_system=True)
    A0, b0 = sys_.assemble(np.zeros(sys_.P))
    assert np.array_equal(A0, r1['A_final']) and np.array_equal(b0, r1['b_final'])
    theta1 = np.concatenate([r1['theta_u'], r1['theta_v'], r1['theta_w'], r1['theta_p']])
    r2 = solve_beltrami(dataclasses.replace(cfg, max_iter=2), verbose=False, return_final_system=True)
    A1, b1 = sys_.assemble(theta1)
    assert np.abs(A1 - r2['A_final']).max() <= 1e-14 * np.abs(r2['A_final']).max()
    assert np.abs(b1 - r2['b_final']).max() <= 1e-14 * np.abs(r2['b_final']).max()


def test_newton_identity_on_a_cc_grid():
    """Check K4's identity: A(theta) theta - b(theta) is the weighted nonlinear residual."""
    cfg = m.config_for('B1')
    sys_ = m.System(cfg, cf.beltrami_grid(7, m.INTERVALS)['blocks'])
    th = np.random.default_rng(3).standard_normal(sys_.P)
    A, b = sys_.assemble(th, out=np.empty((sys_.N, sys_.P)))
    R = sys_.residual(th)
    assert np.linalg.norm(A @ th - b - R) < 1e-12 * np.linalg.norm(R)


def test_b1_level_1_has_the_rows_of_section_3_4():
    sys_ = m.System(m.config_for('B1'), cf.beltrami_grid(13, m.INTERVALS)['blocks'])
    assert (sys_.N, sys_.P, sys_.n_pin) == (160386, 1393, 5)
    assert sys_.s_pin == pytest.approx(np.sqrt(10.0 / 5))                 # the paper's lambda_bc / N_p


def test_delta_P_gives_the_pressure_its_time_levels():
    """A pressure constant in space with any time dependence is in the span once
    the 11 level columns are added."""
    sys_ = m.System(m.config_for('B1'), cf.beltrami_grid(5, m.INTERVALS)['blocks'], pins=False)
    exact = m.exact_on_grid(sys_.physics)
    T = np.meshgrid(*m.eval_axes(sys_.physics), indexing='ij')[3]
    exact['p'] = np.exp(3 * T) * np.sin(7 * T)
    dp, _ = m.delta_P(sys_, exact)
    assert dp['delta_P_p'] < 1e-13 and 0 < dp['delta_P_u'] < 1


def test_tiny_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setitem(m.SIZES, 'B1', TINY)
    for tag, scale in (('', 1.0), ('_pin1e-3', 1e-3), ('_pin1e3', 1e3), ('_rerun', 1.0)):
        m.run('B1', 1, tmp_path, pin_scale=scale, tag=tag, k_iters=3)
    d = tmp_path / m.ITEM / 'B1_L1'
    meta = json.loads((d / 'run.json').read_text())
    assert meta['package3']['status'] == 'complete' and meta['b2_check']['max_rel_err_over_run'] < 1e-10
    assert "paper's runs put lambda on each face" in meta['package3']['weights_note']     # Section 2.2
    assert sorted(p.name for p in (d / 'coefficients').iterdir()) == ['beta_1.npy', 'beta_2.npy', 'beta_3.npy']
    rho = (d / 'rho_r.csv').read_text().splitlines()
    assert len(rho) == 4 and int(rho[1].split(',')[5]) == 129 - 3      # the Y-system: the gauge modes, unpinned
    m._write_csv(tmp_path / m.ITEM / 'constants_B1_L1.csv', m.constants('B1', 1))
    terminal, consts, checks = m.summarize(tmp_path)
    assert len(consts) == 23 and consts[-1]['block'] == 'overall'      # interior, 21 blocks, overall
    assert all(checks[k]['passed'] for k in ('K3', 'K4', 'K6', 'K7'))
    assert len(checks['K3']['runs']) == 2


def test_a_stopped_run_keeps_what_it_did(tmp_path, monkeypatch):
    """Section 2.1: the log and the coefficients are written after every iterate."""
    monkeypatch.setitem(m.SIZES, 'B1', TINY)
    calls = {'n': 0}
    real = m._lstsq

    def stop_on_second(A, b):
        calls['n'] += 1
        if calls['n'] == 2:
            raise KeyboardInterrupt('wall cap')
        return real(A, b)
    monkeypatch.setattr(m, '_lstsq', stop_on_second)
    with pytest.raises(KeyboardInterrupt):
        m.run('B1', 1, tmp_path, k_iters=3, with_rho=False)
    d = tmp_path / m.ITEM / 'B1_L1'
    assert json.loads((d / 'run.json').read_text())['package3']['status'] == 'running'
    assert (d / 'coefficients' / 'beta_1.npy').exists() and len((d / 'iterations.csv').read_text().splitlines()) == 2
