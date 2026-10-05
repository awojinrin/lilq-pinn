"""Package 2, item 4 (P2-14, P2-4): the uniform-scale ELM basis, the dense grid
evaluator, the normal-equation solve and its failure, the Kovasznay options
(the paper's runs unchanged), and the drivers on small sizes."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from lilq.basis import ELMBasis2D_TorchDefault, ELMBasis2D_Uniform, ELMBasis2D_Xavier
from lilq.test_errors import grid_evaluator

REPO = Path(__file__).resolve().parents[1]


def test_uniform_elm_reproduces_the_table_3_initializations():
    x, t = np.linspace(-1, 1, 9), np.linspace(0, 1, 9)
    a = ELMBasis2D_Xavier(40, (-1, 1), (0, 1), seed=3)
    b = ELMBasis2D_Uniform(40, (-1, 1), (0, 1), sigma=np.sqrt(6 / 42), seed=3)
    c = ELMBasis2D_TorchDefault(40, (-1, 1), (0, 1), seed=3)
    d = ELMBasis2D_Uniform(40, (-1, 1), (0, 1), sigma=1 / np.sqrt(2), seed=3)
    assert np.array_equal(a.derivative(x, t, dx=2), b.derivative(x, t, dx=2))
    assert np.array_equal(c.evaluate(x, t), d.evaluate(x, t))
    assert float(b.alpha.abs().max()) <= np.sqrt(6 / 42)


def test_grid_evaluator_for_a_non_tensor_basis():
    basis = ELMBasis2D_Uniform(12, (-0.5, 1.0), (-0.5, 1.5), sigma=1.0, seed=0)
    xs, ys = np.linspace(-0.5, 1, 7), np.linspace(-0.5, 1.5, 5)
    c = np.arange(12.0)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    np.testing.assert_allclose(grid_evaluator(basis, [xs, ys])(c), (basis.evaluate(X.ravel(), Y.ravel()) @ c).reshape(7, 5))


def test_normal_equations_ok_and_failed():
    from problems.kovasznay import _solve_normal_equations
    rng = np.random.default_rng(0)
    A, b = rng.standard_normal((40, 6)), rng.standard_normal(40)
    beta, status, _ = _solve_normal_equations(A, b)
    assert status == 'none' and np.allclose(beta, np.linalg.lstsq(A, b, rcond=None)[0])
    A[:, 5] = A[:, 4]                                         # rank-deficient: A^T A singular
    beta, status, _ = _solve_normal_equations(A, b)
    assert status in ('shifted', 'failed') and (beta is None) == (status == 'failed')


def _kov(**kw):
    from problems.kovasznay import KovasznayConfig
    return KovasznayConfig(N_x=5, N_y=5, k_ratio=4, max_iter=3, tol=1e-9, **kw)


def test_kovasznay_options(tmp_path, monkeypatch):
    import problems.kovasznay as pk
    from lilq.iteration_log import IterationLogger
    q = pk.solve_kovasznay(_kov(), verbose=False, diagnostics=False)
    n = pk.solve_kovasznay(_kov(linear_solver='normal'), verbose=False, diagnostics=False)
    assert np.allclose(np.concatenate([n[f'theta_{f}'] for f in 'uvp']),
                       np.concatenate([q[f'theta_{f}'] for f in 'uvp']), rtol=1e-6)
    elm = pk.solve_kovasznay(_kov(basis_type='elm_uniform', elm_neurons=30, layout_P=75), verbose=False,
                             iteration_logger=IterationLogger(), run_json_path=tmp_path / 'elm' / 'run.json')
    assert elm['basis_u'] is elm['basis_p'] and elm['basis_u'].n_basis == 30
    # the control stopping at k = 0
    monkeypatch.setattr(pk, '_solve_normal_equations', lambda A, b: (None, 'failed', A.T @ A))
    logger = IterationLogger()
    f = pk.solve_kovasznay(_kov(linear_solver='normal'), verbose=False, iteration_logger=logger,
                           run_json_path=tmp_path / 'fail' / 'run.json')
    assert f['normal_failed'] and f['n_outer_iters'] == 0
    run = json.loads((tmp_path / 'fail' / 'run.json').read_text())
    assert run['stopping_reason'] == 'cholesky_failed' and run['solver_driver'] == 'normal_cholesky'
    with pytest.raises(ValueError):
        pk.solve_kovasznay(_kov(linear_solver='normal', use_gpu=True), verbose=False, diagnostics=False)


def test_check_c5_passes():
    import experiments.p2_14_elm as m
    c5 = m.check_c5()
    assert c5['passed'] and c5['rel_difference_k1'] < 1e-8 and c5['cholesky_shift'] == 'none'


def test_burgers_run_records_the_columns():
    import experiments.p2_14_elm as m
    x, t = np.linspace(-1, 1, 11), np.linspace(0, 1, 11)
    X, T = np.meshgrid(x, t, indexing='ij')
    r = m.burgers_run(0.7, 0, ((x, t), -np.sin(np.pi * X) * np.exp(-T)), k_max=2)
    for k in ('final_R_h_squared', 'eps_ref', 'kappa_raw', 'kappa_retained', 'rank_svd', 'rank_gelsy', 'norm_beta',
              'k_target'):
        assert k in r
    assert r['iterations'] == 2 and 0 < r['rank_svd'] <= 625


def test_item_4_job():
    text = (REPO / 'scripts' / 'cluster' / 'package2' / 'p2s2_elm.slurm').read_text()
    assert '# lilq-resources: cpu' in text and 'p2_14_elm.py sweep' in text and 'p2_14_elm.py kovasznay' in text
