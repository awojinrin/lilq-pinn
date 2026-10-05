"""Package 2, item 4 (P2-14, P2-4): the uniform-scale ELM basis, the dense grid
evaluator, the two normal-equation variants (the advisor's reply of 5 October,
Section 3.1) and a failure, the Kovasznay options (the paper's runs unchanged),
and the drivers on small sizes."""

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


def test_normal_shifted(monkeypatch):
    import scipy.linalg
    from problems.kovasznay import NORMAL_SHIFTS, _solve_normal_shifted
    rng = np.random.default_rng(0)
    A, b = rng.standard_normal((40, 6)), rng.standard_normal(40)
    beta, factor, _ = _solve_normal_shifted(A, b)
    assert factor == NORMAL_SHIFTS[0] and np.allclose(beta, np.linalg.lstsq(A, b, rcond=None)[0], rtol=1e-12)
    # a factorization that fails unless the smallest eigenvalue is above 1e-11 of the largest:
    # the smallest shifts fail, and the smallest that works (1e-10) is chosen
    cho = scipy.linalg.cho_factor

    def strict(M, **kw):
        lam = np.linalg.eigvalsh(M)
        if lam.min() < 1e-11 * lam.max():
            raise np.linalg.LinAlgError('not positive definite enough')
        return cho(M, **kw)

    monkeypatch.setattr(scipy.linalg, 'cho_factor', strict)
    A[:, 5] = A[:, 4]                                         # rank-deficient: A^T A singular
    beta, factor, AtA = _solve_normal_shifted(A, b)
    assert factor == 1e-10 and np.isfinite(beta).all()
    # the factor is fixed: passed back, it is used as it is, and fails if it cannot factor
    beta2, factor2, _ = _solve_normal_shifted(A, b, factor)
    assert factor2 == factor and np.array_equal(beta, beta2)
    assert _solve_normal_shifted(A, b, NORMAL_SHIFTS[0])[0] is None


def test_normal_eigh():
    from problems.kovasznay import _solve_normal_eigh
    rng = np.random.default_rng(1)
    A, b = rng.standard_normal((40, 6)), rng.standard_normal(40)
    beta, kept, _ = _solve_normal_eigh(A, b)
    assert kept == 6 and np.allclose(beta, np.linalg.lstsq(A, b, rcond=None)[0], rtol=1e-12)
    A[:, 5] = A[:, 4]                                         # rank 5: the pseudo-inverse, the minimum-norm solution
    beta, kept, _ = _solve_normal_eigh(A, b)
    assert kept == 5 and np.allclose(beta, np.linalg.pinv(A) @ b, rtol=1e-8)


def _kov(**kw):
    from problems.kovasznay import KovasznayConfig
    return KovasznayConfig(N_x=5, N_y=5, k_ratio=4, max_iter=3, tol=1e-9, **kw)


def test_kovasznay_options(tmp_path, monkeypatch):
    import problems.kovasznay as pk
    from lilq.iteration_log import IterationLogger
    q = pk.solve_kovasznay(_kov(), verbose=False, diagnostics=False)
    for solver in ('normal_shifted', 'normal_eigh'):
        n = pk.solve_kovasznay(_kov(linear_solver=solver), verbose=False, diagnostics=False)
        assert np.allclose(np.concatenate([n[f'theta_{f}'] for f in 'uvp']),
                           np.concatenate([q[f'theta_{f}'] for f in 'uvp']), rtol=1e-6)
        assert len(n['history']['normal_shift_or_rank']) == 3
    elm = pk.solve_kovasznay(_kov(basis_type='elm_uniform', elm_neurons=30, layout_P=75), verbose=False,
                             iteration_logger=IterationLogger(), run_json_path=tmp_path / 'elm' / 'run.json')
    assert elm['basis_u'] is elm['basis_p'] and elm['basis_u'].n_basis == 30
    # the control stopping at k = 0
    monkeypatch.setattr(pk, '_solve_normal_shifted', lambda A, b, factor: (None, None, A.T @ A))
    logger = IterationLogger()
    f = pk.solve_kovasznay(_kov(linear_solver='normal_shifted'), verbose=False, iteration_logger=logger,
                           run_json_path=tmp_path / 'fail' / 'run.json')
    assert f['normal_failed'] and f['n_outer_iters'] == 0
    run = json.loads((tmp_path / 'fail' / 'run.json').read_text())
    assert run['stopping_reason'] == 'cholesky_failed' and run['solver_driver'] == 'normal_shifted'
    for solver in ('normal_shifted', 'normal_eigh', 'normal'):
        with pytest.raises(ValueError):
            pk.solve_kovasznay(_kov(linear_solver=solver, use_gpu=solver != 'normal'), verbose=False,
                               diagnostics=False)


def test_check_c5_passes():
    import experiments.p2_14_elm as m
    from problems.kovasznay import NORMAL_SHIFTS
    c5 = m.check_c5()
    v = c5['variants']
    assert c5['passed'] and all(x['rel_difference_k1'] < 1e-8 for x in v.values())
    assert v['normal_shifted']['shift_or_rank'] == NORMAL_SHIFTS[0] and v['normal_eigh']['shift_or_rank'] == 300


def test_kovasznay_driver_small(tmp_path, monkeypatch):
    """The driver on small sizes: every solver on the surrogate and on one ELM
    seed, the logs with the normal-equation columns, the summary, the figure."""
    import csv
    import experiments.p2_14_elm as m
    from problems.kovasznay import KovasznayConfig
    monkeypatch.setattr(m, 'kovasznay_config', lambda sigma, seed, solver='gelsy', k_max=None: KovasznayConfig(
        N_x=5, N_y=5, k_ratio=4, max_iter=3, tol=1e-12, basis_type='elm_uniform', elm_neurons=20, elm_sigma=sigma,
        elm_seed=seed, layout_P=60, linear_solver=solver))
    monkeypatch.setattr(m, '_chebyshev_config', lambda solver='gelsy', k_max=None: KovasznayConfig(
        N_x=5, N_y=5, k_ratio=4, max_iter=3, tol=1e-12, linear_solver=solver))
    c5, sweep, rows = m.kovasznay(tmp_path, sigmas=(1.0,), seeds=(0,), k_max=3)
    out = tmp_path / 'P2_4_elm_kovasznay'
    assert [(r['basis'], r['variant']) for r in rows] == [
        ('chebyshev', 'qr'), ('chebyshev', 'normal_shifted'), ('chebyshev', 'normal_eigh'),
        ('shared', 'qr'), ('shared', 'normal_shifted'), ('shared', 'normal_eigh')]
    with open(out / 'summary.csv') as f:
        summary = list(csv.DictReader(f))
    assert {'variant', 'shift_or_rank', 'basis', 'solver'} <= set(summary[0])
    assert summary[3]['shift_or_rank'] == '' and float(summary[4]['shift_or_rank']) > 0
    with open(out / 'normal_eigh' / 'seed0' / 'iterations.csv') as f:
        log = list(csv.DictReader(f))
    assert int(log[0]['shift_or_rank']) <= 60 and float(log[0]['kappa_AtA']) > 1 and log[-1]['shift_or_rank'] == ''
    assert (out / 'chebyshev_P300' / 'qr' / 'iterations.csv').exists()
    assert (out / 'figures' / 'normal_vs_qr.png').exists()


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
