"""Task B10 (Addendum v2.3): Clenshaw-Curtis row weights on Kovasznay's CGL
grids -- the five tests of its Section 1, the weighted diagnostics, and the
driver."""

import json
import math

import numpy as np
import pytest

from lilq.collocation import clenshaw_curtis_weights, points_1d
from lilq.iteration_log import IterationLogger
from problems import kovasznay as kov
from problems.kovasznay import KovasznayConfig, solve_kovasznay


@pytest.mark.parametrize('n', [2, 3, 4, 5, 8, 13, 20, 41, 57])
def test_univariate_weights_sum_to_one_and_are_exact_to_degree_n_minus_1(n):
    w, x = clenshaw_curtis_weights(n), points_1d(-1.0, 1.0, n, 'cgl')
    assert abs(w.sum() - 1.0) < 1e-13 and (w > 0).all()
    for k in range(n):
        exact = 1.0 / (k + 1) if k % 2 == 0 else 0.0        # (1/2) * integral of x^k over [-1, 1]
        assert abs(np.dot(w, x ** k) - exact) < 1e-13
    assert np.allclose(w, w[::-1], rtol=0, atol=1e-15)     # symmetric: the node order does not matter


def _cc(n_dim, n_edge, **kw):
    config = KovasznayConfig(sampling='cgl', weights='clenshaw_curtis', **kw)
    return config, kov._row_weights(config, n_dim * n_dim, n_dim, {e: n_edge for e in kov.EDGES})


def test_interior_weights_reproduce_the_normalized_area_integral():
    """sum_i w_i f(x_i) = (1 / |Omega|) int_Omega f, for f = exp(x) cos(y):
    spectrally at the CGL nodes; to the 1e-6 inset at the grid the solver uses."""
    config, rw = _cc(21, 10)
    (x0, x1), (y0, y1) = config.x_domain, config.y_domain
    exact = (math.exp(x1) - math.exp(x0)) * (math.sin(y1) - math.sin(y0)) / ((x1 - x0) * (y1 - y0))
    w = rw['mom'] ** 2 / config.lambda_mom
    assert abs(w.sum() - 1.0) < 1e-13 and abs((rw['cont'] ** 2 / config.lambda_cont).sum() - 1.0) < 1e-13
    xx, yy = np.meshgrid(points_1d(x0, x1, 21, 'cgl'), points_1d(y0, y1, 21, 'cgl'))
    assert abs(np.dot(w, np.exp(xx.ravel()) * np.cos(yy.ravel())) - exact) < 1e-13
    xi = points_1d(x0 + 1e-6, x1 - 1e-6, 21, 'cgl')        # the solver's interior grid (inset)
    yi = points_1d(y0 + 1e-6, y1 - 1e-6, 21, 'cgl')
    xx, yy = np.meshgrid(xi, yi)
    assert abs(np.dot(w, np.exp(xx.ravel()) * np.cos(yy.ravel())) - exact) < 1e-5


def test_boundary_weights_reproduce_the_normalized_perimeter_integral():
    """sum over the four edges of w_k g = (1 / |dOmega|) int_dOmega g, with
    g = exp(x) + y^2; each component's weights sum to lambda_bc."""
    config, rw = _cc(11, 15)
    (x0, x1), (y0, y1) = config.x_domain, config.y_domain
    g = lambda x, y: np.exp(x) + y ** 2  # noqa: E731
    xs, ys = points_1d(x0, x1, 15, 'cgl'), points_1d(y0, y1, 15, 'cgl')
    edges = {'bot': (xs, np.full(15, y0)), 'top': (xs, np.full(15, y1)),
             'left': (np.full(15, x0), ys), 'right': (np.full(15, x1), ys)}
    total = sum(np.dot(rw['bc'][e] ** 2 / config.lambda_bc, g(*edges[e])) for e in kov.EDGES)
    lx, ly = x1 - x0, y1 - y0
    integral = (2 * (math.exp(x1) - math.exp(x0)) + lx * (y0 ** 2 + y1 ** 2)            # bottom + top
                + ly * (math.exp(x0) + math.exp(x1)) + 2 * (y1 ** 3 - y0 ** 3) / 3)        # left + right
    assert sum((rw['bc'][e] ** 2).sum() for e in kov.EDGES) == pytest.approx(config.lambda_bc, rel=1e-13)
    assert abs(total - integral / (2 * (lx + ly))) < 1e-12


def test_equal_weights_are_the_scalar_expressions_and_leave_the_solve_unchanged():
    """'equal' is the default and gives exactly the scalars every run has
    used; the solve with it is bitwise the default's. (Bitwise against the
    previous commit: DECISIONS.md, 2026-10-01.)"""
    config = KovasznayConfig(N_x=5, N_y=5, k_ratio=4, max_iter=20, tol=1e-9)
    assert config.weights == 'equal'
    rw = kov._row_weights(config, 100, 10, {e: 12 for e in kov.EDGES})
    assert rw['mom'] == np.sqrt(config.lambda_mom / 100) and rw['bc']['left'] == np.sqrt(config.lambda_bc / 12)
    assert np.ndim(rw['mom']) == 0
    a = solve_kovasznay(config, verbose=False, return_final_system=True)
    b = solve_kovasznay(KovasznayConfig(N_x=5, N_y=5, k_ratio=4, max_iter=20, tol=1e-9, weights='equal'),
                        verbose=False, return_final_system=True)
    for k in ('A_final', 'b_final', 'theta_u', 'theta_v', 'theta_p'):
        assert np.array_equal(a[k], b[k])


def test_clenshaw_curtis_needs_cgl_and_a_known_name():
    with pytest.raises(ValueError, match="sampling='cgl'"):
        solve_kovasznay(KovasznayConfig(N_x=4, N_y=4, weights='clenshaw_curtis'), verbose=False)
    with pytest.raises(ValueError, match='weights must be'):
        solve_kovasznay(KovasznayConfig(N_x=4, N_y=4, sampling='cgl', weights='gauss'), verbose=False)


def test_clenshaw_curtis_solve_logs_consistent_weighted_residuals(tmp_path):
    """The assembly, the logged residual vector and the loss use the same row
    weights: check B2 holds; collocation.npz has the per-row weights and
    run.json the weights option."""
    config = KovasznayConfig(N_x=6, N_y=6, k_ratio=3.0, max_iter=8, tol=1e-9, sampling='cgl',
                             collocation_floor=1, weights='clenshaw_curtis')
    logger = IterationLogger()
    r = solve_kovasznay(config, verbose=False, iteration_logger=logger, run_json_path=tmp_path / 'run.json',
                        collocation_path=tmp_path / 'c.npz', return_final_system=True)
    run = json.loads((tmp_path / 'run.json').read_text())
    assert run['collocation_construction']['weights'] == 'clenshaw_curtis'
    assert run['b2_check']['max_rel_err_over_run'] < 1e-10      # ||weighted residual||^2 == the loss
    with np.load(tmp_path / 'c.npz') as z:
        w, block = z['weight'].copy(), z['block'].copy()
    interior = w[block == 'xmom']
    assert len(np.unique(interior)) > 1                      # per-row, not one scalar
    assert abs((interior ** 2).sum() - config.lambda_mom) < 1e-12
    # The final system's rows carry the same weights as the saved file.
    assert len(r['A_final']) == len(w)
    assert np.isfinite(logger.rows[-1]['norm_R_h'])


def test_b10_driver_smoke(tmp_path):
    import csv
    import experiments.b10_cgl_cc as b10
    paper = tmp_path / 'paper'
    (paper / 'kovasznay_P300_cpu_paper').mkdir(parents=True)
    (paper / 'kovasznay_P300_cpu_paper' / 'summary.json').write_text(json.dumps(
        {'test_eps_u': 2.9e-2, 'test_eps_v': 1e-2, 'test_eps_p': 1e-1, 'test_eps_p_meanfree': 5e-2, 'iterations': 9}))
    b10.main(['--out-root', str(tmp_path / 'w4'), '--paper-grid-root', str(paper), '--smoke'])
    root = tmp_path / 'w4' / 'B_instrumentation' / 'b10'
    rows = list(csv.DictReader(open(root / 'b10.csv')))
    assert [(r['weights'], r['pass']) for r in rows] == [('clenshaw_curtis', 'paper'), ('clenshaw_curtis', 'kmax')] * 2
    assert all(r['stopping_reason'] in ('target', 'iteration_cap') and r['eps_u'] for r in rows)
    assert {int(r['P']) for r in rows} == {300} and {r['ratio_target'] for r in rows} == {'5', '10'}
    for r in rows:
        d = root / r['run']
        assert {'iterations.csv', 'run.json', 'collocation.npz', 'solution.pt', 'hardware.json'} <= {p.name for p in d.iterdir()}
    cmp = list(csv.DictReader(open(root / 'b10_vs_paper_grid.csv')))
    assert len(cmp) == 2 and float(cmp[0]['paper_grid_eps_u']) == 2.9e-2
    # The total boundary weight (the advisor's reply on wave 3, item 2.8).
    assert {r['bc_weight_total'] for r in rows} == {'1'}
    assert (cmp[0]['b10_bc_weight_total'], cmp[0]['paper_grid_bc_weight_total']) == ('1', '4')
    assert '4 lambda_bc' in (root / 'README.md').read_text()
