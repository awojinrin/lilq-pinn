"""Package 2, item 3 (P2-16): the certified grids, their weights, the
quasilinear system, the constants, check C4 and a small run end to end."""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

import experiments.p2_16_certified as m

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('bench', ['bratu', 'burgers'])
def test_grid_counts_and_weights(bench):
    P, r = 100, 10
    g = m.grid(bench, P, r)
    b = g['blocks']
    assert g['M'] == math.ceil(math.sqrt((0.85 if bench == 'bratu' else 0.9) * r * P))
    assert np.isclose(b['interior'][2].sum(), 1.0)                        # CC weights on |Omega| = 1
    aux = sum(v[2].sum() for k, v in b.items() if k != 'interior')
    # Bratu: four edges, the whole boundary. Burgers: three of the four sides (t = 1 carries none),
    # less the t = 0 end weight the lateral lines drop
    if bench == 'bratu':
        assert np.isclose(aux, m.LAM_AUX)
    else:
        w0 = m.cc_weights(math.ceil(0.025 * r * P) + 1)[0]
        assert np.isclose(aux, m.LAM_AUX * (4 - 2 * w0) / 6)
    assert g['N'] == sum(len(v[0]) for v in b.values()) and g['ratio_actual'] >= r
    if bench == 'burgers':
        assert b['left'][1].min() > 0 and np.isclose(b['left'][1].max(), 1.0)  # (0, 1]
        assert len(b['initial'][0]) == math.ceil(0.05 * r * P) and len(b['left'][0]) == math.ceil(0.025 * r * P)
    else:
        assert len(b['bottom'][0]) == math.ceil(0.15 * r * P / 4)


@pytest.mark.parametrize('bench', ['bratu', 'burgers'])
def test_quasilinear_system_is_consistent(bench):
    """A beta - f is the nonlinear residual at beta (the B2 identity), and the
    derivative matrices match finite differences of the values."""
    p = 6
    rows = m.Rows(bench, p, m.grid(bench, p * p, 5)['blocks'])
    beta = np.random.default_rng(0).standard_normal(p * p) * 0.1
    A, f, R = m.assemble(rows, beta)
    np.testing.assert_allclose(A @ beta - f, R, atol=1e-12)
    dom = m.DOMAIN[bench][0]
    z = np.linspace(*dom, 7)
    V, V1, V2 = m.cheb(z, p, *dom)
    h = 1e-5
    np.testing.assert_allclose(V1, (m.cheb(z + h, p, *dom)[0] - m.cheb(z - h, p, *dom)[0]) / (2 * h), atol=1e-6)
    np.testing.assert_allclose(V2, (m.cheb(z + h, p, *dom)[1] - m.cheb(z - h, p, *dom)[1]) / (2 * h), atol=1e-5)


def test_constants_and_c6():
    g = m.grid('bratu', 25, 20)
    c = m.constants('bratu', 25, 20, g, 2)
    # degree 2 p_d = 10 per direction on 21 CGL points: the CC Gram matrix is exact, c1 = c2 = 1
    assert c['computable'] and abs(c['c1'] - 1) < 1e-12 and abs(c['c2'] - 1) < 1e-12
    assert not m.constants('bratu', 25, 5, m.grid('bratu', 25, 5), 2)['computable']   # dim 121 >= 121 points


@pytest.mark.parametrize('bench', ['bratu', 'burgers'])
def test_check_c4_and_its_literal_form_fails(bench):
    g = m.grid(bench, 100, 5)
    assert m.check_cc_exactness(bench, g)['passed']
    # the literal wording (squared norms of degree M - 1) is not exact on M points
    x, y, w2 = g['blocks']['interior']
    (ax, bx), (ay, by) = m.DOMAIN[bench]
    q = g['M'] - 1
    h = m.legendre_orth(x, ax, bx, q)[:, q] * m.legendre_orth(y, ay, by, q)[:, q]
    assert abs(np.sum(w2 * h ** 2) - 1) > 1e-3


def test_delta_P_is_zero_for_a_member_of_the_space():
    x, y = np.linspace(0, 1, 31), np.linspace(0, 1, 29)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = (2 * X - 1) ** 3 * (2 * Y - 1) - 0.5
    assert m.delta_P('bratu', 5, ((x, y), u)) < 1e-13


def test_small_run_end_to_end(tmp_path):
    x, y = np.linspace(0, 1, 41), np.linspace(0, 1, 41)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = 1.2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * 0.115 * 4       # a rough stand-in for the Bratu solution
    ref = ((x, y), u)
    g, log, rho = m.run('bratu', 25, 5, ref, tmp_path, k_max=6)
    d = tmp_path / 'bratu_P25_NP5'
    for f in ('run.json', 'iterations.csv', 'rho_r.csv', 'collocation.npz'):
        assert (d / f).exists()
    run = json.loads((d / 'run.json').read_text())
    assert run['K_max'] == 6 and run['N_total'] == g['N'] and run['b2_check']['max_rel_err_over_run'] < 1e-12
    with open(d / 'iterations.csv') as fh:
        assert [int(r['k']) for r in csv.DictReader(fh)] == list(range(7))
    assert all(r['rho_r'] >= 1 - 1e-9 for r in rho)
    t = m.terminal_row('bratu', 25, 5, g, log, rho, m.delta_P('bratu', 5, ref))
    assert t['eps_ref_60'] == log[-1]['eps_ref'] and t['loss_60'] > 0


def test_item_3_job():
    text = (REPO / 'scripts' / 'cluster' / 'package2' / 'p2s2_certified.slurm').read_text()
    assert '# lilq-resources: cpu' in text and 'p2_16_certified.py' in text
