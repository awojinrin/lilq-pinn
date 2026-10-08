"""Package 3, batch 0: the CC-CGL grids, the sampling constants and checks K1, K2
(``lilq/certified.py``, ``experiments/p3_checks.py``)."""

import math

import numpy as np
import pytest

from lilq import certified as cf
import experiments.p3_checks as pc

BELTRAMI = pc.BELTRAMI_INTERVALS


@pytest.mark.parametrize('key', sorted(pc.BELTRAMI_EXPECTED))
def test_kronecker_reproduces_section_3_3(key):
    nv, npr, M = key
    k = cf.interior_constants_kronecker(M, 2 * nv + 1, 4)
    e = pc.BELTRAMI_EXPECTED[key]
    assert pc.printed(k['c1_squared'], e[0]) and pc.printed(k['c2_squared'], e[1]) and pc.printed(k['c2_over_c1'], e[2])


@pytest.mark.parametrize('key', sorted(pc.BL_EXPECTED))
def test_kronecker_reproduces_section_4_2(key):
    P, r, m = key
    g = cf.bl_grid(P, r, 0.4)
    assert g['M'] == pc.BL_M[(P, r)]
    k = cf.interior_constants_kronecker(g['M'], m * int(round(math.sqrt(P))) + 1, 2)
    assert pc.printed(k['c2_over_c1'], pc.BL_EXPECTED[key])         # None when not computable (C6)


def test_kronecker_reproduces_p2_16():
    for (bench, p, M), text in pc.P2_16.items():
        assert pc.printed(cf.interior_constants_kronecker(M, 2 * p + 1, 2)['c2_over_c1'], text)


def test_direct_equals_kronecker_in_2d_and_4d():
    g = cf.bl_grid(64, 10, 0.175)['blocks']['interior']
    d = cf.interior_constants_direct(g['points'], g['w'], ((0.0, 1.0), (0.0, 0.175)), 17)
    k = cf.interior_constants_kronecker(24, 17, 2)
    assert abs(d['c1'] - k['c1']) < 1e-12 and abs(d['c2'] - k['c2']) < 1e-12
    pts, w = cf.tensor_rule(BELTRAMI, (6,) * 4)                       # a small 4D case
    d = cf.interior_constants_direct(pts, w, BELTRAMI, 4)
    k = cf.interior_constants_kronecker(6, 4, 4)
    assert abs(d['c1'] - k['c1']) < 1e-12 and abs(d['c2'] - k['c2']) < 1e-12


def test_block_constants_reproduce_the_burgers_lines():
    for P, (c1, c2) in pc.BURGERS_LINES.items():
        p, nb = int(round(math.sqrt(P))), math.ceil(0.025 * 10 * P)
        t, w = cf.cgl_rule(0.0, 1.0, nb + 1)
        b = cf.block_constants(t[1:], w[1:], [(0.0, 1.0)], p, n_gl=400)
        assert pc.printed(b['c1'], c1) and pc.printed(b['c2'], c2) and b['dropped'] == 0
    x, w = cf.cgl_rule(-1.0, 1.0, 313)
    b = cf.block_constants(x, w, [(-1.0, 1.0)], 25, data=lambda z: -np.sin(np.pi * z[:, 0]), n_gl=400)
    assert pc.printed(b['c1'], '1.0000') and pc.printed(b['c2'], '1.0000')
    assert b['dropped'] == 1                                          # the datum is numerically in the trace space


def test_block_constants_on_a_face():
    """A 3D face with a full CC rule exact for the trace space's Gram: c1 = c2 = 1;
    a datum inside the span is dropped; one outside it is kept."""
    iv = [(-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)]
    pts, w = cf.tensor_rule(iv, (8, 8, 8))
    b = cf.block_constants(pts, w, iv, 3)                             # degree <= 4 products: exact on 8 nodes
    assert abs(b['c1'] - 1) < 1e-12 and abs(b['c2'] - 1) < 1e-12 and b['n_gl'] == 32
    b = cf.block_constants(pts, w, iv, 3, data=lambda z: z[:, 0] * z[:, 2])
    assert b['dropped'] == 1 and abs(b['c1'] - 1) < 1e-12
    b = cf.block_constants(pts, w, iv, 3, data=lambda z: np.exp(z[:, 0]) * np.sin(3 * z[:, 2]))
    assert b['dropped'] == 0 and b['c1'] < 1 <= b['c2'] + 1e-12


def test_beltrami_grid_rows_and_shares():
    """Section 3.2: shares 4/40 (faces) and 8/40 (initial slab); the row counts
    of Section 3.4 (4 equations inside, 3 velocity rows per face and slab point,
    N_p pins)."""
    for (nv, npr, Ms), rows in (((4, 5, (13, 16)), (160386, None)), ((6, 8, (16, 19)), (348168, 665331))):
        for M, expected in zip(Ms, rows):
            g = cf.beltrami_grid(M, BELTRAMI)
            b = g['blocks']
            assert g['boundary_measure'] == 40.0 and set(b) == {'interior', 'x-', 'x+', 'y-', 'y+', 'z-', 'z+', 'initial'}
            assert all(abs(b[n]['share'] - 0.1) < 1e-15 for n in b if n not in ('interior', 'initial'))
            assert abs(b['initial']['share'] - 0.2) < 1e-15 and b['interior']['share'] == 1.0
            assert all(abs(blk['w'].sum() - 1) < 1e-13 for blk in b.values())
            assert np.all(b['x+']['points'][:, 0] == 1.0) and np.all(b['initial']['points'][:, 3] == 0.0)
            n_rows = 4 * len(b['interior']['points']) + 3 * sum(len(b[n]['points']) for n in b if n != 'interior') + npr
            assert expected is None or n_rows == expected


def test_bl_grid_follows_the_p2_16_rule():
    g = cf.bl_grid(256, 10, 0.4)
    b = g['blocks']
    assert (g['M'], g['n_initial'], g['n_lateral']) == (48, 128, 64)
    assert len(b['left']['points']) == 64 and b['left']['points'][:, 1].min() > 0     # t = 0 left out
    assert abs(b['left']['points'][:, 1].max() - 0.4) < 1e-15 and np.all(b['right']['points'][:, 0] == 1.0)
    _, w = cf.cgl_rule(0.0, 0.4, 65)
    assert abs(b['left']['w'].sum() - (1 - w[0])) < 1e-15             # P2-16: the cut rule's weights as they are
    assert abs(b['initial']['share'] - 1 / 2.8) < 1e-15 and abs(b['left']['share'] - 0.4 / 2.8) < 1e-15


def test_cc_exactness_detects_a_wrong_rule():
    iv = [(0.0, 1.0), (0.0, 0.4)]
    pts, w = cf.tensor_rule(iv, (9, 9))
    assert cf.cc_exactness(pts, w, iv, (9, 9))['passed']
    assert not cf.cc_exactness(pts, np.full(len(w), 1 / len(w)), iv, (9, 9))['passed']       # equal weights
    assert not cf.cc_exactness(pts, w, iv, (11, 11))['passed']                               # beyond its degree


def test_k2_passes_on_every_grid():
    r = pc.k2()
    assert r['passed'] and r['n_rules'] == 6 * 8 + 2 * 12 * 4
