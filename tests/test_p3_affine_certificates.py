"""Package 3, item 3: certificates for the affine problems
(``experiments/p3_3_affine_certificates.py``, ``lilq.certified``'s R-factor helpers)."""

import json

import numpy as np
import pytest
import scipy.linalg

import experiments.p3_3_affine_certificates as m
from experiments.p3_2_bl_certified import section6_constants
from lilq import certified as cf


def test_tsqr_is_the_direct_r_factor():
    A = np.random.default_rng(0).standard_normal((700, 12))
    R = cf.tsqr([A[:100], A[100:450], A[450:]])
    R0 = scipy.linalg.qr(A, mode='r')[0][:12]
    assert np.allclose(np.abs(R), np.abs(R0), atol=1e-12)


def test_section6_from_r_equals_the_dense_steps():
    """Elasticity, compatible, P = 200: the R-factor route gives what the
    dense column-pivoted QR of B_Y gives (item 2's ``section6_constants``),
    and rho_r agrees with a dense least squares."""
    config, physics = m.elasticity_setup('compatible', 10)
    bu, bv = m.elasticity_bases(config)
    points, w2 = m.elasticity_collocation(config, physics)
    rows = m.elasticity_rows(bu, bv, physics, points, w2)
    keep = [n for n in m.ELAST_BLOCKS if n not in m.zero_blocks(rows)]
    B_h = np.vstack([np.column_stack(rows[n]) for n in keep])
    qp, qw = m.elasticity_quadrature(config, w2, 48)
    quad = m.elasticity_rows(bu, bv, physics, qp, qw)
    B_Y = np.vstack([np.column_stack(quad[n]) for n in keep])
    dense, viaR = section6_constants(B_h, B_Y), cf.section6_from_R(B_h, cf.tsqr([B_Y[:500], B_Y[500:]]))
    for k in ('c1', 'c2'):
        assert viaR[k] == pytest.approx(dense[k], rel=1e-10)
    assert viaR['dropped'] == dense['dropped']
    beta = scipy.linalg.lstsq(B_h[:, :-1], B_h[:, -1], lapack_driver='gelsy')[0]
    rho = cf.rho_from_R(cf.tsqr([B_Y]), beta, len(B_Y))
    best = np.linalg.norm(B_Y[:, :-1] @ scipy.linalg.lstsq(B_Y[:, :-1], B_Y[:, -1], lapack_driver='gelsy')[0] - B_Y[:, -1])
    assert rho['rho_r_numerator'] == pytest.approx(np.linalg.norm(B_Y[:, :-1] @ beta - B_Y[:, -1]), rel=1e-10)
    assert rho['rho_r_denominator'] == pytest.approx(best, rel=1e-10)


@pytest.mark.parametrize('solution, zeros, dropped, round_off', [
    ('paper', ['bot_ux', 'top_ux', 'left_sxx', 'left_uy', 'right_sxx', 'right_uy'], 1, True),
    ('compatible', ['bot_ux', 'top_ux', 'left_sxx', 'left_uy', 'right_sxx', 'right_uy'], 0, False),
    ('specified', ['bot_ux', 'top_ux', 'left_uy', 'right_uy'], 0, False)])
def test_elasticity_certificate(solution, zeros, dropped, round_off):
    """The assembly is the paper's, bit for bit. The zero blocks are the
    vanishing Dirichlet edges, plus the lateral sigma_xx where the traction is
    zero ('specified' keeps them: zero in A, not in b). The paper's f is in the
    span (dropped), so its rho_r is round-off."""
    r = m.run_elasticity(solution, 5)
    assert r['assembly_matches_paper_bitwise'] and r['zero_blocks_excluded'] == zeros
    assert r['dropped'] == dropped and r['round_off'] is round_off and r['dropped_A_only'] == 0


def test_darcy_with_one_point_per_cell_is_the_collocation_set():
    """n_q = 1: the Gauss point is the cell centre with weight 1, so the
    Y-system is the collocation system: c1 = c2 = 1 and rho_r = 1."""
    r = m.run_darcy('S1', n_q=1, orders=4)
    assert r['assembly_matches_paper_bitwise']
    assert r['c1'] == pytest.approx(1, abs=1e-10) and r['c2'] == pytest.approx(1, abs=1e-10)
    assert r['rho_r'] == pytest.approx(1, abs=1e-10) and not r['round_off']


def test_captured_solve_leaves_lstsq_as_it_was():
    real = scipy.linalg.lstsq
    A, b = np.eye(3), np.ones(3)
    with m.captured_solve() as cap:
        x = scipy.linalg.lstsq(A, b)[0]
        scipy.linalg.lstsq(2 * A, b)                              # only the first call is kept
    assert scipy.linalg.lstsq is real and np.array_equal(cap['A'], A) and np.array_equal(cap['x'], x)


def test_logged_residuals_and_summary(tmp_path):
    p1 = tmp_path / 'package1' / 'B_instrumentation' / 'elasticity_P50_cpu_paper'
    p1.mkdir(parents=True)
    (p1 / 'iterations.csv').write_text('k,norm_Rlin_h\n0,1.25e-14\n1,\n')
    p210 = tmp_path / 'p2_10' / 'compatible' / 'P50'
    p210.mkdir(parents=True)
    (p210 / 'iterations.csv').write_text('k,norm_Rlin_h\n0,0.5\n1,\n')
    assert m.logged_residual('elasticity', 'paper', 50, tmp_path / 'package1')[0] == 1.25e-14
    assert m.logged_residual('elasticity', 'compatible', 50, None, tmp_path / 'p2_10')[0] == 0.5
    runs = tmp_path / 'out' / m.ITEM / 'runs'
    runs.mkdir(parents=True)
    base = {'problem': 'elasticity', 'case': 'compatible', 'P': 50, 'N': 530, 'N_used': 290, 'c1': 0.9, 'c2': 1.5,
            'c2_over_c1': 1.6, 'dropped': 0, 'rho_r': 1.002, 'round_off': False, 'resolve_residual': 0.5,
            'assembly_matches_paper_bitwise': True, 'quadrature': 'Gauss-Legendre 96', 'tag': ''}
    (runs / 'a.json').write_text(json.dumps(base))
    (runs / 'b.json').write_text(json.dumps({**base, 'c1': 0.9 * (1 + 2e-4), 'rho_r': 1.002 * (1 + 5e-4), 'tag': '_k5'}))
    main, checks = m.summarize(tmp_path / 'out', None, tmp_path / 'p2_10')
    assert main[0]['resolve_identical'] is True and checks['assembly']['all_bitwise']
    assert checks['K5']['passed'] and len(checks['K5']['runs']) == 1


def test_a_failed_configuration_does_not_stop_the_others(tmp_path, monkeypatch):
    calls = []

    def fake(problem, case, N, out_root, threads, n_quad=None, tag=''):
        calls.append((problem, case, N, threads, n_quad, tag))
        if (problem, case) == ('darcy', 'S2'):
            raise RuntimeError('darcy S2 N=None exited 1\nMemoryError')
    monkeypatch.setattr(m, '_run_process', fake)
    failed = m.run_all(tmp_path)
    assert len(calls) == 15 + 4 + 3 + 1 and failed == ['darcy S2 N=None exited 1']
    assert calls[-1] == ('darcy', 'SPE10', None, m.THREADS[('darcy', None)], m.K5_QUAD['darcy'], '_k5')
    assert {c[3] for c in calls if c[1] in ('compatible', 'specified')} == {m.THREADS[('elasticity', 'specified')]}
