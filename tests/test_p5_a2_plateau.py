"""Package 5, Addendum 2 (``experiments/p5_a2_plateau.py``) and ``lilq.xnorm.relative_updates``."""

import csv
import json

import numpy as np
import pytest

import experiments.p5_a2_plateau as a2
from experiments import p5_1_weights as p51
from lilq.xnorm import relative_updates


def test_relative_updates_by_hand():
    one = {'u': np.array([[1.0]])}
    betas = np.array([[0.0], [1.0], [3.0]])
    assert np.allclose(relative_updates([1.0], one, betas, ('u',)), [1.0, 2.0 / 3.0])
    # a lifting of 1 enters ||u^(k+1)|| and cancels in the difference: u = 1, 2, 4
    assert np.allclose(relative_updates([1.0], one, betas, ('u',), lift={'u': np.array([1.0])}), [1.0 / 2.0, 2.0 / 4.0])


def test_relative_updates_sum_the_components_with_the_weights():
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 1.5, 7)
    mats = {'u': rng.standard_normal((7, 3)), 'u_x': rng.standard_normal((7, 3))}
    betas = rng.standard_normal((4, 3))
    got = relative_updates(w, mats, betas, ('u', 'u_x'))
    for k in range(3):
        num = sum(np.sum(w * (mats[c] @ (betas[k + 1] - betas[k])) ** 2) for c in mats)
        den = sum(np.sum(w * (mats[c] @ betas[k + 1]) ** 2) for c in mats)
        assert got[k] == pytest.approx(np.sqrt(num / den), rel=1e-12)


def test_pb_space_from_the_folder_name():
    assert a2.pb_space('p22_seed0') == 22
    rf = a2.pb_space('P100_sigma2_draw1')
    assert (rf.P, rf.sigma, rf.draw) == (100, 2.0, 1)
    with pytest.raises(ValueError):
        a2.pb_space('P100_lam10')


def test_item1_updates_in_place(tmp_path):
    """One item 1 run folder per kind of space: scalar H2 (Bratu) and lifted H^{2,1} (BL sine)."""
    rng = np.random.default_rng(1)
    for family, P in (('bratu', 25), ('bl_sine_gravity', 64)):
        d = tmp_path / p51.ITEM / family / f'P{P}_lam10000'
        d.mkdir(parents=True)
        (d / 'run.json').write_text(json.dumps({'package5': {'P': P, 'n_gl': 12}}))
        np.save(d / 'betas.npy', rng.standard_normal((4, P)))
    assert a2.item1_updates(tmp_path) == 2
    for family, P in (('bratu', 25), ('bl_sine_gravity', 64)):
        d = tmp_path / p51.ITEM / family / f'P{P}_lam10000'
        rows = list(csv.DictReader(open(d / 'updates_by_iterate.csv')))
        assert [r['k'] for r in rows] == ['1', '2', '3'] and list(rows[0]) == list(a2.UPDATE_COLUMNS)
        x, y, w = p51.quadrature(family, 12)
        p = int(round(P ** 0.5))
        if family == 'bratu':
            mats, lift, comps = p51.scalar_mats(family, p, x, y), None, p51.H2
        else:
            mats, lift = p51.bl_mats(p51.p32.TrialSpace('sine', p, p51.p32.CASES['gravity']['T']), x, y)
            comps = p51.H21
        want = relative_updates(w, mats, np.load(d / 'betas.npy'), comps, lift)
        assert [float(r['update_X']) for r in rows] == list(want)


# ---------------------------------------------------------------- the runs and their summary

# The delivered P2_9_bases/bl_gravity_paper_P1024_ic, iterates 0..15: chi, norm_Rlin_h, norm_R_h, eps_ref
P2_9 = [
    (2.3642456267876657, 2.1146492246081507, 14.7962602880674, 0.4287641913205996),
    (7.2756485220355, 0.8619744741456419, 4.751482709422786, 0.4743273460100582),
    (1.6207302832049, 0.8737470838290724, 6.254620000440311, 0.3384424213864755),
    (1.0783898821246092, 0.57804618323598, 1.4482855281143456, 0.3574299784321775),
    (1.3770012419813988, 0.5024662435513085, 0.8013456745188465, 0.31261421643699117),
    (1.7756182603993267, 0.42210060157801843, 0.8108015265318104, 0.26181976838494303),
    (2.1265673896429336, 0.33366438387624797, 0.8174924931957649, 0.20744324792277072),
    (2.8302027816589126, 0.24038934559182565, 0.7485510930887826, 0.1505183690161793),
    (2.437705695455488, 0.15694627467499048, 0.695305991692899, 0.09136909754160714),
    (1.1542420452143096, 0.11150483700770615, 0.3966548956291765, 0.04364318916815642),
    (0.0871178717066943, 0.10315155290384742, 0.16496521847189458, 0.016740635417011977),
    (0.007918013495896426, 0.10186966732004009, 0.10268036127798968, 0.012243301766132725),
    (0.0017652947475601556, 0.10154075320644869, 0.10165657547562987, 0.011810415423781188),
    (0.0004755993175548891, 0.10144944451375223, 0.10148128104647818, 0.011628700569820311),
    (0.0001330083755228486, 0.10142353161679533, 0.10143254965016905, 0.011552961363604348),
    (3.798115417754373e-05, 0.10141615151388395, 0.1014187212469232, 0.011516493342880343),
]


def fake_run(d, log, ref_err=1e-6, eX=None):
    d.mkdir(parents=True)
    with open(d / 'iterations.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['k', 'chi', 'norm_Rlin_h', 'norm_R_h', 'eps_ref'])
        for k, (chi, rl, r, e) in enumerate(log):
            last = k == len(log) - 1
            w.writerow([k, '' if last else chi, '' if last else rl, r, e])
    with open(d / 'errors_by_iterate.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['k', 'eL2', 'eX'])
        for k, row in enumerate(log):
            w.writerow([k, row[3], '' if eX is None else eX])
    (d / 'run.json').write_text(json.dumps({'package5': {'part': 'D', 'space': 'trig', 'P': 1024, 'nu': 0.1,
                                                         'lam': 10.0, 'reference_error': ref_err}}))
    return d


def test_eighteen_runs():
    rs = a2.runs()
    assert len(rs) == 18 and len({a2.run_name(*r) for r in rs}) == 18
    assert [sum(r[0] == p for r in rs) for p in 'ABCD'] == [6, 8, 2, 2]
    assert a2.run_name('B', 'cheb', 576, 0.05, 1e4) == 'B_cheb_P576_nu0.05_lam10000'


def test_stretches():
    R = [1.0, 0.5, 0.49, 0.48, 0.3, 0.29, 0.285, 0.1]
    assert a2.stretches(R, 7) == [(1, 3), (4, 6)]
    assert a2.stretches(R, 5) == [(1, 3), (4, 5)]
    assert a2.stretches([1.0, 0.99, 0.98], 2) == [(0, 2)]


def test_summary_row_on_the_delivered_run(tmp_path):
    """His description of the run: the rule stops at k = 13 with an error of 0.0116 (the returned iterate 14).
    Before the stop, two stretches of two steps tie: 4..6 (the plateau) and 11..13; the earliest is reported."""
    d = fake_run(tmp_path / 'D_trig', P2_9)
    row, found = a2.summary_row(d, {'delta_L2': 0.005})
    assert (row['rule_step'], row['returned_iterate']) == (13, 14)
    assert row['eL2'] == pytest.approx(0.011552961363604348) and round(row['eL2'], 4) == 0.0116
    assert row['eL2_over_delta'] == pytest.approx(0.011552961363604348 / 0.005)
    assert (row['stretch_first'], row['stretch_last'], row['stretch_steps'], row['n_longest']) == (4, 6, 2, 2)
    assert row['R_first'] == pytest.approx(0.8013456745188465) and row['Rlin_last'] == pytest.approx(0.33366438387624797)
    assert [(s['first'], s['last']) for s in found] == [(4, 6), (11, 13)]
    assert 'eX' not in row


def test_summary_row_withholds_errors_below_three_reference_errors(tmp_path):
    d = fake_run(tmp_path / 'B', P2_9, ref_err=0.0116 / 2.9)
    row, found = a2.summary_row(d, {'delta_L2': 0.005})
    assert row['error_reported'] is False and row['eL2'] is None and row['eL2_first'] is None
    assert row['R_first'] is not None and all(s['eL2_first'] is None for s in found)


def test_summary_row_gates_the_x_ratio(tmp_path):
    d = fake_run(tmp_path / 'A', P2_9, eX=0.2)
    row, _ = a2.summary_row(d, {'delta_L2': 0.005, 'delta_X': 0.05}, gate=0.01)
    assert row['x_ratio_reported'] and row['eX_over_delta'] == pytest.approx(4.0)
    row, _ = a2.summary_row(d, {'delta_L2': 0.005, 'delta_X': 0.05}, gate=0.02)
    assert not row['x_ratio_reported'] and row['eX_over_delta'] is None


def test_k2(tmp_path):
    old = fake_run(tmp_path / 'old', P2_9)
    same = fake_run(tmp_path / 'same', P2_9)
    res = a2.k2(same, old)
    assert res['pass'] and res['max_rel_diff_every_iterate'] == 0.0 and res['delivered_first_stall'] == 13
    moved = [tuple(v * (1 + 1e-6) if k == 9 else v for v in r) for k, r in enumerate(P2_9)]
    res = a2.k2(fake_run(tmp_path / 'moved', moved), old)
    assert not res['pass'] and res['first_iterate_over_tolerance'] == 9
