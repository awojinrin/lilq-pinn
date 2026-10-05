"""Package 2, item 1 (Stage 2): the timing protocol, the plateau, the Bratu and
Kovasznay stages on small sizes, the same-day comparison and the jobs."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import baselines.square_chebyshev as sc
import experiments.p2_3_classical as p2

REPO = Path(__file__).resolve().parents[1]


def test_plateau_and_iteration_times():
    assert p2.plateau([1.0, 0.1, 0.0104, 0.01]) == 2              # within 5% of 0.01
    hist = [{'t_assemble_s': 1.0, 't_solve_s': 2.0}, {'t_assemble_s': 1.0, 't_solve_s': 2.0},
            {'t_assemble_s': 1.0, 't_solve_s': 2.0}, {'t_assemble_s': None, 't_solve_s': None}]
    t = p2.iteration_times(hist, k_stop=3, k_plateau=2)
    assert t == {'t_assemble_s': 3.0, 't_solve_s': 6.0, 't_plateau_s': 6.0, 't_stop_s': 9.0}
    assert p2.iteration_times(hist, k_stop=None, k_plateau=1)['t_stop_s'] == 9.0   # stopped at K_max


def test_timed_warms_up_and_takes_the_median():
    calls = []

    def run():
        calls.append(1)
        t = len(calls)
        return {'k_stop': 2, 'history': [{'t_assemble_s': 0.0, 't_solve_s': float(t)}] * 2}
    out = p2.timed(run, k_plateau=1, repeats=3)
    assert len(calls) == 4                                          # one warm-up, three timed
    assert out['t_stop_s'] == 6.0 and json.loads(out['t_stop_runs_s']) == [4.0, 6.0, 8.0]
    assert out['k_stop_clean'] == [2]


def _fake_reference(tmp_path):
    ref = tmp_path / 'reference'
    ref.mkdir()
    r = sc.newton('SQ-CGL', 16, diagnostics=False)
    u = r['space'].grid_values(r['beta'], p2.TEST_AXIS)
    for p in (p2.REF_P, p2.CHECK_P):
        np.savez_compressed(ref / f'bratu_ref_p{p}.npz', u=u)
    return ref


def test_bratu_stage_writes_the_rows_and_histories(tmp_path):
    ref = _fake_reference(tmp_path)
    rows, timing = p2.bratu_stage(tmp_path, reference_dir=ref, runs=[('SQ-CGL', 8), ('LS-hard-CGL-1.5', 6)],
                                  repeats=1, same_day=False)
    out = tmp_path / 'P2_3_classical' / 'bratu'
    with open(out / 'rows.csv') as f:
        got = list(csv.DictReader(f))
    assert [(r['method'], r['p']) for r in got] == [('SQ-CGL', '8'), ('LS-hard-CGL-1.5', '6')]
    for col in ('free_coeff', 'k_stop', 'k_plateau', 'err_ref48', 'err_ref64', 't_assemble_s', 't_solve_s',
                't_plateau_s', 't_stop_s', 'kappa', 'rank'):
        assert col in got[0]
    assert got[0]['free_coeff'] == '36' and got[1]['free_coeff'] == '36'
    assert float(got[0]['t_plateau_s']) <= float(got[0]['t_stop_s'])
    assert (out / 'history_SQ-CGL_p8.csv').exists() and (out / 'run.json').exists()
    assert json.loads((out / 'run.json').read_text())['K_max'] == sc.K_MAX


def test_kovasznay_stage_splits_the_unknowns(tmp_path):
    rows, _ = p2.kovasznay_stage(tmp_path, sizes=(8, 6), repeats=1, same_day=False)
    r, r6 = rows
    assert (r['P_velocity'], r['P_pressure'], r['free_coeff']) == (128, 36, 164)        # 2 x 64 + 36
    assert r['rank'] == 164 and r['stop'] == 'tolerance' and r['k_plateau'] <= r['k_stop']
    assert (r6['stop'], r6['k_stop']) == ('k_max', None)          # p_d = 6 does not converge in 60: recorded, not hidden
    assert (tmp_path / 'P2_3_classical' / 'kovasznay' / 'history_SQ-PNPN2_pd8.csv').exists()


def test_same_day_lilq_compares_with_package1(tmp_path):
    import experiments.clean_timing as ct
    ctd = tmp_path / 'B_instrumentation' / 'clean_timing'
    ctd.mkdir(parents=True)
    with open(ctd / 'clean_timing.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'quantity', 'clean_time_s'])
        w.writerows([['a', 'training_time', '1.0'], ['b', 'training_time', '1.0']])
    seq = iter([9, 1.1, 1.0, 1.2, 9, 2.0, 2.0, 2.0])
    runs = [ct.TimedRun(n, 'bratu', 'P', 'cpu', 60, lambda: {'training_time': next(seq), 'iterations': 3})
            for n in ('a', 'b')]
    rows = p2.same_day_lilq(runs, 'training_time', tmp_path, repeats=3)
    assert rows[0]['time_s'] == pytest.approx(1.1) and rows[0]['within_15pct'] is True
    assert rows[1]['time_s'] == 2.0 and rows[1]['within_15pct'] is False


def test_burgers_square_system():
    import baselines.square_burgers as sb
    s = sb.System(9)
    assert (s.n_pde, s.n_initial, s.n_boundary) == (sb.free_coefficients(9), 7, 18) and s.n == 81
    assert s.n_pde + s.n_initial + s.n_boundary == s.n                       # square
    assert np.all(s.points['pde'][1] > 0) and np.all(np.abs(s.points['pde'][0]) < 1)
    beta = np.random.default_rng(0).standard_normal(81) * 0.1
    A, f, R = s.assemble(beta)
    np.testing.assert_allclose(A @ beta - f, R, atol=1e-12)                 # the quasilinearization identity
    # the residual of a member of the space, by hand: u = T_1(x) T_1(2t - 1) = x (2t - 1)
    b = np.zeros(81)
    b[1 * 9 + 1] = 1.0
    (x, t) = s.points['pde']
    u, ux, ut = x * (2 * t - 1), 2 * t - 1, 2 * x
    np.testing.assert_allclose(s.assemble(b)[2][:s.n_pde], ut + u * ux, atol=1e-12)


def test_burgers_converges_spectrally():
    import baselines.square_burgers as sb
    fine = sb.newton(40, diagnostics=False)
    x, t = np.linspace(-1, 1, 41), np.linspace(0, 1, 21)
    ref = ((x, t), fine['system'].grid_values(fine['beta'], x, t))
    errs = [sb.newton(p, reference=ref)['err_final'] for p in (12, 17, 22)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 2e-3
    r = sb.newton(12, reference=ref)
    assert r['stop'] == 'tolerance' and r['rank'] == 144 and r['k_plateau'] <= r['k_stop']


def test_burgers_stage_writes_the_rows(tmp_path):
    import baselines.square_burgers as sb
    fine = sb.newton(30, diagnostics=False)
    x, t = np.linspace(-1, 1, 21), np.linspace(0, 1, 11)
    ref = tmp_path / 'reference'
    ref.mkdir()
    np.savez_compressed(ref / 'burgers_cole_hopf.npz', x=x, t=t, u=fine['system'].grid_values(fine['beta'], x, t))
    rows, _ = p2.burgers_stage(tmp_path, reference_dir=ref, sizes=(7, 12), repeats=1, same_day=False)
    assert [r['free_coeff'] for r in rows] == [30, 110]
    out = tmp_path / 'P2_3_classical' / 'burgers'
    with open(out / 'rows.csv') as f:
        got = list(csv.DictReader(f))
    for col in ('free_coeff', 'k_stop', 'k_plateau', 'err_ref', 't_assemble_s', 't_solve_s', 't_plateau_s',
                't_stop_s', 'kappa', 'rank'):
        assert col in got[0]
    assert (out / 'history_SQ-CGL_p12.csv').exists() and json.loads((out / 'run.json').read_text())['section'] == '4.3'


def test_item_1_jobs():
    for name in ('p2s2_classical_bratu', 'p2s2_classical_kovasznay', 'p2s2_classical_burgers'):
        text = (REPO / 'scripts' / 'cluster' / 'package2' / f'{name}.slurm').read_text()
        assert '# lilq-resources: timed-cpu' in text and '#SBATCH --time=01:00:00' in text
        assert '--package1 "$RESULTS/package1_v2.0.0/package1"' in text and 'LILQ_WAVE=p2s2' in text
