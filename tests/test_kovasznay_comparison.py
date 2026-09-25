"""Section 4.6 comparison table and figure, on synthetic run folders."""

import csv
import json

import experiments.kovasznay_comparison as kc


def _lilq(b_root, device, P, eps_u, t):
    d = b_root / f'kovasznay_P{P}_{device}_paper'
    d.mkdir(parents=True)
    (d / 'summary.json').write_text(json.dumps({'test_eps_u': eps_u, 'test_eps_v': eps_u, 'test_eps_p': eps_u,
                                                'test_eps_p_meanfree': eps_u, 'solve_time_total': t,
                                                'iterations': 6}))


def _baseline(a_root, stage, rep, seed, eps_series, peak, budget=3600):
    d = a_root / stage / f'{rep}_s{seed}'
    d.mkdir(parents=True)
    with open(d / 'log.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iter', 't_cum_s', 'eps_u', 'eps_v', 'eps_p_meanfree'])
        for i, (t, e) in enumerate(eps_series):
            w.writerow([100 * i, t, e, 2 * e, 3 * e])
    (d / 'run.json').write_text(json.dumps({'seed': seed, 'peak_gpu_bytes': peak, 'eps_u': eps_series[-1][1],
                                            'budget_s': budget, 'wall_s': eps_series[-1][0]}))


def test_table_and_figure(tmp_path):
    b, a = tmp_path / 'B_instrumentation', tmp_path / 'A_calibration'
    for P, e, t in zip(kc.KOVASZNAY_P, [1e-1, 1e-2, 1e-3, 1e-6, 1e-9], [0.1, 0.2, 0.7, 2, 5]):
        _lilq(b, 'cuda', P, e, t)
    (a / 'full').mkdir(parents=True)
    (a / 'full' / 'F1_representative.json').write_text(json.dumps({'representative': 'F1_07'}))
    _baseline(a, 'full', 'F1_07', 0, [(1, 0.5), (10, 5e-2), (100, 5e-4)], 1000)
    _baseline(a, 'full', 'F1_07', 1, [(1, 0.5), (20, 5e-3), (200, 2e-4)], 2000)

    out = kc.build(tmp_path)
    rows = list(csv.DictReader(open(out / 'results' / 'kovasznay_comparison.csv')))
    f1 = next(r for r in rows if r['family'] == 'F1' and r['device'] == 'gpu')
    assert float(f1['eps_u_median']) == (5e-4 + 2e-4) / 2 and float(f1['eps_u_min']) == 2e-4
    assert float(f1['eps_v_max']) == 1e-3 and f1['peak_gpu_bytes_max'] == '2000'
    assert f1['tta_P75_reached'] == '2/2' and float(f1['tta_P75_time_median']) == 15.0   # 1e-1: t = 10, 20
    assert f1['tta_P675_reached'] == '2/2' and float(f1['tta_P675_time_median']) == 150.0
    assert f1['tta_P1200_reached'] == '0/2' and f1['tta_P1200_time_median'] == 'not reached'
    assert sum(r['family'] == 'LiL-Q' for r in rows) == 5
    tta = list(csv.DictReader(open(out / 'results' / 'time_to_accuracy.csv')))
    assert len(tta) == 2 * 5
    assert (out / 'figures' / 'eps_u_vs_time.pdf').stat().st_size > 1000


def test_rows_past_the_budget_are_ignored(tmp_path):
    b, a = tmp_path / 'B_instrumentation', tmp_path / 'A_calibration'
    _lilq(b, 'cuda', 75, 1e-1, 0.1)
    (a / 'full').mkdir(parents=True)
    (a / 'full' / 'F2_representative.json').write_text(json.dumps({'representative': 'F2_03'}))
    _baseline(a, 'full', 'F2_03', 0, [(10, 0.5), (590, 0.2), (605, 1e-3)], None, budget=600)
    rows = list(csv.DictReader(open(kc.build(tmp_path) / 'results' / 'kovasznay_comparison.csv')))
    f2 = next(r for r in rows if r['family'] == 'F2')
    assert float(f2['eps_u_median']) == 0.2            # the 605 s row and the overrunning final value excluded
