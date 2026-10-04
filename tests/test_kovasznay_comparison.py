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
    e = eps_series[-1][1]                  # the final errors, as a real run.json records them
    (d / 'run.json').write_text(json.dumps({'seed': seed, 'peak_gpu_bytes': peak, 'eps_u': e, 'eps_v': 2 * e,
                                            'eps_p_meanfree': 3 * e, 'budget_s': budget,
                                            'wall_s': eps_series[-1][0]}))


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
    assert float(f1['eps_u_best_logged_median']) == (5e-4 + 2e-4) / 2 and f1['n_final_past_budget'] == '0'
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
    assert float(f2['eps_u_best_logged_median']) == 0.2   # the 605 s row and the overrunning final value excluded
    assert float(f2['eps_u_median']) == 1e-3              # the final value, as the manuscript quotes
    assert f2['n_final_past_budget'] == '1' and float(f2['max_final_overrun_s']) == 5.0


def test_medians_are_of_the_final_errors_not_the_best_logged(tmp_path):
    """The advisor's reply to wave 4, item 2(a): F1's eps_u_median was the
    median of each seed's minimum logged eps_u (4.11e-5 for F1_18 on the
    GPU) where the manuscript quotes the median of the final errors in
    run.json (4.19e-5). Here a seed whose last logged value is above its
    best: the median is of the finals, the best logged is beside it."""
    b, a = tmp_path / 'B_instrumentation', tmp_path / 'A_calibration'
    _lilq(b, 'cuda', 75, 1e-1, 0.1)
    (a / 'full').mkdir(parents=True)
    (a / 'full' / 'F1_representative.json').write_text(json.dumps({'representative': 'F1_18'}))
    _baseline(a, 'full', 'F1_18', 0, [(1, 0.5), (100, 2.0e-5), (3600, 2.5e-5)], None)
    _baseline(a, 'full', 'F1_18', 1, [(1, 0.5), (100, 4.0e-5), (3600, 4.0e-5)], None)
    _baseline(a, 'full', 'F1_18', 2, [(1, 0.5), (100, 4.1e-5), (3600, 4.2e-5)], None)
    rows = list(csv.DictReader(open(kc.build(tmp_path) / 'results' / 'kovasznay_comparison.csv')))
    f1 = next(r for r in rows if r['family'] == 'F1')
    assert float(f1['eps_u_median']) == 4.0e-5 and float(f1['eps_u_max']) == 4.2e-5    # finals 2.5, 4.0, 4.2
    assert float(f1['eps_u_best_logged_median']) == 4.0e-5 and float(f1['eps_u_best_logged_min']) == 2.0e-5
    assert float(f1['eps_u_min']) == 2.5e-5


def test_lilq_times_come_from_the_clean_timing_runs(tmp_path):
    """The rule for every quoted LiL-Q time (the advisor's reply to wave 2,
    Section 3): the clean time where there is one, the logged one beside it."""
    b = tmp_path / 'B_instrumentation'
    _lilq(b, 'cuda', 300, 1e-2, 1.42)
    _lilq(b, 'cuda', 1200, 1e-6, 1.10)
    (b / 'clean_timing').mkdir(parents=True)
    with open(b / 'clean_timing' / 'clean_timing.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'quantity', 'clean_time_s'])
        w.writeheader()
        w.writerow({'run': 'kovasznay_P300_cuda_paper', 'quantity': 'solve_time_total', 'clean_time_s': 0.061})
    lil = kc.load_lilq(b, 'cuda')
    assert (lil[300]['time_s'], lil[300]['time_s_logged'], lil[300]['time_source']) == (0.061, 1.42, 'clean')
    assert (lil[1200]['time_s'], lil[1200]['time_source']) == (1.10, 'logged')
