"""Package 5, item 3: the OpenMP library of Package 1's Component A baselines (experiments/p5_3_openmp_kovasznay.py)."""
import csv
import json

import pytest

import experiments.p5_3_openmp_kovasznay as k
import lilq.provenance as prov

MODULE_LIB = '/sw/eb/sw/GCCcore/13.3.0/lib64/libgomp.so.1.0.0'
F1_HEAD = ['iter', 'phase', 't_cum_s', 'loss_total']
F2_HEAD = ['iter', 'phase', 't_cum_s', 'loss_total', 'n_evals']


def _csv(path, head, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(rows)
    return path


def test_the_f1_measures_take_adam_rows_100_to_200_and_lbfgs_from_its_second_row(tmp_path):
    log = _csv(tmp_path / 'f1.csv', F1_HEAD, [[0, 'adam', 5.0, 1], [100, 'adam', 37.0, 1], [200, 'adam', 70.0, 1],
                                              [210, 'lbfgs', 80.0, 1], [220, 'lbfgs', 83.0, 1],
                                              [240, 'lbfgs', 89.0, 1]])
    m = k.f1_measures(log)
    assert m['F1_adam_s_per_iteration'] == (pytest.approx(0.33), 1)
    assert m['F1_lbfgs_s_per_iteration'] == (pytest.approx(0.3), 1)        # 220 -> 240: 6 s over 20
    short = k.f1_measures(_csv(tmp_path / 's.csv', F1_HEAD, [[0, 'adam', 5.0, 1], [100, 'adam', 37.0, 1]]))
    assert short == {'F1_adam_s_per_iteration': (None, 0), 'F1_lbfgs_s_per_iteration': (None, 0)}


def test_the_f2_measure_is_seconds_per_evaluation_from_the_second_row(tmp_path):
    log = _csv(tmp_path / 'f2.csv', F2_HEAD, [[1, 'lm', 15.0, 1, 2], [2, 'lm', 26.0, 1, 3], [3, 'lm', 32.0, 1, 4],
                                              [4, 'lm', 44.0, 1, 6]])
    assert k.f2_measures(log) == {'F2_s_per_evaluation': (pytest.approx(6.0), 2)}
    assert k.f2_measures(_csv(tmp_path / 's.csv', F2_HEAD, [[1, 'lm', 15.0, 1, 2], [2, 'lm', 26.0, 1, 3]])) == \
        {'F2_s_per_evaluation': (None, 0)}


@pytest.fixture
def tiny(monkeypatch):
    """Small networks and budgets: the whole process in a few seconds."""
    monkeypatch.setitem(k.CONFIGS, 'F1', k.CONFIGS['F1'] | {'width': 8, 'depth': 2, 'm': 4, 'n_int': 64})
    monkeypatch.setitem(k.CONFIGS, 'F2', k.CONFIGS['F2'] | {'width': 8, 'depth': 2, 'm': 4, 'n_int': 64})
    monkeypatch.setattr(k, 'F1_BUDGET_S', 8.0)
    monkeypatch.setattr(k, 'F2_BUDGET_S', 2.0)


def test_a_process_records_its_library_runs_both_families_and_writes_its_timing(tiny, tmp_path):
    out = tmp_path / 'out'
    k.main(['run', '--first', 'numpy', '--round', '0', '--out', str(out)])
    root = out / k.ITEM / k.SUB
    proc = json.loads((root / 'numpy-first' / 'process_r0.json').read_text())
    assert proc['first'] == 'numpy' and proc['expected_library'] == 'module' and 'openmp_libraries' in proc
    f1 = json.loads((root / 'numpy-first' / 'r0' / 'F1' / 'run.json').read_text())
    assert f1['adam_iters'] == k.F1_ADAM_ITERATIONS and f1['lbfgs_iters'] > 0 and f1['device'] == 'cpu'
    f2 = json.loads((root / 'numpy-first' / 'r0' / 'F2' / 'run.json').read_text())
    assert f2['end_reason'] == 'budget' and f2['n_evals'] > 0
    with open(root / 'timing_numpy-first_r0.csv') as fh:
        rows = list(csv.DictReader(fh))
    assert [r['measure'] for r in rows] == list(k.MEASURES)
    assert rows[0]['value'] and rows[1]['value'] and int(rows[1]['intervals']) >= 1
    assert all(r['library'] == proc['library'] for r in rows)
    assert (root / 'provenance').is_dir()


def test_require_library_stops_before_any_run(tmp_path, monkeypatch):
    monkeypatch.setattr(prov, 'openmp_libraries', lambda: [MODULE_LIB])
    with pytest.raises(SystemExit, match='not the torch one'):
        k.main(['run', '--first', 'torch', '--round', '0', '--out', str(tmp_path / 'out'), '--require-library'])
    root = tmp_path / 'out' / k.ITEM / k.SUB
    assert (root / 'torch-first' / 'process_r0.json').exists()
    assert not (root / 'torch-first' / 'r0').exists() and not list(root.glob('timing_*'))


def _timing(root, first, rd, values, library):
    _csv(root / f'timing_{first}-first_r{rd}.csv', k.TIMING_COLUMNS,
         [[first, rd, library, m, repr(v), 3, 'budget'] for m, v in zip(k.MEASURES, values)])


def test_the_summary_applies_the_cutoff_and_the_overlap_and_compares_with_package1(tmp_path):
    out = tmp_path / 'out'
    root = out / k.ITEM / k.SUB
    a = tmp_path / 'A_calibration'
    # Package 1's seed-0 logs (the reference) and one more seed.
    for s, scale in ((0, 1.0), (1, 1.1)):
        _csv(a / 'full_cpu' / f'F1_18_s{s}' / 'log.csv', F1_HEAD,
             [[0, 'adam', 5.0, 2.0], [100, 'adam', 5 + 40 * scale, 1.0], [200, 'adam', 5 + 80 * scale, 0.5],
              [5010, 'lbfgs', 1600.0, 0.1], [5020, 'lbfgs', 1603.0, 0.09], [5040, 'lbfgs', 1609.0, 0.08]])
        _csv(a / 'full_cpu' / f'F2_05_s{s}' / 'log.csv', F2_HEAD,
             [[1, 'lm', 15.0, 0.9, 2], [2, 'lm', 26.0, 0.4, 3], [3, 'lm', 38.0, 0.2, 5]])
    for first, rd, vals, lib, f1loss in (('numpy', 0, (0.30, 0.30, 5.0), 'module', 0.5 * (1 + 1e-3)),
                                         ('torch', 0, (0.40, 0.31, 5.0), 'torch', 0.5),
                                         ('numpy', 1, (0.31, 0.32, 5.1), 'module', 0.5),
                                         ('torch', 1, (0.41, 0.30, 5.05), 'torch', 0.5 * (1 + 1e-9))):
        _timing(root, first, rd, vals, lib)
        d = root / f'{first}-first' / f'r{rd}'
        _csv(d / 'F1' / 'log.csv', F1_HEAD, [[0, 'adam', 5.0, 2.0], [100, 'adam', 45.0, 1.0],
                                             [200, 'adam', 85.0, f1loss], [210, 'lbfgs', 90.0, 0.3]])
        _csv(d / 'F2' / 'log.csv', F2_HEAD, [[1, 'lm', 15.0, 0.9, 2], [2, 'lm', 26.0, 0.4, 3]])
    s = k.main(['summary', '--out', str(out), '--package1', str(a)])
    e = s['effect']
    assert e['F1_adam_s_per_iteration']['verdict'] == 'the library changes it'
    assert e['F1_adam_s_per_iteration']['torch_over_numpy'] == pytest.approx(0.405 / 0.305)
    assert e['F1_lbfgs_s_per_iteration']['verdict'] == 'no change beyond the cutoff'       # rounds overlap
    assert e['F2_s_per_evaluation']['verdict'] == 'no change beyond the cutoff'
    r = s['reproduction']
    assert r['torch-first/r0/F1'] == {'rows': 3, 'max_rel_diff': 0.0, 'within_rtol': True}
    assert r['torch-first/r1/F1']['within_rtol'] and not r['numpy-first/r0/F1']['within_rtol']
    assert r['torch-first/r0/F2']['rows'] == 2                     # every LM row Package 1 has too
    assert s['torch_first_reproduces_package1'] is True
    assert s['libraries'] == {'numpy': ['module'], 'torch': ['torch']}
    p1 = s['package1_full_cpu']
    assert p1['F1_adam_s_per_iteration']['values'] == pytest.approx([0.40, 0.44])
    assert p1['F1_lbfgs_s_per_iteration']['median'] == pytest.approx(0.3)
    assert p1['F2_s_per_evaluation']['median'] == pytest.approx(6.0)
    assert json.loads((root / 'summary.json').read_text())['cutoff'] == k.CUTOFF


def test_a_missing_order_is_not_measured(tmp_path):
    out = tmp_path / 'out'
    root = out / k.ITEM / k.SUB
    _timing(root, 'torch', 0, (0.4, 0.3, 5.0), 'torch')
    s = k.main(['summary', '--out', str(out), '--package1', str(tmp_path / 'none')])
    assert {v['verdict'] for v in s['effect'].values()} == {'not measured'}
    assert s['torch_first_reproduces_package1'] is False
