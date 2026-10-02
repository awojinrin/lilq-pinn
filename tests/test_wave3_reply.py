"""The advisor's reply on wave 3 and the wave 4 commit (1 October 2026),
Section 2: the four-method LiL-Q rows checked against their logged passes
(2.2), failures that do not end a job and standard JSON (2.5), wave 2's
superseded Component A runs marked in package1 (2.7), and B10's total
boundary weight (2.8)."""

import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

import experiments.component_a as ca
from baselines import validation

REPO = Path(__file__).resolve().parents[1]


# ---- 2.2 the four-method LiL-Q rows ------------------------------------------

def _clean_table(b, rows):
    (b / 'clean_timing').mkdir(parents=True, exist_ok=True)
    with open(b / 'clean_timing' / 'clean_timing.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'benchmark', 'config', 'device', 'quantity', 'clean_time_s',
                                          'iterations', 'K_max', 'commit', 'error'])
        w.writeheader()
        w.writerows(rows)


def _clean_row(run, iterations, **kw):
    bench, P = run.rsplit('_P', 1)[0], run.split('_P')[1].split('_')[0]
    return dict(dict(run=run, benchmark=bench, config=f'P{P}', device='cpu', quantity='training_time',
                     clean_time_s=0.1, iterations=iterations, K_max=60, commit='c', error=''), **kw)


def _summary(b, run, **s):
    (b / run).mkdir(parents=True, exist_ok=True)
    (b / run / 'summary.json').write_text(json.dumps(s))


def test_lilq_rows_record_the_logged_pass_and_raise_on_differing_iterations(tmp_path):
    import experiments.four_method_tables as fmt
    b = tmp_path / 'B'
    _clean_table(b, [_clean_row('bratu_P25_cpu_paper', 2), _clean_row('burgers_P25_cpu_paper', 3)])
    _summary(b, 'bratu_P25_cpu_paper', final_loss=0.2, converged=True, iterations=2, K_max=25)
    _summary(b, 'burgers_P25_cpu_paper', final_loss=0.05, converged=True, iterations=3, K_max=20)
    rows = fmt.lilq_rows(b, b / 'four_method_lilq.csv')
    assert [(r['total_iterations'], r['logged_iterations'], r['iterations_cap'], r['logged_K_max']) for r in rows] == \
        [(2, 2, 60, 25), (3, 3, 60, 20)]
    written = list(csv.DictReader(open(b / 'four_method_lilq.csv')))
    assert written[0]['logged_K_max'] == '25' and written[0]['logged_summary'] == 'bratu_P25_cpu_paper/summary.json'

    _summary(b, 'burgers_P25_cpu_paper', final_loss=0.05, converged=True, iterations=4, K_max=20)
    with pytest.raises(RuntimeError, match='burgers_P25_cpu_paper: clean 3, logged 4'):
        fmt.lilq_rows(b, b / 'four_method_lilq.csv')
    assert len(list(csv.DictReader(open(b / 'four_method_lilq.csv')))) == 2      # written before raising


def test_lilq_rows_warn_on_a_missing_summary_and_skip_a_failed_clean_run(tmp_path):
    import experiments.four_method_tables as fmt
    b = tmp_path / 'B'
    _clean_table(b, [_clean_row('bratu_P25_cpu_paper', 2),
                     _clean_row('bratu_P100_cpu_paper', '', quantity='', clean_time_s='', error='Traceback ...')])
    with pytest.warns(UserWarning) as record:
        rows = fmt.lilq_rows(b, b / 'four_method_lilq.csv')
    messages = ' '.join(str(w.message) for w in record)
    assert 'summary.json is missing' in messages and 'bratu_P100_cpu_paper: the clean-timing run failed' in messages
    assert len(rows) == 1 and rows[0]['final_loss'] is None and rows[0]['logged_iterations'] is None
    with pytest.warns(UserWarning, match='clean_timing.csv is missing'):
        assert fmt.lilq_rows(tmp_path / 'nowhere') == []


# ---- 2.5 failures and JSON ---------------------------------------------------

def test_an_infinite_validation_residual_is_written_as_null(tmp_path):
    d = tmp_path / 'F2_00_s0'
    d.mkdir()
    (d / 'run.json').write_text(json.dumps({'config': {'family': 'F2'}, 'end_reason': 'failure'}))
    assert validation.ensure(d) == math.inf
    text = (d / validation.VALIDATION_FILE).read_text()
    assert 'Infinity' not in text and json.loads(text)['val_residual'] is None
    assert validation.ensure(d) == math.inf                          # read back as inf
    assert validation.json_safe({'a': [1.0, math.inf, math.nan], 'b': 'x'}) == {'a': [1.0, None, None], 'b': 'x'}
    assert validation.as_residual(None) == math.inf and validation.as_residual(2.5) == 2.5


def _screening(root, family, rows):
    """Screening runs with given (training loss, validation residual); a copy
    of tests/test_validation_residual.py's helper (test files do not import
    each other: test_advisor_reply_fixes.py)."""
    configs = [dict(id=f'{family}_{i:02d}', family=family, width=8 + i) for i in range(len(rows))]
    (root / 'search').mkdir(parents=True, exist_ok=True)
    (root / 'search' / f'{family}_configs.json').write_text(json.dumps({'configs': configs}))
    for c, (train, val) in zip(configs, rows):
        d = root / 'screening' / f"{c['id']}_s0"
        d.mkdir(parents=True)
        key = 'final_loss_unweighted' if family == 'F1' else 'final_loss'
        (d / 'run.json').write_text(json.dumps({key: train, 'val_residual': val}))
    return [c['id'] for c in configs]


def _full_runs(root, ids, vals):
    for cid, val in zip(ids, vals):
        for s in (0, 1, 2):
            d = root / 'full' / f'{cid}_s{s}'
            d.mkdir(parents=True)
            (d / 'run.json').write_text(json.dumps({'final_loss_unweighted': 1e-7, 'val_residual': val}))


def test_one_failed_validation_does_not_end_the_full_stage(tmp_path, monkeypatch):
    ids = _screening(tmp_path, 'F1', [(2e-7, 3.8), (3e-7, 2e-5), (6e-7, 3e-6)])
    ca.select(tmp_path, 'F1', keep_top=ids)
    _full_runs(tmp_path, ids, [3.8, 2e-5, 3e-6])
    real = validation.ensure

    def flaky(run_dir, device='cpu', record_in_run_json=False):
        if Path(run_dir).name == f'{ids[2]}_s1':
            raise RuntimeError('CUDA out of memory')
        return real(run_dir, device, record_in_run_json)
    monkeypatch.setattr(validation, 'ensure', flaky)
    summary = ca.full(tmp_path, 'F1', 'cpu', budget_s=1, seeds=(0, 1, 2))
    assert summary['validation_failures'] == [f'{ids[2]}_s1']
    assert summary['representative'] == ids[2]                       # median of 3e-6, inf, 3e-6
    text = (tmp_path / 'full' / 'F1_representative.json').read_text()
    assert 'Infinity' not in text and json.loads(text)['val_residual'][ids[2]] == [3e-6, None, 3e-6]


def test_the_full_stage_raises_when_no_finalist_can_be_validated(tmp_path, monkeypatch):
    ids = _screening(tmp_path, 'F1', [(2e-7, 3.8), (3e-7, 2e-5), (6e-7, 3e-6)])
    ca.select(tmp_path, 'F1', keep_top=ids)
    _full_runs(tmp_path, ids, [3.8, 2e-5, 3e-6])
    monkeypatch.setattr(validation, 'ensure', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('broken')))
    with pytest.raises(RuntimeError, match='no finalist has a finite median'):
        ca.full(tmp_path, 'F1', 'cpu', budget_s=1, seeds=(0, 1, 2))
    assert not (tmp_path / 'full' / 'F1_representative.json').exists()


def test_the_selection_still_raises_on_a_validation_failure(tmp_path, monkeypatch):
    """Only the full stage tolerates a failure: a broken validation must not
    silently pick finalists."""
    _screening(tmp_path, 'F1', [(2e-7, None), (3e-7, 2e-5), (6e-7, 3e-6)])
    for d in (tmp_path / 'screening').iterdir():
        run = json.loads((d / 'run.json').read_text())
        run.pop('val_residual')
        (d / 'run.json').write_text(json.dumps(run))
    monkeypatch.setattr(validation, 'compute', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('broken')))
    with pytest.raises(RuntimeError, match='broken'):
        ca.select(tmp_path, 'F1')


def test_one_failed_clean_timing_run_does_not_end_the_job(tmp_path, monkeypatch):
    import experiments.clean_timing as ct
    calls = []

    def good():
        calls.append('good')
        return {'training_time': 2.0, 'iterations': 3}

    def bad():
        raise RuntimeError('singular matrix')
    runs = [ct.TimedRun('bratu_P25_cpu_paper', 'bratu', 'P25', 'cpu', 60, bad),
            ct.TimedRun('burgers_P25_cpu_paper', 'burgers', 'P25', 'cpu', 60, good)]
    monkeypatch.setattr(ct, 'build_runs', lambda *a: runs)
    with pytest.raises(SystemExit, match='1 run\\(s\\) failed: bratu_P25_cpu_paper'):
        ct.main(['--out-root', str(tmp_path), '--benchmarks', 'bratu', 'burgers'])
    path = tmp_path / 'B_instrumentation' / 'clean_timing' / 'clean_timing.csv'
    rows = {r['run']: r for r in csv.DictReader(open(path))}
    assert 'singular matrix' in rows['bratu_P25_cpu_paper']['error'] and rows['bratu_P25_cpu_paper']['clean_time_s'] == ''
    assert float(rows['burgers_P25_cpu_paper']['clean_time_s']) == 2.0 and rows['burgers_P25_cpu_paper']['error'] == ''

    # A resubmission runs the failed one again and replaces its row.
    runs[0] = ct.TimedRun('bratu_P25_cpu_paper', 'bratu', 'P25', 'cpu', 60, good)
    calls.clear()
    ct.main(['--out-root', str(tmp_path), '--benchmarks', 'bratu', 'burgers'])
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 2 and all(r['error'] == '' and float(r['clean_time_s']) == 2.0 for r in rows)
    assert calls == ['good'] * 2                                     # warm-up and timed run of bratu only

    # The readers skip a failed row.
    from experiments.kovasznay_comparison import load_clean_times
    assert load_clean_times(tmp_path / 'B_instrumentation') == {}


# ---- 2.7 superseded runs ------------------------------------------------------

def _load_assemble():
    spec = importlib.util.spec_from_file_location('assemble_package', REPO / 'scripts/cluster/assemble_package.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _put(path, text='{}'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_package_marks_wave_2s_superseded_component_a_runs(tmp_path):
    assemble = _load_assemble()
    res = tmp_path / 'results'
    w2, w4 = res / 'wave2' / 'A_calibration', res / 'wave4' / 'A_calibration'
    for cid in ('F1_04', 'F1_15', 'F1_18', 'F2_03', 'F2_16', 'F2_17'):
        _put(w2 / 'full' / f'{cid}_s0' / 'run.json')
    for cid in ('F1_04', 'F2_16'):
        _put(w2 / 'full_cpu' / f'{cid}_s0' / 'run.json')
    _put(w2 / 'float32' / 'F1_04_s0' / 'run.json')
    _put(w4 / 'screening' / 'F1_selection.json', json.dumps({'top': ['F1_04', 'F1_15', 'F1_18']}))
    _put(w4 / 'screening' / 'F2_selection.json', json.dumps({'top': ['F2_10', 'F2_05', 'F2_14']}))
    _put(w4 / 'full' / 'F1_representative.json', json.dumps({'representative': 'F1_18'}))
    _put(w4 / 'full' / 'F2_representative.json', json.dumps({'representative': 'F2_05'}))
    for cid in ('F1_18', 'F2_05'):
        _put(w4 / 'full_cpu' / f'{cid}_s0' / 'run.json')
    _put(w4 / 'float32' / 'F1_18_s0' / 'run.json')
    _put(w4 / 'full' / 'F2_05_s0' / 'run.json')
    record = assemble.assemble(res, res / 'package1')
    marked = sorted(m['path'] for m in record['superseded'])
    assert marked == sorted(['A_calibration/full/F2_03_s0', 'A_calibration/full/F2_16_s0', 'A_calibration/full/F2_17_s0',
                             'A_calibration/full_cpu/F1_04_s0', 'A_calibration/full_cpu/F2_16_s0',
                             'A_calibration/float32/F1_04_s0'])
    pkg = res / 'package1'
    assert json.loads((pkg / 'A_calibration/full/F2_16_s0/SUPERSEDED.json').read_text())['superseded_by_wave'] == 4
    assert not (pkg / 'A_calibration/full/F1_04_s0/SUPERSEDED.json').exists()        # still a wave 4 finalist
    assert not (w2 / 'full' / 'F2_16_s0' / 'SUPERSEDED.json').exists()                # the wave itself untouched
    assert json.loads((pkg / 'WAVES.json').read_text())['superseded'] == record['superseded']

    # Assembled again after the representative changed: the old marker goes.
    _put(w4 / 'full' / 'F1_representative.json', json.dumps({'representative': 'F1_04'}))
    assemble.assemble(res, res / 'package1')
    assert (pkg / 'A_calibration/float32/F1_18_s0/SUPERSEDED.json').exists()
    assert not (pkg / 'A_calibration/float32/F1_04_s0/SUPERSEDED.json').exists()


# ---- 2.8 B10's total boundary weight ------------------------------------------

def test_b10_records_the_total_boundary_weight():
    import experiments.b10_cgl_cc as b10
    assert b10.BC_WEIGHT_TOTAL == {'equal': 4, 'clenshaw_curtis': 1}
    assert 'bc_weight_total' in b10.COLUMNS and 'paper_grid_bc_weight_total' in b10.COMPARE_COLUMNS


def test_the_weight_totals_match_the_assembled_rows():
    """Sum of the squared boundary row weights of one velocity component:
    4 lambda_bc with equal weights, lambda_bc with Clenshaw-Curtis."""
    import dataclasses
    import numpy as np
    from experiments.component_c import _config
    from problems.kovasznay import EDGES, _row_weights
    import experiments.b10_cgl_cc as b10
    config, _ = _config('kovasznay', 10, 2, 'cgl', None)
    n_edge = {e: 21 for e in EDGES}
    for weights, total in b10.BC_WEIGHT_TOTAL.items():
        w = _row_weights(dataclasses.replace(config, weights=weights), 441, 21, n_edge)
        s = sum(float(np.sum(np.broadcast_to(np.asarray(w['bc'][e]) ** 2, (n_edge[e],)))) for e in EDGES)
        assert s == pytest.approx(total * config.lambda_bc, rel=1e-12)


# ---- Section 4: the wave 4 report ------------------------------------------------

def _load_report():
    spec = importlib.util.spec_from_file_location('wave_report', REPO / 'scripts/cluster/wave_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_su_per_job_at_grace_rates_reproduces_wave_3s_charge():
    """Wave 3's jobs as sacct listed them: 1,019 SU charged (myproject)."""
    report = _load_report()
    assert report.su_rate('billing=8,cpu=8,gres/gpu:a100=1,gres/gpu=1,mem=32G,node=1') == 80       # shared A100
    assert report.su_rate('billing=48,cpu=48,gres/gpu:a100=1,gres/gpu=1,mem=360G,node=1') == 192   # timed A100
    assert report.su_rate('billing=48,cpu=48,mem=360G,node=1') == 48                               # timed CPU
    assert report.su_rate('billing=24,cpu=24,mem=32G,node=1') == 24
    sacct = '\n'.join([
        '1_0|lilq-b8|billing=8,cpu=8,gres/gpu:a100=1,gres/gpu=1,mem=32G,node=1|36000|COMPLETED',
        '2|lilq-wave-report|billing=24,cpu=24,mem=32G,node=1|150|RUNNING',
        'garbage line'])
    rows = report.su_rows(sacct)
    assert [r['su'] for r in rows] == [800.0, 1.0] and rows[1]['note'].startswith('still running')


def test_b8_reruns_beside_wave_3(tmp_path):
    report = _load_report()
    res = tmp_path / 'results'
    cols = ['case', 'guess', 'P', 'method', 'iterations', 'final_loss', 'stopping_reason', 'iterations_cap', 'target']
    old = res / 'wave3' / 'B_instrumentation' / 'b8_jobs' / 'gravity_zero' / 'b8_initial_guess.csv'
    old.parent.mkdir(parents=True)
    with open(old, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow(dict(case='gravity', guess='zero', P=64, method='LiL-Q', iterations=20, final_loss=0.95,
                        stopping_reason='iteration_cap', iterations_cap=20, target=0.24))
        w.writerow(dict(case='gravity', guess='zero', P=64, method='NiL-N', iterations=900, final_loss=0.3,
                        stopping_reason='target', iterations_cap=2000, target=0.24))
    root = res / 'wave4'
    new = root / 'B_instrumentation' / 'b8_jobs' / 'lilq_kmax60' / 'b8_initial_guess.csv'
    new.parent.mkdir(parents=True)
    with open(new, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols + ['K_max', 'target_reached'])
        w.writeheader()
        w.writerow(dict(case='gravity', guess='zero', P=64, method='LiL-Q', iterations=43, final_loss=0.2367,
                        stopping_reason='target', iterations_cap=60, target=0.24, K_max=60, target_reached=True))
    (row,) = report.b8_reruns_beside_wave3(res, root)
    assert (row['wave3_K_max'], row['wave3_iterations'], row['wave3_stopping_reason']) == ('20', '20', 'iteration_cap')
    assert (row['wave4_K_max'], row['wave4_iterations'], row['wave4_final_loss']) == ('60', '43', '0.2367')
    assert (root / 'B_instrumentation' / 'b8_kmax60_vs_wave3.csv').exists()


def test_clean_against_logged_with_the_iteration_check(tmp_path):
    report = _load_report()
    root = tmp_path / 'wave4'
    logged = tmp_path / 'wave2' / 'kovasznay_P300_cpu_paper' / 'summary.json'
    _put(logged, json.dumps({'solve_time_total': 0.3, 'iterations': 9, 'K_max': 20}))
    pinned = tmp_path / 'wave1' / 'beltrami_pinned' / 'report.json'
    _put(pinned, json.dumps({'solver_time_s': 267.0, 'n_outer_iters': 5}))
    ct = root / 'B_instrumentation' / 'clean_timing' / 'clean_timing.csv'
    ct.parent.mkdir(parents=True)
    with open(ct, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'benchmark', 'config', 'device', 'quantity', 'clean_time_s',
                                          'warmup_run_time_s', 'iterations', 'K_max', 'logged_time_s',
                                          'logged_source', 'error'])
        w.writeheader()
        w.writerow(dict(run='kovasznay_P300_cpu_paper', benchmark='kovasznay', config='P300', device='cpu',
                        quantity='solve_time_total', clean_time_s=0.15, warmup_run_time_s=0.2, iterations=9,
                        K_max=60, logged_time_s=0.3, logged_source=str(logged)))
        w.writerow(dict(run='beltrami_pinned', benchmark='beltrami_pinned', config='P7984', device='cpu',
                        quantity='solve_time_total', clean_time_s=297.0, warmup_run_time_s=297.0, iterations=4,
                        K_max=60, logged_time_s=267.0, logged_source=str(pinned)))
    kov, pin = report.clean_vs_logged(root)
    assert kov['clean_over_logged'] == pytest.approx(0.5) and kov['iterations_match'] is True
    assert kov['logged_K_max'] == 20
    assert pin['logged_iterations'] == 5 and pin['iterations_match'] is False      # flagged
    assert (ct.parent / 'clean_vs_logged.csv').exists()


def test_package_is_provisional_until_assembled_with_final(tmp_path):
    assemble = _load_assemble()
    res = tmp_path / 'results'
    _put(res / 'wave1' / 'COMMIT', json.dumps({'commit': 'a'}))
    assert assemble.assemble(res, res / 'package1')['status'].startswith('provisional')
    assert json.loads((res / 'package1' / 'WAVES.json').read_text())['status'].startswith('provisional')
    assert assemble.assemble(res, res / 'package1', final=True)['status'] == 'final'
