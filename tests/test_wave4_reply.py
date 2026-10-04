"""The advisor's reply to wave 4 (3 October 2026): option B for the clean
times (the median of three independent clean timings), the final package
(WAVES.json's clean-timing record, the assembling commit, status final),
the stall-based rule's tables in the package, and every log's stall columns
recomputed beside the logged ones."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _row(run, device='cpu', t=1.0, it=5, quantity='training_time'):
    return {'run': run, 'benchmark': run.split('_')[0], 'config': run.split('_')[1], 'device': device, 'quantity': quantity, 'clean_time_s': t, 'iterations': it, 'K_max': 60,
            'commit': 'c', 'error': ''}


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_compose_median_takes_the_median_of_three_cpu_timings():
    import experiments.clean_timing as ct
    base = [_row('bl_gravity_P256_cpu_paper', t=3.169, it=12), _row('darcy_S2_cpu_paper', t=20.7, it=1,
                                                                   quantity='total_time'),
            _row('kovasznay_P75_cuda_paper', device='cuda', t=0.04, it=16, quantity='solve_time_total')]
    rep1 = [_row('bl_gravity_P256_cpu_paper', t=0.717, it=12), _row('darcy_S2_cpu_paper', t=29.0, it=1,
                                                                    quantity='total_time')]
    rep2 = [_row('bl_gravity_P256_cpu_paper', t=0.654, it=12), _row('darcy_S2_cpu_paper', t=28.2, it=1,
                                                                    quantity='total_time')]
    g, d, k = ct.compose_median([{k: str(v) for k, v in r.items()} for r in base],
                                [[{k: str(v) for k, v in r.items()} for r in rep] for rep in (rep1, rep2)])
    assert g['clean_time_s'] == pytest.approx(0.717) and json.loads(g['clean_time_timings_s']) == [3.169, 0.717, 0.654]
    assert g['clean_time_spread'] == pytest.approx(3.169 / 0.654) and 'median of 3' in g['clean_time_rule']
    assert d['clean_time_s'] == pytest.approx(28.2)
    assert k['clean_time_s'] == '0.04' and '13b' in k['clean_time_rule']          # GPU unchanged


def test_compose_median_refuses_a_different_solve_or_a_missing_row():
    import experiments.clean_timing as ct
    base = [{k: str(v) for k, v in _row('bratu_P25_cpu_paper', it=2).items()}]
    with pytest.raises(ValueError, match='not the same solve'):
        ct.compose_median(base, [[{k: str(v) for k, v in _row('bratu_P25_cpu_paper', it=3).items()}]])
    with pytest.raises(ValueError, match='no clean time'):
        ct.compose_median(base, [[]])


def test_compose_package_keeps_the_single_table_and_records_the_rule(tmp_path):
    import experiments.clean_timing as ct
    pkg = tmp_path / 'package1'
    _write(pkg / 'B_instrumentation' / 'clean_timing' / 'clean_timing.csv', [_row('bratu_P25_cpu_paper', t=1.0)])
    for n, t in ((1, 3.0), (2, 2.0)):
        _write(pkg / 'fixes' / f'option_b_replicate_{n}' / 'B_instrumentation' / 'clean_timing' / 'clean_timing.csv',
               [_row('bratu_P25_cpu_paper', t=t)])
    (pkg / 'WAVES.json').write_text(json.dumps({'status': 'final', 'waves': {}}))
    ct.main(['--compose-package', str(pkg)])
    ctd = pkg / 'B_instrumentation' / 'clean_timing'
    assert float(next(csv.DictReader(open(ctd / 'clean_timing.csv')))['clean_time_s']) == 2.0
    assert float(next(csv.DictReader(open(ctd / 'clean_timing_single.csv')))['clean_time_s']) == 1.0
    w = json.loads((pkg / 'WAVES.json').read_text())
    assert w['clean_timing']['timings_s'] == {'bratu_P25_cpu_paper|training_time': [1.0, 3.0, 2.0]}
    assert 'median of three' in w['clean_timing']['rule'] and len(w['clean_timing']['replicates']) == 2
    empty = tmp_path / 'empty'
    _write(empty / 'B_instrumentation' / 'clean_timing' / 'clean_timing.csv', [_row('bratu_P25_cpu_paper')])
    assert ct.compose_package(empty) is None


def test_stall_columns_beside_the_logged_ones(tmp_path):
    import experiments.stopping_rule_table as srt
    log = tmp_path / 'B_instrumentation' / 'b10' / 'run_x' / 'iterations.csv'
    log.parent.mkdir(parents=True)
    # a 5% change in ||R_lin||: a stall at the logged tau_r = 0.1, not at 0.01
    with open(log, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['k', 'chi', 'norm_Rlin_h', 'norm_R_h', 'eps_u', 'stall_flag'])
        rows = [(0, '', 1.0, 1.0, 1, 'False'), (1, 0.05, 0.95, 0.9, 1, 'True'), (2, 0.05, 0.95, 0.9, 1, 'True'),
                (3, 0.05, 0.95, 0.9, 1, 'True'), (4, '', '', 0.9, 1, '')]
        w.writerows(rows)
    (row,) = srt.stall_columns(tmp_path)
    assert row['run'] == 'B_instrumentation/b10/run_x' and row['solve_rows'] == 4
    assert row['first_stall_logged_tau_r_0.1'] == 1 and row['first_stall_tau_r_0.01'] == 2
    assert (row['rule_n_s_2_fires_at'], row['rule_n_s_2_returns'], row['rule_censored']) == (3, 4, False)


def test_package_records_the_assembling_commit_and_final(tmp_path):
    spec = importlib.util.spec_from_file_location('assemble_package', REPO / 'scripts/cluster/assemble_package.py')
    assemble = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assemble)
    res = tmp_path / 'results'
    (res / 'wave1').mkdir(parents=True)
    (res / 'wave1' / 'COMMIT').write_text(json.dumps({'commit': 'a'}))
    record = assemble.assemble(res, res / 'package1', final=True)
    from lilq.source_lock import current_commit
    assert record['status'] == 'final' and record['assembled_by_commit'] == current_commit()


def test_finalize_composes_the_clean_times_before_anything_reads_them():
    text = (REPO / 'scripts/cluster/90_finalize.slurm').read_text()
    compose = text.index('clean_timing.py --compose-package')
    assert compose < text.index('kovasznay_comparison.py') < text.index('stopping_rule_table.py')
    assert compose < text.index('--lilq-from-clean-timing')
    assert '${LILQ_PACKAGE_FINAL:+--final}' in text
    for name in ('14a_gravity_rerun', '14b_clean_timing_replicates'):
        assert '# lilq-resources: timed-cpu' in (REPO / f'scripts/cluster/after_wave4/{name}.slurm').read_text()
