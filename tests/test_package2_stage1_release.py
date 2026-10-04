"""Package 2, Stage 1, release items (Section 2): the stall controls that
depart from their original, the elasticity active-coefficient counts (item 9)
and the reference errors of the saved four-method models (Section 2.6)."""

import csv
import json

import numpy as np
import pytest
import torch


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_stall_control_departures(tmp_path):
    import experiments.stall_control_departures as scd
    hist = lambda its: json.dumps([[i, 1.0, 0.0] for i in its])  # noqa: E731
    base = dict(benchmark='burgers', P='400', method='NiL-Q', device='cuda', variant='')
    _write(tmp_path / 'four_method_tables.csv', [
        dict(base, seed='2', stall_iteration='5814', total_iterations='5814', loss_history_every_10=hist([5800, 5810, 5814])),
        dict(base, seed='0', stall_iteration='5241', total_iterations='5241', loss_history_every_10=hist([5230, 5240, 5241])),
        dict(base, method='LiL-N', seed='', stall_iteration='1320', total_iterations='1320',
             loss_history_every_10=hist([1310, 1320]))])
    ctrl = dict(base, variant='f1_stall_rule')
    _write(tmp_path / 'four_method_controls.csv', [
        dict(ctrl, seed='2', departs_at_iteration='5800'), dict(ctrl, seed='0', departs_at_iteration='5240'),
        dict(ctrl, method='LiL-N', seed='', departs_at_iteration='1320'), dict(ctrl, seed='1', departs_at_iteration='')])
    rows = {r['original_run']: r for r in scd.departures(tmp_path)}
    assert len(rows) == 3
    assert rows['burgers_P400_NiL-Q_s2_cuda']['gap'] == 14
    assert rows['burgers_P400_NiL-Q_s2_cuda']['at_last_logged_row_before_stall'] is False
    assert rows['burgers_P400_NiL-Q_s0_cuda']['at_last_logged_row_before_stall'] is True     # 5240, then the stall row
    assert rows['burgers_P400_LiL-N_sna_cuda']['control_run'] == 'burgers_P400_LiL-N_sna_cuda_f1_stall_rule'


def test_elasticity_counts(tmp_path):
    import experiments.elasticity_counts as ec
    from lilq.basis import create_basis_2d
    from lilq.saved_models import save_solution
    run = tmp_path / 'B_instrumentation' / 'elasticity_P8_cpu_paper'
    basis = create_basis_2d('chebyshev', 2, 2, (0, 1), (0, 1))
    save_solution(run, {'u': (basis, np.array([1.0, 1e-13, 0.0, 0.0])), 'v': (basis, np.array([2.0, 1e-3, 1e-11, 0.0]))})
    (run / 'run.json').write_text(json.dumps({'P_total': 8, 'commit': 'abc'}))
    _write(run / 'iterations.csv', [{'k': 0, 'num_rank_gelsy': 8, 'num_rank_svd': 8}, {'k': 1, 'num_rank_gelsy': '', 'num_rank_svd': ''}])
    u, v = ec.counts(tmp_path)
    assert (u['field'], u['n_active'], u['n_coeff'], u['rank_gelsy']) == ('u_x', 1, 4, 8)
    assert v['n_active'] == 3                                    # 2, 1e-3 and 1e-11 > 1e-12 * 2


def test_network_reference_errors_on_saved_models(tmp_path, monkeypatch):
    import experiments.network_reference_errors as nre
    from lilq.basis import create_basis_2d
    from lilq.nn import MLP
    from lilq.saved_models import save_network, save_solution
    from problems.bratu import BratuConfig
    cfg = BratuConfig()
    models = tmp_path / 'B_instrumentation' / 'four_method_jobs' / 'toy_gpu' / 'models'
    models.mkdir(parents=True)
    (models.parent / 'hardware.json').write_text(json.dumps({'git': {'commit': 'c0ffee'}}))
    torch.manual_seed(0)
    save_network(models / 'toy_P25_NiL-N_s1_cuda', MLP(hidden_dim=4, num_layers=2), cfg)
    save_network(models / 'toy_P25_NiL-N_s1_cuda_f1_stall_rule', MLP(hidden_dim=4, num_layers=2), cfg)   # a control
    basis = create_basis_2d('chebyshev', 2, 2, (0, 1), (0, 1))
    save_solution(models / 'toy_P25_LiL-N_sna_cuda', {'u': (basis, np.array([1.0, 0, 0, 0]))}, cfg)
    axes = [np.linspace(0, 1, 11), np.linspace(0, 1, 11)]
    ref = np.ones((11, 11))                                      # T_0 x T_0 = 1: LiL-N exact
    rows = nre.rows_for(tmp_path, 'toy', lambda c: (axes, ref))
    assert [(r['method'], r['seed']) for r in rows] == [('LiL-N', ''), ('NiL-N', 1)]
    assert rows[0]['eps_ref_final'] == pytest.approx(0.0, abs=1e-14) and rows[1]['eps_ref_final'] > 0
    assert rows[1]['eps_ref_min'] == '' and 'only the final model' in rows[1]['note'] and rows[1]['commit'] == 'c0ffee'


def test_lilq_rows_read_eps_ref_from_the_reruns(tmp_path):
    import experiments.network_reference_errors as nre
    runs = tmp_path / 'P2_12_reference_errors' / 'B_instrumentation'
    (runs.parent / 'hardware.json').parent.mkdir(parents=True)
    (runs.parent / 'hardware.json').write_text(json.dumps({'git': {'commit': 'beef'}}))
    _write(runs / 'toy_P25_cpu_paper' / 'iterations.csv',
           [{'k': 0, 'eps_ref': 1.0}, {'k': 1, 'eps_ref': 1e-3}, {'k': 2, 'eps_ref': 2e-3}])
    _write(runs / 'other_P25_cpu_paper' / 'iterations.csv', [{'k': 0, 'eps_ref': 1.0}])
    [row] = nre.lilq_rows(runs, 'toy')
    assert (row['method'], row['P'], row['device'], row['commit']) == ('LiL-Q', 25, 'cpu', 'beef')
    assert (row['eps_ref_final'], row['eps_ref_min'], row['k_min']) == (2e-3, 1e-3, 1)
    assert row['run'] == 'P2_12_reference_errors/B_instrumentation/toy_P25_cpu_paper'
