"""The advisor's follow-up of 1 October 2026: the default-initialization ELM
row of Table 3 (item 3), the B9 time records (item 4), and the B8 LiL-Q
reruns with K_max = 60 (item 1)."""

import csv
import json
import os

import numpy as np
import pytest
import torch

from lilq.basis import ELMBasis2D, ELMBasis2D_TorchDefault, ELMBasis2D_Xavier


def test_default_init_elm_is_uniform_on_one_over_root_fan_in():
    b = ELMBasis2D_TorchDefault(625, (-1, 1), (0, 1), seed=0)
    bound = 1 / np.sqrt(2)
    for p in (b.alpha, b.beta, b.gamma):
        assert float(p.abs().max()) <= bound and float(p.abs().max()) > 0.95 * bound
    again = ELMBasis2D_TorchDefault(625, (-1, 1), (0, 1), seed=0)
    assert torch.equal(b.alpha, again.alpha) and torch.equal(b.gamma, again.gamma)
    # Not the Xavier (+-0.098 at 625 neurons; Table 3's existing ELM row) nor the
    # LeCun-uniform ELMBasis2D (+-sqrt(3 / fan_in) = +-1.22) variant.
    assert float(ELMBasis2D_Xavier(625, (-1, 1), (0, 1), seed=0).alpha.abs().max()) < 0.1
    assert float(ELMBasis2D(625, (-1, 1), (0, 1), seed=0).alpha.abs().max()) > 1.0
    assert b.evaluate(np.array([0.0, 0.5]), np.array([0.1, 0.9])).shape == (2, 625)


def test_table3_row_has_the_error_and_both_kappas(tmp_path):
    import experiments.run_burgers_basis_comparison as rb
    config = rb.ComparisonConfig(N_x=5, N_t=5, disable_stopping_rule=True, max_quasi_iters=3)
    results = rb.run_table3_study(config, basis_keys=['elm_default', 'sin_cheb'], verbose=False)
    rb.write_table3_csv(results, tmp_path / 't3.csv')
    rows = {r['basis_key']: r for r in csv.DictReader(open(tmp_path / 't3.csv'))}
    assert set(rows) == {'elm_default', 'sin_cheb'}
    for r in rows.values():
        assert float(r['eps_u']) >= 0 and float(r['kappa_raw']) >= float(r['kappa_retained']) > 0
        assert float(r['final_R_h_squared']) > 0
    assert rows['elm_default']['basis_label'] == 'ELM (default init)'


def test_b9_rows_record_allocation_and_resumes(tmp_path):
    import experiments.darcy_fv_comparison as b9
    rows = b9.compare_field('S1', order=6, nil_seeds=(0,), nil_epochs=2, verbose=False, model_root=tmp_path,
                            nil_dtypes=('float32',))
    assert all(r['allocation'] for r in rows)
    assert rows[0]['resumed_at'] == '' and json.loads(rows[1]['resumed_at']) == []


def test_allocation_label_from_a_scheduler_record():
    import experiments.darcy_fv_comparison as b9
    shared = {'slurm': True, 'job': {'NumCPUs': '8'}, 'SLURM_JOB_GPUS': '1', 'exclusive': False}
    assert b9.allocation_label(shared, 'NVIDIA A100-PCIE-40GB') == '8 cores + 1 x NVIDIA A100-PCIE-40GB, shared'
    whole = {'slurm': True, 'job': {'NumCPUs': '48'}, 'SLURM_JOB_GPUS': '0', 'exclusive': False,
             'holds_whole_node': True}
    assert b9.allocation_label(whole, 'NVIDIA A100-PCIE-40GB') == '48 cores + 1 x NVIDIA A100-PCIE-40GB, whole node'
    cpu = {'slurm': True, 'job': {'NumCPUs': '48'}, 'SLURM_JOB_GPUS': None, 'exclusive': True}
    assert b9.allocation_label(cpu) == '48 cores, exclusive'
    assert b9.allocation_label({'slurm': False}) == 'local'


def test_annotate_fills_wave_3_tables(tmp_path):
    """Tables written before the columns existed get them from the job's
    hardware.json and the saved networks."""
    import experiments.darcy_fv_comparison as b9
    job = tmp_path / 'darcy_fv' / 'S1_s0'
    (job / 'models' / 'NiL_S1_s0_float64').mkdir(parents=True)
    torch.save({'resumed_at': [5000]}, job / 'models' / 'NiL_S1_s0_float64' / 'network.pt')
    (job / 'hardware.json').write_text(json.dumps({
        'scheduler': {'slurm': True, 'job': {'NumCPUs': '8'}, 'SLURM_JOB_GPUS': '0', 'exclusive': False},
        'gpu': {'gpus': [{'name': 'NVIDIA A100-PCIE-40GB'}]}}))
    old = [c for c in b9.COLUMNS if c not in ('allocation', 'resumed_at')]
    with open(job / 'darcy_fv_comparison.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=old)
        w.writeheader()
        w.writerow({c: '' for c in old} | {'field': 'S1', 'method': 'LiL', 'seed': '', 'dtype': 'float64', 'delta_fv': 1})
        w.writerow({c: '' for c in old} | {'field': 'S1', 'method': 'NiL', 'seed': '0', 'dtype': 'float64', 'delta_fv': 2})
    # package1 once held wave 3's table by a hard link: annotating it must not
    # change the original (the advisor's reply on wave 3, Section 1, item 2).
    original = tmp_path / 'wave3_original.csv'
    os.link(job / 'darcy_fv_comparison.csv', original)
    before = original.read_bytes()
    assert b9.annotate(tmp_path / 'darcy_fv') == 2
    assert original.read_bytes() == before
    rows = list(csv.DictReader(open(job / 'darcy_fv_comparison.csv')))
    assert {r['allocation'] for r in rows} == {'8 cores + 1 x NVIDIA A100-PCIE-40GB, shared'}
    assert json.loads(rows[1]['resumed_at']) == [5000] and rows[0]['resumed_at'] == ''
    assert b9.annotate(tmp_path / 'darcy_fv') == 0                    # nothing left to fill


def test_b8_reruns_only_the_capped_lilq_rows_with_k_max_60(tmp_path):
    import experiments.b8_initial_guess as b8
    from lilq.solvers import LILQ_PAPER_KMAX
    old = tmp_path / 'old.csv'
    with open(old, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['case', 'guess', 'P', 'method', 'seed', 'stopping_reason'])
        w.writeheader()
        w.writerows([dict(case='gravity', guess='zero', P=64, method='LiL-Q', seed='', stopping_reason='iteration_cap'),
                     dict(case='gravity', guess='zero', P=256, method='LiL-Q', seed='', stopping_reason='target'),
                     dict(case='gravity', guess='zero', P=64, method='NiL-N', seed=0, stopping_reason='iteration_cap')])
    runs = b8.capped_lilq_runs([old, old])
    assert runs == [('gravity', 'zero', 8, 'LiL-Q', '')]
    path = b8.run_b8(tmp_path / 'k60', runs=runs, verbose=False)
    (row,) = list(csv.DictReader(open(path)))
    assert int(row['K_max']) == LILQ_PAPER_KMAX == 60 and row['stopping_reason'] == 'target'
    assert int(row['iterations']) > 20                     # it needed more than the old cap


def test_four_method_lilq_rows_come_from_the_clean_timing_runs(tmp_path):
    """The follow-up's item 1: the four-method tables' LiL-Q rows take their
    time and iterations from the clean-timing runs, their final loss from
    the logged paper pass."""
    import experiments.four_method_tables as fmt
    b = tmp_path / 'B'
    (b / 'clean_timing').mkdir(parents=True)
    with open(b / 'clean_timing' / 'clean_timing.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'benchmark', 'config', 'device', 'quantity', 'clean_time_s',
                                          'iterations', 'K_max', 'commit'])
        w.writeheader()
        w.writerow({'run': 'bl_gravity_P64_cpu_paper', 'benchmark': 'bl_gravity', 'config': 'P64', 'device': 'cpu',
                    'quantity': 'training_time', 'clean_time_s': 0.13, 'iterations': 43, 'K_max': 60, 'commit': 'c'})
        w.writerow({'run': 'kovasznay_P75_cpu_paper', 'benchmark': 'kovasznay', 'config': 'P75', 'device': 'cpu',
                    'quantity': 'solve_time_total', 'clean_time_s': 0.05, 'iterations': 16, 'K_max': 60, 'commit': 'c'})
    (b / 'bl_gravity_P64_cpu_paper').mkdir()
    (b / 'bl_gravity_P64_cpu_paper' / 'summary.json').write_text(json.dumps({'final_loss': 0.2367, 'converged': True}))
    rows = fmt.lilq_rows(b, b / 'four_method_lilq.csv')
    assert len(rows) == 1
    r = rows[0]
    assert (r['benchmark'], r['P'], r['method'], r['total_iterations'], r['training_time_s']) == \
        ('bl_gravity', 64, 'LiL-Q', 43, 0.13)
    assert (r['final_loss'], r['stopping_reason'], r['iterations_cap']) == (0.2367, 'target', 60)
    assert (b / 'four_method_lilq.csv').exists()


def test_table3_solutions_load_without_the_main_module_shim(tmp_path):
    """Wave 1's B6 solutions pickled a ComparisonConfig defined in a script
    run as __main__; the configuration is now a plain dict, so a solution
    loads in any process."""
    import subprocess
    import sys
    import experiments.run_burgers_basis_comparison as rb
    config = rb.ComparisonConfig(N_x=4, N_t=4, disable_stopping_rule=True, max_quasi_iters=2)
    rb.run_table3_study(config, basis_keys=['sin_cheb'], verbose=False, run_root=tmp_path)
    code = ("import sys; sys.path.insert(0, %r); from lilq.saved_models import load_solution; "
            "s = load_solution(%r); print(type(s['config']).__name__, s['config']['N_x'])"
            % (str(__import__('pathlib').Path(rb.__file__).parents[1]), str(tmp_path / 'sin_cheb')))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.split() == ['dict', '4'], out.stderr
