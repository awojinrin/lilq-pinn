"""Clean timing (the advisor's reply to wave 2, Section 3) and K_max = 60 for
every LiL-Q paper pass (his follow-up of 1 October 2026)."""

import json
import time

import numpy as np
import pytest
import torch

from lilq.iteration_log import IterationLogger
from lilq.solvers import LILQ_PAPER_KMAX


def _coefficients(result, fields):
    return np.concatenate([np.ravel(result[f'theta_{f}']) for f in fields])


def test_bratu_diagnostics_off_gives_the_same_coefficients():
    from experiments.run_bratu import paper_setup
    from problems.bratu import run_lil_q
    config, opt = paper_setup(5)
    logged = run_lil_q(config, opt, verbose=False, iteration_logger=IterationLogger())
    clean = run_lil_q(config, opt, verbose=False, diagnostics=False)
    assert np.array_equal(logged[1], clean[1])
    assert logged[-1]['total_iterations'] == clean[-1]['total_iterations']


def test_gravity_bl_diagnostics_off_gives_the_same_coefficients():
    from experiments.run_bl import paper_setup
    from problems.buckley_leverett import run_lil_q
    config, opt = paper_setup(8, True)
    logged = run_lil_q(config, opt, verbose=False, iteration_logger=IterationLogger())
    clean = run_lil_q(config, opt, verbose=False, diagnostics=False)
    assert np.array_equal(logged[1], clean[1])


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='no CUDA device'))])
def test_kovasznay_diagnostics_off_gives_the_same_coefficients(device):
    from problems.kovasznay import KovasznayConfig, solve_kovasznay
    config = KovasznayConfig(N_x=5, N_y=5, k_ratio=4, max_iter=LILQ_PAPER_KMAX, tol=1e-9,
                             use_gpu=(device == 'cuda'))
    logged = solve_kovasznay(config, verbose=False, iteration_logger=IterationLogger())
    clean = solve_kovasznay(config, verbose=False, diagnostics=False)
    assert np.array_equal(_coefficients(logged, 'uvp'), _coefficients(clean, 'uvp'))
    assert logged['n_outer_iters'] == clean['n_outer_iters']
    assert np.isnan(clean['history']['pde_residual']).all() and not np.isnan(logged['history']['pde_residual']).any()


@pytest.mark.parametrize('pins', [1, 3])
def test_beltrami_diagnostics_off_gives_the_same_coefficients(pins):
    from problems.beltrami import BeltramiConfig, solve_beltrami
    config = BeltramiConfig(N_vel=3, N_p=3, N_x=4, N_y=4, N_z=4, N_t=4, N_bc=3, N_t_bc=3, N_ic=3, max_iter=3,
                            n_pressure_pin_levels=pins)
    logged = solve_beltrami(config, verbose=False, iteration_logger=IterationLogger())
    clean = solve_beltrami(config, verbose=False, diagnostics=False)
    assert np.array_equal(_coefficients(logged, 'uvwp'), _coefficients(clean, 'uvwp'))


def test_elasticity_diagnostics_off_gives_the_same_coefficients():
    from problems.elasticity import ElasticityConfig, solve_elasticity
    config = ElasticityConfig(N_x=5, N_y=5, k_ratio=4)
    logged = solve_elasticity(config, verbose=False, iteration_logger=IterationLogger())
    clean = solve_elasticity(config, verbose=False, diagnostics=False)
    assert np.array_equal(_coefficients(logged, 'uv'), _coefficients(clean, 'uv'))


def test_darcy_diagnostics_off_gives_the_same_coefficients_and_stops_the_clock_at_the_solve(monkeypatch):
    """With diagnostics off, total_time ends with the LiL solve: the FVM
    reference (here slowed by 0.5 s) is off the clock."""
    import problems.darcy as darcy
    config = darcy.DarcyConfig(ORDER_H=6, ORDER_U=6, ORDER_V=6, perm_file='perm_field_S1.txt')
    real = darcy.solve_fvm
    monkeypatch.setattr(darcy, 'solve_fvm', lambda physics: (time.sleep(0.5), real(physics))[1])
    logged = darcy.solve_lilq_darcy(config, darcy.DarcyPhysics(config, verbose=False), verbose=False,
                                    iteration_logger=IterationLogger())
    clean = darcy.solve_lilq_darcy(config, darcy.DarcyPhysics(config, verbose=False), verbose=False,
                                   diagnostics=False)
    for k in ('c_h_tilde', 'c_u', 'c_v'):
        assert np.array_equal(logged[k], clean[k])
    assert logged['metrics']['total_time'] >= 0.5 > clean['metrics']['total_time']


def test_diagnostics_off_refuses_a_logger():
    from problems.kovasznay import KovasznayConfig, solve_kovasznay
    from problems.elasticity import ElasticityConfig, solve_elasticity
    with pytest.raises(ValueError, match='diagnostics=False'):
        solve_kovasznay(KovasznayConfig(N_x=4, N_y=4), verbose=False, iteration_logger=IterationLogger(),
                        diagnostics=False)
    with pytest.raises(ValueError, match='diagnostics=False'):
        solve_elasticity(ElasticityConfig(N_x=4, N_y=4), verbose=False, iteration_logger=IterationLogger(),
                         diagnostics=False)


def test_every_lilq_paper_pass_has_k_max_60_and_nil_q_budgets_are_unchanged():
    """LiL-Q's K_max is its own constant: NiL-Q's outer-iteration budgets,
    which shared it, stay 25 (Bratu), 20 (Burgers), 50 (viscous BL) and 20
    (gravity BL)."""
    from experiments.run_bratu import paper_setup as bratu
    from experiments.run_burgers import paper_setup as burgers
    from experiments.run_bl import paper_setup as bl
    from experiments.run_kovasznay import MAX_ITER
    from experiments.run_beltrami_pinned import pinned_config
    import experiments.component_b as cb
    assert LILQ_PAPER_KMAX == 60 and MAX_ITER == 60
    for opt, nil_q in ((bratu(10)[1], 25), (burgers(10)[1], 20), (bl(16, False)[1], 50), (bl(8, True)[1], 20)):
        assert opt.max_quasi_iters_lil == 60 and opt.max_quasi_iters_nn == nil_q
    assert pinned_config().max_iter == 60 and pinned_config(kmax=True).max_iter == 8
    paper = [r for r in cb.build_runs(benchmarks=('beltrami',), passes=('paper',))]
    assert len(paper) == 1
    import inspect
    assert 'max_iter=LILQ_PAPER_KMAX' in inspect.getsource(cb._beltrami_runs)


def test_clean_timing_driver_end_to_end(tmp_path):
    """Warm-up, timed run and record, with the logged time looked up beside
    it; a rerun skips what is done."""
    import experiments.clean_timing as ct
    logged = tmp_path / 'logged'
    (logged / 'bratu_P25_cpu_paper').mkdir(parents=True)
    (logged / 'bratu_P25_cpu_paper' / 'summary.json').write_text(json.dumps({'training_time': 0.123}))
    ct.main(['--out-root', str(tmp_path / 'w4'), '--benchmarks', 'bratu', 'elasticity', 'kovasznay',
             'beltrami_pinned', 'darcy', '--smoke', '--logged-roots', str(logged)])
    import csv
    path = tmp_path / 'w4' / 'B_instrumentation' / 'clean_timing' / 'clean_timing.csv'
    rows = list(csv.DictReader(open(path)))
    by_run = {(r['run'], r['quantity']): r for r in rows}
    assert {q for (run, q) in by_run if run.startswith('elasticity')} == {'time_lil_s', 'solve_time_qr'}
    bratu = by_run[('bratu_P25_cpu_paper', 'training_time')]
    assert float(bratu['logged_time_s']) == 0.123 and float(bratu['clean_time_s']) > 0
    assert all(float(r['warmup_run_time_s']) > 0 for r in rows)
    assert ('beltrami_pinned', 'solve_time_total') in by_run and ('darcy_S1_cpu_paper', 'total_time') in by_run
    assert (tmp_path / 'w4' / 'B_instrumentation' / 'clean_timing' / 'hardware.json').exists()
    before = path.read_text()
    ct.main(['--out-root', str(tmp_path / 'w4'), '--benchmarks', 'bratu', '--smoke'])
    assert path.read_text() == before
