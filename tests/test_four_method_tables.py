"""Tests for experiments/four_method_tables.py (Computational_Package_1_v2.md
Section 3.4). Real, small-scale end-to-end runs -- same rationale as
tests/test_residual_band_figures.py: these are cheap enough (--quick's
tiny caps mean a handful of L-BFGS steps, not real training) to exercise
directly rather than mock, unlike the full paper-scale sweep the script
also supports (never triggered by a test).
"""

import csv
import json

import pytest
import torch

import experiments.four_method_tables as fmt
from lilq.four_method_log import FourMethodLogger


def test_bratu_runs_quick_uses_first_paper_size():
    runs = fmt._bratu_runs(quick=True)
    assert len(runs) == 1
    assert runs[0][0] == 25  # N=5 squared, first of DEFAULT_N_VALUES


def test_burgers_runs_quick_uses_first_paper_size():
    runs = fmt._burgers_runs(quick=True)
    assert len(runs) == 1
    assert runs[0][0] == 25


def test_bl_runs_viscous_vs_gravity_differ():
    viscous = fmt._bl_runs(gravity=False, quick=True)
    gravity = fmt._bl_runs(gravity=True, quick=True)
    assert viscous[0][0] == 64
    assert gravity[0][0] == 64
    assert viscous[0][1].N_g == 0.0
    assert gravity[0][1].N_g != 0.0


def test_bl_gravity_runs_use_cos_fourier_basis():
    """DECISIONS.md, 2026-09-23: the gravity IC's monotonic steep step is
    poorly represented by DEFAULT_BASIS's mode_x='both' split; verified
    against a real run that 'cos_fourier' takes LiL-N from not converging
    at N=24/32 to converging in seconds. Viscous is unaffected."""
    viscous = fmt._bl_runs(gravity=False, quick=True)
    gravity = fmt._bl_runs(gravity=True, quick=True)
    assert viscous[0][1].basis_type == 'fourier'
    assert gravity[0][1].basis_type == 'cos_fourier'


def test_bl_runs_never_pass_explicit_max_line_searches():
    """Regression guard mirroring test_bl_experiment_runner_config.py:
    BLOptConfig's derived max_line_searches (max_iterations * 15) is the
    safe default; an explicit override here previously truncated LiL-N
    before convergence (DECISIONS.md)."""
    import inspect
    source = inspect.getsource(fmt._bl_runs)
    opt_construction = source[source.index("opt = BLOptConfig("):source.index("runs.append")]
    assert "max_line_searches=" not in opt_construction


def test_apply_quick_budgets_shrinks_caps_not_R_tol():
    from problems.bratu import BratuOptConfig
    opt = BratuOptConfig(R_tol=2.5e-7, max_iterations=10000, max_line_searches=30000)
    shrunk = fmt._apply_quick_budgets(opt)
    assert shrunk.R_tol == 2.5e-7  # untouched
    assert shrunk.max_iterations < opt.max_iterations
    assert shrunk.max_line_searches < opt.max_line_searches


def test_run_and_log_nil_n_single_seed_real_run():
    from problems.bratu import BratuConfig, BratuOptConfig, run_nil_n

    config = BratuConfig(N_x=3, N_y=3, k_ratio=5)
    opt = fmt._apply_quick_budgets(BratuOptConfig())
    logger = FourMethodLogger()

    fmt._run_and_log(logger, 'bratu', 9, config, opt, 'NiL-N', run_nil_n,
                     seeds=[0], devices=[torch.device('cpu')], verbose=False)

    assert len(logger) == 1
    row = logger.rows[0]
    assert row['benchmark'] == 'bratu'
    assert row['method'] == 'NiL-N'
    assert row['seed'] == 0
    assert row['device'] == 'cpu'
    assert row['stopping_reason'] in ('target', 'iteration_cap', 'line_search_cap')
    history = row['loss_history_every_10']
    assert history[0][0] == 0  # first recorded iteration is always 0
    assert history[-1][0] == row['total_iterations']  # last row always kept


def test_run_and_log_lil_n_logs_seed_as_none():
    from problems.bratu import BratuConfig, BratuOptConfig, run_lil_n

    config = BratuConfig(N_x=3, N_y=3, k_ratio=5)
    opt = fmt._apply_quick_budgets(BratuOptConfig())
    logger = FourMethodLogger()

    fmt._run_and_log(logger, 'bratu', 9, config, opt, 'LiL-N', run_lil_n,
                     seeds=None, devices=[torch.device('cpu')], verbose=False)

    assert len(logger) == 1
    assert logger.rows[0]['seed'] is None


def test_run_and_log_nil_q_uses_quasi_iter_cap_not_total_iterations():
    from problems.bratu import BratuConfig, BratuOptConfig, run_nil_q

    config = BratuConfig(N_x=3, N_y=3, k_ratio=5)
    opt = fmt._apply_quick_budgets(BratuOptConfig(), max_quasi_iters_nn=2, max_inner_iters_nn=5)
    logger = FourMethodLogger()

    fmt._run_and_log(logger, 'bratu', 9, config, opt, 'NiL-Q', run_nil_q,
                     seeds=[0], devices=[torch.device('cpu')], verbose=False)

    row = logger.rows[0]
    # iterations_cap must reflect the outer quasi-iteration budget (2),
    # not total_iterations' own value (which is quasi_iters * inner_iters,
    # a different, larger number).
    assert row['iterations_cap'] == 2
    assert row['total_iterations'] != row['iterations_cap']


def test_csv_output_is_well_formed(tmp_path):
    from problems.bratu import BratuConfig, BratuOptConfig, run_lil_n

    config = BratuConfig(N_x=3, N_y=3, k_ratio=5)
    opt = fmt._apply_quick_budgets(BratuOptConfig())
    logger = FourMethodLogger()
    fmt._run_and_log(logger, 'bratu', 9, config, opt, 'LiL-N', run_lil_n,
                     seeds=None, devices=[torch.device('cpu')], verbose=False)

    out_path = tmp_path / "four_method_tables.csv"
    logger.to_csv(out_path)

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    history = json.loads(rows[0]["loss_history_every_10"])
    assert isinstance(history, list)
    assert all(len(pair) == 2 for pair in history)


# ─────────────────────────────────────────────────────────────────────────────
# Robustness: failures logged not fatal, incremental writes, resume
# ─────────────────────────────────────────────────────────────────────────────

class _FakeMetrics:
    def to_dict(self):
        return {"iteration": [0, 1], "loss": [1.0, 0.5]}


def _fake_runner_factory(calls, fail_seeds=()):
    def runner(config, opt, device=None, verbose=True):
        calls.append(config.init_seed)
        if config.init_seed in fail_seeds:
            raise RuntimeError("boom")
        return "model", _FakeMetrics(), {
            "total_iterations": 1, "total_line_searches": 2, "training_time": 0.1,
            "final_loss": 0.5, "converged": True,
        }
    return runner


def _cfg_opt():
    from problems.bratu import BratuConfig, BratuOptConfig
    return BratuConfig(N_x=3, N_y=3), BratuOptConfig()


def test_failure_is_logged_with_traceback_and_sweep_continues(tmp_path):
    config, opt = _cfg_opt()
    calls = []
    logger = FourMethodLogger()
    csv_path = tmp_path / "fmt.csv"
    fmt._run_and_log(logger, 'bratu', 9, config, opt, 'NiL-N', _fake_runner_factory(calls, fail_seeds=(0,)),
                     seeds=[0, 1], devices=[torch.device('cpu')], verbose=False, csv_path=csv_path)

    assert calls == [0, 1]
    failed, ok = logger.rows
    assert failed['stopping_reason'] == 'failure'
    assert 'RuntimeError: boom' in failed['error']
    assert ok['stopping_reason'] == 'target'
    assert ok['collocation_seed'] == 42 and ok['seed'] == 1
    with open(csv_path, newline="") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_csv_written_after_every_row(tmp_path):
    config, opt = _cfg_opt()
    csv_path = tmp_path / "fmt.csv"
    logger = FourMethodLogger()
    seen_sizes = []

    def runner(config, opt, device=None, verbose=True):
        seen_sizes.append(len(list(csv.DictReader(open(csv_path)))) if csv_path.exists() else 0)
        return _fake_runner_factory([])(config, opt, device, verbose)

    fmt._run_and_log(logger, 'bratu', 9, config, opt, 'NiL-N', runner,
                     seeds=[0, 1, 2], devices=[torch.device('cpu')], verbose=False, csv_path=csv_path)
    assert seen_sizes == [0, 1, 2]  # each run sees every earlier row already on disk


def test_resume_skips_completed_and_reruns_failures(tmp_path):
    config, opt = _cfg_opt()
    csv_path = tmp_path / "fmt.csv"
    first = FourMethodLogger()
    fmt._run_and_log(first, 'bratu', 9, config, opt, 'NiL-N', _fake_runner_factory([], fail_seeds=(1,)),
                     seeds=[0, 1, 2], devices=[torch.device('cpu')], verbose=False, csv_path=csv_path)

    resumed = FourMethodLogger.from_csv(csv_path)
    assert len(resumed) == 2  # the failure is dropped so it gets rerun
    calls = []
    fmt._run_and_log(resumed, 'bratu', 9, config, opt, 'NiL-N', _fake_runner_factory(calls),
                     seeds=[0, 1, 2], devices=[torch.device('cpu')], verbose=False,
                     csv_path=csv_path, skip_keys=frozenset(resumed.completed_keys()))
    assert calls == [1]
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert sorted(r['seed'] for r in rows) == ['0', '1', '2']
    assert all(r['stopping_reason'] == 'target' for r in rows)
    assert json.loads(rows[0]['loss_history_every_10']) == [[0, 1.0], [1, 0.5]]  # not double-encoded


def test_run_problem_selects_sizes_methods_and_passes(monkeypatch):
    from problems.bratu import BratuConfig, BratuOptConfig
    monkeypatch.setattr(fmt, 'DEVICE', torch.device('cuda'))
    monkeypatch.setattr(fmt, 'warm_up', lambda device, verbose=True: None)
    runs = lambda quick: [(P, BratuConfig(N_x=3, N_y=3), BratuOptConfig()) for P in (25, 100, 225)]
    seen = []

    def recorder(name):
        def runner(config, opt, device=None, verbose=True):
            seen.append((name, str(device), config.init_seed))
            return _fake_runner_factory([])(config, opt, device, verbose)
        return runner

    logger = FourMethodLogger()
    fmt.run_problem('bratu', runs, recorder('NiL-N'), recorder('NiL-Q'), recorder('LiL-N'), logger,
                    seeds=(0,), verbose=False, P_values=[100, 225], methods=['LiL-N'], passes=['cpu'])
    # Only the CPU pass, which exists only at the largest size, and only LiL-N.
    assert seen == [('LiL-N', 'cpu', None)]
    assert logger.rows[0]['P'] == 225
