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
