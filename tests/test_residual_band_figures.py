"""Tests for experiments/residual_band_figures.py (Computational_Package_1_v2.md
Section 3.5). Unlike the source-inspection-only tests for the NiL-N/NiL-Q
experiment runners (test_bl_experiment_runner_config.py's own docstring
explains why: those trigger real gradient-descent training as a side
effect), this script's LiL-Q reruns are fast direct linear-algebra
solves -- a real end-to-end run at small scale is cheap enough to
exercise directly, the same way tests/test_bratu_instrumentation.py etc.
do for the underlying solvers.
"""

import csv
import math

import pytest

from lilq.iteration_log import IterationLogger
from problems.bratu import BratuConfig, BratuOptConfig, run_lil_q as run_bratu_lil_q
import experiments.residual_band_figures as rbf


def test_bratu_runs_quick_uses_first_two_paper_sizes():
    runs = rbf._bratu_runs(quick=True)
    Ps = [P for P, _config, _opt in runs]
    assert Ps == [25, 100]  # N=5,10 squared -- first two of DEFAULT_N_VALUES


def test_burgers_runs_quick_uses_first_two_paper_sizes():
    runs = rbf._burgers_runs(quick=True)
    Ps = [P for P, _config, _opt in runs]
    assert Ps == [25, 100]


def test_bl_runs_viscous_vs_gravity_use_different_configs_and_targets():
    viscous = rbf._bl_runs(gravity=False, quick=True)
    gravity = rbf._bl_runs(gravity=True, quick=True)

    Ps_viscous = [P for P, _c, _o in viscous]
    Ps_gravity = [P for P, _c, _o in gravity]
    assert Ps_viscous == [64, 256]
    assert Ps_gravity == [64, 256]

    # Gravity configs must actually have gravity on; viscous must not.
    for _P, config, _opt in viscous:
        assert config.N_g == 0.0
        assert config.basis_type == 'fourier'
    for _P, config, _opt in gravity:
        assert config.N_g != 0.0
        # DECISIONS.md, 2026-09-23: 'cos_fourier' resolves the gravity
        # IC's steep monotonic step far better than the default mode_x
        # ='both' split -- verified against a real run.
        assert config.basis_type == 'cos_fourier'

    # Distinct R_tol schedules (GRAVITY_TARGET_LOSSES != TARGET_LOSSES).
    opt_viscous = {P: opt.R_tol for P, _c, opt in viscous}
    opt_gravity = {P: opt.R_tol for P, _c, opt in gravity}
    assert opt_viscous != opt_gravity


def test_run_and_log_writes_csv_and_populates_logger(tmp_path):
    def _tiny_runs(quick):
        config = BratuConfig(N_x=3, N_y=3, k_ratio=5)
        opt = BratuOptConfig()
        return [(9, config, opt)]

    loggers = rbf.run_and_log('bratu_tiny', _tiny_runs, run_bratu_lil_q,
                              tmp_path, quick=True, verbose=False)

    assert set(loggers.keys()) == {9}
    logger = loggers[9]
    assert len(logger) > 0

    csv_path = tmp_path / "bratu_tiny_P9_iterations.csv"
    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(logger)
    assert rows[0]["k"] == "0"


def test_plot_residual_bands_produces_pdf_and_png(tmp_path):
    config = BratuConfig(N_x=3, N_y=3, k_ratio=5)
    opt = BratuOptConfig()
    logger = IterationLogger()
    run_bratu_lil_q(config, opt, verbose=False, iteration_logger=logger)

    pdf_path = rbf.plot_residual_bands('bratu_tiny', 'Bratu (tiny)', {9: logger}, tmp_path)

    assert pdf_path == tmp_path / "bratu_tiny_residual_bands.pdf"
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
    assert (tmp_path / "bratu_tiny_residual_bands.png").exists()


def test_end_to_end_quick_bratu_sweep(tmp_path):
    """Real, small-scale end-to-end run of the actual quick-mode pipeline
    (not a hand-rolled tiny config) -- exercises _bratu_runs, run_and_log,
    and plot_residual_bands together exactly as main() calls them."""
    loggers = rbf.run_and_log('bratu', rbf._bratu_runs, run_bratu_lil_q,
                              tmp_path, quick=True, verbose=False)
    assert set(loggers.keys()) == {25, 100}
    for P, logger in loggers.items():
        assert len(logger) > 0
        assert not math.isnan(logger.rows[-1]["norm_R_h"])

    pdf_path = rbf.plot_residual_bands('bratu', 'Bratu', loggers, tmp_path)
    assert pdf_path.exists()

    for P in (25, 100):
        assert (tmp_path / f"bratu_P{P}_iterations.csv").exists()


def test_from_logs_reads_the_section_3_3_run_folders(tmp_path):
    """--from-logs plots the Component B driver's logs; 'bl' must not pick
    up 'bl_gravity' folders, and the read-back rows must plot like live ones."""
    import experiments.component_b as cb
    runs = cb.build_runs(benchmarks=('bl', 'bl_gravity'), passes=('paper',), devices=('cpu',), smoke=True)
    for r in runs:
        assert cb.execute_run(r, tmp_path / 'B_instrumentation', verbose=False) == 'ok'

    loggers = rbf.load_from_logs('bl', tmp_path, 'paper')
    assert list(loggers) == [64]
    rows = loggers[64].rows
    assert [r['k'] for r in rows] == list(range(len(rows)))
    assert rows[-1]['norm_Rlin_h'] is None and isinstance(rows[0]['norm_Rlin_h'], float)
    assert list(rbf.load_from_logs('bl_gravity', tmp_path, 'paper')) == [64]
    assert rbf.load_from_logs('bl', tmp_path, 'kmax') == {}

    pdf = rbf.plot_residual_bands('bl', 'BL', loggers, tmp_path)
    assert pdf.exists() and pdf.stat().st_size > 0
