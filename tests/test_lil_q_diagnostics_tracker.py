"""Tests for lilq.iteration_log.LilQDiagnosticsTracker -- the stateful
helper that turns solve_lil_q's raw per-iteration ingredients into a
full iterations.csv row, in isolation from any real PDE solve.
"""

import math

import numpy as np
import pytest

from lilq.instrumentation import EPS_MACH
from lilq.iteration_log import ITERATION_CSV_COLUMNS, IterationLogger, LilQDiagnosticsTracker


def _synthetic_system(n_rows=20, n_cols=5, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_rows, n_cols))
    b = rng.standard_normal(n_rows)
    return A, b


def test_step_returns_only_valid_schema_columns():
    # The tracker deliberately does NOT populate test-error/GPU-path
    # columns (out of its scope -- see its docstring) -- every key it
    # DOES return must still be a real schema column (no typos), and the
    # result must feed cleanly into IterationLogger.record(**row), which
    # fills in everything the tracker left out.
    A, b = _synthetic_system()
    beta_prev = np.zeros(5)
    beta_new = np.linalg.lstsq(A, b, rcond=None)[0]

    tracker = LilQDiagnosticsTracker(initial_norm_R_h=1.0)
    row = tracker.step(
        k=1, A_stacked=A, b_stacked=b, beta_prev=beta_prev, beta_new=beta_new,
        total_loss=0.5, rank_gelsy=5, t_assemble_s=0.001, t_solve_s=0.002,
        is_final_iterate=False,
    )

    assert set(row.keys()) <= set(ITERATION_CSV_COLUMNS)
    assert row["k"] == 1

    logger = IterationLogger()
    logger.record(**row)  # must not raise
    assert logger.rows[0]["eps_u"] is None  # left for a problem-specific caller to fill in


def test_norm_Rlin_h_matches_direct_computation():
    A, b = _synthetic_system()
    beta_prev = np.zeros(5)
    beta_new = np.linalg.lstsq(A, b, rcond=None)[0]
    expected = float(np.linalg.norm(A @ beta_new - b))

    tracker = LilQDiagnosticsTracker(initial_norm_R_h=1.0)
    row = tracker.step(
        k=1, A_stacked=A, b_stacked=b, beta_prev=beta_prev, beta_new=beta_new,
        total_loss=0.5, rank_gelsy=5, t_assemble_s=0.0, t_solve_s=0.0,
        is_final_iterate=False,
    )
    assert row["norm_Rlin_h"] == pytest.approx(expected)


def test_norm_f_h_is_norm_of_rhs():
    A, b = _synthetic_system()
    beta_new = np.linalg.lstsq(A, b, rcond=None)[0]
    tracker = LilQDiagnosticsTracker(initial_norm_R_h=1.0)
    row = tracker.step(
        k=1, A_stacked=A, b_stacked=b, beta_prev=np.zeros(5), beta_new=beta_new,
        total_loss=0.5, rank_gelsy=5, t_assemble_s=0.0, t_solve_s=0.0,
        is_final_iterate=False,
    )
    assert row["norm_f_h"] == pytest.approx(float(np.linalg.norm(b)))


def test_norm_dbeta_and_rel_dbeta():
    beta_prev = np.array([1.0, 0.0, 0.0])
    beta_new = np.array([1.0, 3.0, 4.0])  # dbeta = [0,3,4], norm 5
    A = np.eye(6, 3)
    b = np.zeros(6)

    tracker = LilQDiagnosticsTracker(initial_norm_R_h=1.0)
    row = tracker.step(
        k=1, A_stacked=A, b_stacked=b, beta_prev=beta_prev, beta_new=beta_new,
        total_loss=0.1, rank_gelsy=3, t_assemble_s=0.0, t_solve_s=0.0,
        is_final_iterate=False,
    )
    assert row["norm_dbeta"] == pytest.approx(5.0)
    assert row["rel_dbeta"] == pytest.approx(5.0 / np.linalg.norm(beta_new))


def test_order_obs_and_stall_flag_undefined_on_first_step():
    A, b = _synthetic_system()
    beta_new = np.linalg.lstsq(A, b, rcond=None)[0]
    tracker = LilQDiagnosticsTracker(initial_norm_R_h=1.0)
    row = tracker.step(
        k=1, A_stacked=A, b_stacked=b, beta_prev=np.zeros(5), beta_new=beta_new,
        total_loss=0.5, rank_gelsy=5, t_assemble_s=0.0, t_solve_s=0.0,
        is_final_iterate=False,
    )
    assert math.isnan(row["order_obs"])
    assert row["stall_flag"] is False


def test_order_obs_and_stall_flag_defined_from_second_step_onward():
    A, b = _synthetic_system()
    beta = np.zeros(5)
    tracker = LilQDiagnosticsTracker(initial_norm_R_h=10.0)

    row1 = tracker.step(
        k=1, A_stacked=A, b_stacked=b, beta_prev=beta, beta_new=beta + 0.1,
        total_loss=4.0, rank_gelsy=5, t_assemble_s=0.0, t_solve_s=0.0,
        is_final_iterate=False,
    )
    row2 = tracker.step(
        k=2, A_stacked=A, b_stacked=b, beta_prev=beta + 0.1, beta_new=beta + 0.15,
        total_loss=1.0, rank_gelsy=5, t_assemble_s=0.0, t_solve_s=0.0,
        is_final_iterate=False,
    )
    assert not math.isnan(row2["order_obs"])
    assert isinstance(row2["stall_flag"], bool)


def test_t_cum_s_accumulates_across_steps():
    A, b = _synthetic_system()
    beta = np.zeros(5)
    tracker = LilQDiagnosticsTracker(initial_norm_R_h=1.0)

    row1 = tracker.step(
        k=1, A_stacked=A, b_stacked=b, beta_prev=beta, beta_new=beta,
        total_loss=1.0, rank_gelsy=5, t_assemble_s=0.01, t_solve_s=0.02,
        is_final_iterate=False,
    )
    row2 = tracker.step(
        k=2, A_stacked=A, b_stacked=b, beta_prev=beta, beta_new=beta,
        total_loss=1.0, rank_gelsy=5, t_assemble_s=0.03, t_solve_s=0.04,
        is_final_iterate=False,
    )
    assert row1["t_cum_s"] == pytest.approx(0.03)
    assert row2["t_cum_s"] == pytest.approx(0.10)


def test_chi_is_nan_without_residual_vector_callback():
    A, b = _synthetic_system()
    beta_new = np.linalg.lstsq(A, b, rcond=None)[0]
    tracker = LilQDiagnosticsTracker(initial_norm_R_h=1.0)
    row = tracker.step(
        k=1, A_stacked=A, b_stacked=b, beta_prev=np.zeros(5), beta_new=beta_new,
        total_loss=0.5, rank_gelsy=5, t_assemble_s=0.0, t_solve_s=0.0,
        is_final_iterate=False,
    )
    assert math.isnan(row["chi"])


def test_chi_is_real_with_residual_vector_callback():
    A, b = _synthetic_system()
    beta_new = np.linalg.lstsq(A, b, rcond=None)[0]
    tracker = LilQDiagnosticsTracker(initial_norm_R_h=1.0)

    def fake_residual_vector_fn(beta):
        return A @ beta - b + 0.01  # deliberately not identical to R_lin_k

    row = tracker.step(
        k=1, A_stacked=A, b_stacked=b, beta_prev=np.zeros(5), beta_new=beta_new,
        total_loss=0.5, rank_gelsy=5, t_assemble_s=0.0, t_solve_s=0.0,
        is_final_iterate=False, compute_residual_vector_fn=fake_residual_vector_fn,
    )
    assert not math.isnan(row["chi"])
    assert row["chi"] > 0  # the +0.01 perturbation guarantees R_next != R_lin_k


def test_small_p_uses_svd_every_iteration_including_non_final():
    A, b = _synthetic_system(n_rows=20, n_cols=5)
    beta_new = np.linalg.lstsq(A, b, rcond=None)[0]
    tracker = LilQDiagnosticsTracker(initial_norm_R_h=1.0, conditioning_svd_threshold=3200)
    row = tracker.step(
        k=1, A_stacked=A, b_stacked=b, beta_prev=np.zeros(5), beta_new=beta_new,
        total_loss=0.5, rank_gelsy=5, t_assemble_s=0.0, t_solve_s=0.0,
        is_final_iterate=False,
    )
    assert row["kappa_method"] == "svd"
    assert row["num_rank_svd"] == 5
    assert not math.isnan(row["kappa"])


def test_large_p_skips_conditioning_except_at_final_iterate():
    A, b = _synthetic_system(n_rows=20, n_cols=5)
    beta_new = np.linalg.lstsq(A, b, rcond=None)[0]
    tracker = LilQDiagnosticsTracker(initial_norm_R_h=1.0, conditioning_svd_threshold=3)  # force "large P" path

    mid_row = tracker.step(
        k=1, A_stacked=A, b_stacked=b, beta_prev=np.zeros(5), beta_new=beta_new,
        total_loss=0.5, rank_gelsy=5, t_assemble_s=0.0, t_solve_s=0.0,
        is_final_iterate=False,
    )
    final_row = tracker.step(
        k=2, A_stacked=A, b_stacked=b, beta_prev=beta_new, beta_new=beta_new,
        total_loss=0.4, rank_gelsy=5, t_assemble_s=0.0, t_solve_s=0.0,
        is_final_iterate=True,
    )

    assert math.isnan(mid_row["kappa"])
    assert mid_row["kappa_method"] is None
    assert mid_row["num_rank_svd"] is None

    assert not math.isnan(final_row["kappa"])
    assert final_row["kappa_method"] == "qr_pivoted"
    assert final_row["num_rank_svd"] is None  # qr_pivoted never reports this


def test_rcond_and_solver_path_are_the_documented_constants():
    A, b = _synthetic_system()
    beta_new = np.linalg.lstsq(A, b, rcond=None)[0]
    tracker = LilQDiagnosticsTracker(initial_norm_R_h=1.0)
    row = tracker.step(
        k=1, A_stacked=A, b_stacked=b, beta_prev=np.zeros(5), beta_new=beta_new,
        total_loss=0.5, rank_gelsy=5, t_assemble_s=0.0, t_solve_s=0.0,
        is_final_iterate=False,
    )
    assert row["rcond"] == EPS_MACH
    assert row["solver_path"] == "cpu_gelsy"


def test_num_rank_gelsy_passes_through_the_given_rank():
    A, b = _synthetic_system()
    beta_new = np.linalg.lstsq(A, b, rcond=None)[0]
    tracker = LilQDiagnosticsTracker(initial_norm_R_h=1.0)
    row = tracker.step(
        k=1, A_stacked=A, b_stacked=b, beta_prev=np.zeros(5), beta_new=beta_new,
        total_loss=0.5, rank_gelsy=3, t_assemble_s=0.0, t_solve_s=0.0,
        is_final_iterate=False,
    )
    assert row["num_rank_gelsy"] == 3


def test_interior_norms_are_nan_documented_limitation():
    A, b = _synthetic_system()
    beta_new = np.linalg.lstsq(A, b, rcond=None)[0]
    tracker = LilQDiagnosticsTracker(initial_norm_R_h=1.0)
    row = tracker.step(
        k=1, A_stacked=A, b_stacked=b, beta_prev=np.zeros(5), beta_new=beta_new,
        total_loss=0.5, rank_gelsy=5, t_assemble_s=0.0, t_solve_s=0.0,
        is_final_iterate=False,
    )
    assert math.isnan(row["norm_R_interior"])
    assert math.isnan(row["norm_Rlin_interior"])
