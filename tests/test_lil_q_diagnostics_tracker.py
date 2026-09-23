"""Tests for lilq.iteration_log.LilQDiagnosticsTracker -- the stateful
helper that turns a LiL-Q solver's raw per-iteration ingredients into
iterations.csv rows, in isolation from any real PDE solve.

Row convention (the package spec's): row k describes outer iteration k,
assembled at beta^(k) and solved for beta^(k+1); it holds ||R^(k)||_h
(the mat-vec A^(k) beta^(k) - f^(k)), ||R_lin^(k)||_h, chi_k, o_k, the
stall flag and kappa(A^(k)). finish() adds a terminal row k=K with
||R^(K)||_h only.
"""

import math

import numpy as np
import pytest

from lilq.instrumentation import EPS_MACH, observed_order, phase_indicator
from lilq.iteration_log import (
    ITERATION_CSV_COLUMNS, IterationLogger, LilQDiagnosticsTracker,
    last_solve_row, solve_rows,
)


def _synthetic_system(n_rows=20, n_cols=5, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_rows, n_cols))
    b = rng.standard_normal(n_rows)
    return A, b


def _step(tracker, A, b, beta_prev, beta_new, k=0, total_loss=0.5, **kw):
    kw.setdefault("rank_gelsy", A.shape[1])
    kw.setdefault("t_assemble_s", 0.0)
    kw.setdefault("t_solve_s", 0.0)
    kw.setdefault("is_final_iterate", False)
    return tracker.step(k=k, A_stacked=A, b_stacked=b, beta_prev=beta_prev,
                        beta_new=beta_new, total_loss=total_loss, **kw)


def test_step_returns_only_valid_schema_columns():
    A, b = _synthetic_system()
    row = _step(LilQDiagnosticsTracker(), A, b, np.zeros(5), np.linalg.lstsq(A, b, rcond=None)[0])
    assert set(row.keys()) <= set(ITERATION_CSV_COLUMNS)
    assert row["k"] == 0
    logger = IterationLogger()
    logger.record(**row)
    assert logger.rows[0]["eps_u"] is None


def test_norm_R_h_is_the_matvec_residual_at_beta_k():
    """Spec item 1: R^(k) = A^(k) beta^(k) - f^(k), at the point the system
    was assembled -- not the post-solve residual."""
    A, b = _synthetic_system()
    beta_k = np.full(5, 0.3)
    beta_next = np.linalg.lstsq(A, b, rcond=None)[0]
    row = _step(LilQDiagnosticsTracker(), A, b, beta_k, beta_next, total_loss=123.0)
    assert row["norm_R_h"] == pytest.approx(float(np.linalg.norm(A @ beta_k - b)))


def test_norm_Rlin_h_is_the_residual_of_the_solution():
    A, b = _synthetic_system()
    beta_next = np.linalg.lstsq(A, b, rcond=None)[0]
    row = _step(LilQDiagnosticsTracker(), A, b, np.zeros(5), beta_next)
    assert row["norm_Rlin_h"] == pytest.approx(float(np.linalg.norm(A @ beta_next - b)))


def test_norm_f_h_is_norm_of_rhs():
    A, b = _synthetic_system()
    row = _step(LilQDiagnosticsTracker(), A, b, np.zeros(5), np.linalg.lstsq(A, b, rcond=None)[0])
    assert row["norm_f_h"] == pytest.approx(float(np.linalg.norm(b)))


def test_norm_dbeta_and_rel_dbeta():
    beta_k = np.array([1.0, 0.0, 0.0])
    beta_next = np.array([1.0, 3.0, 4.0])
    row = _step(LilQDiagnosticsTracker(), np.eye(6, 3), np.zeros(6), beta_k, beta_next)
    assert row["norm_dbeta"] == pytest.approx(5.0)
    assert row["rel_dbeta"] == pytest.approx(5.0 / np.linalg.norm(beta_next))


def test_order_obs_and_stall_flag_undefined_at_k0():
    A, b = _synthetic_system()
    row = _step(LilQDiagnosticsTracker(), A, b, np.zeros(5), np.linalg.lstsq(A, b, rcond=None)[0])
    assert math.isnan(row["order_obs"])
    assert row["stall_flag"] is False


def test_order_obs_at_k1_uses_R_k_plus_1_R_k_and_R_k_minus_1():
    """o_k = ln(R^(k+1)/R^(k)) / ln(R^(k)/R^(k-1)): at k=1, R^(0) and R^(1)
    are the mat-vec residuals of rows 0 and 1, R^(2) = sqrt(total_loss)."""
    A, b = _synthetic_system()
    beta0, beta1, beta2 = np.zeros(5), np.full(5, 0.1), np.full(5, 0.15)
    tracker = LilQDiagnosticsTracker()
    row0 = _step(tracker, A, b, beta0, beta1, k=0, total_loss=4.0)
    row1 = _step(tracker, A, b, beta1, beta2, k=1, total_loss=1.0)
    expected = observed_order(1.0, row1["norm_R_h"], row0["norm_R_h"])
    assert row1["order_obs"] == pytest.approx(expected)
    assert isinstance(row1["stall_flag"], bool)


def test_t_cum_s_accumulates_across_steps():
    A, b = _synthetic_system()
    beta = np.zeros(5)
    tracker = LilQDiagnosticsTracker()
    row0 = _step(tracker, A, b, beta, beta, k=0, t_assemble_s=0.01, t_solve_s=0.02)
    row1 = _step(tracker, A, b, beta, beta, k=1, t_assemble_s=0.03, t_solve_s=0.04)
    assert row0["t_cum_s"] == pytest.approx(0.03)
    assert row1["t_cum_s"] == pytest.approx(0.10)


def test_chi_is_nan_without_residual_vector_callback():
    A, b = _synthetic_system()
    row = _step(LilQDiagnosticsTracker(), A, b, np.zeros(5), np.linalg.lstsq(A, b, rcond=None)[0])
    assert math.isnan(row["chi"])


def test_chi_uses_the_residual_at_beta_k_plus_1():
    A, b = _synthetic_system()
    beta_next = np.linalg.lstsq(A, b, rcond=None)[0]
    fn = lambda beta: A @ beta - b + 0.01  # noqa: E731
    row = _step(LilQDiagnosticsTracker(), A, b, np.zeros(5), beta_next,
                compute_residual_vector_fn=fn)
    assert row["chi"] == pytest.approx(phase_indicator(fn(beta_next), A @ beta_next - b))


def test_b2_is_round_off_for_an_exact_linearization():
    A, b = _synthetic_system()
    tracker = LilQDiagnosticsTracker()
    _step(tracker, A, b, np.full(5, 0.2), np.linalg.lstsq(A, b, rcond=None)[0],
          compute_residual_vector_fn=lambda beta: A @ beta - b)
    assert tracker.b2_check["rel_err"] < 1e-14


def test_b2_check_reports_k1_and_the_run_maximum():
    """rel_err is taken at k=1 (the first iterate away from the start);
    the maximum over the run is reported alongside, not as the check."""
    A, b = _synthetic_system()
    beta_k = np.full(5, 0.2)
    offset = {"value": 0.0}

    def direct(beta):  # "nonlinear operator" disagreeing with A beta - b by a constant
        return A @ beta - b + offset["value"]

    tracker = LilQDiagnosticsTracker()
    for k, off in enumerate([0.0, 1e-3, 0.5]):
        offset["value"] = off
        _step(tracker, A, b, beta_k, beta_k, k=k, compute_residual_vector_fn=direct)

    rel = lambda off: np.linalg.norm(np.full(20, off)) / np.linalg.norm(A @ beta_k - b + off)  # noqa: E731
    check = tracker.b2_check
    assert check["k"] == 1
    assert check["rel_err"] == pytest.approx(rel(1e-3))
    assert check["max_rel_err_over_run"] == pytest.approx(rel(0.5))


def test_b2_check_falls_back_to_k0_for_a_single_solve():
    A, b = _synthetic_system()
    tracker = LilQDiagnosticsTracker()
    _step(tracker, A, b, np.zeros(5), np.zeros(5), k=0,
          compute_residual_vector_fn=lambda beta: A @ beta - b)
    assert tracker.b2_check["k"] == 0


def test_b2_check_is_none_without_residual_vector_callback():
    A, b = _synthetic_system()
    tracker = LilQDiagnosticsTracker()
    _step(tracker, A, b, np.zeros(5), np.zeros(5))
    assert tracker.b2_check is None


def test_small_p_uses_svd_every_iteration_including_non_final():
    A, b = _synthetic_system()
    row = _step(LilQDiagnosticsTracker(conditioning_svd_threshold=3200), A, b,
                np.zeros(5), np.linalg.lstsq(A, b, rcond=None)[0])
    assert row["kappa_method"] == "svd"
    assert row["num_rank_svd"] == 5
    assert not math.isnan(row["kappa"])


def test_large_p_skips_conditioning_except_at_final_iterate_and_keeps_raw_ratio():
    A, b = _synthetic_system()
    beta_next = np.linalg.lstsq(A, b, rcond=None)[0]
    tracker = LilQDiagnosticsTracker(conditioning_svd_threshold=3)  # force the large-P path
    mid = _step(tracker, A, b, np.zeros(5), beta_next, k=0)
    assert tracker.kappa_qr_raw_ratio is None
    final = _step(tracker, A, b, beta_next, beta_next, k=1, is_final_iterate=True)

    assert math.isnan(mid["kappa"]) and mid["kappa_method"] is None and mid["num_rank_svd"] is None
    assert final["kappa_method"] == "qr_pivoted" and not math.isnan(final["kappa"])
    assert final["num_rank_svd"] is None
    assert tracker.kappa_qr_raw_ratio is not None and tracker.kappa_qr_raw_ratio >= final["kappa"]


def test_rcond_and_solver_path_are_the_documented_constants():
    A, b = _synthetic_system()
    row = _step(LilQDiagnosticsTracker(), A, b, np.zeros(5), np.zeros(5))
    assert row["rcond"] == EPS_MACH
    assert row["solver_path"] == "cpu_gelsy"


def test_num_rank_gelsy_passes_through_the_given_rank():
    A, b = _synthetic_system()
    row = _step(LilQDiagnosticsTracker(), A, b, np.zeros(5), np.zeros(5), rank_gelsy=3)
    assert row["num_rank_gelsy"] == 3


def test_interior_norms_are_nan_when_not_configured():
    A, b = _synthetic_system()
    row = _step(LilQDiagnosticsTracker(), A, b, np.zeros(5), np.zeros(5))
    assert math.isnan(row["norm_R_interior"])
    assert math.isnan(row["norm_Rlin_interior"])


def test_interior_norms_unweight_the_leading_rows_when_configured():
    """8 interior rows (weight 2.0) stacked first, then 4 boundary rows --
    every problem's actual convention. Both interior norms come from the
    mat-vec residuals, so no residual callback is needed."""
    n_interior, w = 8, 2.0
    rng = np.random.default_rng(0)
    A = np.vstack([w * rng.standard_normal((n_interior, 3)), 5.0 * rng.standard_normal((4, 3))])
    b = rng.standard_normal(12)
    beta_k = np.full(3, 0.4)
    beta_next = np.linalg.lstsq(A, b, rcond=None)[0]

    row = _step(LilQDiagnosticsTracker(n_interior_rows=n_interior, interior_weight=w),
                A, b, beta_k, beta_next)

    assert row["norm_R_interior"] == pytest.approx(np.linalg.norm((A @ beta_k - b)[:n_interior]) / w)
    assert row["norm_Rlin_interior"] == pytest.approx(np.linalg.norm((A @ beta_next - b)[:n_interior]) / w)


def test_finish_emits_terminal_row_with_final_residual_only():
    n_interior, w = 8, 2.0
    rng = np.random.default_rng(1)
    A = rng.standard_normal((12, 3))
    b = rng.standard_normal(12)
    R_final = rng.standard_normal(12)
    tracker = LilQDiagnosticsTracker(n_interior_rows=n_interior, interior_weight=w)
    _step(tracker, A, b, np.zeros(3), np.ones(3), k=0, total_loss=0.25, t_assemble_s=0.5,
          compute_residual_vector_fn=lambda beta: R_final)

    terminal = tracker.finish(k=1)

    assert terminal["k"] == 1
    assert terminal["norm_R_h"] == pytest.approx(0.5)  # sqrt(total_loss) = ||R^(K)||_h
    assert terminal["norm_R_interior"] == pytest.approx(np.linalg.norm(R_final[:n_interior]) / w)
    assert terminal["t_cum_s"] == pytest.approx(0.5)
    logger = IterationLogger()
    logger.record(**terminal)
    assert logger.rows[0]["norm_Rlin_h"] is None and logger.rows[0]["kappa"] is None


def test_solve_rows_and_last_solve_row_skip_the_terminal_row():
    A, b = _synthetic_system()
    tracker = LilQDiagnosticsTracker()
    logger = IterationLogger()
    for k in range(3):
        logger.record(**_step(tracker, A, b, np.zeros(5), np.zeros(5), k=k, rank_gelsy=k))
    logger.record(**tracker.finish(k=3))

    assert [r["k"] for r in solve_rows(logger.rows)] == [0, 1, 2]
    assert last_solve_row(logger.rows)["num_rank_gelsy"] == 2
