"""Tests for lilq.instrumentation -- the four pure-math per-iteration
diagnostics from Computational_Package_1_v2.md Section 3.1 (items 4-7):
phase indicator, observed order, stall flag, round-off comparison.

Correctness is checked two ways: (1) hand-computed values against the
written formula, including degenerate/edge cases, and (2) a synthetic
residual sequence with a *known* convergence order, confirming
observed_order recovers the mathematically correct answer -- not just
that the code matches the formula as transcribed, but that the formula
means what it's supposed to mean.
"""

import math

import numpy as np
import pytest

from lilq.instrumentation import (
    EPS_MACH,
    observed_order,
    phase_indicator,
    roundoff_comparison,
    stall_flag,
)


# ── phase_indicator ──────────────────────────────────────────────────

def test_phase_indicator_hand_computed():
    R_next = np.array([3.0, 4.0])       # norm 5
    R_lin_k = np.array([0.0, 0.0])      # norm 0 -- difference is R_next itself
    # denom is 0 here, so use a nonzero R_lin_k instead for the real case:
    R_lin_k = np.array([3.0, 0.0])      # norm 3
    diff_norm = np.linalg.norm(R_next - R_lin_k)  # ||[0,4]|| = 4
    expected = diff_norm / 3.0
    assert phase_indicator(R_next, R_lin_k) == pytest.approx(expected)


def test_phase_indicator_zero_when_linear_step_predicted_exactly():
    # R_next == R_lin_k means the linear model was exact -- chi_k must be 0.
    v = np.array([1.5, -2.3, 0.7])
    assert phase_indicator(v, v) == pytest.approx(0.0)


def test_phase_indicator_nan_on_zero_denominator():
    R_next = np.array([1.0, 2.0])
    R_lin_k = np.zeros(2)
    assert math.isnan(phase_indicator(R_next, R_lin_k))


# ── observed_order ───────────────────────────────────────────────────

def test_observed_order_hand_computed():
    # o_k = ln(4/2) / ln(2/1) = ln(2)/ln(2) = 1
    assert observed_order(norm_R_next=4.0, norm_R_k=2.0, norm_R_km1=1.0) == pytest.approx(1.0)


def test_observed_order_recovers_linear_convergence():
    """R_k = r^k (constant ratio r between consecutive norms) must give
    o_k == 1 exactly, at every k -- this is what "linear convergence"
    means, and the formula should say so unambiguously."""
    r = 0.5
    R = [r ** k for k in range(6)]
    for k in range(1, 5):
        o_k = observed_order(R[k + 1], R[k], R[k - 1])
        assert o_k == pytest.approx(1.0, abs=1e-12)


def test_observed_order_recovers_quadratic_convergence():
    """R_k = r^(2^k) is the textbook quadratic-convergence sequence
    (each step squares the previous error, up to the constant r) --
    o_k must come out to 2 exactly, at every k."""
    r = 0.5
    R = [r ** (2 ** k) for k in range(5)]  # R[0]=0.5, R[1]=0.25, R[2]=0.0625, ...
    for k in range(1, 3):
        o_k = observed_order(R[k + 1], R[k], R[k - 1])
        assert o_k == pytest.approx(2.0, abs=1e-9)


def test_observed_order_nan_on_nonpositive_norm():
    assert math.isnan(observed_order(1.0, 0.0, 1.0))
    assert math.isnan(observed_order(1.0, 1.0, -1.0))


def test_observed_order_nan_on_degenerate_zero_denominator_log():
    # norm_R_k == norm_R_km1 -> ln(1) == 0 -> order undefined, not inf.
    assert math.isnan(observed_order(norm_R_next=2.0, norm_R_k=1.0, norm_R_km1=1.0))


# ── stall_flag ────────────────────────────────────────────────────────

def test_stall_flag_true_when_both_conditions_hold():
    # chi_k <= 0.1 and |Rlin_k - Rlin_km1| <= 0.01 * Rlin_k
    assert stall_flag(chi_k=0.05, norm_Rlin_k=1.0, norm_Rlin_km1=1.005) is True


def test_stall_flag_false_when_chi_too_large():
    assert stall_flag(chi_k=0.5, norm_Rlin_k=1.0, norm_Rlin_km1=1.0) is False


def test_stall_flag_false_when_residual_still_moving():
    # chi_k small, but the linearized residual dropped by 50% -- not stalled.
    assert stall_flag(chi_k=0.01, norm_Rlin_k=0.5, norm_Rlin_km1=1.0) is False


def test_stall_flag_respects_custom_tolerances():
    # A 2% change: a stall at the earlier tau_r = 0.1, not at the default 0.01.
    assert stall_flag(chi_k=0.05, norm_Rlin_k=1.0, norm_Rlin_km1=1.02) is False
    assert stall_flag(chi_k=0.05, norm_Rlin_k=1.0, norm_Rlin_km1=1.02, tau_r=0.1) is True
    assert stall_flag(chi_k=0.05, norm_Rlin_k=1.0, norm_Rlin_km1=1.02, tau_chi=0.01, tau_r=0.1) is False


def test_stall_flag_false_on_nan_chi():
    assert stall_flag(chi_k=float("nan"), norm_Rlin_k=1.0, norm_Rlin_km1=1.0) is False


# ── roundoff_comparison ──────────────────────────────────────────────

def test_roundoff_comparison_hand_computed():
    ratio, floor = roundoff_comparison(norm_Rlin_k=1e-8, norm_f_k=2.0, kappa=1e10)
    assert ratio == pytest.approx(5e-9)
    assert floor == pytest.approx(1e10 * EPS_MACH)


def test_roundoff_comparison_nan_ratio_on_zero_rhs():
    ratio, floor = roundoff_comparison(norm_Rlin_k=1e-8, norm_f_k=0.0, kappa=1e10)
    assert math.isnan(ratio)
    assert floor == pytest.approx(1e10 * EPS_MACH)  # floor is still well-defined


def test_eps_mach_matches_ieee_double_precision():
    assert EPS_MACH == pytest.approx(2.2e-16, rel=1e-2)
    assert EPS_MACH == np.finfo(np.float64).eps
