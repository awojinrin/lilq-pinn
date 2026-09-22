"""Tests for the two conditioning methods in lilq.instrumentation
(Computational_Package_1_v2.md Section 3.1 item 8): full-SVD (P<=3200)
and pivoted-QR at the final iterate (Beltrami-scale problems).
"""

import numpy as np
import pytest

from lilq.instrumentation import (
    EPS_MACH,
    conditioning_via_pivoted_qr,
    conditioning_via_svd,
)


def _matrix_with_known_singular_values(singular_values, n_rows, seed=0):
    """Build an (n_rows, len(singular_values)) matrix with exactly the
    given singular values, via A = U @ diag(s) @ V^T for random
    orthonormal U, V -- so the true kappa is known exactly, not just
    self-consistently re-derived from the same SVD call under test."""
    rng = np.random.default_rng(seed)
    n_cols = len(singular_values)
    U, _ = np.linalg.qr(rng.standard_normal((n_rows, n_cols)))
    V, _ = np.linalg.qr(rng.standard_normal((n_cols, n_cols)))
    return U @ np.diag(singular_values) @ V.T


# ── conditioning_via_svd ─────────────────────────────────────────────

def test_svd_kappa_matches_known_singular_values():
    A = _matrix_with_known_singular_values([10.0, 5.0, 2.0, 1.0], n_rows=20)
    result = conditioning_via_svd(A)
    assert result["kappa"] == pytest.approx(10.0, rel=1e-8)
    assert result["kappa_method"] == "svd"


def test_svd_num_rank_is_full_for_a_well_conditioned_matrix():
    A = _matrix_with_known_singular_values([10.0, 5.0, 2.0, 1.0], n_rows=20)
    result = conditioning_via_svd(A)
    assert result["num_rank_svd"] == 4


def test_svd_num_rank_detects_deliberate_rank_deficiency():
    # rank_tol = max(N,P) * sigma_max * EPS_MACH = 20 * 10 * ~2.22e-16
    # ~= 4.4e-14 here -- the three "small" values must sit clearly below
    # that, not just be small in an everyday sense.
    singular_values = [10.0, 5.0, 4.0, 3.0, 2.0, 1e-16, 1e-17, 1e-18]
    A = _matrix_with_known_singular_values(singular_values, n_rows=20)
    result = conditioning_via_svd(A)
    assert result["num_rank_svd"] == 5


# ── conditioning_via_pivoted_qr ──────────────────────────────────────

def test_qr_pivoted_kappa_is_finite_and_reasonable_for_well_conditioned_matrix():
    A = _matrix_with_known_singular_values([10.0, 5.0, 2.0, 1.0], n_rows=20)
    result = conditioning_via_pivoted_qr(A)
    assert result["kappa_method"] == "qr_pivoted"
    assert 1.0 < result["kappa"] < 100.0  # same order of magnitude as the true kappa=10
    assert result["num_rank_svd"] is None


def test_qr_pivoted_retained_ratio_stays_sane_when_raw_ratio_blows_up():
    """The scenario the spec's own retained-vs-raw distinction exists
    for: a matrix with a few near-zero trailing singular values makes
    the naive first/last diagonal ratio meaningless (near-infinite),
    while discarding the pivoted-out columns first recovers a sane
    number describing the well-conditioned retained part."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((20, 5))
    A = np.zeros((20, 8))
    A[:, :5] = base
    A[:, 5] = base[:, 0] * 2.0                                  # exact linear dependence
    A[:, 6] = base[:, 0] * 3.0 + 1e-14 * rng.standard_normal(20)  # near-exact dependence
    A[:, 7] = base[:, 0] * -1.0                                  # exact linear dependence

    result = conditioning_via_pivoted_qr(A)

    assert result["kappa_raw_ratio"] > 1e10       # blows up, as expected
    assert result["kappa"] < 100.0                # retained-only stays sane


def test_qr_pivoted_and_svd_roughly_agree_on_a_well_conditioned_matrix():
    """Not the same algorithm, so not expected to match exactly -- but
    for a genuinely well-conditioned matrix both should land in the same
    order of magnitude, which is the whole point of qr_pivoted being an
    acceptable substitute for svd at a scale where svd is too expensive."""
    A = _matrix_with_known_singular_values([10.0, 5.0, 2.0, 1.0], n_rows=20)
    svd_kappa = conditioning_via_svd(A)["kappa"]
    qr_kappa = conditioning_via_pivoted_qr(A)["kappa"]
    assert qr_kappa == pytest.approx(svd_kappa, rel=2.0)  # same order of magnitude
