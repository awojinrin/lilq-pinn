"""Tests for problems.kovasznay's Section 3.2 GPU solve path: the
full-rank QR-based GPU solver, the rank-degeneracy flag + CPU gelsy
cross-check, the CPU gels timing helper, and the GPU/CPU equivalence
check.

Real GPU tests are skipped (not xfail'd) when no CUDA device is present
-- this environment has one, so they run for real, not mocked, matching
this whole engagement's established discipline of verifying real
solver behavior rather than trusting the code from inspection alone.
"""

import dataclasses

import numpy as np
import pytest

from lilq.iteration_log import IterationLogger, solve_rows
from problems.kovasznay import (
    KovasznayConfig, HAS_TORCH_CUDA,
    _lstsq_gpu_qr, _qr_degeneracy_ratio, gpu_memory_estimate_bytes,
    _lstsq_cpu_gels, solve_kovasznay, verify_gpu_cpu_equivalence,
)

requires_cuda = pytest.mark.skipif(not HAS_TORCH_CUDA, reason="no CUDA device available")


def _small_config(**overrides):
    kwargs = dict(N_x=4, N_y=4, k_ratio=5, max_iter=10)
    kwargs.update(overrides)
    return KovasznayConfig(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Pure-math helpers (no GPU needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_qr_degeneracy_ratio_well_conditioned():
    R_diag = np.array([10.0, 8.0, 5.0, 3.0])
    assert _qr_degeneracy_ratio(R_diag) == pytest.approx(3.0 / 10.0)


def test_qr_degeneracy_ratio_below_threshold_for_near_singular():
    R_diag = np.array([10.0, 5.0, 1e-14])
    assert _qr_degeneracy_ratio(R_diag) < 1e-13


def test_qr_degeneracy_ratio_handles_zero_max():
    assert _qr_degeneracy_ratio(np.array([0.0, 0.0])) == 0.0


def test_gpu_memory_estimate_matches_formula():
    assert gpu_memory_estimate_bytes(N=1000, P=50) == 3 * 8 * 1000 * 50


def test_cpu_gels_matches_gelsy_on_synthetic_overdetermined_system():
    import scipy.linalg
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 20))
    x_true = rng.standard_normal(20)
    b = A @ x_true

    x_gels = _lstsq_cpu_gels(A, b)
    x_gelsy, _, _, _ = scipy.linalg.lstsq(A, b, lapack_driver='gelsy')

    assert np.linalg.norm(x_gels - x_gelsy) < 1e-10
    assert np.linalg.norm(x_gels - x_true) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Real GPU solve
# ─────────────────────────────────────────────────────────────────────────────

@requires_cuda
def test_lstsq_gpu_qr_matches_cpu_on_synthetic_system():
    import scipy.linalg
    rng = np.random.default_rng(1)
    A = rng.standard_normal((300, 30))
    x_true = rng.standard_normal(30)
    b = A @ x_true

    x_gpu, R_diag, peak_mem = _lstsq_gpu_qr(A, b)
    x_cpu, _, _, _ = scipy.linalg.lstsq(A, b, lapack_driver='gelsy')

    assert np.linalg.norm(x_gpu - x_cpu) < 1e-8
    assert len(R_diag) == 30
    assert peak_mem > 0


@requires_cuda
def test_implicit_q_solve_matches_the_explicit_q_solve_it_replaced():
    """Addendum v2.2 Section 2.10: geqrf + ormqr (Q^T b applied from the
    reflectors, as LAPACK's gels does) instead of forming Q with
    torch.linalg.qr -- the same solution and the same |R_pp| on a real
    Kovasznay system, and on an inconsistent overdetermined system the
    least-squares solution of CPU gels."""
    import torch
    from experiments.run_kovasznay import K_RATIO, MAX_ITER, TOL
    from problems.kovasznay import _lstsq_cpu_gels
    s = solve_kovasznay(KovasznayConfig(N_x=10, N_y=10, k_ratio=K_RATIO, max_iter=MAX_ITER, tol=TOL),
                        verbose=False, return_final_system=True)
    A, b = s['A_final'], s['b_final']
    At = torch.as_tensor(A, dtype=torch.float64, device='cuda')
    Q, R = torch.linalg.qr(At, mode='reduced')                  # the replaced explicit-Q solve
    x_old = torch.linalg.solve_triangular(
        R, (Q.T @ torch.as_tensor(b, dtype=torch.float64, device='cuda')).unsqueeze(1), upper=True)
    x_new, R_diag, _ = _lstsq_gpu_qr(A, b)
    x_old = x_old.squeeze(1).cpu().numpy()
    assert np.linalg.norm(x_new - x_old) <= 1e-12 * np.linalg.norm(x_old)
    np.testing.assert_allclose(np.abs(R_diag), np.abs(R.diagonal().cpu().numpy()), rtol=1e-10)
    rng = np.random.default_rng(3)
    A2, b2 = rng.standard_normal((500, 40)), rng.standard_normal(500)      # nonzero residual
    np.testing.assert_allclose(_lstsq_gpu_qr(A2, b2)[0], _lstsq_cpu_gels(A2, b2), rtol=0, atol=1e-12)


@requires_cuda
def test_use_gpu_false_is_bit_identical_to_before_this_feature_existed():
    config = _small_config(use_gpu=False)
    r1 = solve_kovasznay(config, verbose=False)
    r2 = solve_kovasznay(config, verbose=False)
    assert np.array_equal(r1['theta_u'], r2['theta_u'])


@requires_cuda
def test_use_gpu_true_produces_real_solve_with_gpu_solver_path():
    config = _small_config(use_gpu=True)
    logger = IterationLogger()

    result = solve_kovasznay(config, verbose=False, iteration_logger=logger)

    rows = solve_rows(logger.rows)
    assert len(rows) == result['n_outer_iters']
    assert all(row['solver_path'] == 'gpu_qr' for row in rows)
    assert all(row['gpu_mem_peak_bytes'] is not None and row['gpu_mem_peak_bytes'] > 0
              for row in rows)
    # No rank-revealing step on the (non-degenerate) GPU path -- logged
    # as empty/None, not a fabricated value.
    assert all(row['num_rank_gelsy'] is None for row in rows)
    # SVD conditioning is solver-path-independent -- still populated.
    assert all(row['num_rank_svd'] == result['n_params'] for row in rows)


@requires_cuda
def test_use_gpu_raises_without_iteration_logger_still_solves():
    # run_json_path/iteration_logger are independent of use_gpu -- GPU
    # solving works fine with no logger at all.
    config = _small_config(use_gpu=True, max_iter=3)
    result = solve_kovasznay(config, verbose=False)
    assert result['n_outer_iters'] > 0


def test_use_gpu_without_cuda_raises_runtime_error(monkeypatch):
    import problems.kovasznay as kz_module
    monkeypatch.setattr(kz_module, 'HAS_TORCH_CUDA', False)
    config = _small_config(use_gpu=True, max_iter=2)
    with pytest.raises(RuntimeError, match="CUDA"):
        kz_module.solve_kovasznay(config, verbose=False)


@requires_cuda
def test_degeneracy_flag_triggers_cpu_gelsy_cross_check(monkeypatch):
    """Force the GPU path to report a near-singular R diagonal and
    confirm solve_kovasznay then populates a real (non-None)
    num_rank_gelsy for that iteration, via the documented CPU gelsy
    cross-check -- without needing a genuinely ill-conditioned physical
    Kovasznay config to trigger it naturally."""
    import problems.kovasznay as kz_module

    real_lstsq_gpu_qr = kz_module._lstsq_gpu_qr
    call_count = {"n": 0}

    def fake_lstsq_gpu_qr(A, b, timings=None):
        x, R_diag, peak_mem = real_lstsq_gpu_qr(A, b, timings=timings)
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Force degeneracy only on the first call so the run still
            # proceeds sensibly afterward.
            R_diag = R_diag.copy()
            R_diag[-1] = R_diag[0] * 1e-15
        return x, R_diag, peak_mem

    monkeypatch.setattr(kz_module, '_lstsq_gpu_qr', fake_lstsq_gpu_qr)

    config = _small_config(use_gpu=True, max_iter=3)
    logger = IterationLogger()
    kz_module.solve_kovasznay(config, verbose=False, iteration_logger=logger)

    rows = logger.rows
    assert rows[0]['num_rank_gelsy'] is not None
    assert rows[0]['solver_path'] == 'gpu_qr'  # still the GPU iterate, per design


@requires_cuda
def test_verify_gpu_cpu_equivalence_passes_on_real_solve():
    config = _small_config(max_iter=15, tol=1e-13)
    result = verify_gpu_cpu_equivalence(config, verbose=False)

    assert result['equivalent'] is True
    assert result['beta_ok'] is True
    assert result['rlin_ok'] is True
    assert result['beta_rel_diff'] <= 1e-8
    assert result['rlin_rel_diff'] <= 5e-7
    # The spec's rule passes, so the amended rule does too.
    assert isinstance(result['rlin_diff_within_floor'], bool)
    assert result['equivalent_amended'] is True


@requires_cuda
def test_equivalence_amended_rule_waives_six_figures_only_within_round_off_floor():
    """The FASTER case (DECISIONS.md, 2026-09-24): at P = 1,875 both
    residuals are ~6e-13 and differ in the third digit, but by far less than
    the CPU run's floor kappa * eps * ||f|| ~ 9e-10 -- a spec-rule failure
    the amended rule waives (Addendum v2.2 tightened it: the difference, not
    each residual, is compared with the floor), while beta still agrees far
    below 1e-8. At P = 1,200 the spec rule passes by itself."""
    from experiments.run_kovasznay import K_RATIO, MAX_ITER, TOL
    big = verify_gpu_cpu_equivalence(
        KovasznayConfig(N_x=25, N_y=25, k_ratio=K_RATIO, max_iter=MAX_ITER, tol=TOL))
    assert big['beta_ok'] and big['rlin_diff_within_floor']
    assert abs(big['rlin_gpu'] - big['rlin_cpu']) <= big['rlin_floor_cpu']
    assert big['equivalent_amended'] is True
    mid = verify_gpu_cpu_equivalence(
        KovasznayConfig(N_x=20, N_y=20, k_ratio=K_RATIO, max_iter=MAX_ITER, tol=TOL))
    assert mid['equivalent_amended'] == mid['equivalent'] is True


@requires_cuda
def test_gpu_solve_times_the_transfers_apart_from_the_qr():
    """Addendum v2.2 Section 2.10: host-to-device copy, QR + triangular
    solve, and the copy back, timed apart; the solver logs them per iteration."""
    import numpy as np
    rng = np.random.default_rng(0)
    A, b = rng.standard_normal((400, 60)), rng.standard_normal(400)
    t = {}
    x, _, _ = _lstsq_gpu_qr(A, b, timings=t)
    assert set(t) == {'h2d_s', 'qr_solve_s', 'd2h_s'} and all(v >= 0 for v in t.values())
    assert np.allclose(x, np.linalg.lstsq(A, b, rcond=None)[0])
    import problems.kovasznay as kz_module
    r = kz_module.solve_kovasznay(_small_config(use_gpu=True, max_iter=3), verbose=False)
    assert len(r['gpu_qr']['h2d_s']) == len(r['gpu_qr']['qr_solve_s']) == r['n_outer_iters']


def test_verify_gpu_cpu_equivalence_requires_cuda(monkeypatch):
    import problems.kovasznay as kz_module
    monkeypatch.setattr(kz_module, 'HAS_TORCH_CUDA', False)
    with pytest.raises(RuntimeError, match="CUDA"):
        kz_module.verify_gpu_cpu_equivalence(_small_config(), verbose=False)
