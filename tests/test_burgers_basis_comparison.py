"""Regression guard for the Burgers basis-comparison script's ELM basis.

Context (Codebase_v3_Proposal.md S1.2): ``create_comparison_basis('elm', ...)``
instantiated ``ELMBasis2D`` (fixed bound, +/-1.2247 regardless of size), while
the paper's Table 3/4 ELM row (final loss 5.0e-2 at n_hidden=625) was produced
with ``ELMBasis2D_Xavier`` (bound scales with size; ~+/-0.098 at n_hidden=625).
Confirmed by matching the repository's own stored reference result
(``reference_results/burgers_basis_comparison/burgers_basis_results.json``,
final_loss ~1.77e-5) to what the wrong class produces. Fixed by swapping the
instantiated class; this test locks the correct class and bound in place.
"""

import csv
import math

import pytest

from experiments.run_burgers_basis_comparison import (
    BASIS_CONFIGS, ComparisonConfig, create_comparison_basis,
    generate_collocation, run_table3_study, solve_lilq_burgers_comparison,
    write_table3_csv,
)
from lilq.basis import ELMBasis2D_Xavier, Fourier1D, TensorProductBasis2D
from lilq.iteration_log import IterationLogger


def test_elm_basis_uses_xavier_scaled_class():
    basis, n_coeffs, desc = create_comparison_basis(
        "elm", N_x=25, N_t=25, x_domain=(-1.0, 1.0), t_domain=(0.0, 1.0), seed=42,
    )

    assert isinstance(basis, ELMBasis2D_Xavier)
    assert n_coeffs == 625


def test_elm_basis_bound_matches_xavier_formula():
    n_hidden = 25 * 25  # 625, matching the paper's Table 3/4 configuration
    expected_limit = math.sqrt(6.0 / (2 + n_hidden))  # fan_in=2, fan_out=n_hidden

    basis, _, _ = create_comparison_basis(
        "elm", N_x=25, N_t=25, x_domain=(-1.0, 1.0), t_domain=(0.0, 1.0), seed=42,
    )

    observed_bound = basis.alpha.abs().max().item()
    assert observed_bound <= expected_limit
    assert observed_bound == pytest.approx(expected_limit, rel=0.05)


# =============================================================================
# Section 3.6: Table 3 (Burgers basis study) completion
# =============================================================================

def test_sin_fourier_basis_is_sin_x_by_cosplussin_t():
    """New basis key added for Section 3.6's "Sin x {Cos,Sin}" row --
    sin-only in x (respects the homogeneous Dirichlet BCs), full
    cos+sin Fourier in t (unlike sin_cheb/sin_sin, doesn't rely on a
    Chebyshev temporal basis)."""
    basis, n_coeffs, desc = create_comparison_basis(
        "sin_fourier", N_x=25, N_t=25, x_domain=(-1.0, 1.0), t_domain=(0.0, 1.0), seed=42,
    )

    assert isinstance(basis, TensorProductBasis2D)
    assert n_coeffs == 625
    assert isinstance(basis.basis_x, Fourier1D) and basis.basis_x.mode == 'sin'
    assert isinstance(basis.basis_y, Fourier1D) and basis.basis_y.mode == 'both'


def test_sin_fourier_registered_in_basis_configs():
    assert "sin_fourier" in BASIS_CONFIGS
    entry = BASIS_CONFIGS["sin_fourier"]
    assert "label" in entry and "short_label" in entry
    assert "color" in entry and "marker" in entry


def _tiny_pts_and_basis(config, basis_key="sin_cheb"):
    pts = generate_collocation(config)
    basis, n_coeffs, _ = create_comparison_basis(
        basis_key, config.N_x, config.N_t, config.x_domain, (0.0, config.T_final), config.seed,
    )
    return pts, basis, n_coeffs


def test_disable_stopping_rule_runs_full_iteration_budget():
    """Section 3.6's ``disable_stopping_rule`` must skip both the R_tol
    check and the stagnation-window early break, so a trivially-satisfied
    R_tol (1.0) still runs every iteration up to max_quasi_iters -- while
    the default (False) behavior stops as soon as R_tol is satisfied."""
    common = dict(N_x=4, N_t=4, R_tol=1.0, max_quasi_iters=5)
    config_stops_early = ComparisonConfig(disable_stopping_rule=False, **common)
    config_runs_full = ComparisonConfig(disable_stopping_rule=True, **common)

    pts, basis, _ = _tiny_pts_and_basis(config_stops_early)

    logger_early = IterationLogger()
    solve_lilq_burgers_comparison(
        "sin_cheb", basis, config_stops_early, pts, verbose=False, iteration_logger=logger_early,
    )
    logger_full = IterationLogger()
    solve_lilq_burgers_comparison(
        "sin_cheb", basis, config_runs_full, pts, verbose=False, iteration_logger=logger_full,
    )

    assert len(logger_early.rows) < 5
    assert len(logger_full.rows) == 5


def test_run_table3_study_rejects_config_without_disable_stopping_rule():
    bad_config = ComparisonConfig(N_x=4, N_t=4, max_quasi_iters=5,
                                   disable_stopping_rule=False)
    with pytest.raises(ValueError, match="disable_stopping_rule"):
        run_table3_study(bad_config, basis_keys=["sin_cheb"])


def test_run_table3_study_small_scale_populates_real_diagnostics():
    config = ComparisonConfig(N_x=4, N_t=4, disable_stopping_rule=True,
                               max_quasi_iters=5)
    results = run_table3_study(config, basis_keys=["sin_cheb", "sin_fourier", "elm"],
                                verbose=False)

    assert set(results.keys()) == {"sin_cheb", "sin_fourier", "elm"}
    for bk, entry in results.items():
        assert entry["n_coefficients"] == 16
        rows = entry["logger"].rows
        assert len(rows) == 5  # full budget, stopping rule disabled
        last = rows[-1]
        assert math.isfinite(last["norm_R_h"])
        assert math.isfinite(last["kappa"])
        assert last["num_rank_svd"] == 16
        assert last["num_rank_gelsy"] == 16


def test_run_table3_study_defaults_to_all_nine_bases():
    config = ComparisonConfig(N_x=4, N_t=4, disable_stopping_rule=True,
                               max_quasi_iters=2)
    results = run_table3_study(config, verbose=False)
    assert set(results.keys()) == set(BASIS_CONFIGS.keys())
    assert len(results) == 9


def test_write_table3_csv_roundtrip(tmp_path):
    config = ComparisonConfig(N_x=4, N_t=4, disable_stopping_rule=True,
                               max_quasi_iters=3)
    results = run_table3_study(config, basis_keys=["sin_cheb", "cheb_cheb"], verbose=False)

    out_path = tmp_path / "table3_test.csv"
    write_table3_csv(results, out_path)

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    keys = {r["basis_key"] for r in rows}
    assert keys == {"sin_cheb", "cheb_cheb"}
    for r in rows:
        assert r["P"] == "16"
        assert r["K_max"] == "3"
        assert float(r["final_R_h_squared"]) >= 0.0
        assert float(r["kappa"]) > 0.0
