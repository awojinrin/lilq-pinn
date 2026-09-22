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

import math

import pytest

from experiments.run_burgers_basis_comparison import create_comparison_basis
from lilq.basis import ELMBasis2D_Xavier


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
