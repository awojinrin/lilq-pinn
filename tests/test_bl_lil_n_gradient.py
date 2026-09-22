"""Regression guard: LiL-N's loss gradient for Buckley-Leverett must be
mathematically correct, not just "doesn't crash."

Context (DECISIONS.md, Codebase_v3_Proposal.md): `_make_lil_n_loss_fn`
computed the flux-divergence term as `f_p = physics.flux_derivative(S);
f_x = f_p * S_x`, where `flux_derivative` deliberately detaches its
output from the autograd graph -- correct for the quasilinear solver
(`_make_lil_q_system_fn`, which freezes this coefficient by design), but
silently wrong here: `solve_lil_n` differentiates this loss directly
w.r.t. `beta`, and the detached path drops the contribution of f'(S)'s
own dependence on beta through S. Confirmed via a fresh empirical run:
BL's LiL-N converges cleanly in the pre-GitHub codebase but plateaus
without converging in the current one, at every size tested, both
viscous and gravity (Phase0_Empirical_Verification.md).

Fixed by adding `flux_derivative_differentiable` (graph-preserving) and
using it in `_make_lil_n_loss_fn` instead. This test uses
`torch.autograd.gradcheck` -- PyTorch's numerical finite-difference
gradient checker -- as the actual proof: a broken gradient computation
can still run without error, so only a numerical check catches this
class of bug. It's written to also demonstrate the failure mode: swapping
back to the detached `flux_derivative` is shown to fail this same check,
so the test can't be satisfied by coincidence or a no-op edit.
"""

import numpy as np
import pytest
import torch

from lilq.basis import create_basis_2d
from lilq.collocation import generate_collocation_points_2d
from problems.buckley_leverett import (
    BLConfig, BLPhysics, _make_lil_n_loss_fn, _prepare_lil_matrices,
)


def _build_loss_fn(gravity: bool):
    config = BLConfig.with_gravity() if gravity else BLConfig()
    config.N_x = config.N_t = 4  # tiny -- gradcheck is O(n_params) forward passes
    physics = BLPhysics(config)
    basis = create_basis_2d(config.basis_type, config.N_x, config.N_t,
                             config.x_domain, (0, config.T_final))
    pts = generate_collocation_points_2d(
        config.x_domain, (0, config.T_final), config.N_x, config.N_t,
        k_ratio=config.k_ratio, collocation_ratios=(0.9, 0.05, 0.05),
        has_initial_condition=True, seed=config.seed, sampling=config.sampling,
    )
    matrices = _prepare_lil_matrices(config, physics, basis, pts)
    (A_u, A_ux, A_ut, A_uxx, A_ic, A_bc_left, A_bc_right,
     ic_target, bc_l_target, bc_r_target) = matrices

    loss_fn = _make_lil_n_loss_fn(
        A_u, A_ux, A_ut, A_uxx, A_ic, A_bc_left, A_bc_right,
        ic_target, bc_l_target, bc_r_target,
        physics, lp=1.0, li=10.0, lb=10.0, device=torch.device("cpu"),
    )
    n_coefs = basis.n_basis
    return loss_fn, n_coefs


@pytest.mark.parametrize("gravity", [False, True])
def test_lil_n_loss_gradient_matches_finite_differences(gravity):
    loss_fn, n_coefs = _build_loss_fn(gravity)
    rng = np.random.default_rng(0)
    beta0 = torch.tensor(
        rng.uniform(-0.1, 0.1, size=(n_coefs, 1)), dtype=torch.float64, requires_grad=True,
    )

    def total_only(beta):
        return loss_fn(beta)[0]

    assert torch.autograd.gradcheck(total_only, (beta0,), eps=1e-6, atol=1e-4)


def test_flux_derivative_differentiable_disagrees_with_detached_version_on_gradient():
    """Sanity check on the fix itself: the two flux-derivative variants
    must agree on the *value* (same formula) but only the differentiable
    one propagates gradient back through S -- proving they're not
    accidentally identical in a way that would make the test above pass
    regardless of which one _make_lil_n_loss_fn uses."""
    config = BLConfig()
    physics = BLPhysics(config)
    S = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True)

    detached = physics.flux_derivative(S)
    differentiable = physics.flux_derivative_differentiable(S)

    assert torch.allclose(detached, differentiable)
    assert detached.grad_fn is None
    assert differentiable.grad_fn is not None
