"""Task B9 (Addendum v2.1): delta_FV against the TPFA solution, the TPFA
self-residual, and the NiL network's precision and seeding."""

import numpy as np
import pytest
import torch

from problems.darcy import (
    DarcyConfig, DarcyPhysics, DarcyPINN, assemble_fvm, delta_fv, solve_fvm, tpfa_residual,
)


@pytest.fixture(scope='module')
def s1():
    physics = DarcyPhysics(DarcyConfig(perm_file='perm_field_S1.txt'), verbose=False)
    return physics, solve_fvm(physics)


def test_delta_fv_normalizes_by_the_pressure_increment():
    p_fv = np.array([3000.0, 3100.0, 3300.0])
    p_h = p_fv + np.array([0.0, 3.0, 4.0])
    assert delta_fv(p_h, p_fv, 3000.0) == pytest.approx(5.0 / np.sqrt(100.0**2 + 300.0**2))


def test_tpfa_system_is_solved_to_round_off(s1):
    physics, P_fvm = s1
    A, b = assemble_fvm(physics)
    assert A.shape == (60 * 220, 60 * 220) and P_fvm.shape == (60, 220)
    assert tpfa_residual(physics, P_fvm)['tpfa_residual_rel'] < 1e-12


def test_fvm_pressure_lies_between_the_dirichlet_values(s1):
    physics, P_fvm = s1
    cfg = physics.config
    lo, hi = sorted((cfg.P_BOTTOM, cfg.P_TOP))
    assert lo <= P_fvm.min() and P_fvm.max() <= hi


def test_darcy_pinn_is_float64_by_default_and_seeded(s1):
    physics, _ = s1
    a = DarcyPINN(physics.config, physics, hidden_dim=8, num_layers=2, device='cpu', seed=3)
    b = DarcyPINN(physics.config, physics, hidden_dim=8, num_layers=2, device='cpu', seed=3)
    assert all(p.dtype == torch.float64 for p in a.net_P.parameters())
    assert a.xpde.dtype == torch.float64
    for pa, pb in zip(a.net_U.parameters(), b.net_U.parameters()):
        assert torch.equal(pa, pb)
    single = DarcyPINN(physics.config, physics, hidden_dim=8, num_layers=2, device='cpu',
                       dtype=torch.float32, seed=3)
    assert all(p.dtype == torch.float32 for p in single.net_P.parameters())
    assert single.predict(np.array([[10.0]]), np.array([[20.0]]))['P'].dtype == np.float32
