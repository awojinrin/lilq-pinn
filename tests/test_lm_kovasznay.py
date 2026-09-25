"""Component A, family F2: the advisor's checks of the Levenberg-Marquardt
reference (Addendum v2.1 Section 6), reproduced: hard boundary values,
residual against an independent autograd evaluation, the exact solution
satisfying the equations, the Jacobian against central differences, and
check A2 (two runs with the same seed agree over 20 steps)."""

import argparse
import csv

import pytest
import torch
from torch.func import vmap

import baselines.lm_kovasznay as lm


@pytest.fixture(autouse=True)
def float64_default():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.fixture
def small():
    torch.manual_seed(0)
    model = lm.FourierMLP(12, 2, 8, 1.0, 0)
    theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    xy = torch.tensor([[0.1, 0.3], [0.7, -0.2], [-0.3, 1.1]])
    return model, theta, lm.Residual(model, xy, torch.tensor([lm.X0, lm.Y0]))


def _autograd_ns(fields_at, z):
    z = z.clone().requires_grad_(True)
    u, v, p = fields_at(z)
    g = [torch.autograd.grad(q, z, create_graph=True)[0] for q in (u, v, p)]
    second = lambda gq, i: torch.autograd.grad(gq[i], z, create_graph=True)[0][i]  # noqa: E731
    lap_u = second(g[0], 0) + second(g[0], 1)
    lap_v = second(g[1], 0) + second(g[1], 1)
    return torch.stack([u * g[0][0] + v * g[0][1] + g[2][0] - lm.NU * lap_u,
                        u * g[1][0] + v * g[1][1] + g[2][1] - lm.NU * lap_v,
                        g[0][0] + g[1][1]])


def test_hard_boundary_values(small):
    model, theta, res = small
    params = res.unflatten(theta)
    bpts = torch.tensor([[lm.X0, 0.2], [lm.X1, 0.9], [0.4, lm.Y0], [-0.1, lm.Y1],
                         [lm.X0, lm.Y0], [lm.X1, lm.Y1]])
    out = vmap(lambda z: lm.fields(model, params, z))(bpts)
    ue, ve, _ = lm.exact(bpts[:, 0], bpts[:, 1])
    assert (out[:, 0] - ue).abs().max() < 1e-14 and (out[:, 1] - ve).abs().max() < 1e-14


def test_residual_matches_independent_autograd(small):
    model, theta, res = small
    params = res.unflatten(theta)
    ours = res.point_residual(theta, res.xy_int[0])
    ref = _autograd_ns(lambda z: lm.fields(model, params, z), res.xy_int[0])
    assert (ours - ref).abs().max() < 1e-12


def test_exact_solution_satisfies_the_equations():
    r = _autograd_ns(lambda z: torch.stack(lm.exact(z[0], z[1])), torch.tensor([0.3, 0.4]))
    assert r.abs().max() < 1e-12


def test_jacobian_matches_central_differences(small):
    _model, theta, res = small
    J = res.jacobian(theta, 2)
    torch.manual_seed(1)
    worst, h = 0.0, 1e-6
    for i in torch.randperm(theta.numel())[:8]:
        e = torch.zeros_like(theta); e[i] = h
        fd = (res.vector(theta + e, 2) - res.vector(theta - e, 2)) / (2 * h)
        worst = max(worst, ((fd - J[:, i]).abs().max() / (J[:, i].abs().max() + 1e-30)).item())
    assert J.shape == (3 * 3 + 1, theta.numel()) and worst < 1e-7


def test_check_a2_same_seed_runs_are_identical(tmp_path):
    def run(out):
        # 150 points x 3 rows > 273 parameters: a real least-squares fit, not an
        # exact one (an exact fit ends early when the damping overflows).
        args = argparse.Namespace(width=10, depth=2, m=6, sigma_ff=1.0, n_int=150, w_int=1.0, w_pin=1.0,
                                  mu0=1e-3, diag_floor=1e-12, budget_s=600.0, max_steps=20,
                                  max_params=20000, chunk=50, test_every=100, seed=0, device='cpu',
                                  out=str(out))
        lm.lm_train(args)
        with open(out / 'log.csv', newline='') as f:
            return [row['loss_total'] for row in csv.DictReader(f)]
    first, second = run(tmp_path / 'a'), run(tmp_path / 'b')
    assert len(first) == 20 and first == second
