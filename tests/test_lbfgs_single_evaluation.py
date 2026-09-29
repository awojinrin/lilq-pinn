"""Addendum v2.2 Sections 2.1-2.3: each L-BFGS point evaluated once
(bitwise-identical trajectories, fewer real evaluations), the
``optimizer_stall`` stop, and the stopping-reason bookkeeping."""

import functools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lilq.solvers as solvers
import problems.bratu as bratu
from lilq.four_method_log import stopping_fields
from lilq.solvers import LBFGSObjective, solve_lil_n, solve_nil_q


def _run(monkeypatch, runner, solver_name, memoize, config, opt):
    monkeypatch.setattr(bratu, solver_name, functools.partial(getattr(solvers, solver_name), memoize=memoize))
    out = runner(config, opt, device=torch.device('cpu'), verbose=False)
    return out


def _params(out):
    first = out[0]
    if isinstance(first, torch.nn.Module):
        return torch.cat([p.detach().reshape(-1) for p in first.parameters()])
    return torch.as_tensor(out[1])


@pytest.mark.parametrize("method", ["NiL-N", "LiL-N"])
def test_trajectory_is_bitwise_identical_with_fewer_evaluations(monkeypatch, method):
    """Bratu, P = 100, 300 iterations on the CPU (Section 2.1's test)."""
    runner, solver_name = {"NiL-N": (bratu.run_nil_n, "solve_nil_n"),
                           "LiL-N": (bratu.run_lil_n, "solve_lil_n")}[method]
    config = bratu.BratuConfig(N_x=10, N_y=10, init_seed=0)
    opt = bratu.BratuOptConfig(max_iterations=300, R_tol=0.0, pretrain_epochs=50)
    ref = _run(monkeypatch, runner, solver_name, False, config, opt)
    new = _run(monkeypatch, runner, solver_name, True, config, opt)
    ref_m, new_m = ref[-2].data, new[-2].data
    assert ref[-1]['total_iterations'] == new[-1]['total_iterations']
    assert new[-1]['total_iterations'] == 300 or new[-1]['optimizer_stall']
    assert new_m['loss'] == ref_m['loss']                        # bitwise, every iteration
    assert torch.equal(_params(new), _params(ref))
    assert new[-1]['total_line_searches'] < 0.75 * ref[-1]['total_line_searches']


def test_nil_q_trajectory_is_bitwise_identical_with_fewer_evaluations(monkeypatch):
    config = bratu.BratuConfig(N_x=10, N_y=10, init_seed=0)
    opt = bratu.BratuOptConfig(max_quasi_iters_nn=3, max_inner_iters_nn=40, R_tol=0.0, pretrain_epochs=50)
    ref = _run(monkeypatch, bratu.run_nil_q, "solve_nil_q", False, config, opt)
    new = _run(monkeypatch, bratu.run_nil_q, "solve_nil_q", True, config, opt)
    assert new[-2].data['loss'] == ref[-2].data['loss']
    assert torch.equal(_params(new), _params(ref))
    assert new[-1]['total_line_searches'] < ref[-1]['total_line_searches']
    assert new[-1]['monitor_evaluations'] == new[-1]['total_iterations']


def test_objective_counts_only_real_evaluations():
    x = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    calls = []

    def evaluate():
        calls.append(1)
        return (x ** 2).sum() + x[0] * x[1], ()

    obj = LBFGSObjective([x], evaluate)
    obj.closure()
    g = x.grad.clone()
    obj.closure()                                  # same point: reused, gradient restored
    assert len(calls) == obj.n_evals == 1 and torch.equal(x.grad, g)
    assert obj.value()[0] == pytest.approx(5.0 - 2.0) and obj.n_evals == 1
    with torch.no_grad():
        x.add_(1.0)
    obj.value()                                    # new point, value only: one real evaluation
    obj.closure()                                  # needs the gradient too: another
    assert len(calls) == obj.n_evals == 3


def test_lil_n_stops_on_optimizer_stall():
    """At an exact minimizer the gradient is zero, so L-BFGS returns with the
    parameters bitwise unchanged at the first step."""
    target = torch.tensor([0.5, -1.5], dtype=torch.float64)

    def loss_fn(beta):
        r = ((beta - target) ** 2).sum()
        return r, r, 0.0, r * 0

    coeffs, metrics, summary = solve_lil_n(loss_fn, target.numpy().copy(), torch.device('cpu'),
                                           max_iterations=100, R_tol=-1.0, verbose=False)
    assert summary['optimizer_stall'] and summary['stall_iteration'] == 1
    assert summary['total_iterations'] == 1
    opt = SimpleNamespace(max_iterations=100, max_line_searches=None)
    assert stopping_fields('LiL-N', summary, opt)['stopping_reason'] == 'optimizer_stall'


def test_nil_q_stalls_only_when_a_whole_outer_iteration_is_a_no_op():
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1).double()
    x = torch.rand(8, 1, dtype=torch.float64, requires_grad=True)
    y = torch.rand(8, 1, dtype=torch.float64, requires_grad=True)
    flat = lambda m, xx, yy: 0.0 * m(torch.cat([xx, yy], 1))            # noqa: E731
    lin = lambda m, xx, yy, frozen: 0.0 * m(torch.cat([xx, yy], 1))     # noqa: E731
    bc = lambda m, data: (0.0 * m(torch.zeros(1, 2, dtype=torch.float64)).sum(), None)  # noqa: E731
    _, _, summary = solve_nil_q(flat, lin, bc, model, x, y, {}, max_quasi_iters=5, max_inner_iters=7,
                                R_tol=-1.0, verbose=False)
    assert summary['optimizer_stall'] and summary['n_quasi_iters'] == 1
    assert summary['n_inner_stalls'] == 1 and summary['total_iterations'] == 1
    opt = SimpleNamespace(max_quasi_iters_nn=5, max_inner_iters_nn=7, max_line_searches=None)
    fields = stopping_fields('NiL-Q', summary, opt)
    assert fields['stopping_reason'] == 'optimizer_stall' and fields['iterations_cap'] == 35
