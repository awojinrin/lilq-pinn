"""Package 5, item 3: the run probe (``lilq.metrics.RunProbe``) in the L-BFGS loops of ``lilq.solvers``.

- C2: logging the error along a run does not change the iterates (NiL-N, NiL-Q, LiL-N, bit for bit on the CPU).
- The time of the error evaluations is off the clock (``wall_time``, ``time_s``, the budget).
- The budget ends a run; the error is logged every ``every`` iterations, at iteration 0 and at the last iterate.
"""

import math
import time

import numpy as np
import pytest
import torch

from lilq.metrics import MetricsTracker, RunProbe
from lilq.nn import MLP
from lilq.solvers import solve_lil_n, solve_nil_n, solve_nil_q

N_IT = 30


# ---------------------------------------------------------------- small problems with the solvers' interfaces

def _points():
    g = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    x, y = torch.meshgrid(g, g, indexing='ij')
    x, y = x.reshape(-1, 1).clone().requires_grad_(True), y.reshape(-1, 1).clone().requires_grad_(True)
    bc = {'x': torch.tensor([[0.0], [1.0]], dtype=torch.float64), 'y': torch.tensor([[0.5], [0.5]], dtype=torch.float64)}
    return x, y, bc


def _u(model, x, y):
    return model(torch.cat([x, y], dim=1))


def _pde(model, x, y):
    """u_xx + u_yy - exp(u): a Bratu-like nonlinear residual."""
    u = _u(model, x, y)
    ux, uy = torch.autograd.grad(u.sum(), (x, y), create_graph=True)
    uxx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]
    uyy = torch.autograd.grad(uy.sum(), y, create_graph=True)[0]
    return uxx + uyy + 0.5 * torch.exp(u)


def _linearized(model, x, y, frozen):
    """The Newton linearization about the frozen iterate (None: freeze the current one)."""
    if frozen is None:
        return _u(model, x, y).detach()
    u = _u(model, x, y)
    ux, uy = torch.autograd.grad(u.sum(), (x, y), create_graph=True)
    uxx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]
    uyy = torch.autograd.grad(uy.sum(), y, create_graph=True)[0]
    e = 0.5 * torch.exp(frozen)
    return uxx + uyy + e * (u - frozen) + e


def _bc(model, bd):
    return (_u(model, bd['x'], bd['y']) ** 2).mean(), None


def _model(seed=0):
    torch.manual_seed(seed)
    return MLP(hidden_dim=6, num_layers=2)


def _error_fn(model, x, y, calls=None, sleep=0.0):
    """An 'error' of the network: evaluated without touching its parameters or their gradients."""
    def fn():
        if calls is not None:
            calls.append(1)
        if sleep:
            time.sleep(sleep)
        with torch.no_grad():
            return float(_u(model, x.detach(), y.detach()).abs().mean())
    return fn


def _nil_n(probe_of=None, max_iterations=N_IT, stall_rule='f1'):
    x, y, bc = _points()
    model = _model()
    probe = probe_of(model, x, y) if probe_of else None
    model, metrics, summary = solve_nil_n(_pde, _bc, model, x, y, bc, max_iterations=max_iterations,
                                          max_line_searches=10 ** 9, R_tol=0.0, verbose=False,
                                          stall_rule=stall_rule, probe=probe)
    return torch.cat([p.detach().flatten() for p in model.parameters()]), metrics, summary, probe


def _nil_q(probe_of=None):
    x, y, bc = _points()
    model = _model(1)
    probe = probe_of(model, x, y) if probe_of else None
    model, metrics, summary = solve_nil_q(_pde, _linearized, _bc, model, x, y, bc, max_quasi_iters=4,
                                          max_inner_iters=8, max_line_searches=10 ** 9, R_tol=0.0, verbose=False,
                                          stall_rule='f1', probe=probe)
    return torch.cat([p.detach().flatten() for p in model.parameters()]), metrics, summary, probe


A = np.random.default_rng(0).standard_normal((40, 12))
F = np.random.default_rng(1).standard_normal(40)


def _lil_n(probe=None, max_iterations=N_IT):
    At, Ft = torch.from_numpy(A), torch.from_numpy(F)

    def loss(beta):
        r = At @ beta + 0.1 * torch.sin(beta).sum() - Ft
        total = (r ** 2).mean()
        return total, total, 0.0, torch.zeros((), dtype=torch.float64)

    return solve_lil_n(loss, np.zeros(12), torch.device('cpu'), max_iterations=max_iterations,
                       max_line_searches=10 ** 9, R_tol=0.0, verbose=False, stall_rule='f1', probe=probe)


# ---------------------------------------------------------------- C2: the iterates do not change

def test_c2_logging_leaves_the_nil_n_iterates_unchanged():
    plain, m0, s0, _ = _nil_n()
    logged, m1, s1, probe = _nil_n(lambda model, x, y: RunProbe(_error_fn(model, x, y), every=3))
    assert torch.equal(plain, logged)
    assert m0.data['loss'] == m1.data['loss'] and s0['total_iterations'] == s1['total_iterations'] == N_IT
    assert [r['iteration'] for r in probe.rows] == list(range(0, N_IT + 1, 3))
    assert all(math.isfinite(r['error']) for r in probe.rows)


def test_c2_logging_leaves_the_nil_q_iterates_unchanged():
    plain, m0, s0, _ = _nil_q()
    logged, m1, s1, probe = _nil_q(lambda model, x, y: RunProbe(_error_fn(model, x, y), every=5))
    assert torch.equal(plain, logged) and m0.data['loss'] == m1.data['loss']
    its = [r['iteration'] for r in probe.rows]
    assert its[0] == 0 and its[-1] == s1['total_iterations'] and all(i % 5 == 0 for i in its[1:-1])


def test_c2_logging_leaves_the_lil_n_iterates_unchanged():
    b0, m0, s0 = _lil_n()
    seen = []
    probe = RunProbe(lambda: seen.append(1) or 0.0, every=10)
    b1, m1, s1 = _lil_n(probe)
    assert np.array_equal(b0, b1) and m0.data['loss'] == m1.data['loss']
    last = s1['total_iterations']          # this small problem stalls at its minimum before the cap
    expected = sorted(set(range(0, last + 1, 10)) | {last})
    assert [r['iteration'] for r in probe.rows] == expected and len(seen) == len(expected)


# ---------------------------------------------------------------- the clock

def test_off_clock_time_is_not_on_the_clock():
    m = MetricsTracker()
    m.start()
    m.off_clock(time.sleep, 0.2)
    m.record(1, 1, 0.0)
    assert m.off_clock_s >= 0.2 and m.data['wall_time'][-1] < 0.1 and m.elapsed() < 0.1


def test_the_error_evaluations_are_off_the_run_clock():
    t0 = time.perf_counter()
    calls = []
    _, metrics, summary, probe = _nil_n(lambda model, x, y: RunProbe(_error_fn(model, x, y, calls, sleep=0.05),
                                                                     every=10))
    real = time.perf_counter() - t0
    assert len(calls) == 4 and metrics.off_clock_s >= 0.2
    assert summary['training_time'] <= real - 0.2
    assert [r['time_s'] for r in probe.rows] == sorted(r['time_s'] for r in probe.rows)
    assert probe.rows[-1]['time_s'] == metrics.data['wall_time'][-1]
    on_record = dict(zip(metrics.data['iteration'], metrics.data['wall_time']))
    assert all(r['time_s'] == on_record[r['iteration']] for r in probe.rows)


# ---------------------------------------------------------------- the budget

def test_the_budget_ends_the_run_and_the_last_iterate_is_logged():
    _, metrics, summary, probe = _nil_n(lambda model, x, y: RunProbe(_error_fn(model, x, y), every=10,
                                                                     budget_s=0.0), max_iterations=10 ** 6)
    assert summary['budget_reached'] and summary['total_iterations'] == 1 and not summary['optimizer_stall']
    assert [r['iteration'] for r in probe.rows] == [0, 1]


def test_the_budget_ends_nil_q_in_its_outer_loop_too():
    _, _, summary, probe = _nil_q(lambda model, x, y: RunProbe(None, every=10, budget_s=0.0))
    assert summary['budget_reached'] and summary['total_iterations'] == 1 and summary['n_quasi_iters'] == 1
    assert [r['iteration'] for r in probe.rows] == [0, 1] and math.isnan(probe.rows[0]['error'])


def test_the_iteration_stop_counts_nil_q_inner_iterations():
    """max_iterations=11 crosses NiL-Q's inner cap of 8: the run stops inside its second linearization."""
    _, metrics, summary, probe = _nil_q(lambda model, x, y: RunProbe(None, every=10, max_iterations=11))
    assert summary['total_iterations'] == 11 and summary['n_quasi_iters'] == 2
    assert probe.stopped and not summary['budget_reached'] and metrics.data['iteration'][-1] == 11
    assert [r['iteration'] for r in probe.rows] == [0, 10, 11]


def test_the_budget_is_on_the_clock_without_the_error_time():
    """A budget the error evaluations alone would exhaust does not end the run."""
    probe = RunProbe(lambda: time.sleep(0.05) or 0.0, every=1, budget_s=10.0)
    _, _, summary = _lil_n(probe, max_iterations=20)
    assert not summary['budget_reached'] and summary['total_iterations'] == 20


def test_without_a_probe_the_summary_says_no_budget():
    _, _, summary, _ = _nil_n(max_iterations=3)
    assert summary['budget_reached'] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_off_clock_waits_for_queued_gpu_work():
    a = torch.randn(3000, 3000, device='cuda', dtype=torch.float64)
    m = MetricsTracker()
    m.start()
    for _ in range(5):
        a = a @ a / 3000.0          # queued, not finished
    m.off_clock(lambda: None)        # synchronizes first: that work stays on the clock
    on = m.elapsed()
    assert on > 0.0 and m.off_clock_s < on
