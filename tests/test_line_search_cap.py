"""The line-search (function-evaluation) cap is 16x each method's own
iteration budget, for every problem, so it cannot bind before the
iteration cap (Addendum v2.2 Section 2.2; DECISIONS.md, 2026-09-29): NiL-N and
LiL-N budget ``max_iterations``, NiL-Q ``max_quasi_iters_nn *
max_inner_iters_nn``. An explicit ``max_line_searches`` overrides it.

The solver calls are intercepted, so these tests check what each problem
actually passes to the solvers without training anything.
"""

import pytest

import problems.bratu as bratu
import problems.buckley_leverett as bl
import problems.burgers as burgers
from lilq.solvers import LINE_SEARCH_CAP_FACTOR, line_search_cap


def test_line_search_cap_rule():
    assert LINE_SEARCH_CAP_FACTOR == 16
    assert line_search_cap(10000) == 160000
    assert line_search_cap(10000, override=42) == 42


@pytest.mark.parametrize("opt_cls", [bratu.BratuOptConfig, burgers.BurgersOptConfig, bl.BLOptConfig])
def test_no_problem_hard_codes_a_cap(opt_cls):
    assert opt_cls().max_line_searches is None


class _Captured(Exception):
    pass


def _capture_caps(monkeypatch, module, config, opt):
    seen = {}

    def fake(name):
        def solver(*args, **kwargs):
            seen[name] = kwargs['max_line_searches']
            raise _Captured
        return solver

    for name in ('solve_nil_n', 'solve_nil_q', 'solve_lil_n'):
        monkeypatch.setattr(module, name, fake(name))
    for fn in (module.run_nil_n, module.run_nil_q, module.run_lil_n):
        with pytest.raises(_Captured):
            fn(config, opt, verbose=False)
    return seen


@pytest.mark.parametrize("module, config, opt", [
    (bratu, bratu.BratuConfig(N_x=3, N_y=3, k_ratio=2),
     bratu.BratuOptConfig(max_iterations=700, max_quasi_iters_nn=4, max_inner_iters_nn=50, pretrain_epochs=1)),
    (burgers, burgers.BurgersConfig(N_x=3, N_t=3, k_ratio=2),
     burgers.BurgersOptConfig(max_iterations=700, max_quasi_iters_nn=4, max_inner_iters_nn=50, pretrain_epochs=1)),
    (bl, bl.BLConfig(N_x=3, N_t=3, k_ratio=2),
     bl.BLOptConfig(max_iterations=700, max_quasi_iters_nn=4, max_inner_iters_nn=50, pretrain_epochs=1)),
])
def test_each_method_gets_sixteen_times_its_own_budget(monkeypatch, module, config, opt):
    seen = _capture_caps(monkeypatch, module, config, opt)
    assert seen == {'solve_nil_n': 16 * 700, 'solve_nil_q': 16 * 4 * 50, 'solve_lil_n': 16 * 700}
