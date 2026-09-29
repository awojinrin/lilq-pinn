"""Addendum v2.2 Section 2.3: one stopping-reason rule for the four-method
and B8 CSVs (``lilq.four_method_log.classify_stopping_reason`` /
``stopping_fields``)."""

import math
from types import SimpleNamespace

from lilq.four_method_log import classify_stopping_reason, stopping_fields
from lilq.solvers import line_search_cap


def test_classify_stopping_reason_v22_rules():
    c = classify_stopping_reason
    assert c(False, 10, 100, 50, 1600, final_loss=math.inf) == 'failure'
    assert c(False, 10, 100, 50, 1600, final_loss=math.nan) == 'failure'
    assert c(True, 10, 100, 50, 1600, final_loss=0.1) == 'target'
    assert c(False, 10, 100, 50, 1600, optimizer_stall=True) == 'optimizer_stall'
    assert c(False, 10, 100, 1600, 1600) == 'line_search_cap'          # evaluations first when iterations below cap
    assert c(False, 100, 100, 1600, 1600) == 'iteration_cap'
    assert c(False, 60, 100, 50, 1600, budget_exhausted=True) == 'iteration_cap'
    assert c(False, 60, 100, 50, 1600) == 'failure'


def test_nil_q_budget_exhausted_with_stalled_inner_loops_is_an_iteration_cap():
    opt = SimpleNamespace(max_quasi_iters_nn=3, max_inner_iters_nn=10, max_line_searches=None)
    summary = dict(total_iterations=17, total_line_searches=40, n_quasi_iters=3, converged=False,
                   optimizer_stall=False, final_loss=0.2)
    fields = stopping_fields('NiL-Q', summary, opt)
    assert fields['stopping_reason'] == 'iteration_cap'
    assert (fields['iterations'], fields['iterations_cap']) == (17, 30)
    assert fields['evaluations_cap'] == line_search_cap(30)
