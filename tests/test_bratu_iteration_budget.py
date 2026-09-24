"""Regression guard for Bratu's per-size iteration budgets: the GitHub
reference schedule (DECISIONS.md, 2026-09-24). NiL-N/LiL-N budget
MAX_ITERATIONS; NiL-Q budgets MAX_QUASI_ITERS * MAX_LBFGS_PER_QUASI_ITER.
"""

from experiments.run_bratu import MAX_ITERATIONS, MAX_LBFGS_PER_QUASI_ITER, MAX_QUASI_ITERS


def test_bratu_max_iterations_is_the_github_schedule():
    assert MAX_ITERATIONS == {5: 5000, 10: 10000, 15: 10000}


def test_bratu_nil_q_inner_iteration_schedule_is_the_github_schedule():
    assert MAX_LBFGS_PER_QUASI_ITER == {5: 300, 10: 400, 15: 400}


def test_bratu_nil_q_worst_case_iteration_totals():
    expected_totals = {5: 7500, 10: 10000, 15: 10000}
    for n, inner in MAX_LBFGS_PER_QUASI_ITER.items():
        assert MAX_QUASI_ITERS * inner == expected_totals[n]
