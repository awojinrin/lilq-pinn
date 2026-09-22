"""Regression guard for Bratu's per-size NiL iteration and line-search budgets.

Context (Codebase_v3_Proposal.md S5.3(d), DECISIONS.md): the June 2026
consolidation changed three separate per-size Bratu constants, all only at
N=10 or on every entry -- confirmed against
pre-v2-local-codebase/Bratu/run_bratu_experiments.py:

- MAX_ITERATIONS (governs NiL-N/LiL-N): N=10 only, 7,500 -> 10,000.
- MAX_LBFGS_PER_QUASI_ITER (governs NiL-Q's inner loop, combined with the
  flat MAX_QUASI_ITERS=25 outer cap): N=10 only, 300 -> 400 -- missed on
  the first revert pass since it's a wholly separate knob from
  MAX_ITERATIONS despite both nominally being "Bratu's N=10 iteration
  cap"; caught by comparing a fresh v3-dev run's NiL-Q iteration count
  (10,000) against the pre-GitHub reference (7,500).
- MAX_LINE_SEARCHES: every entry differs (see below).

All reverted to the pre-GitHub values.
"""

from experiments.run_bratu import (
    MAX_ITERATIONS, MAX_LINE_SEARCHES, MAX_LBFGS_PER_QUASI_ITER, MAX_QUASI_ITERS,
)


def test_bratu_max_iterations_matches_pre_github_schedule():
    assert MAX_ITERATIONS == {5: 5000, 10: 7500, 15: 10000}


def test_bratu_max_line_searches_matches_pre_github_schedule():
    assert MAX_LINE_SEARCHES == {5: 24000, 10: 30000, 15: 30000}


def test_bratu_nil_q_inner_iteration_schedule_matches_pre_github():
    assert MAX_LBFGS_PER_QUASI_ITER == {5: 300, 10: 300, 15: 400}


def test_bratu_nil_q_worst_case_iteration_totals_match_pre_github():
    # MAX_QUASI_ITERS is flat (25) in both codebases; only the per-quasi-iter
    # inner cap varies by N. Worst-case NiL-Q total = MAX_QUASI_ITERS * inner.
    expected_totals = {5: 7500, 10: 7500, 15: 10000}
    for n, inner in MAX_LBFGS_PER_QUASI_ITER.items():
        assert MAX_QUASI_ITERS * inner == expected_totals[n]
