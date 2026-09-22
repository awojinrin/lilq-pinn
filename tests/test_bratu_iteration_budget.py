"""Regression guard for Bratu's per-size NiL iteration and line-search budgets.

Context (Codebase_v3_Proposal.md S5.3(d), DECISIONS.md): the June 2026
consolidation changed the N=10 (P=100) entry of Bratu's per-size iteration
cap (7,500 -> 10,000) and every entry of its line-search cap (pre-GitHub:
{5:24000,10:30000,15:30000}, GitHub: {5:15000,10:25000,15:25000}) --
confirmed against pre-v2-local-codebase/Bratu/run_bratu_experiments.py.
Both reverted to the pre-GitHub (higher) values. Unlike Buckley-Leverett's
line-search cap, this one has never actually bound in stored results for
either codebase (iteration cap always binds first); reverting is a
"prefer the more generous number" choice, not a correctness fix.
"""

from experiments.run_bratu import MAX_ITERATIONS, MAX_LINE_SEARCHES


def test_bratu_max_iterations_matches_pre_github_schedule():
    assert MAX_ITERATIONS == {5: 5000, 10: 7500, 15: 10000}


def test_bratu_max_line_searches_matches_pre_github_schedule():
    assert MAX_LINE_SEARCHES == {5: 24000, 10: 30000, 15: 30000}
