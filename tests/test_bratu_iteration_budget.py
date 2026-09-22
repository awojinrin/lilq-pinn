"""Regression guard for Bratu's per-size NiL iteration budget.

Context (Codebase_v3_Proposal.md S5.3(d), DECISIONS.md): the June 2026
consolidation changed only the N=10 (P=100) entry of Bratu's per-size
NiL-N/NiL-Q iteration cap, 7,500 -> 10,000, leaving N=5 and N=15 unchanged
-- confirmed against pre-v2-local-codebase/Bratu/run_bratu_experiments.py.
Reverted to match the pre-GitHub schedule exactly.
"""

from experiments.run_bratu import MAX_ITERATIONS


def test_bratu_max_iterations_matches_pre_github_schedule():
    assert MAX_ITERATIONS == {5: 5000, 10: 7500, 15: 10000}
