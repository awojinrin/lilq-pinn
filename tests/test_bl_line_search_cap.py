"""Regression guard for Buckley-Leverett's (removed) line-search cap.

Context (Codebase_v3_Proposal.md S5.3(c), DECISIONS.md): the pre-GitHub BL
implementation had no evaluation-based termination condition at all -- its
training loop was a pure iteration-count loop. The June 2026 consolidation
introduced a separate `max_line_searches` cap (100,000) that could bind
*before* the iteration cap in edge cases, since the true worst case for a
never-converging run is `max_iterations * 15` (each L-BFGS .step() is
capped at max_eval=15 internally, hardcoded in lilq.solvers). BLOptConfig
now derives max_line_searches from max_iterations by default, so it can
never bind first -- a backstop, not an active constraint -- while still
allowing an explicit override for anyone who wants a tighter cap.
"""

from problems.buckley_leverett import BLOptConfig


def test_max_line_searches_defaults_to_true_worst_case():
    opt = BLOptConfig(max_iterations=10000)
    assert opt.max_line_searches == 10000 * 15


def test_max_line_searches_tracks_custom_max_iterations():
    opt = BLOptConfig(max_iterations=500)
    assert opt.max_line_searches == 500 * 15


def test_max_line_searches_explicit_override_is_preserved():
    opt = BLOptConfig(max_iterations=10000, max_line_searches=42)
    assert opt.max_line_searches == 42
