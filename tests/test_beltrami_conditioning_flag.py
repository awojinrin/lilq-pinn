"""Regression guard for Beltrami's conditioning-analysis flag.

Context (DECISIONS.md): solve_beltrami() unconditionally computed
np.linalg.cond(A_sys) -- a full SVD on a matrix with P_total ~ 8000
columns -- every outer iteration, with nothing downstream ever consuming
the result. Confirmed via a same-machine, same-moment comparison against
the pre-GitHub codebase (which gates the identical computation behind
analyze_svd=False by default) to account for roughly half of Beltrami's
total solve time. Fixed by making it opt-in, matching pre-GitHub's
pattern; the computational package spec's own final-iterate-only
pivoted-QR approach for Beltrami (Section 3.1 item 8) is the real
long-term replacement, planned for Phase 1's instrumentation work.

This test uses a tiny configuration (small N) so it stays fast regardless
of which path is exercised -- it checks the flag's *behavior*, not
performance at full scale (that's what the manual before/after timing
comparison in DECISIONS.md establishes).
"""

import math

from problems.beltrami import BeltramiConfig, solve_beltrami


def _tiny_config():
    # Smallest sizes that still produce a well-posed system -- this test
    # only needs the solve to run at all, not to be physically meaningful.
    return BeltramiConfig(N_vel=2, N_p=2, max_iter=1)


def test_conditioning_not_computed_by_default():
    result = solve_beltrami(_tiny_config(), verbose=False)
    cond_values = result['history']['cond_number']
    assert len(cond_values) == 1
    assert math.isnan(cond_values[0])


def test_conditioning_computed_when_requested():
    result = solve_beltrami(_tiny_config(), verbose=False, analyze_conditioning=True)
    cond_values = result['history']['cond_number']
    assert len(cond_values) == 1
    assert math.isfinite(cond_values[0])
    assert cond_values[0] > 0
