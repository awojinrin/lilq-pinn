"""The passive Section 3.1 diagnostics are kept off the solver clock
(DECISIONS.md, 2026-09-24): solvers report their own time in
``solve_time_total`` and the diagnostics separately in ``diagnostics_time``."""

import time

from lilq.iteration_log import IterationLogger
from problems.kovasznay import KovasznayConfig, solve_kovasznay


def test_kovasznay_reports_diagnostics_separately():
    config = KovasznayConfig(N_x=6, N_y=6, max_iter=4)
    plain = solve_kovasznay(config, verbose=False)
    assert plain['diagnostics_time'] == 0.0

    t0 = time.time()
    logged = solve_kovasznay(config, verbose=False, iteration_logger=IterationLogger())
    wall = time.time() - t0
    assert logged['diagnostics_time'] > 0.0
    assert logged['solve_time_total'] + logged['diagnostics_time'] <= wall
    assert logged['solve_time_total'] < wall - 0.5 * logged['diagnostics_time']
