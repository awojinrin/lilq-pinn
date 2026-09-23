"""
Beltrami — Pressure Pinned at Every Temporal Level (Section 3.7)
====================================================================

Computational_Package_1_v2.md Section 3.7: the Beltrami system has an
exact null space of dimension 7 (the 8 pressure temporal modes T_j(t)*1
enter only the single pin row at t=t_domain[0]). Rerun once at the
paper's config (N_vel=6, N_p=8) with the pressure pinned at one spatial
point at each of the 8 Chebyshev-Gauss-Lobatto temporal collocation
levels (8 pin rows instead of 1) -- everything else unchanged -- and
confirm: full column rank, a meaningful kappa from the pivoted QR, and a
smaller pressure error at t=1 than the paper's published 0.752% baseline.
Reports the 5-snapshot error table (paper Table 11 format) and wall-clock
time. Expected runtime: about ten minutes.

Usage::

    python experiments/run_beltrami_pinned.py
"""

import sys, os, json, time
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy/scipy)
import numpy as np

from lilq.utils import set_seed
from lilq.iteration_log import IterationLogger
from lilq.provenance import save_provenance
from problems.beltrami import BeltramiConfig, solve_beltrami

# Matches experiments/run_beltrami.py's paper config for N_vel=6.
N_VEL, N_P = 6, 8
COLLOC_N6 = dict(N_x=8, N_y=8, N_z=8, N_t=8, N_bc=6, N_t_bc=6, N_ic=8)
DEFAULT_BASIS = 'chebyshev'

# Section 3.7: "eight temporal collocation levels" for N_p=8 -- the 8
# Chebyshev-Gauss-Lobatto nodes are the natural collocation set for an
# 8-term (degree-0..7) Chebyshev temporal expansion of pressure, matching
# the null-space argument (8 modes, 1 constraint each removed by 1 pin row).
N_PRESSURE_PIN_LEVELS = N_P

# Paper's published Table 11 pressure error at t=1 (Section 3.7 baseline).
PAPER_T1_PRESSURE_ERROR_PCT = 0.752

OUTPUT_DIR = Path('results/beltrami_pinned')


def run_beltrami_pinned(verbose=True):
    set_seed(42)
    config = BeltramiConfig(
        N_vel=N_VEL, N_p=N_P, basis_type=DEFAULT_BASIS,
        n_pressure_pin_levels=N_PRESSURE_PIN_LEVELS,
        **COLLOC_N6,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = IterationLogger()

    if verbose:
        print("=" * 70)
        print(f"BELTRAMI, PRESSURE PINNED AT {N_PRESSURE_PIN_LEVELS} "
              f"TEMPORAL LEVELS (Section 3.7)")
        print(f"  N_vel={N_VEL}, N_p={N_P}, basis={DEFAULT_BASIS}")
        print("=" * 70)

    t0 = time.time()
    result = solve_beltrami(
        config, verbose=verbose,
        iteration_logger=logger,
        run_json_path=OUTPUT_DIR / 'run.json',
    )
    wall_clock_s = time.time() - t0

    logger.to_csv(OUTPUT_DIR / 'iterations.csv')
    save_provenance(OUTPUT_DIR)

    last_row = logger.rows[-1]
    P_total = result['n_params']
    # At this P_total (7984), num_rank_svd is always None by design --
    # conditioning_via_pivoted_qr (lilq/instrumentation.py) never computes
    # the SVD-based rank, that's the point of using it instead of a full
    # SVD above DEFAULT_SVD_CONDITIONING_THRESHOLD. num_rank_gelsy (the
    # LAPACK gelsy driver's own rank, a byproduct of every lstsq solve
    # regardless of scale) is the one that's always populated -- the
    # right column for the "full column rank" check here.
    full_rank = (last_row['num_rank_gelsy'] == P_total)
    kappa = last_row['kappa']

    snap = result['snapshots']
    t1_p_pct = next(s['p'] for s in snap if s['t'] == 1.0) * 100.0
    improved = t1_p_pct < PAPER_T1_PRESSURE_ERROR_PCT

    report = {
        'N_vel': N_VEL, 'N_p': N_P,
        'n_pressure_pin_levels': N_PRESSURE_PIN_LEVELS,
        'P_total': P_total,
        'n_outer_iters': result['n_outer_iters'],
        'wall_clock_s': wall_clock_s,
        'full_column_rank': bool(full_rank),
        'num_rank_svd': last_row['num_rank_svd'],
        'num_rank_gelsy': int(last_row['num_rank_gelsy']),
        'kappa': kappa,
        'kappa_method': last_row['kappa_method'],
        'snapshots': snap,
        't1_pressure_error_pct': t1_p_pct,
        'paper_t1_pressure_error_pct': PAPER_T1_PRESSURE_ERROR_PCT,
        'improved_over_paper_baseline': bool(improved),
    }
    with open(OUTPUT_DIR / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)

    if verbose:
        print(f"\n  Wall-clock: {wall_clock_s:.1f}s, iters: {result['n_outer_iters']}")
        print(f"  Full column rank ({P_total}/{P_total}): {full_rank}")
        print(f"  kappa: {kappa} ({last_row['kappa_method']})")
        print(f"\n  {'t':>6} {'u':>10} {'v':>10} {'w':>10} {'p':>10}")
        for s in snap:
            print(f"  {s['t']:>6.2f} {s['u']:>10.3e} {s['v']:>10.3e} "
                  f"{s['w']:>10.3e} {s['p']:>10.3e}")
        print(f"\n  t=1 pressure error: {t1_p_pct:.4f}% "
              f"(paper baseline: {PAPER_T1_PRESSURE_ERROR_PCT:.3f}%, "
              f"improved: {improved})")
        print(f"\n  Report written to {OUTPUT_DIR / 'report.json'}")

    return report


if __name__ == '__main__':
    run_beltrami_pinned()
