"""
Beltrami — Pressure Pinned at Every Temporal Level (Section 3.7)
====================================================================

Computational_Package_1_v2.md Section 3.7: the Beltrami system has an
exact null space of dimension 7 (the 8 pressure temporal modes T_j(t)*1
enter only the single pin row at t=t_domain[0]). Rerun once at the
paper's config (N_vel=6, N_p=8) with the pressure pinned at one spatial
point at each of the 8 Chebyshev-Gauss-Lobatto temporal collocation
levels (8 pin rows instead of 1) -- everything else unchanged -- and
confirm full column rank and a meaningful kappa from the pivoted QR. The
t=1 pressure error is reported beside the paper's published 0.752%, without
a better/worse verdict: the paper's figure has three digits, and wave 1's
0.7515% against it is no difference (the advisor's reply to wave 1).
Reports the 5-snapshot error table (paper Table 11 format) and wall-clock
time. Expected runtime: about ten minutes.

Usage::

    python experiments/run_beltrami_pinned.py
"""

import dataclasses, sys, os, json, time
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy/scipy)
import numpy as np

from lilq.iteration_log import IterationLogger, last_solve_row
from lilq.provenance import save_provenance
from problems.beltrami import BeltramiConfig, solve_beltrami
from lilq.saved_models import save_solution

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

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'results' / 'beltrami_pinned'
KMAX_ITERS = 8


def run_beltrami_pinned(verbose=True, out_dir=OUTPUT_DIR, kmax=False):
    """Section 3.7's run; ``kmax=True``: its K_max pass (8 iterations, zero
    coefficient-change tolerance; Addendum v2.2 Section 2.7)."""
    out_dir = Path(out_dir)
    if (out_dir / 'report.json').exists():   # completed: a resubmitted job does not redo it
        if verbose:
            print(f"  {out_dir}: done, skipping")
        return json.loads((out_dir / 'report.json').read_text())
    np.random.seed(42)  # NumPy only: this solve is scipy on the CPU (Addendum v2.1 fix 3.3)
    config = BeltramiConfig(
        N_vel=N_VEL, N_p=N_P, basis_type=DEFAULT_BASIS,
        n_pressure_pin_levels=N_PRESSURE_PIN_LEVELS,
        **COLLOC_N6,
    )
    if kmax:
        config = dataclasses.replace(config, tol=0.0, max_iter=KMAX_ITERS)

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = IterationLogger()

    if verbose:
        print("=" * 70)
        print(f"BELTRAMI, PRESSURE PINNED AT {N_PRESSURE_PIN_LEVELS} "
              f"TEMPORAL LEVELS (Section 3.7)")
        print(f"  N_vel={N_VEL}, N_p={N_P}, basis={DEFAULT_BASIS}")
        print("=" * 70)

    t0 = time.perf_counter()
    result = solve_beltrami(
        config, verbose=verbose,
        iteration_logger=logger,
        run_json_path=out_dir / 'run.json',
    )
    wall_clock_s = time.perf_counter() - t0

    logger.to_csv(out_dir / 'iterations.csv')
    save_solution(out_dir, {f: (result[f'basis_{f}'], result[f'theta_{f}']) for f in 'uvwp'}, config)
    save_provenance(out_dir)

    last_row = last_solve_row(logger.rows)
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

    report = {
        'N_vel': N_VEL, 'N_p': N_P,
        'n_pressure_pin_levels': N_PRESSURE_PIN_LEVELS,
        'P_total': P_total,
        'n_outer_iters': result['n_outer_iters'],
        # The method's own time (collocation, assembly, solves), with the
        # passive Section 3.1 diagnostics off the clock; the paper's figure.
        'solver_time_s': result['solve_time_total'],
        'diagnostics_time_s': result['diagnostics_time'],
        'wall_clock_s': wall_clock_s,
        'full_column_rank': bool(full_rank),
        'num_rank_svd': last_row['num_rank_svd'],
        'num_rank_gelsy': int(last_row['num_rank_gelsy']),
        'kappa': kappa,
        'kappa_method': last_row['kappa_method'],
        'snapshots': snap,
        't1_pressure_error_pct': t1_p_pct,
        # Without the per-time-level mean shift, which removes exactly the
        # null-space modes and so cannot show their removal (0.7515% against
        # 0.752%); the pins fix the gauge here (Addendum v2.2 2.11).
        't1_pressure_error_pin_gauge_pct': next(s['p_pin_gauge'] for s in snap if s['t'] == 1.0) * 100.0,
        'rel_l2_p_pin_gauge': result.get('rel_l2_p_pin_gauge'),
        'deviations_from_spec': [
            'the pins are at Chebyshev-Gauss-Lobatto times in [0, 1], not at the temporal collocation levels',
            'the pin rows have weight sqrt(lambda_bc / 8) each (the single pin had sqrt(lambda_bc))',
        ],
        'paper_t1_pressure_error_pct': PAPER_T1_PRESSURE_ERROR_PCT,
    }
    with open(out_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)

    if verbose:
        print(f"\n  Solver time: {result['solve_time_total']:.1f}s (diagnostics "
              f"{result['diagnostics_time']:.1f}s, process wall-clock {wall_clock_s:.1f}s), "
              f"iters: {result['n_outer_iters']}")
        print(f"  Full column rank ({P_total}/{P_total}): {full_rank}")
        print(f"  kappa: {kappa} ({last_row['kappa_method']})")
        print(f"\n  {'t':>6} {'u':>10} {'v':>10} {'w':>10} {'p':>10}")
        for s in snap:
            print(f"  {s['t']:>6.2f} {s['u']:>10.3e} {s['v']:>10.3e} "
                  f"{s['w']:>10.3e} {s['p']:>10.3e}")
        print(f"\n  t=1 pressure error: {t1_p_pct:.4f}% "
              f"(paper: {PAPER_T1_PRESSURE_ERROR_PCT:.3f}%)")
        print(f"\n  Report written to {out_dir / 'report.json'}")

    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Section 3.7: Beltrami with the pressure "
                                                 "pinned at every temporal level")
    parser.add_argument('--out-dir', type=str, default=str(OUTPUT_DIR),
                        help='Output directory (default: results/beltrami_pinned/).')
    parser.add_argument('--kmax', action='store_true',
                        help='The K_max pass: 8 iterations, zero tolerance (Addendum v2.2 2.7).')
    args = parser.parse_args()
    run_beltrami_pinned(out_dir=args.out_dir, kmax=args.kmax)
