"""
Package 4: what items 1 and 2 share
==================================

- the termination rule without diagnostics (timed runs, enrichment);
- the logged LiL-Q run: ``iterations.csv`` and ``betas.npy`` after every
  iterate, the scaled identity of check M3, and optional per-iterate extras;
- the timing protocol of Section 2.3;
- the round-off class of a stall, and CSV helpers.
"""

import csv
import statistics
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla

from lilq.instrumentation import EPS_MACH, phase_indicator, roundoff_comparison, stall_flag
from lilq.iteration_log import IterationLogger, LilQDiagnosticsTracker

K_MAX = 60
N_S = 2
REPEATS = 3
SHORT_RUN_S, SHORT_REPEATS = 1.0, 5


def iterate_to_rule(rows, beta, n_s=N_S, k_max=K_MAX, on_iterate=None):
    """Algorithm 1 without the converged exit, diagnostics off: solve until
    the rule fires (tau_chi = 0.1, tau_r = 0.01). ``rows`` has
    ``assemble(beta)`` and ``residual(beta)``. Returns the returned iterate,
    the step at which the rule fired (or None), the stall level ||r|| there
    and chi there. ``on_iterate(beta)``, if given, sees every new iterate
    (Package 5, item 2: the run that gives the coefficients stores them all)."""
    r_prev, run, rn, chi = None, 0, None, None
    for k in range(k_max):
        A, f = rows.assemble(beta)
        new = sla.lstsq(A, f, lapack_driver='gelsy', cond=EPS_MACH, check_finite=False)[0]
        if on_iterate is not None:
            on_iterate(new)
        r = A @ new - f
        rn = float(np.linalg.norm(r))
        chi = phase_indicator(rows.residual(new), r)
        run = run + 1 if (r_prev is not None and stall_flag(chi, rn, r_prev)) else 0
        beta, r_prev = new, rn
        if run >= n_s:
            return beta, k, rn, chi
    return beta, None, rn, chi


def identity_scaled(A, f, beta, N_direct):
    """Check M3 without cancellation: ||(A beta - f) - N(beta)|| over
    || |A||beta| + |f| ||, the scale at which the two vectors are formed
    (Section 4.1 of the manuscript). Its round-off bound is about P eps. The
    plain relative difference of ``b2_check`` divides by ||N||, which is
    small near the plateau, so there it measures cancellation."""
    scale = float(np.linalg.norm(np.abs(A) @ np.abs(beta) + np.abs(f)))
    return float(np.linalg.norm(A @ beta - f - N_direct)) / scale if scale > 0 else 0.0


def m3_summary(run_json):
    """M3 for one run: the K4 reading of Package 3 (k = 1) and the scaled maximum."""
    p4 = run_json['package4']
    P = run_json['P_total']
    return {'b2_at_k1': run_json['b2_check']['rel_err'], 'b2_max': run_json['b2_check']['max_rel_err_over_run'],
            'scaled_max': p4['identity_scaled_max'], 'bound_P_eps': P * EPS_MACH,
            'pass': p4['identity_scaled_max'] <= P * EPS_MACH}


def logged_run(rows, beta, run_dir, error_fn, k_max=K_MAX, stop_n_s=None, extra_fn=None):
    """One logged LiL-Q run on ``rows`` from ``beta``.
    - ``stop_n_s`` None: termination disabled, ``k_max`` iterations.
      Otherwise the run stops when the rule fires with that n_s.
    - ``error_fn(beta)``: the ``eps_ref`` of the iterate.
    - ``extra_fn(beta)``: a dict written to ``errors_by_iterate.csv``.
    Writes ``iterations.csv``, ``betas.npy`` (beta^(0..K)) and the extras
    after every iterate. Returns (iterate at the rule or the last, the
    rule's step, stall level, chi there, tracker, logger, identity maxima)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker = LilQDiagnosticsTracker(test_error_fn=lambda b: {'eps_ref': error_fn(b)}, n_s=stop_n_s or N_S)
    logger = IterationLogger()
    betas = [np.asarray(beta, float).copy()]
    extras = [{'k': 0, **extra_fn(beta)}] if extra_fn else None
    k_s, scaled = None, []
    for k in range(k_max):
        t0 = time.perf_counter()
        A, f = rows.assemble(beta)
        t1 = time.perf_counter()
        new, _, rank, _ = sla.lstsq(A, f, lapack_driver='gelsy', cond=EPS_MACH, check_finite=False)
        t2 = time.perf_counter()
        scaled.append(identity_scaled(A, f, beta, rows.residual(beta)))
        R_new = rows.residual(new)
        row = tracker.step(k=k, A_stacked=A, b_stacked=f, beta_prev=beta, beta_new=new,
                           total_loss=float(R_new @ R_new), rank_gelsy=int(rank), t_assemble_s=t1 - t0,
                           t_solve_s=t2 - t1, is_final_iterate=False, compute_residual_vector_fn=rows.residual)
        logger.record(**row)
        logger.to_csv(run_dir / 'iterations.csv')
        betas.append(new.copy())
        np.save(run_dir / 'betas.npy', np.array(betas))
        if extras is not None:
            extras.append({'k': k + 1, **extra_fn(new)})
            write_csv(run_dir / 'errors_by_iterate.csv', extras)
        beta = new
        if k_s is None and row['stall_rule_fires']:
            k_s = k
            if stop_n_s is not None:
                break
    logger.record(**tracker.finish(k=len(betas) - 1))
    logger.to_csv(run_dir / 'iterations.csv')
    np.save(run_dir / 'beta_last.npy', beta)
    solve_rows = [r for r in logger.rows if r.get('norm_Rlin_h') is not None]
    i = k_s if k_s is not None else len(solve_rows) - 1
    return beta, k_s, float(solve_rows[i]['norm_Rlin_h']), float(solve_rows[i]['chi']), tracker, logger, max(scaled)


def kappa_returned(solve_rows, k_s):
    """kappa by SVD of the system assembled at the iterate the rule returns (k_s + 1): row k_s + 1
    of a logged run's solve rows (Section 2.2). The ``*_at_rule`` columns are the system of the
    step at which the rule fires (row k_s). NaN if the log ends at the rule."""
    if k_s is None or k_s + 1 >= len(solve_rows):
        return {'kappa_retained_returned': float('nan'), 'kappa_raw_returned': float('nan')}
    r = solve_rows[k_s + 1]
    return {'kappa_retained_returned': to_float(r['kappa_retained']), 'kappa_raw_returned': to_float(r['kappa_raw'])}


def stall_class(row):
    """``C`` (conditioning-limited) or ``A`` at an ``iterations.csv`` row: the
    round-off comparison with kappa of the numerically nonsingular part, as
    ``stopping_rule_table.classify``."""
    kr = to_float(row['kappa_retained'])
    if not np.isfinite(kr):
        kr = to_float(row['kappa'])
    ratio, level = roundoff_comparison(to_float(row['norm_Rlin_h']), to_float(row['norm_f_h']), kr)
    return 'C' if ratio < 10 * level else 'A'


def timed(fn, warmup=None):
    """Section 2.3: an untimed warm-up, then three timings, each the median of
    five runs if the first is shorter than 1 s. ``fn`` returns (seconds,
    solves). ``warmup``: the (seconds, solves) of a run already done, which
    then serves as the warm-up (Package 5, item 2: the run that gives the
    coefficients)."""
    warm_s, warm_solves = warmup if warmup is not None else fn()
    timings, solves = [], set()
    for _ in range(REPEATS):
        s, n = fn()
        solves.add(n)
        if s < SHORT_RUN_S:
            reps = [fn() for _ in range(SHORT_REPEATS)]
            solves.update(r[1] for r in reps)
            timings.append({'time_s': statistics.median(r[0] for r in reps), 'runs_s': [r[0] for r in reps]})
        else:
            timings.append({'time_s': s, 'runs_s': [s]})
    times = [t['time_s'] for t in timings]
    return {'warmup_s': warm_s, 'warmup_solves': warm_solves, 'timings': timings,
            'median_s': statistics.median(times), 'spread': max(times) / min(times) - 1,
            'solves': sorted(x for x in solves if x is not None)}


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def to_float(v):
    return float(v) if v not in ('', None) else float('nan')


def rel(a, b):
    return abs(a - b) / abs(b) if b else abs(a - b)
