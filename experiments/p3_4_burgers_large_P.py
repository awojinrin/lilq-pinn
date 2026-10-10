"""
Package 3, Addendum 1 (item 4): Burgers LiL-Q at larger sizes
=============================================================

The advisor's addendum of 8 October 2026. The network trained by LM reaches a
Burgers error of 1.6e-5 at P = 625 (Package 2). LiL-Q's smallest at the
paper's sizes is 3.7e-5 (P = 625, iterate 3). This item finds what LiL-Q
costs to reach 1.6e-5, by running the paper's LiL-Q Burgers configuration at
larger P. It is ``experiments.run_burgers.paper_setup(N)`` unchanged but for
the size:
- ``sin_fourier``, random-tensor collocation (seed 42, k_ratio 10);
- the code's row weights, zero initial coefficients, ``gelsy``;
- the driver P2-12 used: ``problems.burgers.run_lil_q`` with the Cole-Hopf
  reference (``reference/burgers_cole_hopf.npz``).

**Sizes:** modes_x = modes_t = 30, 32, 35 (P = 900, 1,024, 1,225), and the
P = 625 control (25). 40 (P = 1,600) runs only if no iterate at the three
sizes reaches an error of 1.6e-5 or less.

**(a) The logged pass**, per size:
- ``R_tol = 0`` (no loss target at these sizes; for N > 25 ``paper_setup``
  would otherwise default it to 1e-4) and K_max = 60;
- the full ``iterations.csv`` (norms, chi, the coefficient change, eps_ref,
  the stall flags, the rank), kappa by SVD at every iterate (the tracker's
  rule for P <= 3,200), the last one included;
- ``losses.csv``: the loss at every iterate, which check L4 compares.

The P = 625 pass reproduces P2-12's to all digits for k <= 4 (check L1). P2-12's
pass stopped at k = 4 only because it met the paper's target.

**(b) Clean timing** (Package 1 wave 4, option B; on Grace an exclusive node,
48 threads). For each size and each stopping iterate: one untimed warm-up,
then three clean runs (``diagnostics=False``), reported as their median.
- **k_e:** ``max_iter`` is the first iterate of the logged pass with error
  <= 1.6e-5 (none: not timed).
- **k_r:** ``max_iter`` is the iterate the termination rule returns
  (n_s = 2, tau_chi = 0.1, tau_r = 0.01).
- **The P = 625 control:** the paper pass itself (its target, ``max_iter`` 4).

Checks:
- **L1:** the reproduction above.
- **L2:** the control's median within 15% of the paper's 1.18 s.
- **L3:** the B2 identity at every iterate.
- **L4:** every clean run's final loss equals the logged pass's at the same
  iterate, bit for bit.

Output: ``P3_4_burgers_large_P/`` holds one folder per size (``run.json``,
``iterations.csv``, ``losses.csv``, ``timing.json``), ``terminal.csv`` and
``checks_item4.json``.

Usage::

    python experiments/p3_4_burgers_large_P.py all --out <root> --reference-dir <dir> --p2-12 <P2_12_reference_errors>
"""

import argparse
import csv
import dataclasses
import json
import math
import os
import statistics
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)
import numpy as np

from lilq.instrumentation import EPS_MACH, stall_rule_index
from lilq.provenance import capture_blas_thread_env
from lilq.source_lock import current_commit

ITEM = 'P3_4_burgers_large_P'
SIZES = (30, 32, 35)
CONDITIONAL = 40
CONTROL = 25
CONTROL_KMAX = 4
ERROR_TARGET = 1.6e-5
K_MAX = 60
REPEATS = 3
PAPER_CONTROL_TIME_S = 1.18
L2_TOL = 0.15


def setup(N, logged=True, max_iter=None):
    """The paper's configuration at N. ``logged``: no loss target, K_max = 60.
    Otherwise ``max_iter`` iterations with no target, or the paper pass itself
    (its target) for the control."""
    from experiments.run_burgers import paper_setup
    config, opt = paper_setup(N)
    if logged:
        return config, dataclasses.replace(opt, R_tol=0.0, max_quasi_iters_lil=K_MAX)
    if N == CONTROL and max_iter == CONTROL_KMAX:
        return config, dataclasses.replace(opt, max_quasi_iters_lil=CONTROL_KMAX)    # the paper pass
    return config, dataclasses.replace(opt, R_tol=0.0, max_quasi_iters_lil=max_iter)


def _floats(rows, col):
    return np.array([np.nan if r.get(col) in ('', None) else float(r[col]) for r in rows])


def logged_pass(N, out_root, reference_dir):
    from lilq.iteration_log import IterationLogger
    from problems.burgers import run_lil_q
    config, opt = setup(N)
    run_dir = Path(out_root) / ITEM / f'burgers_P{N * N}'
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = IterationLogger()
    _, coef, metrics, summary = run_lil_q(config, opt, verbose=False, iteration_logger=logger,
                                          run_json_path=run_dir / 'run.json',
                                          reference_npz=Path(reference_dir) / 'burgers_cole_hopf.npz')
    logger.to_csv(run_dir / 'iterations.csv')
    # solve_lil_q's own record (QuasilinearMetrics.total_loss): entry k is the
    # loss at iterate k, k = 0 the initial one; a clean run's final_loss is its last
    losses = [float(v) for v in metrics.data['total_loss']]
    with open(run_dir / 'losses.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['iterate', 'loss'])
        w.writerows([(k, repr(v)) for k, v in enumerate(losses)])
    np.save(run_dir / 'beta_last.npy', coef)
    return run_dir


def read_log(run_dir):
    with open(Path(run_dir) / 'iterations.csv') as fh:
        log = list(csv.DictReader(fh))
    with open(Path(run_dir) / 'losses.csv') as fh:
        losses = {int(r['iterate']): float(r['loss']) for r in csv.DictReader(fh)}
    return log, losses


def stopping_iterates(log):
    """``(k_e, k_r, i)``: the first iterate with eps_ref <= 1.6e-5 (None if
    none); the iterate the termination rule returns (None if it never fires),
    and the rule's row i (k_r = i + 1)."""
    eps = _floats(log, 'eps_ref')
    k_e = next((k for k in range(len(eps)) if eps[k] <= ERROR_TARGET), None)
    K = sum(1 for r in log if r.get('rel_dbeta') not in ('', None))
    i = stall_rule_index(_floats(log, 'chi')[:K], _floats(log, 'norm_Rlin_h')[:K])
    return k_e, (i + 1 if i is not None else None), i


def clean_timing(N, max_iter, repeats=REPEATS):
    """One untimed warm-up and ``repeats`` clean runs to ``max_iter``:
    ``training_time``, iterations and the final loss of each."""
    from problems.burgers import run_lil_q
    config, opt = setup(N, logged=False, max_iter=max_iter)

    def once():
        s = run_lil_q(config, opt, verbose=False, diagnostics=False)[-1]
        return {'time_s': float(s['training_time']), 'iterations': int(s['total_iterations']),
                'final_loss': float(s['final_loss'])}
    warm = once()
    runs = [once() for _ in range(repeats)]
    return {'max_iter': max_iter, 'warmup_time_s': warm['time_s'], 'times_s': [r['time_s'] for r in runs],
            'median_s': statistics.median(r['time_s'] for r in runs), 'iterations': [r['iterations'] for r in runs],
            'final_losses': [repr(r['final_loss']) for r in runs]}


def time_size(N, run_dir):
    log, _ = read_log(run_dir)
    k_e, k_r, _ = stopping_iterates(log)
    timing = {'threads': capture_blas_thread_env().get('OMP_NUM_THREADS'), 'commit': current_commit()}
    if N == CONTROL:
        timing['paper_pass'] = clean_timing(N, CONTROL_KMAX)
    if k_e is not None:
        timing['k_e'] = clean_timing(N, k_e)
    if k_r is not None:
        timing['k_r'] = clean_timing(N, k_r)
    (Path(run_dir) / 'timing.json').write_text(json.dumps(timing, indent=2))
    return timing


def run_all(out_root, reference_dir, timing=True):
    """The logged passes (the control and the three sizes; 40 x 40 if none of
    the three reaches 1.6e-5), then the clean timings."""
    dirs = {N: logged_pass(N, out_root, reference_dir) for N in (CONTROL,) + SIZES}
    if all(stopping_iterates(read_log(dirs[N])[0])[0] is None for N in SIZES):     # the rule of Section 5
        dirs[CONDITIONAL] = logged_pass(CONDITIONAL, out_root, reference_dir)
    if timing:
        for N, d in dirs.items():
            time_size(N, d)
    return dirs


def terminal_row(run_dir):
    log, losses = read_log(run_dir)
    meta = json.loads((Path(run_dir) / 'run.json').read_text())
    timing = json.loads((Path(run_dir) / 'timing.json').read_text()) if (Path(run_dir) / 'timing.json').exists() else {}
    eps = _floats(log, 'eps_ref')
    k_e, k_r, i = stopping_iterates(log)
    k_min = int(np.nanargmin(eps))
    cls = ''
    if i is not None:
        kr = float(log[i]['kappa_retained'] or log[i]['kappa'])
        cls = 'C' if float(log[i]['roundoff_ratio']) < 10 * kr * EPS_MACH else 'A'
    K = sum(1 for r in log if r.get('rel_dbeta') not in ('', None))
    last = log[K - 1]
    tm = lambda key, f: timing.get(key, {}).get(f, '')  # noqa: E731
    # rank: gelsy's (rcond = eps_mach). num_rank_svd: at the paper's threshold max(N, P) sigma_1 eps_mach,
    # at the last iterate and its minimum over the run (the advisor's reply of 10 October, item 3.1)
    svd_ranks = [int(float(r['num_rank_svd'])) for r in log[:K] if r.get('num_rank_svd') not in ('', None)]
    return {'P': meta['P_total'], 'modes': int(round(meta['P_total'] ** 0.5)), 'N': meta['N_total'],
            'kappa': last['kappa'], 'kappa_retained': last['kappa_retained'], 'rank': last['num_rank_gelsy'],
            'num_rank_svd': last.get('num_rank_svd', ''), 'num_rank_svd_min': min(svd_ranks) if svd_ranks else '',
            'eps_min': float(eps[k_min]), 'k_min': k_min,
            'k_e': k_e if k_e is not None else 'none', 'eps_k_e': float(eps[k_e]) if k_e is not None else '',
            'k_r': k_r if k_r is not None else 'never', 'rule_class': cls,
            'eps_k_r': float(eps[k_r]) if k_r is not None else '', 'eps_60': float(eps[K]),
            'timings_k_e_s': json.dumps(tm('k_e', 'times_s')) if 'k_e' in timing else '',
            'median_k_e_s': tm('k_e', 'median_s'),
            'timings_k_r_s': json.dumps(tm('k_r', 'times_s')) if 'k_r' in timing else '',
            'median_k_r_s': tm('k_r', 'median_s'),
            'timings_paper_pass_s': json.dumps(tm('paper_pass', 'times_s')) if 'paper_pass' in timing else '',
            'median_paper_pass_s': tm('paper_pass', 'median_s'),
            'b2_rel_err_k1': (meta.get('b2_check') or {}).get('rel_err'),
            'b2_max_rel_err': (meta.get('b2_check') or {}).get('max_rel_err_over_run'), 'threads': timing.get('threads')}


def check_l1(control_dir, p2_12_run):
    """L1: the control's logged pass against P2-12's, k <= 4: the residual
    norms and the errors, bit for bit (as written)."""
    def load(d):
        with open(Path(d) / 'iterations.csv') as fh:
            return {int(r['k']): r for r in csv.DictReader(fh)}
    a, b = load(control_dir), load(p2_12_run)
    _, losses = read_log(control_dir)
    diffs = []
    for k in range(CONTROL_KMAX + 1):
        terminal = b[k].get('norm_Rlin_h', '') in ('', None)        # P2-12 stopped here: no solve at k
        for col in ('norm_R_h', 'norm_Rlin_h', 'eps_ref'):
            va, vb = a[k].get(col, ''), b[k].get(col, '')
            if vb in ('', None):
                continue                                  # P2-12's terminal row has no linear residual
            if terminal and col == 'norm_R_h':            # a terminal row's ||R|| is sqrt(the solver's loss),
                va = repr(math.sqrt(losses[k]))           # a solve row's ||A beta - f||: compare like with like
            if float(va) != float(vb):
                diffs.append({'k': k, 'column': col, 'here': va, 'p2_12': vb})
    return {'compared': f'k = 0..{CONTROL_KMAX}: norm_R_h, norm_Rlin_h, eps_ref', 'differences': diffs,
            'passed': not diffs}


def check_l4(run_dir):
    _, losses = read_log(run_dir)
    timing = json.loads((Path(run_dir) / 'timing.json').read_text())
    out = []
    for key in ('k_e', 'k_r', 'paper_pass'):
        if key in timing:
            t = timing[key]
            logged = losses[t['max_iter']]
            out.append({'which': key, 'max_iter': t['max_iter'], 'logged_loss': repr(logged),
                        'clean_losses': t['final_losses'], 'iterations': t['iterations'],
                        'identical': all(float(v) == logged for v in t['final_losses'])
                        and all(n == t['max_iter'] for n in t['iterations'])})
    return out


def summarize(out_root, p2_12=None):
    d = Path(out_root) / ITEM
    runs = sorted((x for x in d.iterdir() if x.is_dir() and (x / 'iterations.csv').exists()),
                  key=lambda x: int(x.name.split('_P')[1]))
    rows = [terminal_row(x) for x in runs]
    with open(d / 'terminal.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    checks = {}
    control = d / f'burgers_P{CONTROL * CONTROL}'
    if p2_12:
        checks['L1'] = check_l1(control, Path(p2_12) / 'B_instrumentation' / f'burgers_P{CONTROL * CONTROL}_cpu_paper'
                                if (Path(p2_12) / 'B_instrumentation').exists() else Path(p2_12) / 'burgers_P625')
    ctl = json.loads((control / 'timing.json').read_text()).get('paper_pass') if (control / 'timing.json').exists() else None
    if ctl:
        ratio = ctl['median_s'] / PAPER_CONTROL_TIME_S
        checks['L2'] = {'median_s': ctl['median_s'], 'paper_s': PAPER_CONTROL_TIME_S, 'ratio': ratio,
                        'times_s': ctl['times_s'], 'passed': abs(ratio - 1) <= L2_TOL}
    b2 = [r['b2_rel_err_k1'] for r in rows if r['b2_rel_err_k1'] is not None]
    checks['L3'] = {'max_rel_err_at_k1': max(b2) if b2 else None, 'tolerance': 1e-10,
                    'max_rel_err_over_runs': max((r['b2_max_rel_err'] for r in rows if r['b2_max_rel_err'] is not None),
                                                 default=None),
                    'passed': bool(b2) and max(b2) < 1e-10,
                    'rule': "as in Package 1: judged at k = 1, the run maximum reported (cancellation near convergence)"}
    l4 = {x.name: check_l4(x) for x in runs if (x / 'timing.json').exists()}
    checks['L4'] = {'runs': l4, 'passed': bool(l4) and all(c['identical'] for v in l4.values() for c in v)}
    (d / 'checks_item4.json').write_text(json.dumps(checks, indent=2))
    return rows, checks


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 3, item 4: Burgers LiL-Q at larger sizes.')
    ap.add_argument('stage', choices=('all', 'logged', 'summarize'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference-dir')
    ap.add_argument('--p2-12', dest='p2_12', help="P2-12's reruns (P2_12_reference_errors), for L1")
    ap.add_argument('--no-timing', action='store_true', help='the logged passes only (a laptop rehearsal)')
    args = ap.parse_args(argv)
    if args.stage in ('all', 'logged'):
        run_all(args.out, args.reference_dir, timing=not args.no_timing and args.stage == 'all')
    rows, checks = summarize(args.out, args.p2_12)
    for r in rows:
        print(f"  P={r['P']:5d}: eps_min {r['eps_min']:.3e} at k={r['k_min']}, k_e {r['k_e']}, k_r {r['k_r']} "
              f"({r['rule_class']}), median to k_e {r['median_k_e_s']}, to k_r {r['median_k_r_s']}")
    for name, c in checks.items():
        print(f"  {name}: {'passed' if c.get('passed') else 'FAILED'}")


if __name__ == '__main__':
    main()
