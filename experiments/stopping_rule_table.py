"""
The stall-based stopping rule, recomputed from the logs (Table 16 and the held-out evaluation)
=============================================================================================

The manuscript's termination rule (the advisor's reply to wave 4, item
2(b)): both conditions of the stall detector,

    chi_k <= tau_chi   and   | ||R_lin^(k)|| - ||R_lin^(k-1)|| | <= tau_r ||R_lin^(k)||,

must hold at ``n_s`` consecutive steps, with tau_chi = 0.1, tau_r = 0.01
and n_s = 2 by default; the rule returns u^(k+1) at the first step k where
it holds (``lilq.instrumentation.stall_rule_index``). Nothing is rerun: the
rule is evaluated on ``chi`` and ``norm_Rlin_h`` of existing
``iterations.csv`` files (the logged ``stall_flag`` is not used -- logs
written before this change used tau_r = 0.1).

**Table 16** (the kmax passes, stopping rule disabled, of the Section 3.3
CPU runs): for each run and each n_s in ``--n-s``, the returned iterate
k + 1, its class (``C``: round-off limited, ``roundoff_ratio`` < 10 kappa
eps with kappa the retained one where finite; ``A`` otherwise), R = its
||R||_h over the smallest in the pass, and E = its eps_u over the eps_u at
the paper pass's stop (the paper pass's ``k`` maximum). Over the runs:
``early``, how many return R > 1.02; ``errstop``, how many return an
eps_u above twice the pass's smallest at an iterate before that smallest;
and the largest R.

**Held out** (Component C's oversampling runs, except the paper-grid
Kovasznay r = 3 and Bratu r = 10 runs, which are in Table 16's
configurations): how many runs the rule fires within the log, how many are
censored (the single-step conditions are first met at the last logged step,
so n_s steps never fit), the largest R of the fired runs, and for the
fired Kovasznay runs E over the run's smallest eps_u and over its last.

This reproduces the advisor's ``ns_eval.py`` (his
``analysis_scripts/termfix/``), whose n_s = 2 output the manuscript quotes:
on the wave 1 logs, 69 of 192 held-out runs fire, 76 are censored, and the
returned residuals are within 0.04% of the smallest
(``tests/test_stopping_rule_table.py``).

Usage::

    python experiments/stopping_rule_table.py --b-root <package1>/B_instrumentation \\
        --oversampling-root <package1>/C_oversampling/runs --out-dir <package1>/B_instrumentation/stopping_rule
"""

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import numpy as np

from lilq.instrumentation import (
    DEFAULT_N_S, DEFAULT_TAU_CHI, DEFAULT_TAU_R, EPS_MACH, stall_rule_index,
)

TABLE16_RUNS = ('bratu_P25', 'bratu_P100', 'bratu_P225',
                'burgers_P25', 'burgers_P100', 'burgers_P225', 'burgers_P400', 'burgers_P625',
                'bl_P64', 'bl_P256', 'bl_P576', 'bl_P1024',
                'bl_gravity_P64', 'bl_gravity_P256', 'bl_gravity_P576', 'bl_gravity_P1024',
                'kovasznay_P75', 'kovasznay_P300', 'kovasznay_P675', 'kovasznay_P1200', 'kovasznay_P1875',
                'beltrami_P7984')
EARLY_R = 1.02            # a returned ||R||_h more than 2% above the pass's smallest
ERRSTOP_FACTOR = 2.0      # a returned eps_u more than twice the pass's smallest, before it


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def read_log(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    cols = {c: np.array([_f(r.get(c)) for r in rows]) for c in
            ('k', 'chi', 'norm_Rlin_h', 'norm_R_h', 'eps_u', 'roundoff_ratio', 'kappa', 'kappa_retained')}
    return cols


def rule_index(d, n_s, tau_chi, tau_r):
    """The rule on the solve rows (the terminal row K has no system)."""
    K = int(np.nanmax(d['k']))
    return stall_rule_index(d['chi'][:K], d['norm_Rlin_h'][:K], n_s=n_s, tau_chi=tau_chi, tau_r=tau_r), K


def classify(d, i):
    """``C`` (round-off limited) or ``A`` at row i."""
    kr = d['kappa_retained'][i] if np.isfinite(d['kappa_retained'][i]) else d['kappa'][i]
    return 'C' if d['roundoff_ratio'][i] < 10 * kr * EPS_MACH else 'A'


def table16(b_root, n_s_values, tau_chi, tau_r, runs=TABLE16_RUNS):
    rows, summary = [], {}
    for n_s in n_s_values:
        summary[n_s] = {'early': 0, 'errstop': 0, 'R_max': 0.0, 'never_fires': 0}
    for name in runs:
        kmax = read_log(Path(b_root) / f'{name}_cpu_kmax' / 'iterations.csv')
        paper = read_log(Path(b_root) / f'{name}_cpu_paper' / 'iterations.csv')
        stop = int(np.nanmax(paper['k']))
        e_stop = paper['eps_u'][-1]
        R, e = kmax['norm_R_h'], kmax['eps_u']
        e_min = np.nanmin(e[1:])
        k_min = int(np.nanargmin(e[1:])) + 1
        for n_s in n_s_values:
            i, K = rule_index(kmax, n_s, tau_chi, tau_r)
            if i is None:
                summary[n_s]['never_fires'] += 1
                rows.append({'run': name, 'paper_stop': stop, 'n_s': n_s, 'fires_at': '', 'returned': ''})
                continue
            kr = i + 1
            r_ratio = R[kr] / np.nanmin(R)
            s = summary[n_s]
            s['R_max'] = max(s['R_max'], float(r_ratio))
            early = bool(r_ratio > EARLY_R)
            errstop = bool(e[kr] > ERRSTOP_FACTOR * e_min and kr < k_min)
            s['early'] += early
            s['errstop'] += errstop
            rows.append({'run': name, 'paper_stop': stop, 'n_s': n_s, 'fires_at': i, 'returned': kr,
                         'class': classify(kmax, i), 'R_over_min': float(r_ratio),
                         'eps_u_returned': float(e[kr]), 'E_over_paper_stop': float(e[kr] / e_stop),
                         'eps_u_min': float(e_min), 'k_eps_u_min': k_min, 'early': early, 'errstop': errstop})
    return rows, summary


def held_out(oversampling_root, n_s_values, tau_chi, tau_r):
    run_dirs = sorted(p for p in Path(oversampling_root).iterdir() if (p / 'iterations.csv').exists())
    run_dirs = [p for p in run_dirs if not ((p.name.startswith('kovasznay') and '_r3_paper' in p.name)
                                            or (p.name.startswith('bratu') and '_r10_paper' in p.name))]
    rows, summary = [], {}
    for n_s in n_s_values:
        fired = censored = 0
        R_ratios, kov_min, kov_last = [], [], []
        by_class = {'A': [], 'C': []}
        for p in run_dirs:
            d = read_log(p / 'iterations.csv')
            i, K = rule_index(d, n_s, tau_chi, tau_r)
            i1, _ = rule_index(d, 1, tau_chi, tau_r)
            row = {'run': p.name, 'n_s': n_s, 'K': K}
            if i is None:
                row['outcome'] = 'censored' if i1 is not None else 'never'
                censored += i1 is not None
                rows.append(row)
                continue
            fired += 1
            kr = i + 1
            R = d['norm_R_h']
            row.update(outcome='fired', fires_at=i, returned=kr, cls=classify(d, i),
                       R_over_min=float(R[kr] / np.nanmin(R)))
            R_ratios.append(row['R_over_min'])
            if p.name.startswith('kovasznay'):
                e = d['eps_u']
                row.update(E_over_min=float(e[kr] / np.nanmin(e[1:])), E_over_last=float(e[kr] / e[K]))
                kov_min.append(row['E_over_min'])
                kov_last.append(row['E_over_last'])
                if 'P1200' in p.name:
                    by_class[row['cls']].append(float(e[kr]))
            rows.append(row)
        summary[n_s] = {
            'runs': len(run_dirs), 'fired': fired, 'censored': censored,
            'R_max': max(R_ratios) if R_ratios else None,
            'kovasznay_fired': len(kov_min),
            'kovasznay_E_over_min_max': max(kov_min) if kov_min else None,
            'kovasznay_E_over_min_median': statistics.median(kov_min) if kov_min else None,
            'kovasznay_E_over_last_range': [min(kov_last), max(kov_last)] if kov_last else None,
            'kovasznay_P1200_by_class': {c: {'n': len(v), 'eps_u_range': [min(v), max(v)] if v else None}
                                         for c, v in by_class.items()},
        }
    return rows, summary


def _write(path, rows):
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, restval='')
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Table 16 and the held-out evaluation of the stall-based rule.")
    ap.add_argument('--b-root', required=True, help='B_instrumentation with the *_cpu_kmax and *_cpu_paper runs')
    ap.add_argument('--oversampling-root', default=None, help="C_oversampling/runs (the held-out evaluation)")
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n-s', type=int, nargs='+', default=[1, DEFAULT_N_S, 3])
    ap.add_argument('--tau-chi', type=float, default=DEFAULT_TAU_CHI)
    ap.add_argument('--tau-r', type=float, default=DEFAULT_TAU_R)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_rows, t_sum = table16(args.b_root, args.n_s, args.tau_chi, args.tau_r)
    _write(out / 'stopping_rule_table16.csv', t_rows)
    result = {'tau_chi': args.tau_chi, 'tau_r': args.tau_r, 'n_s': args.n_s, 'table16': t_sum}
    print(f"Table 16 (tau_chi = {args.tau_chi}, tau_r = {args.tau_r}):")
    for n_s, s in t_sum.items():
        print(f"  n_s = {n_s}: early {s['early']}, errstop {s['errstop']}, R max {s['R_max']:.4f}, "
              f"never fires {s['never_fires']}")
    if args.oversampling_root:
        h_rows, h_sum = held_out(args.oversampling_root, args.n_s, args.tau_chi, args.tau_r)
        _write(out / 'stopping_rule_heldout.csv', h_rows)
        result['held_out'] = h_sum
        print("Held out:")
        for n_s, s in h_sum.items():
            print(f"  n_s = {n_s}: {s['fired']} of {s['runs']} fire, {s['censored']} censored, "
                  f"R max {s['R_max']:.5f}")
    (out / 'stopping_rule_summary.json').write_text(json.dumps(result, indent=2))
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
