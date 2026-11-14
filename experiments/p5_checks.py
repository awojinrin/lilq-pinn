"""
Package 5, items 1, 2 and 4: the checks against the delivered runs
==================================================================

Section 6 of the advisor's Package 5 instructions defines checks for item 3
only. For items 1, 2 and 4 the plan (DECISIONS.md, Package 5, reading 7) is:
every changed driver, at its old settings, reproduces the delivered rows. Each
check below compares a Package 5 run with a delivered run of the same
configuration (both from Grace). They run at the assembly, on the laptop, where
the delivered results are.

**C1, item 1 at weight 10 against Packages 2 and 3.** Item 1's runs at the
delivered weight (lambda = 10) are the delivered configurations: Bratu and
Burgers are P2-16's runs at N/P = 10, Buckley-Leverett is P3-2's (2a: tensor
Chebyshev, 2b: lifted sine x Chebyshev) at N/P = 10 with the delivered initial
guess.
- Compared up to the delivered run's first stall: ||N|| (``norm_R_h``) and
  the L2 error (``eps_ref``) at every iterate up to the stall's, and the
  linearized residual (``norm_Rlin_h``) of every step before the stall step
  (that step's own linear residual is already at round-off). The linearized
  residual's difference is taken relative to the step's right-hand side
  ||f||, the scale of its system: a step can solve its system to round-off
  (Burgers' first step, ||r|| about 1e-15, differs by 60% between the laptop
  and Grace in relative terms).
- Before the stall these are fixed by the problem, not by round-off. They must
  agree to ``C1_TOL`` = 1e-8 relative, far above round-off (the laptop gives
  1e-13) and far below the two or three figures any table quotes.
- After the stall the values sit at round-off and differ between machines and
  thread counts, as in Package 4 (M5). The rule step and the stall level are
  recorded beside the delivered ones, without a verdict.

**C2, item 2's LiL-Q against Package 4.** Item 2's LiL-Q runs are Package 4's
runs (1a: Burgers p = 10, 18, 26, 30; 2a: Poisson-Boltzmann, seed 0, p = 10,
18, 22), run to the rule.
- The number of solves equals Package 4's returned iterate.
- The error at the returned iterate (Burgers: relative L2; Poisson-Boltzmann:
  L2 and H2) agrees to ``C2_TOL`` = 5e-3 relative, the same cutoff as C3: the
  tables quote ratios to delta_P to two figures. On the rank-deficient
  Poisson-Boltzmann spaces the error at the stall carries round-off in its
  fourth figure (p = 22, laptop against Grace: 5e-5 in L2), as delta_P does.
- ||N|| at the returned iterate is recorded with its relative difference.

**C3, delta_P of Package 4 by gelsy.** Item 4's ``delta_P/cheb_p{p}.json`` for
p = 10 to 30 recompute Package 4's delta_L2 and delta_H2 with the same driver.
They must agree to ``C3_TOL`` = 5e-3 relative: the manuscript quotes ratios to
delta_P to two figures (the advisor's reply to Package 4), and Package 4
measured delta_P across machines to about three digits at P >= 484.

**collect** writes ``checks.json``: each check's rows and verdict.

Usage::

    python experiments/p5_checks.py --out <package5 root> --package2 <P2 stage 2 root>
        --package3 <package3 root> --package4 <package4 root>
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from experiments.p4_common import read_csv, rel, to_float
from lilq.instrumentation import stall_rule_index

C1_TOL = 1e-8
C2_TOL = 5e-3
C3_TOL = 5e-3
C1_COLUMNS = ('norm_R_h', 'norm_Rlin_h', 'eps_ref')
LAM_DELIVERED = 10.0
RATIO = 10
P3_ITEMS = {'cheb': 'P3_2a_bl_chebyshev_certified', 'sine': 'P3_2b_bl_lifted_sine_cheb'}
P3_GUESS = {'viscous': 'ic', 'gravity': 'zero'}
P5_ITEMS = {'1': 'P5_1_weights', '2': 'P5_2_nlls', '4': 'P5_4_pb'}


def _finite(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def delivered_dir_c1(family, P, package2, package3):
    """The delivered run of item 1's family at size P and weight 10."""
    if family in ('bratu', 'burgers'):
        return Path(package2) / 'P2_16_certified' / f'{family}_P{P}_NP{RATIO}'
    _, variant, case = family.split('_')
    return Path(package3) / P3_ITEMS[variant] / f'{case}_P{P}_NP{RATIO}_{P3_GUESS[case]}'


def _rule(rows):
    solve = [r for r in rows if r['norm_Rlin_h'] not in ('', None)]
    k = stall_rule_index([to_float(r['chi']) for r in solve], [to_float(r['norm_Rlin_h']) for r in solve])
    return k, (None if k is None else to_float(solve[k]['norm_Rlin_h']))


def _diff(a, b, c):
    """The relative difference of column c; the linearized residual's is taken relative to the
    right-hand side ||f|| of its step, the system's own scale, since a step can solve its system
    to round-off (Burgers' first step: ||r|| about 1e-15)."""
    if c == 'norm_Rlin_h':
        return abs(to_float(a[c]) - to_float(b[c])) / to_float(b['norm_f_h'])
    return rel(to_float(a[c]), to_float(b[c]))


def compare_logs(here_dir, delivered_dir):
    """C1 for one pair of runs: iterates 0..first stall of the delivered run."""
    here, there = read_csv(Path(here_dir) / 'iterations.csv'), read_csv(Path(delivered_dir) / 'iterations.csv')
    meta = json.loads((Path(delivered_dir) / 'run.json').read_text())
    stall = meta.get('first_stall_iteration')
    last = min(len(here), len(there)) - 1 if stall is None else min(stall, len(here) - 1, len(there) - 1)
    worst = {}
    for c in C1_COLUMNS:
        # the iterate values up to the stall's iterate; the linear residual of the steps before it
        # (the stall step's own linear residual is already at round-off)
        top = last if c == 'norm_Rlin_h' else last + 1
        d = [_diff(here[k], there[k], c) for k in range(top)
             if _finite(to_float(here[k][c])) and _finite(to_float(there[k][c]))]
        worst[c] = max(d) if d else None
    k_here, s_here = _rule(here)
    k_there, s_there = _rule(there)
    diffs = [v for v in worst.values() if v is not None]
    return {'iterates_compared': last + 1, 'first_stall_delivered': stall,
            **{f'max_rel_{c}': v for c, v in worst.items()},
            'rule_step': k_here, 'rule_step_delivered': k_there,
            'stall_level': s_here, 'stall_level_delivered': s_there,
            'stall_rel': None if s_here is None or s_there is None else rel(s_here, s_there),
            'pass': bool(diffs) and max(diffs) <= C1_TOL}


def c1(out_root, package2, package3):
    from experiments import p5_1_weights as w
    rows = []
    for family in w.families():
        for P in w.sizes(family):
            here = w.run_dir_of(out_root, family, P, LAM_DELIVERED)
            there = delivered_dir_c1(family, P, package2, package3)
            if not (here / 'iterations.csv').exists():
                continue
            if not (there / 'iterations.csv').exists():
                rows.append({'family': family, 'P': P, 'delivered': str(there), 'pass': None,
                             'note': 'no delivered run'})
                continue
            rows.append({'family': family, 'P': P, **compare_logs(here, there)})
    return {'check': 'C1', 'tolerance': C1_TOL, 'rows': rows, 'pass': _verdict(rows)}


def _p4_rows(package4):
    p4 = Path(package4)
    burgers = {int(r['p']): r for r in read_csv(p4 / 'P4_1a_burgers_gauss' / 'terminal.csv')}
    pb = {int(r['p']): r for r in read_csv(p4 / 'P4_2_pb_inclusions' / 'lilq' / 'terminal.csv') if int(r['seed']) == 0}
    return burgers, pb


def c2(out_root, package4):
    from experiments import p5_2_nlls as n
    burgers, pb = _p4_rows(package4)
    rows = []
    for problem, sizes in n.SIZES.items():
        for p in sizes:
            f = n.run_dir_of(out_root, problem, p, 'lilq') / 'result.json'
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            d = (burgers if problem == 'burgers' else pb).get(p)
            if d is None:
                rows.append({'problem': problem, 'p': p, 'pass': None, 'note': 'no Package 4 row'})
                continue
            errs = {'eL2': to_float(d['error'])} if problem == 'burgers' else \
                {'eL2': to_float(d['eL2']), 'eH2': to_float(d['eH2'])}
            diffs = {k: rel(r[k], v) for k, v in errs.items()}
            solves_p4 = int(d['returned_iterate'])
            rows.append({'problem': problem, 'p': p, 'solves': r['solves'], 'solves_package4': solves_p4,
                         **{f'{k}': r[k] for k in errs}, **{f'{k}_package4': v for k, v in errs.items()},
                         **{f'rel_{k}': v for k, v in diffs.items()},
                         'norm_N': r['norm_N'], 'norm_N_package4': to_float(d['norm_N_returned']),
                         'rel_norm_N': rel(r['norm_N'], to_float(d['norm_N_returned'])),
                         'pass': r['solves'] == solves_p4 and max(diffs.values()) <= C2_TOL})
    return {'check': 'C2', 'tolerance': C2_TOL, 'rows': rows, 'pass': _verdict(rows)}


def c3(out_root, package4):
    from experiments import p5_4_pb as m
    _, pb = _p4_rows(package4)
    rows = []
    for p in m.SIZES_P4:
        f = Path(out_root) / m.ITEM / 'delta_P' / f'cheb_p{p}.json'
        if not f.exists() or p not in pb:
            continue
        d = json.loads(f.read_text())
        ref = {'L2': to_float(pb[p]['delta_L2']), 'H2': to_float(pb[p]['delta_H2'])}
        diffs = {k: rel(d[f'delta_{k}_gelsy'], v) for k, v in ref.items()}
        rows.append({'p': p, **{f'delta_{k}_gelsy': d[f'delta_{k}_gelsy'] for k in ref},
                     **{f'delta_{k}_package4': v for k, v in ref.items()},
                     **{f'rel_{k}': v for k, v in diffs.items()},
                     'pass': max(diffs.values()) <= C3_TOL})
    return {'check': 'C3', 'tolerance': C3_TOL, 'rows': rows, 'pass': _verdict(rows)}


def _verdict(rows):
    """True if every row passes; False if one fails; None if there is nothing to judge."""
    v = [r['pass'] for r in rows]
    if any(x is False for x in v):
        return False
    if not v or any(x is None for x in v):
        return None
    return True


def collect(out_root, package2, package3, package4):
    checks = {'C1': c1(out_root, package2, package3), 'C2': c2(out_root, package4), 'C3': c3(out_root, package4)}
    summary = {k: v['pass'] for k, v in checks.items()}
    (Path(out_root) / 'checks.json').write_text(json.dumps({'summary': summary, 'checks': checks}, indent=2,
                                                           default=str))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 5, items 1, 2 and 4: the checks against the delivered runs.')
    ap.add_argument('--out', required=True, help='the Package 5 results root')
    ap.add_argument('--package2', required=True, help="Package 2 stage 2's root (P2_16_certified/)")
    ap.add_argument('--package3', required=True, help="Package 3's root (P3_2a_..., P3_2b_...)")
    ap.add_argument('--package4', required=True, help="Package 4's root (P4_1a_..., P4_2_...)")
    a = ap.parse_args(argv)
    summary = collect(a.out, a.package2, a.package3, a.package4)
    print(json.dumps(summary))
    return 0 if all(v is not False for v in summary.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
