"""
Stall controls that depart from their original run (Package 2, Section 2.5)
===========================================================================

For every stall control (``four_method_controls.csv``) whose logged loss
history departs from its original's (``departs_at_iteration`` set), the
original's stall iteration (``four_method_tables.csv``), the gap between the
two, whether the departure is at the original's last logged row before the
stall (its history is logged every 10 iterations plus the final row), and the run
ids of both. The manuscript's statement ("retrace their original exactly up to
the stall") holds where the departure lies within the stall window.

Usage::

    python experiments/stall_control_departures.py --b-root <package1>/B_instrumentation --out departures.csv
"""

import argparse
import csv
import json
from pathlib import Path

COLUMNS = ('benchmark', 'P', 'method', 'seed', 'device', 'departs_at_iteration', 'original_stall_iteration',
           'original_total_iterations', 'gap', 'at_last_logged_row_before_stall', 'original_run', 'control_run')


def departures(b_root):
    b_root = Path(b_root)
    with open(b_root / 'four_method_controls.csv', newline='') as f:
        controls = [r for r in csv.DictReader(f) if r.get('departs_at_iteration')]
    with open(b_root / 'four_method_tables.csv', newline='') as f:
        originals = {(r['benchmark'], r['P'], r['method'], r['seed'], r['device']): r
                     for r in csv.DictReader(f) if not r.get('variant')}
    rows = []
    for c in controls:
        o = originals[(c['benchmark'], c['P'], c['method'], c['seed'], c['device'])]
        stall = int(o['stall_iteration'] or o['total_iterations'])
        dep = int(c['departs_at_iteration'])
        logged = [int(it) for it, *_ in json.loads(o['loss_history_every_10'])]
        tag = f"s{c['seed']}" if c['seed'] != '' else 'sna'           # LiL-N: deterministic, no seed
        name = f"{c['benchmark']}_P{c['P']}_{c['method']}_{tag}_{c['device']}"
        rows.append({'benchmark': c['benchmark'], 'P': int(c['P']), 'method': c['method'],
                     'seed': int(c['seed']) if c['seed'] != '' else '',
                     'device': c['device'], 'departs_at_iteration': dep, 'original_stall_iteration': stall,
                     'original_total_iterations': int(o['total_iterations']), 'gap': stall - dep,
                     # the history is every 10 iterations plus the final row (the stall): the departure
                     # is 'at the last logged row' when no logged row lies between it and the stall
                     'at_last_logged_row_before_stall': dep == stall or dep == max(i for i in logged if i < stall),
                     'original_run': name,
                     'control_run': f'{name}_f1_stall_rule'})
    return sorted(rows, key=lambda r: (r['gap'], r['benchmark'], r['P']))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stall controls that depart from their original run.")
    ap.add_argument('--b-root', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    rows = departures(args.b_root)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['control_run']}: departs at {r['departs_at_iteration']}, original stall {r['original_stall_iteration']} "
              f"(gap {r['gap']}; at the last logged row before the stall: {r['at_last_logged_row_before_stall']})")


if __name__ == '__main__':
    main()
