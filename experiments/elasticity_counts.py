"""
Elasticity: active coefficients per field and the gelsy rank (Package 2, item 9, P2-6)
======================================================================================

From the saved elasticity solutions of the Section 3.3 paper passes
(``B_instrumentation/elasticity_P<P>_cpu_paper/solution.pt``; no solve is
rerun): per field (u_x, u_y) the number of coefficients above
``1e-12 * max|beta|`` of that field, and the numerical rank of the system as
LAPACK's ``gelsy`` determined it in the run (``iterations.csv``,
``num_rank_gelsy``), beside the SVD rank where it was logged.

Usage::

    python experiments/elasticity_counts.py --package <package1> --out <package2_results>/P2_6_elasticity_counts.csv
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import numpy as np

from lilq.saved_models import load_solution

THRESHOLD = 1e-12
FIELDS = (('u_x', 'u'), ('u_y', 'v'))
COLUMNS = ('P', 'field', 'n_coeff', 'n_active', 'threshold', 'max_abs_coeff', 'rank_gelsy', 'rank_svd', 'run', 'commit')


def counts(package):
    B = Path(package) / 'B_instrumentation'
    runs = sorted(B.glob('elasticity_P*_cpu_paper'), key=lambda p: int(p.name.split('_P')[1].split('_')[0]))
    rows = []
    for run in runs:
        sol = load_solution(run)
        with open(run / 'iterations.csv', newline='') as f:
            solve = [r for r in csv.DictReader(f) if r.get('num_rank_gelsy')]
        rank_gelsy = int(solve[-1]['num_rank_gelsy']) if solve else None
        rank_svd = int(solve[-1]['num_rank_svd']) if solve and solve[-1].get('num_rank_svd') else None
        P = json.loads((run / 'run.json').read_text())['P_total']
        commit = json.loads((run / 'run.json').read_text()).get('commit')
        for label, name in FIELDS:
            beta = np.asarray(sol['fields'][name]['coefficients'])
            m = float(np.abs(beta).max())
            rows.append({'P': P, 'field': label, 'n_coeff': beta.size,
                         'n_active': int((np.abs(beta) > THRESHOLD * m).sum()), 'threshold': THRESHOLD,
                         'max_abs_coeff': m, 'rank_gelsy': rank_gelsy, 'rank_svd': rank_svd,
                         'run': run.relative_to(package).as_posix(), 'commit': commit})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Elasticity active-coefficient counts.")
    ap.add_argument('--package', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    rows = counts(Path(args.package))
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"P = {r['P']:5d} {r['field']}: {r['n_active']} of {r['n_coeff']} active; gelsy rank {r['rank_gelsy']}, "
              f"SVD rank {r['rank_svd']}")


if __name__ == '__main__':
    main()
