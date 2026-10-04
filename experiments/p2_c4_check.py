"""
Check C4 for the scalar reruns (Package 2, Sections 6.2 and 12.3)
=================================================================

Each reported LiL-Q configuration of Bratu and Burgers was rerun with the
``eps_ref`` column added. The iteration is deterministic, so the rerun's
``norm_R_h`` must equal ``package1``'s at every k to 1e-10 relative;
differences beyond BLAS round-off would mean a changed configuration. This
compares the two ``iterations.csv`` of each run, row by row, and also the
iteration counts. Writes ``check_c4.csv`` (one row per run) and
``check_c4.json``.

Usage::

    python experiments/p2_c4_check.py --reruns <stage1>/P2_12_reference_errors/B_instrumentation \\
        --package1 <package1 or wave2>/B_instrumentation --out <stage1>/P2_12_reference_errors
"""

import argparse
import csv
import json
from pathlib import Path

RUNS = [f'bratu_P{P}_cpu_paper' for P in (25, 100, 225)] + [f'burgers_P{P}_cpu_paper' for P in (25, 100, 225, 400, 625)]
TOLERANCE = 1e-10


def _norms(path):
    with open(path, newline='') as f:
        return [(int(r['k']), float(r['norm_R_h']), r.get('eps_ref', '')) for r in csv.DictReader(f)]


def compare(reruns, package1, runs=RUNS, tol=TOLERANCE):
    rows = []
    for run in runs:
        new, old = _norms(Path(reruns) / run / 'iterations.csv'), _norms(Path(package1) / run / 'iterations.csv')
        same_k = [k for k, _, _ in new] == [k for k, _, _ in old]
        rel = max(abs(a - b) / max(abs(b), 1e-300) for (_, a, _), (_, b, _) in zip(new, old))
        eps = [e for _, _, e in new if e not in ('', None)]
        rows.append({'run': run, 'rows_rerun': len(new), 'rows_package1': len(old), 'same_iterations': same_k,
                     'max_rel_diff_norm_R_h': rel, 'within_1e-10': same_k and rel <= tol,
                     'eps_ref_logged': len(eps) == len(new), 'eps_ref_final': float(eps[-1]) if eps else None})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check C4: the scalar reruns against package1.")
    ap.add_argument('--reruns', required=True)
    ap.add_argument('--package1', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    rows = compare(args.reruns, args.package1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'check_c4.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    passed = all(r['within_1e-10'] and r['eps_ref_logged'] for r in rows)
    (out / 'check_c4.json').write_text(json.dumps({'check': 'C4 (reruns)', 'tolerance': TOLERANCE, 'rows': rows,
                                                   'passed': passed}, indent=2))
    for r in rows:
        print(f"{r['run']:24s} rows {r['rows_rerun']}/{r['rows_package1']}  max rel diff {r['max_rel_diff_norm_R_h']:.1e}  "
              f"eps_ref final {r['eps_ref_final']}")
    print(f"C4 {'passed' if passed else 'FAILED'}")


if __name__ == '__main__':
    main()
