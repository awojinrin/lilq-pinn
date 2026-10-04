"""
Reference errors of the saved four-method models (Package 2, Section 2.6 and Section 6.2)
=========================================================================================

Evaluates the saved NiL-N, NiL-Q and LiL-N models of the four-method runs
(``four_method_jobs/<benchmark>_<cpu|gpu>/models/``; seeds 0-2 for NiL, every
P, both devices) on a benchmark's test grid against its reference solution,
and writes one row per model: the relative discrete L2 error of the final
model (``eps_ref_final``).

**Only the final models were saved**, so ``eps_ref_min`` and ``k_min`` (the
smallest error over the iterations and where it occurred) cannot be computed
without rerunning; they are left empty and the reason is in ``note``. The stall
controls (``*_f1_stall_rule``) are not among the reported runs and are skipped.

**LiL-Q** (``--lilq-runs``): the Package 2 reruns of the reported Bratu and
Burgers configurations log ``eps_ref`` at every iteration, so their rows have
all three columns, read from each run's ``iterations.csv``.

Benchmarks:
- ``bl`` and ``bl_gravity``: the finite-difference reference of
  ``problems.buckley_leverett.reference_solution`` on its 201 x 201 grid
  (``bl_reference_errors_networks.csv``, Package 2 Section 2.6).
- ``bratu`` and ``burgers``: the Package 2 references (Section 6.2) in
  ``--reference-dir`` (``scalar_reference_errors.csv``, with
  ``--first-column benchmark``).

Usage::

    python experiments/network_reference_errors.py --package <package1> --benchmarks bl bl_gravity \\
        --out <package2_results>/G1_release/bl_reference_errors_networks.csv
    python experiments/network_reference_errors.py --package <package1> --benchmarks bratu burgers \\
        --reference-dir <stage1>/reference --first-column benchmark \\
        --lilq-runs <stage1>/P2_12_reference_errors/B_instrumentation \\
        --out <package2_results>/P2_12_reference_errors/scalar_reference_errors.csv
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import numpy as np
import torch

from lilq.saved_models import load_network, load_solution
from lilq.test_errors import tensor_grid_values

LILQ_RE = re.compile(r'^(?P<bench>.+)_P(?P<P>\d+)_(?P<device>cpu|cuda)_paper$')
LILQ_NOTE = 'Package 2 rerun of the reported configuration; eps_ref logged at every iteration'
MODEL_RE = re.compile(r'^(?P<bench>.+)_P(?P<P>\d+)_(?P<method>NiL-N|NiL-Q|LiL-N)_(?P<seed>s\d+|sna)_(?P<device>cpu|cuda)$')
NOTE = 'only the final model was saved: eps_ref_min and k_min need the intermediate iterates'
COLUMNS = ('case', 'P', 'method', 'seed', 'device', 'eps_ref_final', 'eps_ref_min', 'k_min', 'run', 'commit', 'note')


def saved_models(package, bench):
    """``(match, path)`` for every reported four-method model of ``bench``."""
    jobs = Path(package) / 'B_instrumentation' / 'four_method_jobs'
    out = []
    for path in sorted(jobs.glob(f'{bench}_*/models/*')):
        m = MODEL_RE.match(path.name)
        if m and m['bench'] == bench and path.is_dir():
            out.append((m, path))
    return sorted(out, key=lambda t: (int(t[0]['P']), t[0]['method'], t[0]['seed'], t[0]['device']))


def evaluate(method_dir, method, axes):
    """The model's field on the tensor grid ``axes`` (array [i, j])."""
    if method == 'LiL-N':
        sol = load_solution(method_dir)
        e = sol['fields']['u']
        return tensor_grid_values(e['basis'], e['coefficients'], axes)
    model = load_network(method_dir)['model']
    A, B = np.meshgrid(*axes, indexing='ij')
    dtype = next(model.parameters()).dtype
    pts = torch.tensor(np.stack([A.ravel(), B.ravel()], axis=1), dtype=dtype)
    with torch.no_grad():
        return model(pts).reshape(A.shape).double().numpy()


def _commit(path, package):
    import json
    d = Path(path)
    while d != Path(package) and d.parent != d:
        if (d / 'hardware.json').exists():
            return (json.loads((d / 'hardware.json').read_text()).get('git') or {}).get('commit')
        d = d.parent
    return None


def bl_reference(config):
    from problems.buckley_leverett import TEST_GRID, reference_solution
    axes = [np.linspace(*config.x_domain, TEST_GRID[0]), np.linspace(0.0, config.T_final, TEST_GRID[1])]
    return axes, reference_solution(config)


def rows_for(package, bench, reference):
    """Rows for one benchmark; ``reference(config) -> (axes, field)``."""
    rows, cache = [], {}
    for m, path in saved_models(package, bench):
        cfg = (load_network(path) if m['method'] != 'LiL-N' else load_solution(path))['config']
        key = repr(cfg)
        if key not in cache:
            cache[key] = reference(cfg)
        axes, ref = cache[key]
        u = evaluate(path, m['method'], axes)
        rows.append({'case': bench, 'P': int(m['P']), 'method': m['method'],
                     'seed': '' if m['seed'] == 'sna' else int(m['seed'][1:]), 'device': m['device'],
                     'eps_ref_final': float(np.linalg.norm(u - ref) / np.linalg.norm(ref)),
                     'eps_ref_min': '', 'k_min': '', 'run': path.relative_to(package).as_posix(),
                     'commit': _commit(path, package), 'note': NOTE})
    return rows


def lilq_rows(runs_dir, bench):
    """LiL-Q rows of ``bench`` from the reruns in ``runs_dir``: the final and
    smallest ``eps_ref`` over the iterations, and the k of the smallest."""
    runs_dir = Path(runs_dir)
    rows = []
    for path in sorted(runs_dir.iterdir()):
        m = LILQ_RE.match(path.name)
        if not (m and m['bench'] == bench and (path / 'iterations.csv').exists()):
            continue
        with open(path / 'iterations.csv', newline='') as f:
            log = [(int(r['k']), float(r['eps_ref'])) for r in csv.DictReader(f) if r.get('eps_ref', '') != '']
        k_min, eps_min = min(log, key=lambda t: t[1])
        rows.append({'case': bench, 'P': int(m['P']), 'method': 'LiL-Q', 'seed': '', 'device': m['device'],
                     'eps_ref_final': log[-1][1], 'eps_ref_min': eps_min, 'k_min': k_min,
                     'run': path.relative_to(runs_dir.parent.parent).as_posix(),
                     'commit': _commit(path, Path(path.anchor)), 'note': LILQ_NOTE})
    return rows


def npz_reference(reference_dir, benchmark):
    """Bratu or Burgers: the Package 2 reference (Section 6.2) on its grid,
    checked against the model's domain."""
    from lilq.references import load_reference, reference_path
    axes, ref, _ = load_reference(reference_path(reference_dir, benchmark))

    def reference(config):
        second = getattr(config, 'y_domain', None) or (0.0, config.T_final)
        assert np.allclose([axes[0][0], axes[0][-1]], config.x_domain), 'x domain'
        assert np.allclose([axes[1][0], axes[1][-1]], second), 'second domain'
        return axes, ref

    return reference


REFERENCES = {'bl': bl_reference, 'bl_gravity': bl_reference, 'bratu': None, 'burgers': None}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reference errors of the saved four-method models.")
    ap.add_argument('--package', required=True)
    ap.add_argument('--benchmarks', nargs='+', default=['bl', 'bl_gravity'], choices=list(REFERENCES))
    ap.add_argument('--reference-dir', default=None, help="package2_results/reference (bratu, burgers)")
    ap.add_argument('--lilq-runs', default=None,
                    help="the Package 2 LiL-Q reruns (B_instrumentation), whose eps_ref is in iterations.csv")
    ap.add_argument('--first-column', default='case', choices=('case', 'benchmark'),
                    help="'case' for bl_reference_errors_networks.csv, 'benchmark' for scalar_reference_errors.csv")
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    rows = []
    for bench in args.benchmarks:
        ref = REFERENCES[bench] or npz_reference(args.reference_dir, bench)
        both = rows_for(Path(args.package), bench, ref) + (lilq_rows(args.lilq_runs, bench) if args.lilq_runs else [])
        rows += sorted(both, key=lambda r: (r['P'], r['method'], str(r['seed']), r['device']))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = (args.first_column,) + COLUMNS[1:]
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{args.first_column: r['case'], **{k: r[k] for k in COLUMNS[1:]}} for r in rows])
    print(f"Wrote {out}: {len(rows)} models")


if __name__ == '__main__':
    main()
