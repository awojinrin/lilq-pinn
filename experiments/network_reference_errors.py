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

Benchmarks:
- ``bl`` and ``bl_gravity``: the finite-difference reference of
  ``problems.buckley_leverett.reference_solution`` on its 201 x 201 grid
  (``bl_reference_errors_networks.csv``, Package 2 Section 2.6).
- ``bratu`` and ``burgers``: the Package 2 references (Section 6.2), passed in
  as ``reference`` -- added with them.

Usage::

    python experiments/network_reference_errors.py --package <package1> --benchmarks bl bl_gravity \\
        --out <package2_results>/G1_release/bl_reference_errors_networks.csv
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


REFERENCES = {'bl': bl_reference, 'bl_gravity': bl_reference}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reference errors of the saved four-method models.")
    ap.add_argument('--package', required=True)
    ap.add_argument('--benchmarks', nargs='+', default=['bl', 'bl_gravity'], choices=list(REFERENCES))
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    rows = []
    for bench in args.benchmarks:
        rows += rows_for(Path(args.package), bench, REFERENCES[bench])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}: {len(rows)} models")


if __name__ == '__main__':
    main()
