"""
Package 4, check M0 and the combined checks.json
================================================

**M0** (the advisor's instructions of 12 November 2026, Section 5). After all
code changes, the paper's Burgers run at P = 625 and its Bratu run at P = 225
must reproduce ``package1`` to all digits: the number of rows of
``iterations.csv``, and ``norm_R_h`` and ``norm_Rlin_h`` at every row.
- The runs are component B's paper passes, rebuilt from
  ``experiments.residual_band_figures`` exactly as ``component_b.py`` builds
  them, on a whole Grace node at 48 threads, as ``package1``'s were.
- The comparison is Package 3's K8 (``experiments.p3_k0_k8.compare_with_package1``).

**collect** merges ``checks_item1.json``, ``checks_item2.json`` and
``m0.json`` into ``checks.json`` (M0 to M9).

**provenance** writes ``hardware.json`` and ``environment.txt`` of the node a
job runs on into ``<root>/provenance/<name>/`` (each timed job records its own).

Usage::

    python experiments/p4_checks.py m0 --out <root> --package1 <package1>
    python experiments/p4_checks.py collect --out <root>
    python experiments/p4_checks.py provenance --out <root> --name <job>
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import lilq.blas_threads  # noqa: F401  (must import before numpy)

from lilq.provenance import capture_blas_thread_env
from lilq.source_lock import current_commit

M0_RUNS = (('burgers', 625), ('bratu', 225))


def _paper_pass(benchmark, P, out_dir):
    import experiments.residual_band_figures as rbf
    from lilq.iteration_log import IterationLogger
    from problems.bratu import run_lil_q as bratu_q
    from problems.burgers import run_lil_q as burgers_q
    build, run_lil_q = {'burgers': (rbf._burgers_runs, burgers_q), 'bratu': (rbf._bratu_runs, bratu_q)}[benchmark]
    config, opt = next((c, o) for p, c, o in build(quick=False) if p == P)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = IterationLogger()
    run_lil_q(config, opt, verbose=False, iteration_logger=logger, run_json_path=out_dir / 'run.json')
    logger.to_csv(out_dir / 'iterations.csv')
    return out_dir / 'iterations.csv'


def m0(out_root, package1):
    from experiments.p3_k0_k8 import compare_with_package1
    root = Path(out_root) / 'P4_checks' / 'm0'
    B = Path(package1) / 'B_instrumentation'
    runs = {}
    for benchmark, P in M0_RUNS:
        name = f'{benchmark}_P{P}_cpu_paper'
        t0 = time.perf_counter()
        here = _paper_pass(benchmark, P, root / name)
        runs[name] = {**compare_with_package1(here, B / name / 'iterations.csv'),
                      'seconds': time.perf_counter() - t0, 'package1': str(B / name / 'iterations.csv')}
    return {'check': 'M0', 'runs': runs, 'pass': all(r['passed'] for r in runs.values()),
            'threads': capture_blas_thread_env().get('OMP_NUM_THREADS'), 'commit': current_commit()}


def combined_pass(verdicts):
    """False if any part failed; None (not evaluated here) if any part has no verdict, as
    M6 without the advisor's expected values; True otherwise."""
    if any(v is False for v in verdicts):
        return False
    if not verdicts or any(v is None for v in verdicts):
        return None
    return True


def collect(out_root):
    root = Path(out_root)
    out = {}
    m0_path = root / 'P4_checks' / 'm0.json'
    if m0_path.exists():
        out['M0'] = json.loads(m0_path.read_text())
    items = {}
    for item, name in (('item1', 'checks_item1.json'), ('item2', 'P4_2_pb_inclusions/checks_item2.json')):
        f = root / name
        if f.exists():
            items[item] = json.loads(f.read_text())
    for item, checks in items.items():
        for k, v in checks.items():
            key = k.split('_item')[0]                   # M9_item1, M9_item2 -> M9
            out.setdefault(key, {})[item] = v
    for key, v in list(out.items()):                   # a check over both items passes if each item's does
        if key != 'M0':
            v['pass'] = combined_pass([x.get('pass') for x in v.values() if isinstance(x, dict)])
    summary = {k: (v.get('pass') if isinstance(v, dict) else None) for k, v in out.items()}
    (root / 'checks.json').write_text(json.dumps({'summary': summary, 'checks': out}, indent=2, default=str))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 4: check M0 and the combined checks.json.')
    ap.add_argument('stage', choices=('m0', 'collect', 'provenance'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--package1')
    ap.add_argument('--name', help='provenance: the folder under <root>/provenance/')
    a = ap.parse_args(argv)
    if a.stage == 'provenance':
        if not a.name:
            ap.error('--name is required for provenance')
        from lilq.provenance import save_provenance
        out = Path(a.out) / 'provenance' / a.name
        out.mkdir(parents=True, exist_ok=True)
        save_provenance(out)
        print(out)
        return 0
    if a.stage == 'm0':
        if not a.package1:
            ap.error('--package1 is required for m0')
        result = m0(a.out, a.package1)
        path = Path(a.out) / 'P4_checks' / 'm0.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        for name, r in result['runs'].items():
            print(f"  {name}: {'identical' if r['passed'] else 'DIFFERS'} ({r['rows_here']} rows; package1 "
                  f"{r['rows_package1']})")
        print(f"M0 {'passed' if result['pass'] else 'FAILED'}")
        return 0 if result['pass'] else 1
    print(json.dumps(collect(a.out), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
