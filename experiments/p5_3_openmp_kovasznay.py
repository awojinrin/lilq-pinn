"""
Package 5, item 3: the OpenMP library of Package 1's Component A baselines on the CPU (a diagnostic; the user's decision, 17 November 2026)
=========================================================================================================================================

Every Package 1 Component A process (``experiments/component_a.py``: Kovasznay, the F1 and F2 baselines)
imported torch before numpy and loaded the PyTorch wheel's own OpenMP library; its CPU runs
(``A_calibration/full_cpu/``, the representatives F1_18 and F2_05, seeds 0-4, 3,600 s each) are the
baselines the Kovasznay comparison times against LiL-Q on the CPU, whose times (Parts B and C) come
from processes with the GCC module's library. This driver runs the same two representatives on the
CPU with either library, chosen by which package a fresh process imports first
(``p5_3_openmp.start_process``), and records the time per iteration of each phase:

- **F1** (F1_18, seed 0): ``F1_ADAM_ITERATIONS`` Adam iterations (log rows at 0, 100 and 200),
  then L-BFGS to ``F1_BUDGET_S``. Measures: Adam seconds per iteration between the rows at 100
  and 200; L-BFGS seconds per iteration between its second and last log row (a row every 10
  iterations; the first row carries the objective's set-up).
- **F2** (F2_05, seed 0), to ``F2_BUDGET_S``: seconds per evaluation between the second and the
  last log row (one row per LM step; the first carries the start-up).

Everything else as the Package 1 runs: ``component_a``'s thread set-up (``pin_torch`` at its import),
float64, ``test_every=1`` (test errors off the clock), the runs' own budgets and clocks.

Decided before the job (DECISIONS.md, 17 November 2026), as for the diagnostic p5o:
- **Reproduction:** a torch-first process reproduces Package 1's run if the loss on each of its log
  rows that Package 1's seed-0 run also has (F1: the Adam rows 0, 100, 200; F2: the LM steps)
  agrees within ``RTOL`` (C1's tolerance). If it does, the library's effect measured here is the
  effect on Package 1's runs. The numpy-first processes are compared in the same way and reported.
- **Effect:** the library changes the timing of a measure if the medians over the rounds of the two
  orders differ by more than ``CUTOFF`` (C4's and the pairing check's cutoff) and the two orders'
  rounds do not overlap.

Outputs under ``<out>/P5_3_cost/openmp_kovasznay/``: ``<first>-first/process_r<round>.json``,
``<first>-first/r<round>/{F1,F2}/`` (each run's ``log.csv`` and ``run.json``),
``timing_<first>-first_r<round>.csv``, ``provenance/``; ``summary`` (at the assembly, on the laptop:
Package 1's runs are not on Grace) writes ``summary.json`` there.

Usage::

    python experiments/p5_3_openmp_kovasznay.py run --first numpy --round 0 --out <root> [--require-library]
    python experiments/p5_3_openmp_kovasznay.py summary --out <root> --package1 <package1>/A_calibration
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

from experiments import p5_3_openmp as om   # noqa: E402  (imports nothing beyond the standard library)

ITEM = 'P5_3_cost'
SUB = 'openmp_kovasznay'
# Package 1's representatives (A_calibration/full/<family>_representative.json), as its full_cpu runs used them.
CONFIGS = {
    'F1': {'id': 'F1_18', 'family': 'F1', 'width': 128, 'depth': 4, 'trunk': 'separate', 'm': 64,
           'sigma_ff': 0.5, 'bc': 'soft', 'lambda_bc': 1.0, 'balancing': True, 'eta': 0.001, 't_adam': 5000,
           'n_int': 2000, 'resample': True, 'n_theta': 198531},
    'F2': {'id': 'F2_05', 'family': 'F2', 'width': 32, 'depth': 3, 'm': 32, 'sigma_ff': 0.5, 'n_int': 8000,
           'n_theta': 4291},
}
SEED = 0
F1_ADAM_ITERATIONS = 201       # Adam log rows at 0, 100 and 200 (a row every 100 iterations)
F1_BUDGET_S = 140.0            # about 65 s of Adam, the rest L-BFGS (Package 1: about 0.32 s per iteration each)
F2_BUDGET_S = 90.0             # about 7 LM steps (Package 1: about 11 s per step)
RTOL = 1e-6                    # C1's tolerance
CUTOFF = 0.05                  # C4's and the pairing check's cutoff
MEASURES = ('F1_adam_s_per_iteration', 'F1_lbfgs_s_per_iteration', 'F2_s_per_evaluation')
TIMING_COLUMNS = ['first', 'round', 'library', 'measure', 'value', 'intervals', 'end_reason']


def _rows(log_csv):
    with open(log_csv, newline='') as fh:
        return list(csv.DictReader(fh))


def f1_measures(log_csv):
    """Adam seconds per iteration (rows 100 to 200) and L-BFGS seconds per iteration (second to last
    L-BFGS row) of one F1 log; None where the log has too few rows."""
    rows = _rows(log_csv)
    adam = {int(r['iter']): float(r['t_cum_s']) for r in rows if r['phase'] == 'adam'}
    adam_s = (adam[200] - adam[100]) / 100 if 100 in adam and 200 in adam else None
    lb = [(int(r['iter']), float(r['t_cum_s'])) for r in rows if r['phase'] == 'lbfgs'][1:]
    lbfgs_s = (lb[-1][1] - lb[0][1]) / (lb[-1][0] - lb[0][0]) if len(lb) >= 2 and lb[-1][0] > lb[0][0] else None
    return {'F1_adam_s_per_iteration': (adam_s, 1 if adam_s is not None else 0),
            'F1_lbfgs_s_per_iteration': (lbfgs_s, max(len(lb) - 1, 0))}


def f2_measures(log_csv):
    """Seconds per evaluation of one F2 log, second to last row; None with fewer than two such rows."""
    rows = _rows(log_csv)[1:]
    if len(rows) < 2 or int(rows[-1]['n_evals']) <= int(rows[0]['n_evals']):
        return {'F2_s_per_evaluation': (None, 0)}
    dt = float(rows[-1]['t_cum_s']) - float(rows[0]['t_cum_s'])
    return {'F2_s_per_evaluation': (dt / (int(rows[-1]['n_evals']) - int(rows[0]['n_evals'])), len(rows) - 1)}


def run(args):
    root = Path(args.out) / ITEM / SUB
    d = root / f'{args.first}-first'
    proc = om.start_process(args.first, args.round, d, args.require_library)
    from lilq.provenance import job_dir, save_provenance
    save_provenance(job_dir(root / 'provenance' / f'{args.first}-first_r{args.round}'))
    from experiments import component_a as ca   # after the order is set; pin_torch() at its import, as in Package 1
    out = d / f'r{args.round}'
    f1_run = ca.f1.f1_train(CONFIGS['F1'], SEED, F1_BUDGET_S, str(out / 'F1'), device='cpu', test_every=1,
                            max_adam_iters=F1_ADAM_ITERATIONS)
    print(f"F1: {f1_run.get('adam_iters')} Adam + {f1_run.get('lbfgs_iters')} L-BFGS iterations, "
          f"end {f1_run.get('end_reason')}", flush=True)
    f2_run = ca.lm.lm_train(ca.lm_args(CONFIGS['F2'], SEED, F2_BUDGET_S, out / 'F2', 'cpu'))
    print(f"F2: {f2_run.get('steps')} steps, {f2_run.get('n_evals')} evaluations, end {f2_run.get('end_reason')}",
          flush=True)
    found = f1_measures(out / 'F1' / 'log.csv') | f2_measures(out / 'F2' / 'log.csv')
    ends = {'F1': f1_run.get('end_reason'), 'F2': f2_run.get('end_reason')}
    rows = [{'first': args.first, 'round': args.round, 'library': proc['library'], 'measure': m,
             'value': repr(found[m][0]) if found[m][0] is not None else '', 'intervals': found[m][1],
             'end_reason': ends[m[:2]]} for m in MEASURES]
    with open(root / f'timing_{args.first}-first_r{args.round}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=TIMING_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"timing {r['measure']}: {r['value']} s ({r['intervals']} intervals)", flush=True)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def reproduction(log_csv, reference_csv, family):
    """Each log row (F1: Adam rows; F2: every row) that the reference has too, by iteration: the largest
    relative difference of loss_total, the number of rows compared, and whether all are within RTOL."""
    ref = {(r['phase'], int(r['iter'])): float(r['loss_total']) for r in _rows(reference_csv)}
    pairs = [(float(r['loss_total']), ref[(r['phase'], int(r['iter']))]) for r in _rows(log_csv)
             if (r['phase'], int(r['iter'])) in ref and (family == 'F2' or r['phase'] == 'adam')]
    worst = max((_rel(a, b) for a, b in pairs), default=None)
    return {'rows': len(pairs), 'max_rel_diff': worst, 'within_rtol': bool(pairs) and worst <= RTOL}


def summary(args):
    root = Path(args.out) / ITEM / SUB
    a = Path(args.package1)
    refs = {f: a / 'full_cpu' / f"{CONFIGS[f]['id']}_s{SEED}" / 'log.csv' for f in CONFIGS}
    values = {m: {'numpy': [], 'torch': []} for m in MEASURES}
    libraries, repro = {}, {}
    for t in sorted(root.glob('timing_*-first_r*.csv')):
        for r in _rows(t):
            libraries.setdefault(r['first'], set()).add(r['library'])
            if r['value']:
                values[r['measure']][r['first']].append(float(r['value']))
    for first in ('numpy', 'torch'):
        for rd in sorted((root / f'{first}-first').glob('r*')):
            for f in CONFIGS:
                if (rd / f / 'log.csv').exists():
                    repro[f'{first}-first/{rd.name}/{f}'] = reproduction(rd / f / 'log.csv', refs[f], f)
    effect = {}
    for m, v in values.items():
        if not (v['numpy'] and v['torch']):
            effect[m] = {'numpy': v['numpy'], 'torch': v['torch'], 'verdict': 'not measured'}
            continue
        mn, mt = statistics.median(v['numpy']), statistics.median(v['torch'])
        overlap = max(v['numpy']) >= min(v['torch']) and max(v['torch']) >= min(v['numpy'])
        changed = abs(mt / mn - 1) > CUTOFF and not overlap
        effect[m] = {'numpy': v['numpy'], 'torch': v['torch'], 'median_numpy': mn, 'median_torch': mt,
                     'torch_over_numpy': mt / mn, 'rounds_overlap': overlap,
                     'verdict': 'the library changes it' if changed else 'no change beyond the cutoff'}
    torch_runs = [r for k, r in repro.items() if k.startswith('torch-')]
    # Package 1's own CPU runs, the same measures over their whole logs (seeds 0-4): the diagnostic's
    # torch-first values should be theirs (same library, same node type).
    p1 = {m: [] for m in MEASURES}
    for f in CONFIGS:
        for log in sorted((a / 'full_cpu').glob(f"{CONFIGS[f]['id']}_s*/log.csv")):
            for m, (v, _) in (f1_measures(log) if f == 'F1' else f2_measures(log)).items():
                if v is not None:
                    p1[m].append(v)
    package1 = {m: {'values': v, 'median': statistics.median(v) if v else None} for m, v in p1.items()}
    out = {'libraries': {k: sorted(v) for k, v in libraries.items()},
           'torch_first_reproduces_package1': bool(torch_runs) and all(r['within_rtol'] for r in torch_runs),
           'reproduction': repro, 'effect': effect, 'package1_full_cpu': package1, 'rtol': RTOL, 'cutoff': CUTOFF}
    (root / 'summary.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def _parse(argv):
    ap = argparse.ArgumentParser(description="Package 5, item 3: the OpenMP library of Component A's CPU baselines")
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--first', required=True, choices=('numpy', 'torch'), help='the package this process imports first')
    r.add_argument('--round', type=int, required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--require-library', action='store_true',
                   help='stop unless the OpenMP library loaded is the one --first should give (Grace)')
    s = sub.add_parser('summary')
    s.add_argument('--out', required=True)
    s.add_argument('--package1', required=True, help="Package 1's A_calibration/ directory")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    return run(args) if args.cmd == 'run' else summary(args)


if __name__ == '__main__':
    main()
