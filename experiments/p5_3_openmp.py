"""
Package 5, item 3: the OpenMP library of a network process (a diagnostic; the user's decision, 17 November 2026)
==============================================================================================================

The CPU set's network processes (``experiments/p5_3_networks.py``, which imports torch before
numpy) loaded the PyTorch wheel's own OpenMP library (``torch/lib/libgomp.so.1``). Package 1's
four-method runs (``experiments/four_method_tables.py``, numpy first) and every Grace job since
then loaded the GCC module's (``GCCcore/13.3.0/lib64/libgomp.so.1.0.0``). Check C1 (the loss over
iterations 0 to 100 against Package 1, within 1e-6) failed on 15 of the 18 runs it compares: the
losses agree to round-off at iteration 1 and drift apart from there. This driver runs the same
network runs with either library, chosen by which package a fresh process imports first:

- **C1 runs** (``--round 0`` only): NiL-N, seed 0, at the largest size of each problem,
  ``C1_RUN_ITERATIONS`` iterations, no budget, no error log (C2: the log does not change the
  iterates). Each writes ``run.json`` and ``metrics.csv`` as a budget run does, so that
  ``p5_3_networks.c1`` compares it with Package 1 at the assembly (Package 1 is not on Grace),
  and the torch-first runs also with the CPU set's own runs.
- **Timing runs** (every round): NiL-N and LM, seed 0, at the largest size of Bratu and
  Buckley-Leverett, each to a budget of ``TIMING_BUDGET_S`` without the error log: the time per
  iteration under each library. The job alternates the two orders over its rounds.

Decided before the job (DECISIONS.md, 17 November 2026):
- The library is the cause if the numpy-first C1 runs pass C1 against Package 1 and the
  torch-first ones reproduce the CPU set's runs.
- The library changes the timing of a method and problem if the medians over the rounds of the
  time per iteration differ by more than 5% (C4's and the pairing check's cutoff) and the two
  orders' rounds do not overlap.

Each process records the OpenMP libraries it loaded; with ``--require-library`` (the Grace job) it
stops unless that is the one its ``--first`` should give.

Outputs under ``<out>/P5_3_cost/openmp/``: ``<first>-first/c1/<bench>_P<P>_nil_n_s0_cpu/``,
``<first>-first/process_r<round>.json``, ``timing_<first>-first_r<round>.csv``, ``provenance/``.

Usage::

    python experiments/p5_3_openmp.py --first numpy --round 0 --out <root> --reference-dir <dir> \\
        [--budget 30] [--require-library]
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

C1_BENCHES = ('bratu', 'burgers', 'bl')
TIMING_BENCHES = ('bratu', 'bl')
TIMING_METHODS = ('nil_n', 'lm')
TIMING_BUDGET_S = 30.0
C1_RUN_ITERATIONS = 110            # C1 reads iterations 0..100
TIMING_COLUMNS = ['first', 'round', 'library', 'benchmark', 'P', 'method', 'iterations', 'training_time_s',
                  's_per_iteration', 'stopping_reason']


def _parse(argv):
    ap = argparse.ArgumentParser(description='Package 5, item 3: the OpenMP library of a network process')
    ap.add_argument('--first', required=True, choices=('numpy', 'torch'), help='the package this process imports first')
    ap.add_argument('--round', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference-dir', required=True)
    ap.add_argument('--budget', type=float, default=TIMING_BUDGET_S)
    ap.add_argument('--require-library', action='store_true',
                    help='stop unless the OpenMP library loaded is the one --first should give (Grace)')
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    import lilq.blas_threads  # noqa: F401  (must import before numpy)
    if args.first == 'numpy':
        import numpy  # noqa: F401  (first: its BLAS brings the module's OpenMP library)
    import torch  # noqa: F401
    import numpy  # noqa: F401,F811
    from experiments import p5_3_networks as n3
    from lilq.provenance import job_dir, openmp_libraries, openmp_library_kind, save_provenance
    from lilq.source_lock import current_commit

    libs = openmp_libraries()
    kind = openmp_library_kind(libs)
    want = 'torch' if args.first == 'torch' else 'module'
    root = Path(args.out) / n3.ITEM / 'openmp'
    d = root / f'{args.first}-first'
    d.mkdir(parents=True, exist_ok=True)
    proc = {'first': args.first, 'round': args.round, 'openmp_libraries': libs, 'library': kind,
            'expected_library': want, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'commit': current_commit()}
    (d / f'process_r{args.round}.json').write_text(json.dumps(proc, indent=2))
    print(json.dumps(proc), flush=True)
    if args.require_library and kind != want:
        raise SystemExit(f'--first {args.first} loaded the {kind} OpenMP library ({libs}), not the {want} one')
    save_provenance(job_dir(root / 'provenance' / f'{args.first}-first_r{args.round}'))

    ref = args.reference_dir
    tag = f'{args.first}{args.round}'
    for bench in sorted(set(C1_BENCHES) | set(TIMING_BENCHES)):
        n3.warm_up(bench, 'cpu', ref, TIMING_METHODS, scratch=root, tag=tag)
    if args.round == 0:
        for bench in C1_BENCHES:
            P = n3.sizes(bench)[-1]
            _, rec, _, metrics = n3.solve(bench, P, 'nil_n', 0, ref, 'cpu', budget_s=None,
                                          max_iterations=C1_RUN_ITERATIONS, log_error=False)
            c = d / 'c1' / f'{bench}_P{P}_nil_n_s0_cpu'
            c.mkdir(parents=True, exist_ok=True)
            metrics.to_dataframe().to_csv(c / 'metrics.csv', index=False, float_format='%.17g')
            (c / 'run.json').write_text(json.dumps(rec | proc, indent=2, default=str))
            print(f'C1 run {bench} P={P}: {rec["iterations"]} iterations, loss {rec["final_loss"]:.6e}', flush=True)
    rows = []
    logs = d / 'logs'
    logs.mkdir(exist_ok=True)
    for bench in TIMING_BENCHES:
        P = n3.sizes(bench)[-1]
        for method in TIMING_METHODS:
            _, rec, _, _ = n3.solve(bench, P, method, 0, ref, 'cpu', budget_s=args.budget, log_error=False,
                                    log_path=logs / f'{bench}_{method}_r{args.round}.csv')
            it = max(rec['iterations'], 1)
            rows.append({'first': args.first, 'round': args.round, 'library': kind, 'benchmark': bench, 'P': P,
                         'method': method, 'iterations': rec['iterations'],
                         'training_time_s': repr(rec['training_time_s']),
                         's_per_iteration': repr(rec['training_time_s'] / it),
                         'stopping_reason': rec['stopping_reason']})
            print(f'timing {bench} {method}: {rec["iterations"]} iterations in {rec["training_time_s"]:.2f} s', flush=True)
    with open(root / f'timing_{args.first}-first_r{args.round}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=TIMING_COLUMNS)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
