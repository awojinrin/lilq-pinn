"""
Package 5, item 3: why C4 failed for Buckley-Leverett LM in the CPU rerun (a diagnostic; the user's decision, 17 November 2026)
================================================================================================================================

Check C4 times a run of fixed length without and with the error log; the log is off the clock and
must not change the clock by more than 5%. In the CPU rerun (stage p5r, the GCC module's OpenMP
library) Buckley-Leverett LM at P = 1,024 took 28.17 s without the log and 33.28 s with it
(+18.1%, 213 iterations); in the CPU set (the PyTorch wheel's library) the same check passed
(+2.5%, 128 iterations). The A100 set's LM runs log at every iteration, so a leak there would put
error-log time on their clocks. The A100 set waits for this (the user's decision, 17 November 2026).

Each process times the pair (without the log, then with it) ``REPEATS`` times at the same length,
after the warm-up the rerun's checks process ran (NiL-N and LM). Process kinds:

- ``numpy``: numpy imported first, so the GCC module's OpenMP library (as the rerun).
- ``torch``: torch imported first, so the PyTorch wheel's (as the CPU set).
- ``active``: as ``numpy``, with ``OMP_WAIT_POLICY=ACTIVE`` set by the job: the OpenMP threads spin
  between parallel regions instead of going to sleep. Every other process must run without it.

On the CPU each pair has the rerun's length, ``RERUN_C4_ITERATIONS``; on the A100 the length comes
from C4's own pilot (about 30 s), as the rerun sized it on its node.

Decided before the job (DECISIONS.md, 17 November 2026). For a group of processes, the median over
its pairs of the signed relative difference (with the log against without):
- the log leaks if the median is above 5% (C4's cutoff); it is clean if the median is within 5%.
- cpu numpy leaks and cpu torch is clean: the leak depends on the library.
- both leak: the log leaks with either library.
- cpu numpy is clean: the rerun's +18% was not reproduced.
- cpu numpy leaks and cpu active is clean: the threads going to sleep are the cause.
- the A100 group (numpy, two processes side by side as the A100 set) leaks: stop and bring a fix
  before the A100 set; clean: the A100 set may go. The gate is that group alone (both processes'
  pairs, each with the module's library and no wait policy); C2 and the CPU groups are reported
  beside it.

Outputs under ``<out>/P5_3_cost/c4_diagnostic/``: ``<device>_<kind>_r<round>/`` (``process_r<round>.json``,
``environment.json``, ``result.json`` from ``p5_3_networks.c2_c4``, the logs), ``provenance/``,
``summary.json``.

Usage::

    python experiments/p5_3_c4_diagnostic.py run --kind numpy --round 0 --device cpu --out <root> \\
        --reference-dir <dir> [--require-library]
    python experiments/p5_3_c4_diagnostic.py summary --out <root>
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

BENCH = 'bl'
P = 1024
METHOD = 'lm'
REPEATS = 3
RERUN_C4_ITERATIONS = 213          # the rerun's C4 length at this size (its pilot on that node)
WARMUP_METHODS = ('nil_n', 'lm')   # as the rerun's checks process (both methods it checked, in its order)
RTOL = 0.05                        # C4's cutoff (p5_3_networks.C4_RTOL; checked in run)
KINDS = {'numpy': ('numpy', None), 'torch': ('torch', None), 'active': ('numpy', 'ACTIVE')}  # kind: (first, OMP_WAIT_POLICY)
EXPECTED_LIBRARY = {'numpy': 'module', 'torch': 'torch', 'active': 'module'}
ENV_KEYS = ('OMP_WAIT_POLICY', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'GOMP_SPINCOUNT',
            'CUDA_VISIBLE_DEVICES', 'SLURM_JOB_ID', 'SLURM_CPUS_PER_TASK')


def root_of(out):
    return Path(out) / 'P5_3_cost' / 'c4_diagnostic'      # p5_3_networks.ITEM: not imported before the order is set


def label_of(device, kind, round_):
    return f'{device}_{kind}_r{round_}'


def check_wait_policy(kind, environ=os.environ):
    """The wait policy this kind needs, before anything is imported: 'active' needs
    OMP_WAIT_POLICY=ACTIVE, every other kind needs it unset (the library's default)."""
    want = KINDS[kind][1]
    got = environ.get('OMP_WAIT_POLICY')
    if got != want:
        raise SystemExit(f'--kind {kind} needs OMP_WAIT_POLICY={want or "(unset)"}, not {got or "(unset)"}')


def run(args):
    check_wait_policy(args.kind)
    first = KINDS[args.kind][0]
    root = root_of(args.out)
    label = label_of(args.device, args.kind, args.round)
    d = root / label
    (d / 'result.json').unlink(missing_ok=True)     # a resubmitted job: no earlier attempt's result beside this one's records
    from experiments.p5_3_openmp import start_process
    proc = start_process(first, args.round, d, args.require_library)
    from experiments import p5_3_networks as n3
    from lilq.provenance import job_dir, save_provenance
    assert root.parent.name == n3.ITEM and n3.C4_RTOL == RTOL
    (d / 'environment.json').write_text(json.dumps(
        {'kind': args.kind, 'device': args.device, 'round': args.round, 'library': proc['library'],
         'environment': {k: os.environ.get(k) for k in ENV_KEYS}}, indent=2))
    save_provenance(job_dir(root / 'provenance' / label))
    n3.warm_up(BENCH, args.device, args.reference_dir, WARMUP_METHODS, scratch=root, tag=label)
    iterations = RERUN_C4_ITERATIONS if args.device == 'cpu' else None
    res = n3.c2_c4(BENCH, P, METHOD, args.reference_dir, args.out, args.device, iterations=iterations,
                   repeats=REPEATS, d=d)
    signed = [t['rel_diff_with_vs_without'] for t in res['timings']]
    print(f'{label}: {res["iterations"]} iterations, with the log against without: '
          + ', '.join(f'{s:+.1%}' for s in signed) + f'; C2 {"passed" if res["c2_passed"] else "FAILED"}', flush=True)


# ---------------------------------------------------------------- the summary (no torch needed)

def verdict(median):
    if median is None:
        return 'not measured'
    if median > RTOL:
        return 'leaks'
    if median < -RTOL:
        return 'faster with the log'
    return 'clean'


def _process(d):
    """One process's record: kind, device, library, wait policy, its pairs, C2; or None if it left no result."""
    env_f, res_f = d / 'environment.json', d / 'result.json'
    if not (env_f.exists() and res_f.exists()):
        return None
    env, res = json.loads(env_f.read_text()), json.loads(res_f.read_text())
    return {'process': d.name, 'kind': env['kind'], 'device': env['device'], 'round': env['round'],
            'library': env['library'], 'expected_library': EXPECTED_LIBRARY[env['kind']],
            'omp_wait_policy': env['environment'].get('OMP_WAIT_POLICY'),
            'omp_num_threads': env['environment'].get('OMP_NUM_THREADS'),
            'iterations': res['iterations'], 'c2_passed': res['c2_passed'],
            'pairs': [[t['time_without_s'], t['time_with_s'], t['rel_diff_with_vs_without']]
                      for t in res.get('timings', [])],
            'error_time_s': res['error_time_s']}


def summary(out):
    root = root_of(out)
    started = sorted(p for p in root.glob('*_r*') if p.is_dir() and p.name.split('_')[0] in ('cpu', 'cuda'))
    procs = [_process(p) for p in started]
    missing = [p.name for p, r in zip(started, procs) if r is None]
    procs = [r for r in procs if r is not None]
    groups = {}
    for name, device, kind in (('cpu numpy (module)', 'cpu', 'numpy'), ('cpu torch (wheel)', 'cpu', 'torch'),
                               ('cpu active (module, OMP_WAIT_POLICY=ACTIVE)', 'cpu', 'active'),
                               ('A100 numpy (module, two processes side by side)', 'cuda', 'numpy')):
        mine = [r for r in procs if r['device'] == device and r['kind'] == kind]
        diffs = [p[2] for r in mine for p in r['pairs']]
        med = statistics.median(diffs) if diffs else None
        groups[name] = {'processes': len(mine), 'pairs': len(diffs), 'median': med,
                        'min': min(diffs) if diffs else None, 'max': max(diffs) if diffs else None,
                        'pairs_outside_rtol': sum(abs(x) > RTOL for x in diffs), 'verdict': verdict(med),
                        'iterations': sorted({r['iterations'] for r in mine}),
                        'libraries_as_expected': all(r['library'] == r['expected_library'] for r in mine),
                        'wait_policy_as_expected': all(r['omp_wait_policy'] == KINDS[kind][1] for r in mine),
                        'c2_passed': all(r['c2_passed'] for r in mine)}
    v = {k.split(' (')[0]: g['verdict'] for k, g in groups.items()}
    mod, whl, act, gpu = v['cpu numpy'], v['cpu torch'], v['cpu active'], v['A100 numpy']
    findings = []
    if mod == 'clean':
        findings.append("cpu numpy is clean: the rerun's +18% was not reproduced")
    elif mod == 'leaks':
        if whl == 'clean':
            findings.append('the leak depends on the library (module leaks, wheel clean)')
        elif whl == 'leaks':
            findings.append('the log leaks with either library')
        if act == 'clean':
            findings.append('ACTIVE removes it: the threads going to sleep are the cause')
        elif act == 'leaks':
            findings.append('ACTIVE does not remove it: not the threads going to sleep')
    if any(x == 'faster with the log' for x in v.values()):
        findings.append('a group is more than 5% faster with the log: not foreseen, to be explained')
    records_ok = all(g['libraries_as_expected'] and g['wait_policy_as_expected'] for g in groups.values())
    c2_ok = all(g['c2_passed'] for g in groups.values())
    # The A100 set's gate is the A100 group alone: both processes' three pairs, the module's library, no
    # wait policy, and clean. C2 and the CPU groups are reported beside it, not part of it.
    a = groups['A100 numpy (module, two processes side by side)']
    a100_may_go = (gpu == 'clean' and a['pairs'] == 2 * REPEATS and a['libraries_as_expected']
                   and a['wait_policy_as_expected'] and not any(m.startswith('cuda_') for m in missing))
    res = {'check': 'C4 diagnostic: Buckley-Leverett LM, P = 1024, the pair timed three times per process',
           'rtol': RTOL, 'groups': groups, 'findings': findings, 'records_as_expected': records_ok,
           'c2_passed': c2_ok, 'processes_without_result': missing, 'a100_set_may_go': a100_may_go,
           'processes': procs}
    root.mkdir(parents=True, exist_ok=True)
    (root / 'summary.json').write_text(json.dumps(res, indent=2))
    return res


def print_summary(res):
    for name, g in res['groups'].items():
        if g['pairs']:
            print(f"{name}: {g['processes']} processes, {g['pairs']} pairs, median {g['median']:+.1%} "
                  f"(from {g['min']:+.1%} to {g['max']:+.1%}), {g['pairs_outside_rtol']} outside 5%: {g['verdict']}")
        else:
            print(f'{name}: not measured')
    for f in res['findings']:
        print('finding:', f)
    if res['processes_without_result']:
        print('PROCESSES WITHOUT A RESULT:', ', '.join(res['processes_without_result']))
    if not res['records_as_expected']:
        print('RECORDS NOT AS EXPECTED (an OpenMP library or wait policy): see summary.json')
    if not res['c2_passed']:
        print('C2 FAILED IN A PROCESS (the log changed the iterates, or the runs are not repeatable): see summary.json')
    print('A100 SET MAY GO:', 'yes' if res['a100_set_may_go'] else 'NO', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Package 5, item 3: the C4 diagnostic (Buckley-Leverett LM)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--kind', required=True, choices=tuple(KINDS))
    r.add_argument('--round', type=int, required=True)
    r.add_argument('--device', required=True, choices=('cpu', 'cuda'))
    r.add_argument('--out', required=True)
    r.add_argument('--reference-dir', required=True)
    r.add_argument('--require-library', action='store_true',
                   help="stop unless the OpenMP library loaded is the one --kind should give (Grace)")
    s = sub.add_parser('summary')
    s.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'run':
        run(args)
    else:
        print_summary(summary(args.out))


if __name__ == '__main__':
    main()
