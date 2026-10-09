"""
Package 3: the results folder, from Grace's tarball
====================================================

Assembles ``package3_results/`` in the layout of the advisor's instructions of
8 October 2026 (Section 9) and Addendum 1 (Section 6), from the extracted
``results/package3.tar.gz`` of the last report job:

- **The run folders and their summaries,** as Grace wrote them, at the runs'
  commit (``COMMIT``): ``P3_1_beltrami_certified/`` to ``P3_4_burgers_large_P/``,
  ``P3_checks/`` (K0, K1, K2, K8), ``checks_item2.json``, ``slurm_logs/``,
  ``sacct.txt``, ``su_per_job.csv``, ``report_log.txt``.
- **Check L1 re-evaluated at this commit,** on Grace's files: it reads only the
  logged CSVs, and its comparison of P2-12's terminal row was corrected after
  wave 1 (DECISIONS.md, "L1 compares a terminal row like with like").
  ``checks_item4.json`` keeps Grace's verdict beside the new one.
- **The records of Section 9:**
  - ``checks.json``: K0-K8 and L1-L4 in one file, each with its source;
  - ``su.csv``: every Package 3 job at Grace's rates (``wave_report.su_rows``),
    with the last report's own row, which its tarball cannot hold
    (``--sacct-extra``: ``sacct -X -P -n`` lines);
  - ``environment.txt`` and ``hardware.json``: each job's node, threads and
    allocation, from its log and ``sacct.txt``. The run jobs did not call
    ``save_provenance``; ``--provenance`` adds the record a short job wrote after the
    runs (``p3_provenance.slurm``: the runs' class, environment and source lock), the
    CPU from it, and its files in ``provenance_cpu/``;
  - ``DECISIONS_package3.md``: the Package 3 entries of ``DECISIONS.md``;
  - ``provenance.json``: the runs' commit and tree hash, this commit, and the
    commits between them.

Refuses a Grace tree whose commit is not an ancestor of this checkout's HEAD.
Can be run again: it rewrites what it writes.

Usage::

    python experiments/p3_assemble.py --grace <extracted>/package3 --out ../package3_results \\
        --p2-12 <package2_stage1>/P2_12_reference_errors --sacct-extra <file>
"""

import argparse
import csv
import datetime
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

GRACE_ITEMS = ('P3_1_beltrami_certified', 'P3_2a_bl_chebyshev_certified', 'P3_2b_bl_lifted_sine_cheb',
               'P3_3_affine_certificates', 'P3_4_burgers_large_P', 'P3_checks', 'slurm_logs')
GRACE_FILES = ('COMMIT', 'checks_item2.json', 'sacct.txt', 'su_per_job.csv', 'report_log.txt')
LOG_HEAD = re.compile(r'^job (\d+) on (\S+): (\d+) threads, (\d+) cores usable, OverSubscribe=(\S+)')
P2_HARDWARE = ('Package 2, Stage 2, job 20017335 (lilq-p2s2-scaling-cpu-a) on c528, partition medium: '
               'Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz, 48 cores, RealMemory 368640 MB '
               '(package2_results/hardware.json)')


def _git(*args):
    out = subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, text):
    Path(path).write_bytes(text.encode())                 # LF on every platform


def copy_grace(grace, out, log):
    for name in GRACE_ITEMS:
        if (out / name).exists():
            shutil.rmtree(out / name)
        shutil.copytree(grace / name, out / name)
        log.append(f'copied {name}/')
    for name in GRACE_FILES:
        shutil.copy2(grace / name, out / name)
        log.append(f'copied {name}')


def recheck_l1(out, p2_12, log):
    """L1 at this commit, on Grace's files; Grace's own verdict is kept beside it."""
    import experiments.p3_4_burgers_large_P as item4
    path = out / 'P3_4_burgers_large_P' / 'checks_item4.json'
    checks = _json(path)
    grace_l1 = checks['L1'].get('grace', checks['L1'])     # a second assembly keeps the first's record
    p2 = Path(p2_12)
    ref = p2 / 'B_instrumentation' / 'burgers_P625_cpu_paper' if (p2 / 'B_instrumentation').exists() else p2 / 'burgers_P625'
    new = item4.check_l1(out / 'P3_4_burgers_large_P' / 'burgers_P625', ref)
    checks['L1'] = {**new, 'reevaluated_at': _git('rev-parse', 'HEAD'),
                    'note': "re-evaluated at the assembly on Grace's files: on P2-12's terminal row (k = 4), "
                            "norm_R_h is compared with sqrt(the control's loss), as that row computes it "
                            '(DECISIONS.md, 2026-10-09)', 'grace': grace_l1}
    _write(path, json.dumps(checks, indent=2) + '\n')
    log.append(f"L1 re-evaluated: {'passed' if new['passed'] else 'FAILED'} (Grace's run: "
               f"{'passed' if grace_l1['passed'] else 'failed'})")
    return checks


def checks_json(out, item4):
    """K0-K8 and L1-L4, each with its verdict, key numbers and source file."""
    c = out / 'P3_checks'
    k0, k1, k2, k8 = (_json(c / f'k{i}.json') for i in (0, 1, 2, 8))
    i1 = _json(out / 'P3_1_beltrami_certified' / 'checks_item1.json')
    i2 = _json(out / 'checks_item2.json')
    i3 = _json(out / 'P3_3_affine_certificates' / 'checks_item3.json')
    k5_bl = next(v for k, v in i2.items() if k.startswith('K5'))
    k6_bl = next(v for k, v in i2.items() if k.startswith('K6'))
    resolves = i3['resolves']
    return {
        'K0': {'passed': k0['passed'], 'source': 'P3_checks/k0.json', 'shape': k0['shape'],
               'gelsy': {q: k0['gelsy'][q] for q in ('rel_residual', 'rel_error')},
               'pivoted_qr': {q: k0['pivoted_qr'][q] for q in ('rel_residual', 'rel_error')}},
        'K1': {'passed': k1['passed'], 'source': 'P3_checks/k1.json'},
        'K2': {'passed': k2['passed'], 'source': 'P3_checks/k2.json', 'rules': k2['n_rules'],
               'worst_error': k2['worst_error']},
        'K3': {'passed': i1['K3']['passed'], 'source': 'P3_1_beltrami_certified/checks_item1.json'},
        'K4': {'passed': i1['K4']['passed'] and i2['K4 (cheb)']['passed'] and i2['K4 (sine)']['passed'],
               'source': 'checks_item1.json (item 1), checks_item2.json (item 2)',
               'max_rel_err_at_k1': max(i1['K4']['max_rel_err_at_k1'], i2['K4 (cheb)']['max_rel_err_at_k1'],
                                        i2['K4 (sine)']['max_rel_err_at_k1']), 'rule': i1['K4']['rule']},
        'K5': {'passed': k5_bl['passed'] and i3['K5']['passed'],
               'source': 'checks_item2.json (BL), P3_3_affine_certificates/checks_item3.json (elasticity, Darcy)'},
        'K6': {'passed': i1['K6']['passed'] and k6_bl['passed'],
               'source': 'checks_item1.json (B1 level 1), checks_item2.json (BL viscous P = 256)'},
        'K7': {'passed': i1['K7']['passed'] and i2['K7 (cheb)']['passed'] and i2['K7 (sine)']['passed'],
               'source': 'checks_item1.json, checks_item2.json'},
        'K8': {'passed': k8['passed'], 'source': 'P3_checks/k8.json',
               'runs': {n: {'passed': r['passed'], 'rows': r['rows_here']} for n, r in k8['runs'].items()}},
        'item 3 re-solves (Section 5)': {'passed': all(v['identical'] is True for v in resolves.values()),
                                         'identical': sum(v['identical'] is True for v in resolves.values()),
                                         'of': len(resolves), 'assembly_bitwise': i3['assembly']['all_bitwise'],
                                         'source': 'P3_3_affine_certificates/checks_item3.json'},
        **{f'{k} (Addendum 1)': {'passed': v['passed'], 'source': 'P3_4_burgers_large_P/checks_item4.json'}
           for k, v in item4.items()},
    }


def su_rows(out, sacct_extra):
    wave_report = _module('wave_report', REPO / 'scripts' / 'cluster' / 'wave_report.py')
    with open(out / 'su_per_job.csv') as fh:
        rows = list(csv.DictReader(fh))
    have = {r['job_id'] for r in rows}
    if sacct_extra:
        for r in wave_report.su_rows(Path(sacct_extra).read_text()):
            if r['job_id'] not in have:
                rows.append({k: r[k] for k in wave_report.SU_COLUMNS}
                            | {'note': 'not in the last tarball: from sacct after it ended'})
    total = sum(float(r['su'] or 0) for r in rows)
    return rows, total, wave_report.SU_COLUMNS


def copy_provenance(prov, out, log):
    """The record ``save_provenance`` wrote in a short job after the runs (``p3_provenance.slurm``):
    ``hardware.json`` and ``environment.txt`` to ``provenance_cpu/``, its log to ``slurm_logs/``."""
    prov, dst = Path(prov), out / 'provenance_cpu'
    dst.mkdir(exist_ok=True)
    for name in ('hardware.json', 'environment.txt'):
        shutil.copy2(prov / name, dst / name)
    for p in prov.glob('*.out'):
        shutil.copy2(p, out / 'slurm_logs' / p.name)
    for p in prov.glob('*.slurm'):                       # the job script, run from outside the bundle
        shutil.copy2(p, dst / p.name)
    captured = _json(dst / 'hardware.json')
    log.append(f"copied provenance_cpu/ (job {captured['scheduler']['SLURM_JOB_ID']} on {captured['cpu']['hostname']}, "
               f"commit {captured['git']['commit'][:7]})")
    return captured


def hardware(out, captured=None, extra_rows=()):
    """Per job: the node, the threads and cores env.sh reported, and the allocation. The CPU
    from ``captured`` (``copy_provenance``) when given; otherwise Package 2's record is cited."""
    sacct = {}
    for line in (out / 'sacct.txt').read_text().splitlines()[2:]:
        parts = line.split()
        if parts and parts[0].isdigit():
            sacct[parts[0]] = {'name': parts[1], 'partition': parts[2], 'alloc_tres': parts[3], 'elapsed': parts[4]}
    for r in extra_rows:
        sacct.setdefault(r['job_id'], {'name': r['job_name'], 'alloc_tres': r['alloc_tres'],
                                       'elapsed_h': r['elapsed_h']})
    jobs = {}
    for p in sorted((out / 'slurm_logs').glob('*.out')):
        for line in p.read_text(errors='replace').splitlines()[:5]:
            m = LOG_HEAD.match(line)
            if m:
                job, node, threads, cores, oversub = m.groups()
                jobs[job] = {'log': p.name, 'node': node, 'threads': int(threads), 'cores_usable': int(cores),
                             'oversubscribe': oversub, **sacct.get(job, {})}
                break
    if captured is None:
        return {'cluster': 'Grace (TAMU HPRC)', 'jobs': jobs, 'cpu_model': None,
                'cpu_model_note': ('not recorded by the Package 3 jobs (they did not call save_provenance). Every job '
                                   "ran in partition medium, on a whole node for the timed-cpu class; Package 2's "
                                   f'record of a node of the same partition: {P2_HARDWARE}'),
                'gpu': None}
    s = captured['scheduler']
    return {'cluster': 'Grace (TAMU HPRC)', 'cpu': captured['cpu'],
            'cpu_source': (f"save_provenance in job {s['SLURM_JOB_ID']} ({s['SLURM_JOB_NAME']}) on "
                           f"{captured['cpu']['hostname']}, after the runs: the runs' class (timed-cpu, a whole node, "
                           f"partition {s['SLURM_JOB_PARTITION']}), their environment and their source lock (commit "
                           f"{captured['git']['commit'][:7]}). Its full record: provenance_cpu/hardware.json"),
            'jobs': jobs, 'gpu': None}


def environment(out, runs_commit, captured=False):
    pre = next((out / 'slurm_logs').glob('lilq-p3-preflight.*.out')).read_text(errors='replace').splitlines()
    keep = [l for l in pre if l.startswith(('job ', 'torch ', 'numpy ', 'provenance lock', 'inputs OK', 'K1 ', 'K2 ',
                                            'PREFLIGHT')) or re.match(r'\d+ passed', l)]
    env_sh = (REPO / 'scripts' / 'cluster' / 'env.sh').read_text()
    modules = re.search(r'LILQ_MODULES="\$\{LILQ_MODULES:-([^}]+)\}"', env_sh).group(1)
    return '\n'.join([
        "Package 3's software environment, from its jobs' own logs (the preflight's).",
        f'Code: commit {runs_commit} (the bundle, under the source lock of every job).',
        f'Modules (scripts/cluster/env.sh): {modules}; the venv $SCRATCH/lilq-run/venv first on PYTHONPATH.',
        'BLAS and LAPACK: FlexiBLAS 3.4.4 (GCC 13.3.0), through the modules\' numpy and scipy.',
        'Threads: OMP, OpenBLAS and MKL set to the cores the job holds (48 on a whole node, 24 for the cpu class);',
        '  the runs that set their own count (item 2: 4 per run, six side by side; item 3: as the original run)',
        '  record it in their run.json.',
        '',
        'The preflight log (slurm_logs/lilq-p3-preflight.*.out):',
        *('  ' + l for l in keep),
        *(['', "The full record (numpy's and scipy's build configuration, the BLAS thread pools, the packages),",
           'written by save_provenance in a whole-node job after the runs, in their environment and under',
           'their source lock: provenance_cpu/environment.txt.'] if captured else []),
        ''])


def decisions():
    text = (REPO / 'DECISIONS.md').read_text()
    parts = re.split(r'(?m)^(?=## \d{4}-\d{2}-\d{2} -- )', text)
    mine = [p.rstrip().rstrip('-').rstrip() for p in parts if re.match(r'## \d{4}-\d{2}-\d{2} -- Package 3', p)]
    return ('# DECISIONS.md: the Package 3 entries\n\nCopied from the repository\'s `DECISIONS.md` at the '
            f"assembly's commit ({_git('rev-parse', '--short', 'HEAD')}), newest first.\n\n---\n\n"
            + '\n\n---\n\n'.join(mine) + '\n')


def assemble(grace, out, p2_12, sacct_extra=None, provenance=None):
    grace, out = Path(grace), Path(out)
    runs = _json(grace / 'COMMIT')
    head = _git('rev-parse', 'HEAD')
    if head is None:
        sys.exit('the assembly needs a git checkout (it records the commits)')
    if subprocess.run(['git', 'merge-base', '--is-ancestor', runs['commit'], head], cwd=REPO).returncode != 0:
        sys.exit(f"the runs' commit {runs['commit'][:7]} is not an ancestor of HEAD {head[:7]}")
    out.mkdir(parents=True, exist_ok=True)
    log = [f'Package 3 assembly, {datetime.datetime.now(datetime.timezone.utc).isoformat()}',
           f'from {grace.name} (runs at {runs["commit"][:7]}), assembled at {head[:7]}']
    copy_grace(grace, out, log)
    captured = copy_provenance(provenance, out, log) if provenance else None
    item4 = recheck_l1(out, p2_12, log)
    checks = checks_json(out, item4)
    _write(out / 'checks.json', json.dumps(checks, indent=2) + '\n')
    log.append('checks.json: ' + ', '.join(f"{k} {'passed' if v['passed'] else 'FAILED'}" for k, v in checks.items()))
    rows, total, cols = su_rows(out, sacct_extra)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator='\n', extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
    _write(out / 'su.csv', buf.getvalue())
    log.append(f'su.csv: {len(rows)} jobs, {total:.1f} SU charged at Grace\'s rates')
    have = {r['job_id'] for r in csv.DictReader(io.StringIO((out / 'su_per_job.csv').read_text()))}
    _write(out / 'hardware.json', json.dumps(hardware(out, captured, [r for r in rows if r['job_id'] not in have]),
                                             indent=2) + '\n')
    _write(out / 'environment.txt', environment(out, runs['commit'], captured is not None))
    _write(out / 'DECISIONS_package3.md', decisions())
    between = _git('log', '--format=%h %s', f"{runs['commit']}..{head}") or ''
    _write(out / 'provenance.json', json.dumps({
        'runs': {'commit': runs['commit'], 'tree_hash': runs['tree_hash'], 'source': runs['source'],
                 'locked_at_utc': runs['locked_at_utc']},
        'assembly': {'commit': head, 'branch': _git('rev-parse', '--abbrev-ref', 'HEAD')},
        'commits_between': between.splitlines(),
        'note': "Every run and every summary but L1's is Grace's, at the runs' commit; L1 is re-evaluated "
                'at the assembly (checks_item4.json).'}, indent=2) + '\n')
    log.append(f"provenance.json: {len(between.splitlines())} commit(s) between the runs and the assembly")
    _write(out / 'assembly_log.txt', '\n'.join(log) + '\n')
    print('\n'.join(log))
    return checks, total


def main(argv=None):
    ap = argparse.ArgumentParser(description="Package 3's results folder, from Grace's tarball.")
    ap.add_argument('--grace', required=True, help='the extracted package3/ folder')
    ap.add_argument('--out', required=True)
    ap.add_argument('--p2-12', dest='p2_12', required=True, help="Stage 1's P2_12_reference_errors (L1's reference)")
    ap.add_argument('--sacct-extra', default=None, help='sacct -X -P -n lines of jobs the tarball lacks')
    ap.add_argument('--provenance', default=None,
                    help="p3_provenance.slurm's hardware.json, environment.txt and log, downloaded")
    args = ap.parse_args(argv)
    checks, _ = assemble(args.grace, args.out, args.p2_12, args.sacct_extra, args.provenance)
    return 0 if all(v['passed'] for v in checks.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
