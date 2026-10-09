"""Package 3, wave 2: may B3 level 2 be launched? (the advisor's instructions of 8
October 2026, Sections 3.4 and 8; Addendum 1, Section 7)

First, wave 1 must be over: no Package 3 job (``lilq-p3-*``) pending or running in
``squeue``, the report aside, and B3 level 2 neither queued nor run already. Then all three
must hold:

1. **K0 passed** (``P3_checks/k0.json``). B3 level 2's system has 5.3e9 entries, and K0 is
   the check of gelsy and the pivoted QR above 2^32. A failed K0 is a stop condition
   (Section 8.1).
2. **The launch rule** (Section 3.4), from B3 level 1's ``run.json``:
   1.91 x (8 x level-1 seconds per iteration + level-1 kappa time) < 6.5 h. A level 1 that
   did not complete (stopped by its cap) is reported as logged, and level 2 is not launched
   (Section 8.2).
3. **The budget guard** (``su_plan.csv``; cap 850 SU after Addendum 1): the SU charged by
   every Package 3 job so far (``sacct``, at Grace's rates, ``wave_report.su_rows``), plus
   B3 level 2's 336 SU requested (7 h x 48), is at most 850. Its run stops at 6 h 55 min,
   which leaves about 2 SU for the report after it (about 1 SU).

Nothing is written. Exit 0 to launch, 1 not to, 2 if it cannot be decided yet (or ``squeue``
or ``sacct`` cannot be read). Prints the decision and its numbers.

Usage (from the repository, with the project's Python)::

    python scripts/cluster/package3/b3l2_gate.py --results results --logs logs
"""
import argparse
import getpass
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROW_RATIO = 1.91                  # 665,331 / 348,168, as Section 3.4 writes it
LIMIT_H = 6.5
CAP_SU = 850
L2_REQUEST_SU = 336
L2_JOB = 'lilq-p3-beltrami-b3-l2'
REPORT_JOB = 'lilq-p3-report'
REPO = Path(__file__).resolve().parents[3]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def squeue():
    """``[(job_id, job_name, state)]`` of this user's queued jobs, or None if squeue fails."""
    try:
        out = subprocess.run(['squeue', '-h', '-u', os.environ.get('USER') or getpass.getuser(), '-o', '%i|%j|%T'],
                             capture_output=True, text=True)
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return [tuple(line.split('|')[:3]) for line in out.stdout.splitlines() if line.count('|') >= 2]


def sacct(job_ids):
    """``sacct -X -P -n`` in ``wave_report.su_rows``'s format, or None if sacct fails."""
    try:
        out = subprocess.run(['sacct', '-X', '-P', '-n', '-j', ','.join(job_ids), '--format',
                              'JobID,JobName,AllocTRES,ElapsedRaw,State'], capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout if out.returncode == 0 else None


def launch_rule(p3):
    """(holds, message) from B3 level 1's ``run.json['package3']``."""
    est_h = ROW_RATIO * (8 * p3['seconds_per_iteration'] + p3['kappa_time_s']) / 3600
    return est_h < LIMIT_H, (f"launch rule (Section 3.4): 1.91 x (8 x {p3['seconds_per_iteration']:.1f} s + "
                             f"{p3['kappa_time_s']:.1f} s) = {est_h:.2f} h, {'<' if est_h < LIMIT_H else 'not <'} {LIMIT_H} h")


def decide(results, logs_dir, queue=squeue, accounting=sacct):
    """(exit code, lines): 0 launch, 1 do not, 2 not yet decidable."""
    pkg = Path(results) / 'package3'
    stage_report = _module('stage_report', REPO / 'scripts' / 'cluster' / 'package2' / 'stage_report.py')
    wave_report = _module('wave_report', REPO / 'scripts' / 'cluster' / 'wave_report.py')
    logs = stage_report.stage_logs('p3', logs_dir)
    if any(p.name.startswith(L2_JOB + '.') for p in logs.values()):
        return 1, [f'B3 level 2 has run already ({logs_dir}/{L2_JOB}.*.out): nothing to launch']
    q = queue()
    if q is None:
        return 2, ['squeue could not be read (run this on a login node)']
    mine = [(i, n, s) for i, n, s in q if n.startswith('lilq-p3-')]
    if any(n == L2_JOB for _, n, _ in mine):
        return 1, ['B3 level 2 is queued already: nothing to launch']
    waiting = [f'{i} {n} ({s})' for i, n, s in mine if n != REPORT_JOB]
    if waiting:
        return 2, ['wave 1 is not over; still queued: ' + ', '.join(waiting)]
    if not logs:
        return 2, [f'no Package 3 job has run yet (no {logs_dir}/lilq-p3-*.out)']

    lines, ok = [], True
    k0 = pkg / 'P3_checks' / 'k0.json'
    if not k0.exists():
        return 1, [f'K0 did not finish ({k0} missing; see its log): B3 level 2 needs K0. '
                   'A failed K0 is a stop condition (Section 8.1)']
    k0_ok = bool(json.loads(k0.read_text())['passed'])
    lines.append('K0: ' + ('passed' if k0_ok else 'FAILED: a stop condition (Section 8.1), write to the advisor'))
    ok &= k0_ok

    l1 = pkg / 'P3_1_beltrami_certified' / 'B3_L1' / 'run.json'
    if not l1.exists():
        return 1, lines + [f'B3 level 1 did not run ({l1} missing; see its log)']
    p3 = json.loads(l1.read_text())['package3']
    if p3.get('status') != 'complete':
        return 1, lines + [f"B3 level 1 did not complete (status {p3.get('status')!r}: stopped by its cap or failed). "
                           'Section 8.2: its logged iterates are reported, and level 2 is not launched']
    rule, msg = launch_rule(p3)
    lines.append(msg)
    ok &= rule

    text = accounting(list(logs))
    if text is None:
        return 2, lines + ['sacct could not be read']
    rows = wave_report.su_rows(text)
    missing = sorted(set(logs) - {r['job_id'] for r in rows})
    if missing:
        return 2, lines + [f"sacct has no record of job(s) {' '.join(missing)}"]
    running = [r['job_id'] for r in rows if r['state'] in ('RUNNING', 'PENDING', 'REQUEUED', 'COMPLETING')
               and r['job_name'] != REPORT_JOB]
    if running:
        return 2, lines + [f"sacct still shows job(s) {' '.join(running)} running"]
    su = sum(r['su'] or 0 for r in rows)
    lines.append(f'budget: {su:.1f} SU charged by {len(rows)} Package 3 jobs + {L2_REQUEST_SU} = '
                 f"{su + L2_REQUEST_SU:.1f}, {'<=' if su + L2_REQUEST_SU <= CAP_SU else 'over'} {CAP_SU}")
    ok &= su + L2_REQUEST_SU <= CAP_SU
    return (0 if ok else 1), lines


def main(argv=None):
    if sys.version_info < (3, 8):
        sys.exit("run this with the project's Python (module load ...; source the venv), not the system python3")
    ap = argparse.ArgumentParser(description='Package 3: may B3 level 2 be launched?')
    ap.add_argument('--results', default=str(REPO / 'results'))
    ap.add_argument('--logs', default=str(REPO / 'logs'))
    args = ap.parse_args(argv)
    code, lines = decide(args.results, args.logs)
    for line in lines:
        print('  ' + line)
    print({0: 'LAUNCH B3 level 2', 1: 'DO NOT LAUNCH B3 level 2',
           2: 'NOT YET: wait, and run this again'}[code])
    return code


if __name__ == '__main__':
    sys.exit(main())
