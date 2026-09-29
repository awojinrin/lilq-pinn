"""
A wave's report for the advisor (Addendum v2.2 Section 4.2)
===========================================================

Run by ``91_wave_report.slurm`` at the end of each wave, on that wave's
package root (``results/wave<N>``). It builds the wave's summary tables from
whatever the wave produced -- the run index, check B1's reproduction table,
the merged four-method table -- and packs, into ``results/wave<N>_report.tar.gz``,
everything the advisor asked to see after wave 1, and its analogue after
waves 2 and 3:

* the wave's ``COMMIT`` (commit and source-tree hash);
* ``reproduction_check.csv``, ``gpu_cpu_equivalence.csv``, ``runs_index.csv``;
* ``iterations.csv`` / ``run.json`` / ``summary.json`` of the K_max passes and
  of both Beltrami runs (and their Section 3.7 reports);
* the four-method rows with the saved per-iteration histories, and the
  stall controls (``four_method_controls.csv``; the advisor's reply to wave 1,
  item 2.5);
* Component A's check results, search lists, screening files (each run's
  ``run.json`` and ``log.csv``) and selections, representative and tuning log, and ``run_endings.csv``: how every
  Component A run ended, flagging any that ended before its budget (the
  reply to wave 1, Section 3);
* ``oversampling.csv``, the B8 and B9 tables;
* every ``hardware.json``, the Slurm logs, and ``sacct`` for the wave's jobs
  (elapsed time and resources per job; the SUs charged come from
  ``myproject``, which the submission scripts print).

Trained models, checkpoints and collocation files stay on the cluster (they
are large); nothing here is needed to read the results. Each step runs only
when its inputs exist, and a failing step is recorded, not fatal.

Usage::

    python scripts/cluster/wave_report.py --wave 1
"""

import argparse
import csv
import datetime
import glob
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(cmd, log):
    """Run a step; record its outcome in ``log`` instead of stopping."""
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    log.append(f"$ {' '.join(map(str, cmd))}\n  exit {out.returncode}\n"
               + (out.stdout[-2000:] + out.stderr[-2000:]).strip())
    return out.returncode == 0


def report_files(root: Path):
    """The files the report packs, relative to ``root``."""
    B = root / 'B_instrumentation'
    pats = [
        'COMMIT', '**/hardware.json',
        'B_instrumentation/*.csv',
        'B_instrumentation/*_kmax/*', 'B_instrumentation/beltrami_*/*.json',
        'B_instrumentation/beltrami_*/*.csv',
        'B_instrumentation/beltrami_pinned*/*',
        'B_instrumentation/four_method_jobs/*/four_method_tables.csv',
        'B_instrumentation/four_method_jobs/*/models/*/history.csv',
        'B_instrumentation/b8_jobs/*/b8_initial_guess.csv', 'B_instrumentation/b8_jobs/*/lilq_logs/*/*',
        'B_instrumentation/b8_jobs/*/models/*/history.csv',
        'B_instrumentation/darcy_fv/*/darcy_fv_comparison.csv',
        'B_instrumentation/basis_study/*.csv', 'B_instrumentation/basis_study/runs/*/iterations.csv',
        'A_calibration/checks/*', 'A_calibration/tuning_log.md', 'A_calibration/search/*',
        'A_calibration/run_endings.csv',
        'A_calibration/screening/*_selection.json', 'A_calibration/full/*_representative.json',
        'A_calibration/screening/*/log.csv',
        'A_calibration/*/*/run.json',
        'C_oversampling/results/*', 'C_oversampling/figures/*',
    ]
    files = set()
    for pat in pats:
        files.update(p for p in root.glob(pat) if p.is_file())
    skip = ('.pt', '.npz', '.tmp')
    return sorted(p.relative_to(root) for p in files if not p.name.endswith(skip))


ENDINGS_COLUMNS = ('stage', 'run', 'family', 'end_reason', 'budget_s', 'wall_s', 'ended_before_budget')


def component_a_endings(root: Path):
    """``A_calibration/run_endings.csv``: one row per Component A run
    (screening, full, CPU, float32 and the checks) with its end reason,
    budget and wall time; ``ended_before_budget`` when it stopped for any
    reason other than its budget (the F1 criterion, a failure, F2's
    convergence or step limit). Returns the rows that did."""
    A = root / 'A_calibration'
    rows = []
    for path in sorted(A.glob('*/**/run.json')):
        run = json.loads(path.read_text())
        name = path.parent.name
        rows.append({'stage': path.relative_to(A).parts[0], 'run': path.parent.relative_to(A).as_posix(),
                     'family': (run.get('config') or {}).get('family')
                               or (name[:2] if name[:2] in ('F1', 'F2') else ''),
                     'end_reason': run.get('end_reason'), 'budget_s': run.get('budget_s'),
                     'wall_s': run.get('wall_s'),
                     'ended_before_budget': run.get('end_reason') != 'budget'})
    if rows:
        with open(A / 'run_endings.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ENDINGS_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    return [r for r in rows if r['ended_before_budget']]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build and pack a wave's report.")
    ap.add_argument('--wave', type=int, required=True, choices=(1, 2, 3))
    ap.add_argument('--results', default=str(REPO / 'results'))
    args = ap.parse_args(argv)
    results = Path(args.results)
    root = results / f'wave{args.wave}'
    B = root / 'B_instrumentation'
    log = [f"wave {args.wave} report, {datetime.datetime.now(datetime.timezone.utc).isoformat()}"]

    if list(B.glob('*/summary.json')):
        _run([sys.executable, '-c', "from pathlib import Path; from experiments.component_b import write_index; "
                                    f"write_index(Path({str(B)!r}))"], log)
        _run([sys.executable, 'experiments/component_b.py', '--reproduction-check', '--out-root', str(root)], log)
    jobs = sorted(glob.glob(str(B / 'four_method_jobs' / '*') + os.sep))
    if jobs:
        _run([sys.executable, 'experiments/four_method_tables.py', '--out-dir', str(B), '--merge-from', *jobs], log)

    if (root / 'A_calibration').is_dir():
        early = component_a_endings(root)
        log.append(f"Component A runs ended before their budget: {len(early)}"
                   + ''.join(f"\n  {r['run']}: {r['end_reason']} after {r['wall_s']} s" for r in early))

    job_ids = sorted({p.stem.rsplit('.', 1)[-1].split('_')[0] for p in (REPO / 'logs').glob('*.out')})
    sacct = subprocess.run(['sacct', '-X', '-j', ','.join(job_ids), '--format',
                            'JobID,JobName%24,Partition,AllocTRES%60,Elapsed,State,ExitCode'],
                           capture_output=True, text=True) if job_ids else None
    (root / 'sacct.txt').write_text(sacct.stdout if sacct and sacct.returncode == 0 else
                                    'sacct unavailable\n')
    (root / 'report_log.txt').write_text('\n\n'.join(log) + '\n')

    out = results / f'wave{args.wave}_report.tar.gz'
    files = report_files(root) + [Path('sacct.txt'), Path('report_log.txt')]
    with tarfile.open(out, 'w:gz') as tar:
        for rel in files:
            tar.add(root / rel, arcname=f'wave{args.wave}/{rel.as_posix()}')
        for p in sorted((REPO / 'logs').glob('*.out')):
            tar.add(p, arcname=f'wave{args.wave}/slurm_logs/{p.name}')
    print(f"Wrote {out} ({len(files)} files, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
