"""
Package 2: a stage's report on the cluster
==========================================

Run by ``p2s2_report.slurm`` after every job of a Package 2 stage (afterany):

* ``sacct.txt`` and ``su_per_job.csv`` in the stage folder, for the stage's
  jobs. Those are the jobs whose Slurm log is ``logs/lilq-<stage>-*.out``,
  this report's own excepted (it is still running). The SUs are at Grace's
  rates, ``wave_report.su_rows``, which reproduced waves 2 and 3's charges;
* ``results/package2_<stage name>.tar.gz``: the whole stage folder (the stage
  results are small; Stage 1's was downloaded whole too) and the stage's
  Slurm logs, under ``slurm_logs/``.

A failing step is recorded in ``report_log.txt``, not fatal.

Usage::

    python scripts/cluster/package2/stage_report.py --stage p2s2 --results results
"""

import argparse
import datetime
import importlib.util
import os
import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STAGES = {'p2s1': 'package2_stage1', 'p2s2': 'package2_stage2'}


def _wave_report():
    spec = importlib.util.spec_from_file_location('wave_report', REPO / 'scripts' / 'cluster' / 'wave_report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_logs(stage, logs_dir, exclude_job=None):
    """``{job_id: log path}`` of the stage's jobs (``lilq-<stage>-<name>.<id>.out``)."""
    out = {}
    for p in sorted(Path(logs_dir).glob(f'lilq-{stage}-*.out')):
        job = p.stem.rsplit('.', 1)[-1].split('_')[0]
        if job.isdigit() and job != exclude_job:
            out[job] = p
    return out


def report(stage, results, logs_dir=REPO / 'logs', sacct=True):
    root = Path(results) / STAGES[stage]
    log = [f'{stage} report, {datetime.datetime.now(datetime.timezone.utc).isoformat()}']
    jobs = stage_logs(stage, logs_dir, exclude_job=os.environ.get('SLURM_JOB_ID'))
    log.append(f"jobs: {' '.join(jobs) or 'none found'}")
    if sacct and jobs:
        out = subprocess.run(['sacct', '-X', '-j', ','.join(jobs), '--format',
                              'JobID,JobName%32,Partition,AllocTRES%60,Elapsed,State,ExitCode'],
                             capture_output=True, text=True)
        (root / 'sacct.txt').write_text(out.stdout if out.returncode == 0 else 'sacct unavailable\n')
        rows = _wave_report().su_per_job(list(jobs), root)
        if rows:
            log.append(f"SUs by job (Grace's rates; su_per_job.csv): {sum(r['su'] or 0 for r in rows):,.0f} in all"
                       + ''.join(f"\n  {r['job_id']} {r['job_name']}: {r['su']} ({r['elapsed_h']:.2f} h at "
                                 f"{r['su_per_hour']}/h), {r['state']}" for r in rows))
        else:
            log.append('su_per_job.csv not written (sacct unavailable, or not on Grace)')
    (root / 'report_log.txt').write_text('\n\n'.join(log) + '\n')
    tar_path = Path(results) / f'{STAGES[stage]}.tar.gz'
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(root, arcname=STAGES[stage])
        for job, p in jobs.items():
            tar.add(p, arcname=f'{STAGES[stage]}/slurm_logs/{p.name}')
    print(f'Wrote {tar_path} ({tar_path.stat().st_size / 1e6:.1f} MB; {len(jobs)} job logs)')
    return tar_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="A Package 2 stage's report.")
    ap.add_argument('--stage', choices=tuple(STAGES), required=True)
    ap.add_argument('--results', default=str(REPO / 'results'))
    args = ap.parse_args(argv)
    report(args.stage, args.results)


if __name__ == '__main__':
    main()
