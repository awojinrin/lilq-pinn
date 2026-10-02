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
  (elapsed time and resources per job), with ``su_per_job.csv``: each job's
  SUs from its ``sacct`` record at Grace's rates (``su_rate``), which
  reproduce the charges of waves 2 and 3 (``myproject``);
* wave 4 (the advisor's reply on wave 3, Section 4): the three B8 LiL-Q
  reruns beside wave 3's rows (``b8_kmax60_vs_wave3.csv``), and every
  quoted LiL-Q time clean against logged, with both runs' iterations
  (``clean_timing/clean_vs_logged.csv``).

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
        'A_calibration/run_endings.csv', 'A_calibration/representative_thresholds.csv',
        'A_calibration/*/*/validation.json',
        # wave 4: clean timing, B10, the logged gravity BL rerun
        'B_instrumentation/clean_timing/*', 'B_instrumentation/clean_timing/*/*',
        'B_instrumentation/b10/*.csv', 'B_instrumentation/b10/README.md', 'B_instrumentation/b10/*/iterations.csv',
        'B_instrumentation/b10/*/run.json', 'B_instrumentation/*_paper/summary.json',
        'B_instrumentation/*_paper/iterations.csv', 'B_instrumentation/*_paper/run.json',
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


THRESHOLDS = (1e-4, 1e-6, 1e-8, 1e-9)
THRESHOLD_COLUMNS = ('family', 'representative', 'stage', 'run', 'seed', 'device', 'precision') + tuple(
    f't_eps_u_le_{t:g}' for t in THRESHOLDS) + ('final_eps_u', 'final_eps_p_meanfree', 'wall_s', 'end_reason')


def representative_thresholds(root: Path):
    """``A_calibration/representative_thresholds.csv`` (the advisor's reply to
    wave 2, Section 2): for each family's representative and each of its
    runs -- the full stage (GPU), the CPU reruns and the float32 run -- the
    first logged time (``t_cum_s``) at which eps_u <= 1e-4, 1e-6, 1e-8 and
    1e-9 (empty if never), and the final eps_u and mean-free pressure error.
    Read from the runs' ``log.csv``. Returns the rows."""
    A = root / 'A_calibration'
    rows = []
    for family in ('F1', 'F2'):
        rep_path = A / 'full' / f'{family}_representative.json'
        if not rep_path.exists():
            continue
        rep = json.loads(rep_path.read_text())['representative']
        for stage in ('full', 'full_cpu', 'float32'):
            for run_dir in sorted((A / stage).glob(f'{rep}_s*')):
                log, run_json = run_dir / 'log.csv', run_dir / 'run.json'
                if not log.exists() or not run_json.exists():
                    continue
                run = json.loads(run_json.read_text())
                first = {t: None for t in THRESHOLDS}
                with open(log, newline='') as f:
                    for r in csv.DictReader(f):
                        try:
                            eps, t_cum = float(r['eps_u']), float(r['t_cum_s'])
                        except (KeyError, TypeError, ValueError):
                            continue
                        for t in THRESHOLDS:
                            if first[t] is None and eps <= t:
                                first[t] = t_cum
                rows.append({'family': family, 'representative': rep, 'stage': stage, 'run': run_dir.name,
                             'seed': run.get('seed'), 'device': run.get('device'), 'precision': run.get('precision', ''),
                             **{f't_eps_u_le_{t:g}': first[t] for t in THRESHOLDS},
                             'final_eps_u': run.get('eps_u'), 'final_eps_p_meanfree': run.get('eps_p_meanfree'),
                             'wall_s': run.get('wall_s'), 'end_reason': run.get('end_reason')})
    if rows:
        with open(A / 'representative_thresholds.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=THRESHOLD_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    return rows


B8_COMPARE_COLUMNS = ('case', 'guess', 'P', 'target', 'wave3_K_max', 'wave3_iterations', 'wave3_final_loss',
                      'wave3_stopping_reason', 'wave4_K_max', 'wave4_iterations', 'wave4_final_loss',
                      'wave4_stopping_reason', 'wave4_target_reached')


def b8_reruns_beside_wave3(results: Path, root: Path):
    """``B_instrumentation/b8_kmax60_vs_wave3.csv`` (the advisor's reply on
    wave 3, Section 4): each B8 LiL-Q row rerun with K_max = 60 in this wave
    (``b8_jobs/lilq_kmax60``) beside wave 3's row of the same case, guess and
    P -- K_max, iterations, final loss and stopping reason. Wave 3's tables
    have no K_max column; its cap is ``iterations_cap``. Returns the rows."""
    new = root / 'B_instrumentation' / 'b8_jobs' / 'lilq_kmax60' / 'b8_initial_guess.csv'
    if not new.exists():
        return []
    old = {}
    for path in sorted((results / 'wave3' / 'B_instrumentation' / 'b8_jobs').glob('*/b8_initial_guess.csv')):
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                if r['method'] == 'LiL-Q':
                    old[(r['case'], r['guess'], str(r['P']))] = r
    rows = []
    with open(new, newline='') as f:
        for r in csv.DictReader(f):
            o = old.get((r['case'], r['guess'], str(r['P'])), {})
            rows.append({'case': r['case'], 'guess': r['guess'], 'P': r['P'], 'target': r.get('target'),
                         'wave3_K_max': o.get('K_max') or o.get('iterations_cap', ''),
                         'wave3_iterations': o.get('iterations', ''), 'wave3_final_loss': o.get('final_loss', ''),
                         'wave3_stopping_reason': o.get('stopping_reason', ''),
                         'wave4_K_max': r.get('K_max'), 'wave4_iterations': r.get('iterations'),
                         'wave4_final_loss': r.get('final_loss'), 'wave4_stopping_reason': r.get('stopping_reason'),
                         'wave4_target_reached': r.get('target_reached')})
    with open(root / 'B_instrumentation' / 'b8_kmax60_vs_wave3.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=B8_COMPARE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


CLEAN_VS_LOGGED_COLUMNS = ('run', 'benchmark', 'config', 'device', 'quantity', 'clean_time_s', 'logged_time_s',
                           'clean_over_logged', 'warmup_run_time_s', 'clean_iterations', 'logged_iterations',
                           'iterations_match', 'K_max', 'logged_K_max', 'logged_source', 'error')


def clean_vs_logged(root: Path):
    """``B_instrumentation/clean_timing/clean_vs_logged.csv`` (the advisor's
    reply on wave 3, Section 4): every quoted LiL-Q time -- each row of
    ``clean_timing.csv``, Beltrami and the pinned Beltrami run included --
    clean against logged, their ratio, the warm-up run's time, and both
    runs' iteration counts and K_max (the logged ones from the logged run's
    ``summary.json``, or the pinned run's ``report.json``). The iteration
    counts must agree: the two runs are the same solve. Returns the rows."""
    path = root / 'B_instrumentation' / 'clean_timing' / 'clean_timing.csv'
    if not path.exists():
        return []
    with open(path, newline='') as f:
        clean = list(csv.DictReader(f))
    rows = []
    for r in clean:
        src = r.get('logged_source') or ''
        logged = json.loads(Path(src).read_text()) if src and Path(src).is_file() else {}
        logged_it = logged.get('iterations', logged.get('n_outer_iters'))
        try:
            ratio = float(r['clean_time_s']) / float(r['logged_time_s'])
        except (TypeError, ValueError, ZeroDivisionError):
            ratio = ''
        match = '' if logged_it is None or r.get('iterations') in (None, '') else int(r['iterations']) == int(logged_it)
        rows.append({'run': r['run'], 'benchmark': r['benchmark'], 'config': r['config'], 'device': r['device'],
                     'quantity': r.get('quantity'), 'clean_time_s': r.get('clean_time_s'),
                     'logged_time_s': r.get('logged_time_s'), 'clean_over_logged': ratio,
                     'warmup_run_time_s': r.get('warmup_run_time_s'), 'clean_iterations': r.get('iterations'),
                     'logged_iterations': logged_it, 'iterations_match': match, 'K_max': r.get('K_max'),
                     'logged_K_max': logged.get('K_max', ''), 'logged_source': src, 'error': r.get('error', '')})
    with open(path.parent / 'clean_vs_logged.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CLEAN_VS_LOGGED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


# Grace's charge per hour (TAMU HPRC): one SU per core-hour, plus 72 per
# A100-hour; a job holding a whole A100 node (all 48 cores) is charged for
# the node's two A100s, 192 per hour, even with one requested. These rates
# reproduce wave 3's charge exactly (1,019 SU) and wave 2's to 1% (8,402 of
# 8,498: the rest were jobs outside its sacct record).
GRACE_GPU_SU_PER_HOUR = 72
GRACE_WHOLE_GPU_NODE_SU_PER_HOUR = 192
GRACE_GPU_NODE_CORES = 48
SU_COLUMNS = ('job_id', 'job_name', 'alloc_tres', 'elapsed_h', 'state', 'su_per_hour', 'su', 'note')


def su_rate(alloc_tres: str):
    """SUs per hour on Grace of a job with this ``sacct`` ``AllocTRES``."""
    tres = dict(kv.split('=', 1) for kv in alloc_tres.split(',') if '=' in kv)
    cpu, gpu = int(tres.get('cpu', 0)), int(tres.get('gres/gpu', 0))
    if gpu:
        return GRACE_WHOLE_GPU_NODE_SU_PER_HOUR if cpu >= GRACE_GPU_NODE_CORES else cpu + GRACE_GPU_SU_PER_HOUR * gpu
    return cpu


def su_rows(sacct_parsable: str):
    """``su_per_job.csv``'s rows from ``sacct -X -P -n --format
    JobID,JobName,AllocTRES,ElapsedRaw,State``."""
    rows = []
    for line in sacct_parsable.splitlines():
        parts = line.split('|')
        if len(parts) < 5 or not parts[3].isdigit():
            continue
        job_id, name, tres, elapsed, state = parts[:5]
        hours = int(elapsed) / 3600
        rate = su_rate(tres) if tres else None
        rows.append({'job_id': job_id, 'job_name': name, 'alloc_tres': tres, 'elapsed_h': round(hours, 4),
                     'state': state, 'su_per_hour': rate, 'su': round(rate * hours, 1) if rate is not None else '',
                     'note': 'still running when the report was built' if state == 'RUNNING' else ''})
    return rows


def su_per_job(job_ids, root: Path):
    """``su_per_job.csv`` in ``root`` for ``job_ids``, at Grace's rates
    (``su_rate``; no rates on another cluster). The finalize job runs after
    the report, so it is not in it. Returns the rows."""
    if not job_ids or os.environ.get('CLUSTER', 'grace') != 'grace':
        return []
    out = subprocess.run(['sacct', '-X', '-P', '-n', '-j', ','.join(job_ids), '--format',
                          'JobID,JobName,AllocTRES,ElapsedRaw,State'], capture_output=True, text=True)
    if out.returncode != 0:
        return []
    rows = su_rows(out.stdout)
    with open(root / 'su_per_job.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SU_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build and pack a wave's report.")
    ap.add_argument('--wave', type=int, required=True, choices=(1, 2, 3, 4))
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

    if (root / 'A_calibration' / 'full').is_dir():
        log.append(f"Representative threshold times: {len(representative_thresholds(root))} runs")
    if (root / 'A_calibration').is_dir():
        early = component_a_endings(root)
        log.append(f"Component A runs ended before their budget: {len(early)}"
                   + ''.join(f"\n  {r['run']}: {r['end_reason']} after {r['wall_s']} s" for r in early))
    b8 = b8_reruns_beside_wave3(results, root)
    if b8:
        log.append("B8 LiL-Q reruns with K_max = 60, beside wave 3:" + ''.join(
            f"\n  {r['case']} {r['guess']} P = {r['P']}: wave 3 {r['wave3_iterations']} iterations "
            f"({r['wave3_stopping_reason']}, K_max {r['wave3_K_max']}), loss {r['wave3_final_loss']}; "
            f"wave 4 {r['wave4_iterations']} ({r['wave4_stopping_reason']}), loss {r['wave4_final_loss']}" for r in b8))
    clean = clean_vs_logged(root)
    if clean:
        bad = [r for r in clean if r['iterations_match'] is False]
        failed = [r for r in clean if r['error']]
        log.append(f"Clean against logged times: {len(clean)} rows; iteration counts differ in {len(bad)}"
                   + ''.join(f"\n  {r['run']}: clean {r['clean_iterations']}, logged {r['logged_iterations']}"
                             for r in bad)
                   + f"; failed clean runs: {len(failed)}" + ''.join(f"\n  {r['run']}" for r in failed))

    job_ids = sorted({p.stem.rsplit('.', 1)[-1].split('_')[0] for p in (REPO / 'logs').glob('*.out')})
    sacct = subprocess.run(['sacct', '-X', '-j', ','.join(job_ids), '--format',
                            'JobID,JobName%24,Partition,AllocTRES%60,Elapsed,State,ExitCode'],
                           capture_output=True, text=True) if job_ids else None
    (root / 'sacct.txt').write_text(sacct.stdout if sacct and sacct.returncode == 0 else
                                    'sacct unavailable\n')
    su = su_per_job(job_ids, root)
    if su:
        log.append(f"SUs by job (Grace's rates; su_per_job.csv): {sum(r['su'] or 0 for r in su):,.0f} in all"
                   + ''.join(f"\n  {r['job_id']} {r['job_name']}: {r['su']} ({r['elapsed_h']:.2f} h at "
                             f"{r['su_per_hour']}/h){' -- ' + r['note'] if r['note'] else ''}" for r in su))
    (root / 'report_log.txt').write_text('\n\n'.join(log) + '\n')

    out = results / f'wave{args.wave}_report.tar.gz'
    files = report_files(root) + [Path('sacct.txt'), Path('report_log.txt')] + (
        [Path('su_per_job.csv')] if (root / 'su_per_job.csv').exists() else [])
    with tarfile.open(out, 'w:gz') as tar:
        for rel in files:
            tar.add(root / rel, arcname=f'wave{args.wave}/{rel.as_posix()}')
        for p in sorted((REPO / 'logs').glob('*.out')):
            tar.add(p, arcname=f'wave{args.wave}/slurm_logs/{p.name}')
    print(f"Wrote {out} ({len(files)} files, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
