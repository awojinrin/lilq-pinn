"""
Package 5, items 1, 2 and 4: the results folder, from Grace's tarball
=====================================================================

Assembles ``package5_results/`` in the layout of the advisor's instructions of
13 November 2026 (Section 8), from the extracted ``results/package5.tar.gz`` of
the report job. The helpers every package's assembly shares are in
``assemble_common.py``.

- **The run folders, as Grace wrote them** at the runs' commit (``COMMIT``):
  ``P5_1_weights/`` (item 1), ``P3_1_beltrami_certified/`` (item 1's Beltrami
  runs), ``P5_2_nlls/`` (item 2), ``P5_4_pb/`` (item 4), ``provenance/timed/``
  (the timed job's record), ``slurm_logs/``, ``sacct.txt``, ``su_per_job.csv``,
  ``report_log.txt``.
- **The comparisons with the advisor's values.** The cluster had none of his
  files, so the assembly reruns the three items' summaries on Grace's files
  with them: item 1 beside ``expected_weight_bratu.json`` and
  ``expected_weight_bl.json`` (``compare_expected.csv``), item 2 beside
  ``expected_nlls_liln.json``, 4b beside ``expected_rf_pb.json``; and the
  Beltrami runs beside the delivered rows of Package 3
  (``P3_1_beltrami_certified/beside_delivered.csv``). The summaries also
  rewrite Grace's tables: those stay Grace's, and the assembly stops if a
  number differs by more than ``TABLE_RTOL`` (relative), but for the columns
  of delta_P, whose round-off is recorded (``table_rerun_differences.json``).
  Item 1's ``references.json`` is compared the same way.
- **The checks of the delivered runs** (``checks.json``: C1 to C3,
  ``p5_checks.collect``), on Package 2's, 3's and 4's delivered results.
- **The records of Section 8:** ``su.csv`` (every job at Grace's rates, with
  the report job's own row from ``--sacct-extra``), ``hardware.json`` and
  ``environment.txt``, ``DECISIONS_package5.md``, ``provenance.json`` (the
  runs' commit, this one, those between, and the SHA-256 of the advisor's
  files, which are not copied).
- **The re-time** (``--retime``, stage p5t): one item 2 run re-timed after an
  untimed warm-up, copied as Grace wrote it to ``package5_retime/``; its run
  must be the timed job's (the assembly stops otherwise), and
  ``P5_2_nlls/retime.csv`` sets the two times side by side. Grace's own
  tables keep the timed job's time.

- **Addendum 2** (``--a2``, stage p5a): the 18 plateau runs, K1 and K2, copied
  as Grace wrote it to ``package5_a2/``; its summary rerun on Grace's stored
  iterates (Grace's kept; delta_P's round-off in
  ``package5_a2_rerun_differences.json``); K1 and K2 join the exit code.
  Section 3a's update histories of the delivered runs are written here too.

Refuses a Grace tree whose commit is not an ancestor of this checkout's HEAD.
Can be run again: it rewrites what it writes.

Usage::

    python experiments/p5_assemble.py --grace <extracted>/package5 --out ../package5_results \\
        --reference-dir <package2_stage2>/reference --package2 <package2_stage2> --package3 <package3> \\
        --package4 <package4> --advisor <folder of the advisor's expected_*.json> [--sacct-extra <file>] \\
        [--retime <extracted>/package5_retime] [--a2 <extracted>/package5_a2 [--a2-logs <job logs>]]
"""

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.assemble_common import (blas_line, cell_diff, compare_tables, copy_grace,  # noqa: E402
                                         decisions, extra_rows, job_table, log_start, modules, preflight_lines,
                                         read_json, runs_and_head, su_rows, tables, write_lf, write_provenance,
                                         write_su)

GRACE_FOLDERS = ('P5_1_weights', 'P3_1_beltrami_certified', 'P5_2_nlls', 'P5_4_pb', 'provenance', 'slurm_logs')
GRACE_FILES = ('COMMIT', 'sacct.txt', 'su_per_job.csv', 'report_log.txt')
ADVISOR_FILES = ('expected_weight_bratu.json', 'expected_weight_bl.json', 'expected_nlls_liln.json',
                 'expected_rf_pb.json')
REFERENCES = 'P5_1_weights/references.json'
TIMED = 'provenance/timed'
TABLE_RTOL = 1e-8
ROUNDOFF_COLUMNS = (lambda column: 'delta' in column)   # delta_P (delta_L2, delta_X, ...) and the ratios to it
RECORD = 'table_rerun_differences.json'
RETIME = 'package5_retime'                       # stage p5t: one item 2 run re-timed (DECISIONS, 14 November 2026)
RETIMED = 'P5_2_nlls/burgers_p10/lbfgs'
RETIME_TABLE = 'P5_2_nlls/retime.csv'
A2 = 'package5_a2'                               # stage p5a: Addendum 2's plateau runs (DECISIONS, 16 November 2026)
A2_RUN_FILES = ('run.json', 'iterations.csv', 'betas.npy', 'errors_by_iterate.csv', 'updates_by_iterate.csv')
A2_JSON = ('trig_delta_P.json', 'item1_references.json', 'checks.json')   # what the summary writes besides its CSVs
A2_RECORD = 'package5_a2_rerun_differences.json'
A2_DELIVERED = 'P2_9_bases/bl_gravity_paper_P1024_ic'                     # K2's delivered run, in Package 2 stage 2
# references.json's round-off quantities: the Burgers PDE check's residual is about 1e-12 (5e-13 of the
# equation's terms, against the check's 1e-8), so it differs between machines in its first digits. Its
# values are recorded; its verdict (``passed``) and everything else must be Grace's.
REFERENCE_ROUNDOFF = ('/burgers_pde_check/max_abs_pde_residual', '/burgers_pde_check/relative')


def _flat(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flat(v, f'{path}/{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flat(v, f'{path}[{i}]')
    else:
        yield path, obj


def compare_references(out, grace_bytes):
    """Item 1's references.json as rewritten against Grace's: the same keys, every number within
    TABLE_RTOL but the round-off quantities (``REFERENCE_ROUNDOFF``), whose values are returned;
    Grace's put back. Returns (the largest other difference, {key: Grace, here, difference})."""
    path = Path(out) / REFERENCES
    old, new = dict(_flat(json.loads(grace_bytes))), dict(_flat(json.loads(path.read_text())))
    if set(old) != set(new):
        sys.exit(f'{REFERENCES}: the rewritten keys differ from Grace\'s: {sorted(set(old) ^ set(new))[:5]}')
    path.write_bytes(grace_bytes)
    recorded = {k: {'grace': old[k], 'assembly': new[k], 'relative_difference': cell_diff(str(old[k]), str(new[k]))}
                for k in old if k.endswith(REFERENCE_ROUNDOFF)}
    worst, worst_at = 0.0, None
    for k in old:
        d = cell_diff(str(old[k]), str(new[k]))
        if k not in recorded and d > worst:
            worst, worst_at = d, k
    if worst > TABLE_RTOL:
        sys.exit(f"{REFERENCES} differs from Grace's by {worst:.1e} relative ({worst_at})")
    return worst, recorded


def rerun_summaries(out, reference_dir, package3, advisor, log):
    """The summaries with the advisor's files; Grace's tables must hold. Writes the comparisons."""
    import experiments.p5_1_weights as item1
    import experiments.p5_2_nlls as item2
    import experiments.p5_4_pb as item4
    advisor = Path(advisor)
    before = tables(out, GRACE_FOLDERS)
    refs = (Path(out) / REFERENCES).read_bytes()
    item1.summarize(out, reference_dir, expected_bratu=advisor / 'expected_weight_bratu.json',
                    expected_bl=advisor / 'expected_weight_bl.json')
    item2.summarize(out, advisor / 'expected_nlls_liln.json')
    item4.summarize(out, advisor / 'expected_rf_pb.json')
    new = {p for d in GRACE_FOLDERS for p in (Path(out) / d).rglob('*.csv')} - set(before)
    _, recorded = compare_tables(before, log, TABLE_RTOL, ROUNDOFF_COLUMNS, RECORD)
    worst_refs, refs_recorded = compare_references(out, refs)
    log.append(f"{REFERENCES}: rewritten within {worst_refs:.0e} relative of Grace's, but the Burgers PDE check's "
               f"round-off residual (recorded in {RECORD}); Grace's kept")
    rows = item1.beltrami_beside_delivered(out, package3)
    log.append(f'P3_1_beltrami_certified/beside_delivered.csv: {len(rows)} rows (delivered and Package 5)')
    log.append('written with the advisor\'s files: ' + ', '.join(sorted(
        str(p.relative_to(out)).replace('\\', '/') for p in new)))
    write_lf(Path(out) / RECORD, json.dumps({
        'note': ("The summaries rerun at the assembly (with the advisor's files) recompute delta_P, a least-squares "
                 "projection (gelsy, rcond = eps), on this machine. Every other number equals Grace's to "
                 f"{TABLE_RTOL:g} relative; delta_P and the ratios to it differ by the round-off below. The tables "
                 "are Grace's."),
        'largest_relative_difference_grace_vs_assembly': {f'{t} {c}': d for (t, c), d in sorted(recorded.items())},
        'references_json_roundoff': {
            'note': ("The Burgers PDE check's residual is round-off (about 1e-12, 5e-13 of the equation's terms; "
                     "the check asks 1e-8): it differs between machines in its first digits. The verdict is the "
                     "same."), **refs_recorded}},
        indent=2) + '\n')


def run_checks(out, package2, package3, package4, log):
    from experiments.p5_checks import collect
    summary = collect(out, package2, package3, package4)
    write_lf(Path(out) / 'checks.json', json.dumps(json.loads((Path(out) / 'checks.json').read_text()), indent=2)
             + '\n')
    log.append('checks.json: ' + ', '.join(
        f"{k} {({True: 'passed', False: 'FAILED', None: 'no verdict'})[v]}" for k, v in sorted(summary.items())))
    return summary


def hardware(out, extra):
    captured = read_json(Path(out) / TIMED / 'hardware.json')
    s = captured['scheduler']
    return {'cluster': 'Grace (TAMU HPRC)', 'cpu': captured['cpu'],
            'cpu_source': (f"save_provenance in the timed job {s['SLURM_JOB_ID']} ({s['SLURM_JOB_NAME']}) on "
                           f"{captured['cpu']['hostname']}, a whole node of partition {s['SLURM_JOB_PARTITION']}: "
                           'it ran every timing of Package 5 (4a, then item 2). The other jobs (class cpu-mem, 24 '
                           'cores and 96 GB of a shared node) ran no timings; their nodes are in jobs. Its full '
                           f'record: {TIMED}/hardware.json'),
            'jobs': job_table(out, extra), 'gpu': None}


def environment(out, runs_commit):
    captured = read_json(Path(out) / TIMED / 'hardware.json')
    return '\n'.join([
        "Package 5's software environment (items 1, 2 and 4), from its jobs' own logs (the preflight's) and the "
        "timed job's record.",
        f'Code: commit {runs_commit} (the bundle, under the source lock of every job).',
        f'Modules (scripts/cluster/env.sh): {modules()}; the venv $SCRATCH/lilq-run/venv first on PYTHONPATH.',
        blas_line(captured, f'{TIMED}/hardware.json', 'the timed job'),
        'Threads: OMP, OpenBLAS and MKL set to the cores the job holds (48 for the timed job, a whole node; 24 for',
        '  the cpu-mem class: the references and the Beltrami runs). Item 1 and item 4 run six processes side by',
        '  side at 4 threads each (each job log\'s first line reports the 24 the job holds).',
        '',
        'The preflight log (slurm_logs/lilq-p5-preflight.*.out):',
        *('  ' + l for l in preflight_lines(out, 'lilq-p5-preflight.*.out')),
        '',
        "The full record (numpy's and scipy's build configuration, the BLAS thread pools, the packages),",
        f'written by save_provenance in the timed job: {TIMED}/environment.txt.',
        ''])


def add_retime(out, retime, log):
    """The re-timed run (stage p5t, ``RETIME``): copied as Grace wrote it, beside the timed job's run.
    The re-time changes the clock only, so its run must be the timed job's: the same counts and
    status, the same errors and coefficients within ``TABLE_RTOL`` of their scale (else this stops).
    Writes ``RETIME_TABLE`` (both times, per iteration, and the warm-up) and returns its row and
    the re-time job's record."""
    import numpy as np
    from experiments.p5_2_nlls import WARMUP_IT
    out, retime = Path(out), Path(retime)
    runs, _ = runs_and_head(retime)
    dest = out / RETIME
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(retime, dest)
    log.append(f"copied {RETIME}/ (the re-time, commit {runs['commit'][:7]})")
    old, new = read_json(out / RETIMED / 'result.json'), read_json(dest / RETIMED / 'result.json')
    for k in ('problem', 'p', 'P', 'solver', 'iterations', 'evaluations', 'restarts', 'status', 'timing_counts'):
        if old[k] != new[k]:
            sys.exit(f'{RETIME}/{RETIMED}: {k} is {new[k]!r}, the timed job had {old[k]!r}')
    for k in ('norm_N', 'eL2', 'delta_eL2'):
        if abs(new[k] - old[k]) > TABLE_RTOL * abs(old[k]):
            sys.exit(f'{RETIME}/{RETIMED}: {k} is {new[k]!r}, the timed job had {old[k]!r}')
    b_old, b_new = np.load(out / RETIMED / 'beta.npy'), np.load(dest / RETIMED / 'beta.npy')
    beta_diff = float(np.abs(b_new - b_old).max() / np.abs(b_old).max())
    if beta_diff > TABLE_RTOL:
        sys.exit(f'{RETIME}/{RETIMED}: the coefficients differ from the timed job\'s by {beta_diff:.1e} of their scale')
    captured = read_json(dest / TIMED / 'hardware.json')
    s = captured['scheduler']
    it = new['iterations']
    row = {'problem': new['problem'], 'p': new['p'], 'solver': new['solver'], 'iterations': it,
           'time_s_timed_job': old['time_s'], 'ms_per_iteration_timed_job': 1e3 * old['time_s'] / it,
           'time_s_retime': new['time_s'], 'ms_per_iteration_retime': 1e3 * new['time_s'] / it,
           'warmup_s_retime': new['warmup_s'], 'warmup_iterations': WARMUP_IT,
           'retime_job': s['SLURM_JOB_ID'], 'retime_node': captured['cpu']['hostname'],
           'coefficients_relative_difference': beta_diff}
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(row), lineterminator='\n')
    w.writeheader()
    w.writerow(row)
    write_lf(out / RETIME_TABLE, buf.getvalue())
    log.append(f"{RETIME_TABLE}: {new['problem']} p = {new['p']} {new['solver']}: {old['time_s']:.2f} s in the timed "
               f"job, {new['time_s']:.3f} s re-timed after a {new['warmup_s']:.1f} s warm-up of {WARMUP_IT} iterations; "
               f'the same {it} iterations, coefficients within {beta_diff:.0e}')
    job = {'job': s['SLURM_JOB_ID'], 'name': s['SLURM_JOB_NAME'], 'node': captured['cpu']['hostname'],
           'partition': s['SLURM_JOB_PARTITION'], 'whole_node': captured['scheduler'].get('holds_whole_node'),
           'commit': runs['commit'], 'record': f'{RETIME}/{TIMED}/hardware.json'}
    return row, job


def add_a2(out, a2, reference_dir, package2, package4, log, a2_logs=()):
    """Addendum 2 (stage p5a), as Grace wrote it: its runs, summary and checks to ``P5_A2_initial_profile/``
    (the advisor's Section 6 path), the stage's own records (``COMMIT``, ``provenance/``, ``slurm_logs/``,
    ``sacct.txt``, ...) to ``A2``, with ``a2_logs`` (job logs the tarball lacks: it is written before its
    own job ends) added to its ``slurm_logs``. Every run of ``p5_a2_plateau.runs()`` must have its files
    (else this stops). Section 3a's update histories of the delivered runs are written here
    (post-processing, into the item 1 run folders and ``P5_A2_initial_profile/updates_package4``). The
    summary is rerun on Grace's stored iterates: its tables and JSON files must equal Grace's within
    ``TABLE_RTOL``, but delta_P's (round-off, recorded in ``A2_RECORD``); Grace's are kept.
    Returns ({'K1', 'K2'}: Grace's verdicts) and the job's record."""
    import experiments.p5_a2_plateau as a2m
    out, a2 = Path(out), Path(a2)
    runs, _ = runs_and_head(a2)
    item, dest = out / a2m.ITEM, out / A2
    for d in (item, dest):
        if d.exists():
            shutil.rmtree(d)
    shutil.copytree(a2 / a2m.ITEM, item)
    shutil.copytree(a2, dest, ignore=lambda folder, names: [a2m.ITEM] if Path(folder) == a2 else [])
    log.append(f"copied {a2m.ITEM}/ and the stage's records to {A2}/ (Addendum 2, commit {runs['commit'][:7]})")
    for f in a2_logs:
        (dest / 'slurm_logs').mkdir(exist_ok=True)
        shutil.copy2(f, dest / 'slurm_logs' / Path(f).name)
        log.append(f'{A2}/slurm_logs/{Path(f).name}: added (the tarball was written before its job ended)')
    names = [a2m.run_name(*r) for r in a2m.runs()]
    missing = [f'{n}/{f}' for n in names for f in A2_RUN_FILES if not (item / n / f).exists()]
    if missing:
        sys.exit(f'{a2m.ITEM}: {len(missing)} run files missing, e.g. {missing[:3]}')
    before = tables(out, [a2m.ITEM])
    counts = a2m.all_updates(out, package4)
    log.append(f'Section 3a, the update histories of delivered runs (updates_by_iterate.csv): {counts}')
    grace_json = {name: (item / name).read_bytes() for name in A2_JSON}
    a2m.summarize(out, reference_dir, out, Path(package2) / A2_DELIVERED)
    _, recorded = compare_tables(before, log, TABLE_RTOL, ROUNDOFF_COLUMNS, A2_RECORD)
    worst, worst_at = 0.0, None
    for name, grace_bytes in grace_json.items():
        old, new = dict(_flat(json.loads(grace_bytes))), dict(_flat(json.loads((item / name).read_text())))
        (item / name).write_bytes(grace_bytes)
        if set(old) != set(new):
            sys.exit(f"{a2m.ITEM}/{name}: the rerun's keys differ from Grace's: {sorted(set(old) ^ set(new))[:5]}")
        for k in old:
            d = cell_diff(str(old[k]), str(new[k]))
            if ROUNDOFF_COLUMNS(k):
                recorded[(name, k)] = max(recorded.get((name, k), 0.0), d)
            elif d > worst:
                worst, worst_at = d, f'{name} {k}'
    if worst > TABLE_RTOL:
        sys.exit(f"{a2m.ITEM}: the rerun summary differs from Grace's by {worst:.1e} relative ({worst_at})")
    log.append(f"{a2m.ITEM}: the summary rerun on Grace's iterates; terminal.csv, stretches.csv and the JSON files within "
               f"{worst:.0e} relative of Grace's but delta_P's (up to {max(recorded.values(), default=0.0):.1e}, "
               f"{A2_RECORD}); Grace's kept")
    write_lf(out / A2_RECORD, json.dumps({
        'note': ("Addendum 2's summary rerun at the assembly on Grace's stored iterates. delta_P (a least-squares "
                 "projection) is recomputed on this machine and differs by the round-off below; every other number "
                 f"equals Grace's to {TABLE_RTOL:g} relative. The files are Grace's."),
        'largest_relative_difference_grace_vs_assembly': {f'{t} {c}': d for (t, c), d in sorted(recorded.items())}},
        indent=2) + '\n')
    checks = json.loads(grace_json['checks.json'])
    verdict = {k: checks[k]['pass'] for k in ('K1', 'K2')}
    log.append(f"{a2m.ITEM}: K1 {'passed' if verdict['K1'] else 'FAILED'}, K2 {'passed' if verdict['K2'] else 'FAILED'} "
               '(Grace\'s checks.json)')
    captured = read_json(dest / 'provenance' / 'a2' / 'hardware.json')
    s = captured['scheduler']
    job = {'job': s['SLURM_JOB_ID'], 'name': s['SLURM_JOB_NAME'], 'node': captured['cpu']['hostname'],
           'partition': s['SLURM_JOB_PARTITION'], 'commit': runs['commit'], 'record': f'{A2}/provenance/a2/hardware.json'}
    return verdict, job


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assemble(grace, out, reference_dir, package2, package3, package4, advisor, sacct_extra=None, retime=None,
             a2=None, a2_logs=()):
    grace, out = Path(grace), Path(out)
    runs, head = runs_and_head(grace)
    out.mkdir(parents=True, exist_ok=True)
    log = log_start(5, grace, runs, head)
    copy_grace(grace, out, GRACE_FOLDERS, GRACE_FILES, log)
    rerun_summaries(out, reference_dir, package3, advisor, log)
    summary = run_checks(out, package2, package3, package4, log)
    retime_job = None
    if retime:
        _, retime_job = add_retime(out, retime, log)
    elif (out / RETIME).exists():
        shutil.rmtree(out / RETIME)
    a2_job = None
    if a2:
        a2_verdict, a2_job = add_a2(out, a2, reference_dir, package2, package4, log, a2_logs)
        summary.update(a2_verdict)
    else:
        import experiments.p5_a2_plateau as a2m
        for name in (A2, a2m.ITEM, A2_RECORD):          # Addendum 2's folder and Section 3a's output, if any
            if (out / name).is_dir():
                shutil.rmtree(out / name)
            elif (out / name).exists():
                (out / name).unlink()
    rows, total, cols = su_rows(out, sacct_extra)
    write_su(out, rows, cols)
    log.append(f"su.csv: {len(rows)} jobs, {total:.1f} SU charged at Grace's rates"
               + ('' if sacct_extra else "; the report job is not in it (it ran after its tarball's table: "
                                         'pass its sacct line with --sacct-extra)'))
    if retime_job and retime_job['job'] not in {r['job_id'] for r in rows}:
        log.append(f"su.csv: the re-time job {retime_job['job']} is not in it (pass its sacct line with --sacct-extra)")
    if a2_job and a2_job['job'] not in {r['job_id'] for r in rows}:
        log.append(f"su.csv: the Addendum 2 job {a2_job['job']} is not in it (pass its sacct line with --sacct-extra)")
    hw = hardware(out, extra_rows(out, rows))
    if retime_job:
        hw['retime'] = retime_job
    if a2_job:
        hw['addendum2'] = a2_job
    write_lf(out / 'hardware.json', json.dumps(hw, indent=2) + '\n')
    write_lf(out / 'environment.txt', environment(out, runs['commit']))
    write_lf(out / 'DECISIONS_package5.md', decisions(5))
    write_provenance(out, runs, head, (
        "Every run, table and log is Grace's, at the runs' commit. The comparisons with the advisor's files and "
        "with the delivered Beltrami rows, and checks.json (C1 to C3, against Packages 2, 3 and 4 as delivered), "
        "are written at the assembly on Grace's files."
        + (f" {RETIME}/ is the re-time of one item 2 run (stage p5t, its own commit lock), as Grace wrote it; "
           f"{RETIME_TABLE} sets its time beside the timed job's." if retime_job else '')
        + (f" P5_A2_initial_profile/ holds Addendum 2's plateau runs as Grace wrote them (stage p5a, its "
           f"own commit lock; the stage's records in {A2}/); its summary was rerun at the assembly on Grace's "
           f"iterates ({A2_RECORD}). Section 3a's update histories "
           "(updates_by_iterate.csv in the delivered run folders, P5_A2_initial_profile/updates_package4/) are "
           "written at the assembly from the stored coefficients." if a2_job else '')), log,
        advisor_files={name: sha256(Path(advisor) / name) for name in ADVISOR_FILES},
        **({'retime': {'commit': retime_job['commit'], 'job': retime_job['job']}} if retime_job else {}),
        **({'addendum2': {'commit': a2_job['commit'], 'job': a2_job['job']}} if a2_job else {}))
    write_lf(out / 'assembly_log.txt', '\n'.join(log) + '\n')
    print('\n'.join(log))
    return summary, total


def main(argv=None):
    ap = argparse.ArgumentParser(description="Package 5's results folder (items 1, 2, 4), from Grace's tarball.")
    ap.add_argument('--grace', required=True, help='the extracted package5/ folder')
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference-dir', required=True, help="Package 2 stage 2's reference folder")
    ap.add_argument('--package2', required=True, help="Package 2 stage 2's results root (C1)")
    ap.add_argument('--package3', required=True, help="Package 3's results root (C1; the delivered Beltrami rows)")
    ap.add_argument('--package4', required=True, help="Package 4's results root (C2, C3)")
    ap.add_argument('--advisor', required=True, help="the folder of the advisor's expected_*.json")
    ap.add_argument('--sacct-extra', default=None, help='sacct -X -P -n lines of jobs the tarball lacks')
    ap.add_argument('--retime', default=None, help='the extracted package5_retime/ folder (stage p5t)')
    ap.add_argument('--a2', default=None, help="the extracted package5_a2/ folder (stage p5a, Addendum 2)")
    ap.add_argument('--a2-logs', nargs='*', default=(), help="Addendum 2's job logs its tarball lacks")
    a = ap.parse_args(argv)
    summary, _ = assemble(a.grace, a.out, a.reference_dir, a.package2, a.package3, a.package4, a.advisor,
                          a.sacct_extra, a.retime, a.a2, a.a2_logs)
    return 0 if all(v is not False for v in summary.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
