"""
Package 4: the results folder, from Grace's tarball
====================================================

Assembles ``package4_results/`` in the layout of the advisor's instructions of
12 November 2026 (Section 7), from the extracted ``results/package4.tar.gz``
of the report job. The helpers every package's assembly shares are in
``assemble_common.py``.

- **The run folders, as Grace wrote them** at the runs' commit (``COMMIT``):
  ``P4_1a_burgers_gauss/`` to ``P4_2_pb_inclusions/``, ``P4_checks/`` (M0),
  ``provenance/timed/`` (the timed job's record), ``slurm_logs/``,
  ``sacct.txt``, ``su_per_job.csv``, ``report_log.txt``.
- **The checks with the advisor's expected values.** The cluster had no
  ``expected_burgers.json`` or ``expected_pb.json``, so it could not evaluate
  M1, M8, and M6's values at the five points. The assembly reruns both items'
  summaries on Grace's files with them, then ``p4_checks collect``:
  - Grace's ``checks.json``, ``checks_item1.json`` and ``checks_item2.json``
    are kept as ``*_grace.json``; every verdict Grace gave must come out the
    same, or the assembly stops;
  - the summaries also rewrite the tables (``terminal.csv`` and the others)
    from Grace's logs. Those tables stay Grace's: the assembly compares the
    rewritten ones with them and stops if a number differs by more than
    ``TABLE_RTOL`` (relative). A difference below it cannot change a digit the
    report gives (it gives four).
- **The records of Section 7:**
  - ``checks.json``: M0 to M9 (``p4_checks collect``);
  - ``su.csv``: every job at Grace's rates, with the report job's own row, which
    its tarball cannot hold (``--sacct-extra``: ``sacct -X -P -n`` lines);
  - ``hardware.json`` and ``environment.txt``: each job's node, threads and
    allocation, the CPU and BLAS from the timed job's ``save_provenance``
    record, and the preflight's lines;
  - ``DECISIONS_package4.md``: the Package 4 entries of ``DECISIONS.md``;
  - ``provenance.json``: the runs' commit and tree hash, this commit, the
    commits between them, and the SHA-256 of the advisor's two files (they
    are his, and are not copied).

Refuses a Grace tree whose commit is not an ancestor of this checkout's HEAD.
Can be run again: it rewrites what it writes. Exits 0 when the assembly is
complete; the verdicts are in ``checks.json`` (M9's spread is a recorded
deviation, not a failure of the assembly).

Usage::

    python experiments/p4_assemble.py --grace <extracted>/package4 --out ../package4_results \\
        --reference-dir <package2_stage2>/reference --expected-burgers expected_burgers.json \\
        --expected-pb expected_pb.json --sacct-extra <file>
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.assemble_common import compare_tables as _compare_tables  # noqa: E402
from experiments.assemble_common import (blas_line, copy_grace, decisions, extra_rows, job_table,  # noqa: E402
                                         log_start, modules, preflight_lines, read_csv, read_json, runs_and_head,
                                         su_rows, tables, write_lf, write_provenance, write_su)

GRACE_FOLDERS = ('P4_1a_burgers_gauss', 'P4_1b_burgers_enrichment', 'P4_1c_burgers_constants', 'P4_2_pb_inclusions',
                 'P4_checks', 'provenance', 'slurm_logs')
GRACE_FILES = ('COMMIT', 'sacct.txt', 'su_per_job.csv', 'report_log.txt')
CHECK_FILES = ('checks.json', 'checks_item1.json', 'P4_2_pb_inclusions/checks_item2.json')
TIMED = 'provenance/timed'
TABLE_RTOL = 1e-8
ROUNDOFF_COLUMNS = (lambda column: 'delta' in column)   # delta_P and the ratios to it


def _grace_name(rel):
    return rel[:-len('.json')] + '_grace.json'


def keep_grace_checks(grace, out, log):
    """Grace's three checks files, as ``*_grace.json`` beside the ones the assembly writes."""
    for rel in CHECK_FILES:
        shutil.copy2(Path(grace) / rel, out / _grace_name(rel))
        log.append(f'kept {rel} as {_grace_name(rel)}')


def _tables(out):
    """Every CSV the summaries rewrite: Grace's bytes, by path."""
    return tables(out, GRACE_FOLDERS)


def compare_tables(before, log):
    """The rewritten tables against Grace's (``assemble_common.compare_tables``); delta_P's columns
    are recorded in ``delta_P_across_machines.json``."""
    return _compare_tables(before, log, TABLE_RTOL, ROUNDOFF_COLUMNS, 'delta_P_across_machines.json')


def delta_P_record(out, rewritten, recorded, expected_pb):
    """delta_P of item 2 (lilq/terminal.csv, seed 0) from Grace's run, this machine's rerun of the
    summary and the advisor's expected_pb.json ("best"), by P; and the largest relative difference
    of every delta_P column between Grace's tables and the rerun."""
    t = out / 'P4_2_pb_inclusions' / 'lilq' / 'terminal.csv'
    grace = {r['P']: r for r in read_csv(t) if r['seed'] == '0'}
    here = {r['P']: r for r in rewritten[t] if r['seed'] == '0'}
    best = json.loads(Path(expected_pb).read_text()).get('best', {})
    by_P = {}
    for P in sorted(grace, key=int):
        row = {'grace': {k: float(grace[P][k]) for k in ('delta_L2', 'delta_H2')},
               'assembly_machine': {k: float(here[P][k]) for k in ('delta_L2', 'delta_H2')}}
        if P in best:
            row['advisor'] = dict(zip(('delta_L2', 'delta_H2'), best[P]))
        sources = list(row.values())
        for k in ('delta_L2', 'delta_H2'):
            vals = [v[k] for v in sources]
            row[f'{k}_spread'] = (max(vals) - min(vals)) / min(vals)
        by_P[P] = row
    write_lf(out / 'delta_P_across_machines.json', json.dumps({
        'note': ("delta_P is the error of the best approximation in the basis, a least-squares projection (gelsy, "
                 "rcond = eps). Item 2's basis on the domain with holes is ill-conditioned, and at P >= 484 the "
                 "projection's round-off reaches the fourth digit: Grace, the assembly's machine and the advisor's "
                 "expected_pb.json give three values. The tables are Grace's; this records the spread."),
        'item2_by_P': by_P,
        'largest_relative_difference_grace_vs_assembly': {f'{tab} {col}': d
                                                          for (tab, col), d in sorted(recorded.items())}},
        indent=2) + '\n')
    return by_P


def rerun_checks(out, reference_dir, expected_burgers, expected_pb, log):
    """Both items' summaries with the advisor's files, then ``collect``; Grace's verdicts must hold."""
    import experiments.p4_1_burgers_gauss as item1
    import experiments.p4_2_pb_inclusions as item2
    from experiments.p4_checks import collect
    before = _tables(out)
    item1.summarize(out, reference_dir, expected_burgers)
    item2.summarize(out, expected_pb)
    summary = collect(out)
    rewritten, recorded = compare_tables(before, log)
    delta_P_record(out, rewritten, recorded, expected_pb)
    for rel in CHECK_FILES:                               # LF, as Grace's
        p = out / rel
        write_lf(p, json.dumps(json.loads(p.read_text()), indent=2) + '\n')
    grace = read_json(out / _grace_name('checks.json'))['summary']
    changed = {k: (v, summary.get(k)) for k, v in grace.items() if v is not None and summary.get(k) != v}
    if changed:
        sys.exit(f"verdicts differ from Grace's: {changed}")
    log.append('checks.json: ' + ', '.join(
        f"{k} {({True: 'passed', False: 'FAILED', None: 'no verdict'})[v]}" for k, v in sorted(summary.items()))
        + f" (Grace's: {', '.join(f'{k} {v}' for k, v in grace.items())})")
    return summary


def hardware(out, extra):
    captured = read_json(out / TIMED / 'hardware.json')
    s = captured['scheduler']
    return {'cluster': 'Grace (TAMU HPRC)', 'cpu': captured['cpu'],
            'cpu_source': (f"save_provenance in the timed job {s['SLURM_JOB_ID']} ({s['SLURM_JOB_NAME']}) on "
                           f"{captured['cpu']['hostname']}, a whole node of partition {s['SLURM_JOB_PARTITION']}: "
                           f'it ran M0 and every timing of Package 4 (1a, 1b, 2a). The other jobs (class cpu, 24 '
                           f'cores of a shared node) ran no timings; their nodes are in jobs. Its full record: '
                           f'{TIMED}/hardware.json'),
            'jobs': job_table(out, extra), 'gpu': None}


def environment(out, runs_commit):
    captured = read_json(out / TIMED / 'hardware.json')
    return '\n'.join([
        "Package 4's software environment, from its jobs' own logs (the preflight's) and the timed job's record.",
        f'Code: commit {runs_commit} (the bundle, under the source lock of every job).',
        f'Modules (scripts/cluster/env.sh): {modules()}; the venv $SCRATCH/lilq-run/venv first on PYTHONPATH.',
        blas_line(captured, f'{TIMED}/hardware.json', 'the timed job'),
        'Threads: OMP, OpenBLAS and MKL set to the cores the job holds (48 for the timed job, a whole node;',
        '  24 for the cpu class). Item 2f runs its three seeds side by side at 8 threads each.',
        '',
        'The preflight log (slurm_logs/lilq-p4-preflight.*.out):',
        *('  ' + l for l in preflight_lines(out, 'lilq-p4-preflight.*.out')),
        '',
        "The full record (numpy's and scipy's build configuration, the BLAS thread pools, the packages),",
        f'written by save_provenance in the timed job: {TIMED}/environment.txt.',
        ''])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assemble(grace, out, reference_dir, expected_burgers, expected_pb, sacct_extra=None):
    grace, out = Path(grace), Path(out)
    runs, head = runs_and_head(grace)
    out.mkdir(parents=True, exist_ok=True)
    log = log_start(4, grace, runs, head)
    copy_grace(grace, out, GRACE_FOLDERS, GRACE_FILES, log)
    keep_grace_checks(grace, out, log)
    summary = rerun_checks(out, reference_dir, expected_burgers, expected_pb, log)
    rows, total, cols = su_rows(out, sacct_extra)
    write_su(out, rows, cols)
    log.append(f"su.csv: {len(rows)} jobs, {total:.1f} SU charged at Grace's rates"
               + ('' if sacct_extra else "; the report job is not in it (it ran after its tarball's table: "
                                         'pass its sacct line with --sacct-extra)'))
    write_lf(out / 'hardware.json', json.dumps(hardware(out, extra_rows(out, rows)), indent=2) + '\n')
    write_lf(out / 'environment.txt', environment(out, runs['commit']))
    write_lf(out / 'DECISIONS_package4.md', decisions(4))
    write_provenance(out, runs, head, (
        "Every run, table and log is Grace's, at the runs' commit. checks.json and the two items' checks files "
        "are re-evaluated at the assembly on Grace's files, with the advisor's expected values (M1, M6, M8); "
        "Grace's are kept as *_grace.json."), log,
        advisor_files={Path(expected_burgers).name: sha256(expected_burgers),
                       Path(expected_pb).name: sha256(expected_pb)})
    write_lf(out / 'assembly_log.txt', '\n'.join(log) + '\n')
    print('\n'.join(log))
    return summary, total


def main(argv=None):
    ap = argparse.ArgumentParser(description="Package 4's results folder, from Grace's tarball.")
    ap.add_argument('--grace', required=True, help='the extracted package4/ folder')
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference-dir', required=True, help="Package 2's reference folder (burgers_cole_hopf.npz)")
    ap.add_argument('--expected-burgers', required=True, help="the advisor's expected_burgers.json")
    ap.add_argument('--expected-pb', required=True, help="the advisor's expected_pb.json")
    ap.add_argument('--sacct-extra', default=None, help='sacct -X -P -n lines of jobs the tarball lacks')
    a = ap.parse_args(argv)
    assemble(a.grace, a.out, a.reference_dir, a.expected_burgers, a.expected_pb, a.sacct_extra)
    return 0


if __name__ == '__main__':
    sys.exit(main())
