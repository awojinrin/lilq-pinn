"""
What every package's assembly shares
====================================

A package's *assembly* turns the extracted tarball of its last Grace job into
the results folder its instructions ask for. ``p3_assemble.py`` (Package 3) and
``p4_assemble.py`` (Package 4) each keep only what is particular to their
package; this module holds the rest:

- the commit check: the runs' commit (Grace's ``COMMIT``) must be an ancestor
  of the checkout's HEAD;
- copying Grace's folders and files;
- files written with LF on every platform, and CSVs rewritten with Grace's own
  line ending;
- ``su.csv``: Grace's ``su_per_job.csv``, plus jobs its tarball cannot hold
  (``sacct -X -P -n`` lines);
- each job's node, threads and allocation, from its log and ``sacct.txt``;
- the BLAS line, the preflight lines and the modules for ``environment.txt``;
- the package's entries of ``DECISIONS.md``;
- ``provenance.json``: the runs' commit, the assembly's, and those between.
"""

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
LOG_HEAD = re.compile(r'^job (\d+) on (\S+): (\d+) threads, (\d+) cores usable, OverSubscribe=(\S+)')
PREFLIGHT_KEEP = ('job ', 'torch ', 'numpy ', 'provenance lock', 'inputs OK', 'K1 ', 'K2 ', 'PREFLIGHT')


def git(*args):
    out = subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_json(path):
    return json.loads(Path(path).read_text())


def write_lf(path, text):
    Path(path).write_bytes(text.encode())                 # LF on every platform


def read_csv(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def write_rows(path, rows):
    """Rewrite a CSV of Grace's with its own line ending (the csv module's CRLF), so a diff shows
    only the values that changed."""
    ending = '\r\n' if b'\r\n' in Path(path).read_bytes() else '\n'
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator=ending)
    w.writeheader()
    w.writerows(rows)
    write_lf(path, buf.getvalue())


def runs_and_head(grace):
    """Grace's ``COMMIT`` and this checkout's HEAD; exits if the runs' commit is not an ancestor."""
    runs = read_json(Path(grace) / 'COMMIT')
    head = git('rev-parse', 'HEAD')
    if head is None:
        sys.exit('the assembly needs a git checkout (it records the commits)')
    if subprocess.run(['git', 'merge-base', '--is-ancestor', runs['commit'], head], cwd=REPO).returncode != 0:
        sys.exit(f"the runs' commit {runs['commit'][:7]} is not an ancestor of HEAD {head[:7]}")
    return runs, head


def log_start(package, grace, runs, head):
    return [f'Package {package} assembly, {datetime.datetime.now(datetime.timezone.utc).isoformat()}',
            f'from {Path(grace).name} (runs at {runs["commit"][:7]}), assembled at {head[:7]}']


def copy_grace(grace, out, folders, files, log):
    grace, out = Path(grace), Path(out)
    for name in folders:
        if (out / name).exists():
            shutil.rmtree(out / name)
        shutil.copytree(grace / name, out / name)
        log.append(f'copied {name}/')
    for name in files:
        shutil.copy2(grace / name, out / name)
        log.append(f'copied {name}')


def su_rows(out, sacct_extra):
    """Grace's ``su_per_job.csv`` rows, plus the jobs of ``sacct_extra`` (``sacct -X -P -n`` lines of
    jobs the tarball lacks, as its own report job). Returns (rows, total SU, columns)."""
    wave_report = load_module('wave_report', REPO / 'scripts' / 'cluster' / 'wave_report.py')
    with open(Path(out) / 'su_per_job.csv') as fh:
        rows = list(csv.DictReader(fh))
    have = {r['job_id'] for r in rows}
    if sacct_extra:
        for r in wave_report.su_rows(Path(sacct_extra).read_text()):
            if r['job_id'] not in have:
                rows.append({k: r[k] for k in wave_report.SU_COLUMNS}
                            | {'note': 'not in the last tarball: from sacct after it ended'})
    total = sum(float(r['su'] or 0) for r in rows)
    return rows, total, wave_report.SU_COLUMNS


def write_su(out, rows, cols):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator='\n', extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
    write_lf(Path(out) / 'su.csv', buf.getvalue())


def extra_rows(out, rows):
    """The rows of ``su_rows`` that are not in Grace's ``su_per_job.csv``."""
    have = {r['job_id'] for r in csv.DictReader(io.StringIO((Path(out) / 'su_per_job.csv').read_text()))}
    return [r for r in rows if r['job_id'] not in have]


def job_table(out, extra=()):
    """Per job: the node, the threads and cores env.sh reported (the log's first lines), and the
    allocation (``sacct.txt``; jobs in ``extra`` from their sacct rows)."""
    out = Path(out)
    sacct = {}
    for line in (out / 'sacct.txt').read_text().splitlines()[2:]:
        parts = line.split()
        if parts and parts[0].isdigit():
            sacct[parts[0]] = {'name': parts[1], 'partition': parts[2], 'alloc_tres': parts[3], 'elapsed': parts[4]}
    for r in extra:
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
    return jobs


def blas_line(captured, where='provenance_cpu/hardware.json', job='the provenance job'):
    """The BLAS the jobs used: FlexiBLAS and the library it dispatched to, from the thread pools
    the captured record lists (threadpoolctl, in the runs' environment). ``where``: that record;
    ``job``: the job that wrote it."""
    pools = {p['internal_api']: p for p in (captured or {}).get('threads', {}).get('threadpools', [])}
    if 'flexiblas' in pools and 'openblas' in pools:
        o = pools['openblas']
        return (f"BLAS and LAPACK: OpenBLAS {o['version']} ({Path(o['filepath']).name}), through FlexiBLAS "
                f"{pools['flexiblas']['version']}, at {o['num_threads']} threads on a whole node: the thread pools "
                f'loaded in {job} ({where}).')
    return "BLAS and LAPACK: FlexiBLAS 3.4.4 (GCC 13.3.0), through the modules' numpy and scipy."


def preflight_lines(out, pattern):
    """The preflight log's lines that describe the environment (``pattern``: its glob in slurm_logs)."""
    pre = next((Path(out) / 'slurm_logs').glob(pattern)).read_text(errors='replace').splitlines()
    return [l for l in pre if l.startswith(PREFLIGHT_KEEP) or re.match(r'\d+ passed', l)]


def modules():
    env_sh = (REPO / 'scripts' / 'cluster' / 'env.sh').read_text()
    return re.search(r'LILQ_MODULES="\$\{LILQ_MODULES:-([^}]+)\}"', env_sh).group(1)


def decisions(package):
    """The ``DECISIONS.md`` entries whose heading names the package, newest first."""
    text = (REPO / 'DECISIONS.md').read_text()
    parts = re.split(r'(?m)^(?=## \d{4}-\d{2}-\d{2} -- )', text)
    mine = [p.rstrip().rstrip('-').rstrip() for p in parts
            if re.match(rf'## \d{{4}}-\d{{2}}-\d{{2}} -- Package {package}\b', p)]
    return (f'# DECISIONS.md: the Package {package} entries\n\nCopied from the repository\'s `DECISIONS.md` at the '
            f"assembly's commit ({git('rev-parse', '--short', 'HEAD')}), newest first.\n\n---\n\n"
            + '\n\n---\n\n'.join(mine) + '\n')


def write_provenance(out, runs, head, note, log, **extra):
    between = git('log', '--format=%h %s', f"{runs['commit']}..{head}") or ''
    write_lf(Path(out) / 'provenance.json', json.dumps({
        'runs': {'commit': runs['commit'], 'tree_hash': runs['tree_hash'], 'source': runs['source'],
                 'locked_at_utc': runs['locked_at_utc']},
        'assembly': {'commit': head, 'branch': git('rev-parse', '--abbrev-ref', 'HEAD')},
        'commits_between': between.splitlines(), **extra, 'note': note}, indent=2) + '\n')
    log.append(f"provenance.json: {len(between.splitlines())} commit(s) between the runs and the assembly")
