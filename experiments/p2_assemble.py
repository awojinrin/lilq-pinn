"""
The Package 2 results layout (Package 2, Section 12.1)
======================================================

Each stage ran into its own root on Grace (``results/package2_stage1``,
locked to its commit). This copies a stage's results into the layout of
Section 12.1, ``package2_results/``; nothing is recomputed.

``stage1`` copies:

- ``reference/`` (the Bratu and Burgers references written by the stage);
- each LiL-Q rerun, ``P2_12_reference_errors/B_instrumentation/<benchmark>_P<P>_cpu_paper/``,
  to ``P2_12_reference_errors/<benchmark>_P<P>/`` (Section 6.2), every file;
- check C4 (``check_c4.csv``, ``check_c4.json``), the run index, ``hardware.json`` and
  ``environment.txt`` to ``P2_12_reference_errors/``;
- the stage's ``COMMIT`` to ``reference/COMMIT`` and ``P2_12_reference_errors/COMMIT``.

Files already in the output that the stage does not carry (for example
``scalar_reference_errors.csv``) are kept. An existing file that differs is
replaced, and the replacements are listed.

``stage2`` (after downloading ``results/package2_stage2.tar.gz``):

- **The commit.** It refuses to run unless the local checkout is at the
  commit the stage is locked to (``COMMIT``): the summaries below are part of
  the results.
- **The item folders** (``P2_*``) are copied, every file.
- **References.** Stage 2 rebuilt the references Stage 1's errors were
  computed against. Stage 2's go to ``reference/``, because Stage 2's results
  used them; where Stage 1's copy differs, it moves to ``reference/stage1/``,
  and both are compared in ``reference/stage2_vs_stage1.json``. The BL
  references are new.
- **Root files:** ``su_per_job.csv``, ``stage2_sacct.txt``,
  ``stage2_report_log.txt`` and the Slurm logs (``slurm_logs/stage2/``).
- **Hardware.** ``hardware.json`` and ``environment.txt`` are the CPU scaling
  job's: a whole Grace CPU node. Every item folder keeps its own job's;
  ``provenance.json`` says so and names both stages' commits.
- **``code/``:** ``COMMIT_stage1``, ``COMMIT_stage2``, and every file changed
  between ``v2.0.0`` and the stage's commit, at that commit (Section 12.1:
  "commit hash; new scripts").
- **Summaries** on the assembled copy:
  - item 1's work-precision figure (``p2_3_classical.py figures``);
  - item 2's rows, medians and figures, with the L-BFGS reference errors of
    Stage 1 (``p2_8_lm_networks.py summarize``);
  - item 7's scaling table, exponents and check C8'
    (``p2_1_scaling.py summarize``), against ``--package1``;
  - item 2's ``gpu-list``: a configuration named there needs the A100
    reruns before the package is complete;
  - item 5's rows (``p2_15_darcy_hardbc.py summarize``), beside the paper's
    NiL and LiL delta_FV from ``--package1``.
- **FASTER's part** (item 5; ``--faster``, the extracted
  ``package2_stage2_faster.tar.gz``): it must be locked to the same commit
  as Grace's. Its item folder is copied like the others; its ``sacct.txt``,
  ``report_log.txt`` and Slurm logs go to ``stage2_faster_sacct.txt``,
  ``stage2_faster_report_log.txt`` and ``slurm_logs/stage2_faster/``.

  Each step is logged in ``assembly_log.txt``; a failing step is recorded,
  not fatal.

Usage::

    python experiments/p2_assemble.py stage1 --stage <downloaded results/package2_stage1> --out <package2_results>
    python experiments/p2_assemble.py stage2 --stage <downloaded results/package2_stage2> --out <package2_results> \\
        --package1 <package1> [--faster <FASTER's package2_stage2>]
"""

import argparse
import datetime
import filecmp
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

RERUN_RE = re.compile(r'^(?P<bench>[a-z_]+)_P(?P<P>\d+)_cpu_paper$')
STAGE1_FILES = ('check_c4.csv', 'check_c4.json', 'hardware.json', 'environment.txt')


def _copy(src, dst, replaced):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not filecmp.cmp(src, dst, shallow=False):
        replaced.append(dst)
    shutil.copy2(src, dst)


def _copy_tree(src, dst, replaced):
    for f in sorted(p for p in src.rglob('*') if p.is_file()):
        _copy(f, dst / f.relative_to(src), replaced)


def stage1(stage, out):
    """Copy Stage 1 into ``out``; returns ``(copied runs, replaced files)``."""
    stage, out = Path(stage), Path(out)
    replaced, runs = [], []
    _copy_tree(stage / 'reference', out / 'reference', replaced)
    _copy(stage / 'COMMIT', out / 'reference' / 'COMMIT', replaced)
    p212, reruns = stage / 'P2_12_reference_errors', stage / 'P2_12_reference_errors' / 'B_instrumentation'
    for run in sorted(p for p in reruns.iterdir() if p.is_dir()):
        m = RERUN_RE.match(run.name)
        if m:
            name = f"{m['bench']}_P{m['P']}"
            _copy_tree(run, out / 'P2_12_reference_errors' / name, replaced)
            runs.append(name)
    for f in STAGE1_FILES:
        _copy(p212 / f, out / 'P2_12_reference_errors' / f, replaced)
    _copy(reruns / 'runs_index.csv', out / 'P2_12_reference_errors' / 'runs_index.csv', replaced)
    _copy(stage / 'COMMIT', out / 'P2_12_reference_errors' / 'COMMIT', replaced)
    return runs, replaced


STAGE2_REFERENCES_SHARED = ('bratu_ref_p48.npz', 'bratu_ref_p64.npz', 'burgers_cole_hopf.npz')


def _git(*args):
    out = subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def _npz_difference(a, b):
    """The largest absolute difference of the arrays two reference files share."""
    with np.load(a) as x, np.load(b) as y:
        keys = [k for k in x.files if k in y.files and x[k].dtype.kind in 'fiu' and x[k].shape == y[k].shape]
        return {k: float(np.abs(x[k] - y[k]).max()) if x[k].size else 0.0 for k in keys}


def stage2_references(stage, out):
    """Stage 2's references into ``reference/``; Stage 1's copies that differ go
    to ``reference/stage1/``. Returns the comparison record."""
    src, dst = Path(stage) / 'reference', Path(out) / 'reference'
    dst.mkdir(parents=True, exist_ok=True)
    record = {}
    for f in sorted(p for p in src.iterdir() if p.is_file()):
        target = dst / f.name
        if target.exists() and not filecmp.cmp(f, target, shallow=False):
            keep = dst / 'stage1' / f.name
            if not keep.exists():
                keep.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, keep)
            if f.suffix == '.npz':
                record[f.name] = {'max_abs_difference_to_stage1': _npz_difference(f, keep)}
            else:
                record[f.name] = {'differs_from_stage1': True}
        elif target.exists():
            record[f.name] = {'identical_to_stage1': True}
        else:
            record[f.name] = {'new_in_stage2': True}
        shutil.copy2(f, target)
    (dst / 'stage2_vs_stage1.json').write_text(json.dumps(record, indent=2))
    return record


def stage2_code(commit, out, base='v2.0.0'):
    """``code/``: both stages' commits and every file changed since ``base``, at ``commit``."""
    code = Path(out) / 'code'
    code.mkdir(parents=True, exist_ok=True)
    files = (_git('diff', '--name-only', f'{base}..{commit}') or '').splitlines()
    written = []
    for name in files:
        content = subprocess.run(['git', 'show', f'{commit}:{name}'], cwd=REPO, capture_output=True)
        if content.returncode != 0:                 # deleted since base
            continue
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.stdout)
        written.append(name)
    (code / 'FILES_CHANGED_SINCE_v2.0.0.txt').write_text('\n'.join(written) + '\n')
    return written


def _step(cmd, log):
    out = subprocess.run([sys.executable, *cmd], cwd=REPO, capture_output=True, text=True)
    log.append(f"$ python {' '.join(cmd)}\n(exit {out.returncode})\n{out.stdout[-4000:]}{out.stderr[-4000:]}")
    return out


def stage2(stage, out, package1=None, summaries=True, check_commit=True, faster=None):
    """Assemble Stage 2 into ``out``; returns the assembly record."""
    stage, out = Path(stage), Path(out)
    lock = json.loads((stage / 'COMMIT').read_text())
    commit = lock['commit']
    head = _git('rev-parse', 'HEAD')
    if check_commit and head != commit:
        raise SystemExit(f'the local checkout is at {head}, the stage is locked to {commit}: '
                         f'check out {commit} (git checkout {commit[:7]}) and run again')
    out.mkdir(parents=True, exist_ok=True)
    replaced, items = [], []
    for item in sorted(p for p in stage.iterdir() if p.is_dir() and p.name.startswith('P2_')):
        _copy_tree(item, out / item.name, replaced)
        items.append(item.name)
    refs = stage2_references(stage, out)
    for name, target in (('su_per_job.csv', 'su_per_job.csv'), ('sacct.txt', 'stage2_sacct.txt'),
                         ('report_log.txt', 'stage2_report_log.txt'), ('COMMIT', 'code/COMMIT_stage2')):
        if (stage / name).exists():
            _copy(stage / name, out / target, replaced)
    if (stage / 'slurm_logs').is_dir():
        _copy_tree(stage / 'slurm_logs', out / 'slurm_logs' / 'stage2', replaced)
    faster_commit = None
    if faster is not None:
        faster = Path(faster)
        faster_commit = json.loads((faster / 'COMMIT').read_text())['commit']
        if faster_commit != commit:
            raise SystemExit(f"FASTER's part is locked to {faster_commit}, Grace's to {commit}: "
                             'Stage 2 must come from one commit')
        for item in sorted(p for p in faster.iterdir() if p.is_dir() and p.name.startswith('P2_')):
            _copy_tree(item, out / item.name, replaced)
            items.append(item.name)
        for name, target in (('sacct.txt', 'stage2_faster_sacct.txt'), ('report_log.txt', 'stage2_faster_report_log.txt')):
            if (faster / name).exists():
                _copy(faster / name, out / target, replaced)
        if (faster / 'slurm_logs').is_dir():
            _copy_tree(faster / 'slurm_logs', out / 'slurm_logs' / 'stage2_faster', replaced)
    stage1_commit = out / 'reference' / 'stage1' / 'COMMIT'
    stage1_commit = stage1_commit if stage1_commit.exists() else out / 'P2_12_reference_errors' / 'COMMIT'
    if stage1_commit.exists():
        _copy(stage1_commit, out / 'code' / 'COMMIT_stage1', replaced)
    hardware = sorted((out / 'P2_1_scaling').glob('provenance_cpu*/hardware.json'))
    if hardware:
        _copy(hardware[0], out / 'hardware.json', replaced)
        if (hardware[0].parent / 'environment.txt').exists():
            _copy(hardware[0].parent / 'environment.txt', out / 'environment.txt', replaced)
    code_files = stage2_code(commit, out)
    log = [f"Stage 2 assembled {datetime.datetime.now(datetime.timezone.utc).isoformat()} at {commit}"]
    gpu_list = None
    if summaries:
        s1 = out / 'P2_12_reference_errors' / 'scalar_reference_errors.csv'
        bl = out / 'G1_release' / 'bl_reference_errors_networks.csv'
        if (out / 'P2_3_classical').is_dir():
            _step(['experiments/p2_3_classical.py', 'figures', '--out', str(out), '--package1', str(package1),
                   '--lilq-errors', str(s1)], log)
        if (out / 'P2_8_lm_networks').is_dir():
            _step(['experiments/p2_8_lm_networks.py', 'summarize', '--out', str(out), '--package1', str(package1),
                   '--lbfgs-errors', str(s1), str(bl)], log)
            g = _step(['experiments/p2_8_lm_networks.py', 'gpu-list', '--out', str(out)], log)
            gpu_list = g.stdout.split() if g.returncode == 0 else None
        if (out / 'P2_15_darcy_hardbc').is_dir():
            _step(['experiments/p2_15_darcy_hardbc.py', 'summarize', '--out', str(out)]
                  + (['--package1', str(package1)] if package1 else []), log)
        if (out / 'P2_1_scaling').is_dir():
            _step(['experiments/p2_1_scaling.py', 'summarize', '--out', str(out), '--package1', str(package1)], log)
    (out / 'assembly_log.txt').write_text('\n\n'.join(log) + '\n')
    record = {'stage2_commit': commit, 'stage2_lock': lock,
              'stage1_commit': json.loads(stage1_commit.read_text())['commit'] if stage1_commit.exists() else None,
              'assembled_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'items': items, 'references': refs, 'replaced': [str(p.relative_to(out)) for p in replaced],
              'code_files': len(code_files),
              'hardware': 'hardware.json and environment.txt are the CPU scaling job\'s (a whole Grace CPU node); '
                          'every item folder holds its own job\'s',
              'item_2_gpu_list': gpu_list, 'faster_commit': faster_commit}
    (out / 'provenance.json').write_text(json.dumps(record, indent=2))
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description="Copy a Package 2 stage into the Section 12.1 layout.")
    ap.add_argument('stage_name', choices=['stage1', 'stage2'])
    ap.add_argument('--stage', required=True, help="the stage's results root (results/package2_stage1 or _stage2)")
    ap.add_argument('--out', required=True, help='package2_results')
    ap.add_argument('--package1', help="stage2: package1 (item 7's check C8', the figures)")
    ap.add_argument('--faster', help="stage2: FASTER's package2_stage2 (item 5), from package2_stage2_faster.tar.gz")
    args = ap.parse_args(argv)
    if args.stage_name == 'stage2':
        r = stage2(args.stage, args.out, args.package1, faster=args.faster)
        print(f"Stage 2 at {r['stage2_commit'][:7]}: {len(r['items'])} item folders, {r['code_files']} code files")
        for name, rec in r['references'].items():
            print(f"  reference/{name}: {rec}")
        print(f"  item 2 gpu-list: {r['item_2_gpu_list'] or 'empty (no A100 reruns needed)'}")
        for f in r['replaced']:
            print(f"  replaced (differed): {f}")
        return
    runs, replaced = stage1(args.stage, args.out)
    print(f"{len(runs)} reruns: {', '.join(runs)}")
    for f in replaced:
        print(f"replaced (differed): {f}")


if __name__ == '__main__':
    main()
