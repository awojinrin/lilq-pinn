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

Usage::

    python experiments/p2_assemble.py stage1 --stage <downloaded results/package2_stage1> --out <package2_results>
"""

import argparse
import filecmp
import re
import shutil
from pathlib import Path

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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Copy a Package 2 stage into the Section 12.1 layout.")
    ap.add_argument('stage_name', choices=['stage1'])
    ap.add_argument('--stage', required=True, help="the stage's results root (results/package2_stage1)")
    ap.add_argument('--out', required=True, help='package2_results')
    args = ap.parse_args(argv)
    runs, replaced = stage1(args.stage, args.out)
    print(f"{len(runs)} reruns: {', '.join(runs)}")
    for f in replaced:
        print(f"replaced (differed): {f}")


if __name__ == '__main__':
    main()
