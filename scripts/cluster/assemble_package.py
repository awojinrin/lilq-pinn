"""
Assemble package1_results from the waves (Addendum v2.2 Section 4.2)
=====================================================================

Each wave wrote its own package root, ``results/wave<N>``, locked to its own
commit (``lilq/source_lock.py``). This builds ``results/package1`` -- the
layout of Package 1 v2.0 Section 6 -- by copying every file of wave 1, then
wave 2, 3 and 4 into it. Copies, not hard links: ``90_finalize`` rewrites
files in ``package1`` (the run index, the reproduction check, B9's tables),
and through a hard link that would rewrite the wave's own file, which is
locked to its commit (the advisor's reply on wave 3, Section 1, item 2).
The waves come to under 1 GB. Component A runs that wave 4's selection
replaced are marked with a ``SUPERSEDED.json`` (``mark_superseded``).
The components come from different waves (B4 from waves 1 and 2, each
benchmark in its own folder), so paths rarely meet; where they do, an
identical file is kept once and a differing one is taken from the later
wave and listed. ``WAVES.json`` records each wave's commit and tree hash
(its ``COMMIT``) and those overrides, so every result stays traceable to
the code that produced it.

Usage (``90_finalize.slurm``)::

    python scripts/cluster/assemble_package.py --results results --out results/package1
"""

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


PROVISIONAL = ("provisional: the advisor reviews the wave 4 report first (his reply on wave 3, Section 4); "
               "assembled again with --final once he has approved it")


def assemble(results: Path, out: Path, waves=(1, 2, 3, 4), final: bool = False) -> dict:
    """Build ``out`` from the wave roots under ``results``. An earlier
    assembly there (it has a ``WAVES.json``) is removed first, so that a
    rerun starts clean: no stale files, no hard links left from assemblies
    made before copies (removing a hard link leaves its wave's file). Any
    other non-empty ``out`` is refused."""
    if out.exists() and any(out.iterdir()):
        if not (out / 'WAVES.json').exists():
            raise FileExistsError(f"{out} is not empty and holds no earlier assembly (WAVES.json); not touching it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    from lilq.source_lock import current_commit
    record = {'status': 'final' if final else PROVISIONAL, 'assembled_by_commit': current_commit(),
              'waves': {}, 'overrides': [], 'files': 0}
    for n in waves:
        src = results / f'wave{n}'
        if not src.is_dir():
            record['waves'][str(n)] = None
            continue
        lock = src / 'COMMIT'
        record['waves'][str(n)] = json.loads(lock.read_text()) if lock.exists() else {'commit': None}
        for path in sorted(p for p in src.rglob('*') if p.is_file()):
            rel = path.relative_to(src)
            if rel.as_posix() == 'COMMIT':
                continue
            dst = out / rel
            if dst.exists():
                if filecmp.cmp(path, dst, shallow=False):
                    continue
                record['overrides'].append({'path': rel.as_posix(), 'from_wave': n})
                dst.unlink()
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            record['files'] += 1
    record['superseded'] = mark_superseded(results, out)
    (out / 'WAVES.json').write_text(json.dumps(record, indent=2))
    return record


SUPERSEDED_FILE = 'SUPERSEDED.json'


def mark_superseded(results: Path, out: Path, wave: int = 4) -> list:
    """Mark the Component A runs of earlier waves that ``wave``'s selection
    replaced (the advisor's reply on wave 3, item 2.7): a ``SUPERSEDED.json``
    in each such run's folder in ``out``, and the list returned (for
    ``WAVES.json``). Per family, from ``wave``'s own files:

    * ``full/<id>_s*`` whose configuration is not among its finalists
      (``screening/<family>_selection.json``'s ``top``);
    * ``full_cpu/<id>_s*`` and ``float32/<id>_s*`` of any configuration but
      its representative (``full/<family>_representative.json``).

    In wave 4 that is wave 2's F2_16, F2_03 and F2_17 full runs, and its
    F1_04 and F2_16 CPU and float32 runs. Markers from an earlier assembly
    are removed first."""
    A = out / 'A_calibration'
    for old in A.glob(f'*/*/{SUPERSEDED_FILE}'):
        old.unlink()
    src = results / f'wave{wave}' / 'A_calibration'
    marked = []

    def mark(run_dir, reason):
        (run_dir / SUPERSEDED_FILE).write_text(json.dumps({'superseded_by_wave': wave, 'reason': reason}, indent=2))
        marked.append({'path': run_dir.relative_to(out).as_posix(), 'reason': reason})

    for family in ('F1', 'F2'):
        sel_path = src / 'screening' / f'{family}_selection.json'
        if sel_path.exists():
            top = json.loads(sel_path.read_text())['top']
            for d in sorted(p for p in A.glob(f'full/{family}_*_s*') if p.is_dir()):
                if d.name.rsplit('_s', 1)[0] not in top:
                    mark(d, f"not among wave {wave}'s {family} finalists {top}")
        rep_path = src / 'full' / f'{family}_representative.json'
        if rep_path.exists():
            rep = json.loads(rep_path.read_text())['representative']
            for stage in ('full_cpu', 'float32'):
                for d in sorted(p for p in A.glob(f'{stage}/{family}_*_s*') if p.is_dir()):
                    if d.name.rsplit('_s', 1)[0] != rep:
                        mark(d, f"not wave {wave}'s {family} representative ({rep})")
    return marked


def main(argv=None):
    ap = argparse.ArgumentParser(description="Assemble results/package1 from the wave roots.")
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--final', action='store_true',
                    help="Mark the package final (WAVES.json's status), once the advisor has approved it.")
    args = ap.parse_args(argv)
    record = assemble(Path(args.results), Path(args.out), final=args.final)
    commits = {n: (w or {}).get('commit') for n, w in record['waves'].items()}
    print(f"Assembled {args.out}: {record['files']} files; commits by wave {commits}; "
          f"{len(record['overrides'])} overrides; {len(record['superseded'])} runs marked superseded; "
          f"status: {record['status']}")


if __name__ == '__main__':
    main()
