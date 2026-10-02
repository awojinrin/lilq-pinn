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
The waves come to under 1 GB.
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
from pathlib import Path


def assemble(results: Path, out: Path, waves=(1, 2, 3, 4)) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    record = {'waves': {}, 'overrides': [], 'files': 0}
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
    (out / 'WAVES.json').write_text(json.dumps(record, indent=2))
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description="Assemble results/package1 from the wave roots.")
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    record = assemble(Path(args.results), Path(args.out))
    commits = {n: (w or {}).get('commit') for n, w in record['waves'].items()}
    print(f"Assembled {args.out}: {record['files']} files; commits by wave {commits}; "
          f"{len(record['overrides'])} overrides")


if __name__ == '__main__':
    main()
