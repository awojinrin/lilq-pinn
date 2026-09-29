"""
Build the upload bundle for running experiments on a cluster (TAMU HPRC).

The bundle is the git-tracked source (minus ``reference_results/``, which
isn't needed to run anything) under a single top-level ``lilq-pinn/``
folder, plus a ``PROVENANCE.json`` recording the commit hash, branch, the
file list and the source-tree hash (``lilq.source_lock``; every cluster job
checks the files on disk against it). A tree with uncommitted changes is
refused unless ``--allow-dirty`` (Addendum v2.2 Section 2.8.4: the bundle
is built from a clean tree at the final commit). The cluster copy has no ``.git``, so
``lilq.provenance.capture_git_info`` reads this file instead -- every
``hardware.json`` written on the cluster still names the exact code that
produced it (Computational_Package_1_v2.md Section 2).

Usage (run locally, from anywhere)::

    python scripts/make_hprc_bundle.py
    python scripts/make_hprc_bundle.py --out C:/path/to/bundle.tar.gz

Then on the cluster::

    tar -xzf lilq-pinn-<sha>.tar.gz -C $SCRATCH/lilq-run
"""

import argparse
import datetime
import io
import json
import platform
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from lilq.source_lock import EXCLUDED_PREFIXES, normalized_bytes, tree_hash  # noqa: E402

TOP_LEVEL = "lilq-pinn"


def _git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout


def tracked_files():
    files = [f for f in _git("ls-files").splitlines() if f]
    return [f for f in files if not f.startswith(EXCLUDED_PREFIXES)]


def provenance_record() -> dict:
    diff = _git("diff", "HEAD")
    files = tracked_files()
    return {
        "commit": _git("rev-parse", "HEAD").strip(),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "dirty": bool(diff),
        "diff": diff or None,
        "files": files,
        "tree_hash": tree_hash(REPO_ROOT, files),
        "bundled_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "bundled_on_host": platform.node(),
    }


def _add_bytes(tar, arcname, data, mode=0o644):
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mode = mode
    info.mtime = int(datetime.datetime.now().timestamp())
    tar.addfile(info, io.BytesIO(data))


def build_bundle(out_path: Path) -> dict:
    """Text files are written with LF line endings: a Windows checkout
    (core.autocrlf=true) has CRLF, and bash on the cluster fails on the
    carriage returns in job scripts. Binary files (any NUL byte) are copied
    unchanged."""
    record = provenance_record()
    files = record["files"]
    with tarfile.open(out_path, "w:gz") as tar:
        for rel in files:
            src = REPO_ROOT / rel
            if not src.is_file():
                continue
            data = normalized_bytes(src)
            executable = rel.endswith((".sh", ".slurm"))
            _add_bytes(tar, f"{TOP_LEVEL}/{rel}", data, 0o755 if executable else 0o644)
        _add_bytes(tar, f"{TOP_LEVEL}/PROVENANCE.json", json.dumps(record, indent=2).encode())
    return {"record": record, "n_files": len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=str, default=None,
                        help="Output .tar.gz (default: lilq-pinn-<short sha>.tar.gz next to the repo)")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Bundle uncommitted changes anyway (the cluster jobs will then refuse to run).")
    args = parser.parse_args()
    if _git("status", "--porcelain", "--untracked-files=no").strip() and not args.allow_dirty:
        sys.exit("Uncommitted changes to tracked files: commit first (or --allow-dirty for a test bundle).")

    short_sha = _git("rev-parse", "--short", "HEAD").strip()
    out_path = Path(args.out) if args.out else REPO_ROOT.parent / f"lilq-pinn-{short_sha}.tar.gz"

    result = build_bundle(out_path)
    record = result["record"]
    print(f"Wrote {out_path} ({result['n_files']} files, {out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  commit {record['commit']} on {record['branch']}")
    if record["dirty"]:
        print("  WARNING: uncommitted changes -- bundled as-is, diff recorded in PROVENANCE.json. "
              "Commit first if these runs are meant to be the record.", file=sys.stderr)


if __name__ == "__main__":
    main()
