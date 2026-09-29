"""
Source identity and the package's provenance lock (Addendum v2.2 Section 2.8.4)
===============================================================================

Every result of Package 1 must come from one known version of the code.
This module names that version -- the commit and a hash of the source
tree -- and pins a package root to it:

* :func:`source_identity` -- the commit (from git, or from the bundle's
  ``PROVENANCE.json`` on a cluster copy without ``.git``) and the hash of
  the source files as they are on disk now.
* :func:`check_lock` -- at the first job of a package root, write
  ``$PKG/COMMIT``; at every later job, refuse to run if the commit or the
  tree hash differs from it. On a bundle copy it also refuses when the
  files on disk no longer hash to what the bundle recorded (a file edited
  on the cluster after upload), and everywhere when the commit is unknown.

The tree hash covers the files the upload bundle ships (git-tracked, minus
``reference_results/``), with text files' CRLF normalized to LF as the
bundle writes them, so the same commit hashes the same on Windows and on
the cluster. Standard library only: ``env.sh`` runs it before every job.

Usage (``scripts/cluster/env.sh``)::

    python -m lilq.source_lock --pkg "$PKG"
"""

import argparse
import datetime
import functools
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PROVENANCE_FILE = "PROVENANCE.json"
EXCLUDED_PREFIXES = ("reference_results/",)
LOCK_FILE = "COMMIT"


def _git(*args, cwd=REPO_ROOT) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _bundle_record(root: Path) -> Optional[dict]:
    try:
        return json.loads((Path(root) / BUNDLE_PROVENANCE_FILE).read_text())
    except (OSError, ValueError):
        return None


def tracked_files(root: Path = REPO_ROOT) -> Optional[List[str]]:
    """The source files, relative to ``root``: git-tracked minus the
    excluded prefixes, or the bundle's recorded list; ``None`` if neither."""
    listed = _git("ls-files", cwd=root)
    if listed is not None:
        return sorted(f for f in listed.splitlines() if f and not f.startswith(EXCLUDED_PREFIXES))
    record = _bundle_record(root)
    if record and record.get("files"):
        return sorted(record["files"])
    return None


def normalized_bytes(path: Path) -> bytes:
    """A file's content as the bundle ships it: text files (no NUL byte)
    with CRLF normalized to LF, binary files unchanged."""
    data = Path(path).read_bytes()
    return data if b"\0" in data else data.replace(b"\r\n", b"\n")


def tree_hash(root: Path = REPO_ROOT, files: Optional[List[str]] = None) -> Optional[str]:
    """SHA-256 over the (path, content hash) of every source file; a
    missing listed file hashes as missing, so deleting one changes it."""
    files = tracked_files(root) if files is None else sorted(files)
    if files is None:
        return None
    h = hashlib.sha256()
    for rel in files:
        p = Path(root) / rel
        digest = hashlib.sha256(normalized_bytes(p)).hexdigest() if p.is_file() else "missing"
        h.update(f"{rel}\0{digest}\n".encode())
    return h.hexdigest()


def source_identity(root: Path = REPO_ROOT) -> Dict[str, Optional[str]]:
    """``{'commit', 'tree_hash', 'source', 'problem'}``: ``problem`` is
    ``None`` when the identity is sound, otherwise why it is not."""
    root = Path(root)
    commit = _git("rev-parse", "HEAD", cwd=root)
    if commit is not None:
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no", cwd=root))
        return {"commit": commit.strip(), "tree_hash": tree_hash(root), "source": "git",
                "problem": "uncommitted changes to tracked files" if dirty else None}
    record = _bundle_record(root)
    if record is None or not record.get("commit"):
        return {"commit": None, "tree_hash": None, "source": None,
                "problem": "no git checkout and no PROVENANCE.json: the code's version is unknown"}
    now = tree_hash(root)
    problem = None
    if record.get("dirty"):
        problem = "the bundle was built from a tree with uncommitted changes"
    elif record.get("tree_hash") is None:
        problem = "the bundle records no tree hash (built before the provenance lock)"
    elif now != record["tree_hash"]:
        problem = "source files differ from the bundle as uploaded (edited on the cluster?)"
    return {"commit": record["commit"], "tree_hash": now, "source": "bundle", "problem": problem}


@functools.lru_cache(maxsize=None)
def current_commit() -> Optional[str]:
    """The commit of the running code (for every run's records)."""
    return source_identity()["commit"]


def _read_lock(lock: Path, attempts: int = 20) -> dict:
    """The lock's record, retrying a moment if the file is not yet readable."""
    for i in range(attempts):
        try:
            return json.loads(lock.read_text())
        except (OSError, ValueError):
            if i == attempts - 1:
                raise
            time.sleep(0.5)


def check_lock(pkg: Path, root: Path = REPO_ROOT) -> dict:
    """Write ``<pkg>/COMMIT`` at the first call; afterwards raise unless the
    code is the same. Raises ``RuntimeError`` with the reason."""
    ident = source_identity(root)
    if ident["problem"]:
        raise RuntimeError(f"provenance: {ident['problem']}")
    pkg = Path(pkg)
    lock = pkg / LOCK_FILE
    if not lock.exists():
        pkg.mkdir(parents=True, exist_ok=True)
        record = {"commit": ident["commit"], "tree_hash": ident["tree_hash"], "source": ident["source"],
                  "locked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "locked_on_host": platform.node()}
        # Jobs of a wave can start together: each writes its own temporary
        # file, and the lock is created by a hard link, which fails if it
        # already exists -- the first job's lock stands, never overwritten,
        # and never read half-written (the advisor's reply, item 2.2).
        tmp = lock.with_name(f"{LOCK_FILE}.{platform.node()}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(record, indent=2))
        try:
            os.link(tmp, lock)
        except FileExistsError:
            pass
        finally:
            tmp.unlink(missing_ok=True)
    record = _read_lock(lock)
    for key in ("commit", "tree_hash"):
        if record.get(key) != ident[key]:
            raise RuntimeError(f"provenance: this package root ({pkg}) is locked to {key} "
                               f"{record.get(key)}, but the code is at {ident[key]}. Results from two "
                               f"versions of the code must not mix; use a new package root.")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check (or create) the package's provenance lock.")
    ap.add_argument("--pkg", required=True, help="The package root ($PKG).")
    args = ap.parse_args(argv)
    try:
        record = check_lock(Path(args.pkg))
    except RuntimeError as e:
        print(f"PROVENANCE LOCK FAILED: {e}", file=sys.stderr)
        return 1
    print(f"provenance lock OK: commit {record['commit'][:12]}, tree {record['tree_hash'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
