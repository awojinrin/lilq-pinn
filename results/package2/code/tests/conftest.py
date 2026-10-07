"""Shared pytest fixtures and import-path setup.

The package is not assumed to be pip-installed (editable or otherwise); tests
resolve ``lilq``/``problems``/``experiments`` the same way the experiment
scripts under ``experiments/`` do -- by inserting the repository root onto
``sys.path``.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


import json  # noqa: E402

import pytest  # noqa: E402

# Set by a submission and inherited by every cluster job (sbatch --export=ALL), so
# present when the preflight runs the tests: a profile honours CLUSTER and
# CPU_PARTITION, and the submission scripts LILQ_WAVE, DRY_RUN and YES. A test that
# runs those scripts must see none of them (Grace's preflight of 6 October failed
# on CPU_PARTITION=medium; FASTER's would have on CLUSTER=faster).
_SUBMISSION_ENV = ('CLUSTER', 'CPU_PARTITION', 'LILQ_WAVE', 'LILQ_MODULES', 'DRY_RUN', 'YES')


@pytest.fixture(autouse=True)
def _no_submission_environment(monkeypatch):
    for name in _SUBMISSION_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_bundle(tmp_path):
    """A minimal upload bundle (two source files and a PROVENANCE.json with
    their tree hash, commit 'abc123') in ``tmp_path/bundle``; returns its root.
    Shared here, not imported from another test file: ``from tests.x import``
    depends on how the interpreter resolves ``tests`` and failed on Grace."""
    from lilq.source_lock import REPO_ROOT, normalized_bytes, tree_hash
    files = ('lilq/source_lock.py', 'scripts/cluster/env.sh')
    root = tmp_path / 'bundle'
    for rel in files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(normalized_bytes(REPO_ROOT / rel))
    record = {'commit': 'abc123', 'dirty': False, 'files': list(files), 'tree_hash': tree_hash(root, list(files))}
    (root / 'PROVENANCE.json').write_text(json.dumps(record))
    return root
