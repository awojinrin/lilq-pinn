"""No tracked file carries a path of someone's machine (a notebook's saved
output, say) or an HPRC allocation account: those belong in the environment
(``LILQ_ACCOUNT``), not in the public repository."""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PATTERNS = [re.compile(p, re.I) for p in (r'[a-z]:\\{1,2}users\\{1,2}', r'/c/users/', r'/home/(?!claude/)\w+/',
                                          r'/scratch/user/\w', r'\b1[34]269895\d{4}\b')]
SKIP = ('reference_results/', 'tests/test_no_local_paths.py')


def _tracked():
    try:
        out = subprocess.run(['git', 'ls-files'], cwd=REPO, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('not a git checkout')
    return [REPO / f for f in out.splitlines() if not f.startswith(SKIP)]


def test_no_machine_paths_or_accounts_in_tracked_files():
    hits = []
    for path in _tracked():
        if path.suffix in ('.pt', '.npz', '.png', '.pdf', '.npy') or not path.is_file():
            continue
        text = path.read_text(encoding='utf-8', errors='ignore')
        hits += [f'{path.relative_to(REPO)}: {m.group(0)}' for p in PATTERNS for m in p.finditer(text)]
    assert hits == []
