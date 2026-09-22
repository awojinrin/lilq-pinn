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
