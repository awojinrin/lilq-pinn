"""Tests for lilq.provenance -- the hardware/environment/git capture
utility implementing Computational_Package_1_v2.md Section 2's
hardware.json / environment.txt requirement (Codebase_v3_Proposal.md S2.5).
"""

import json
import subprocess

from lilq import provenance


def test_run_returns_none_for_missing_executable():
    assert provenance._run(["this-executable-does-not-exist-anywhere"]) is None


def test_run_returns_stdout_for_a_real_command():
    # python is guaranteed available -- this process is running under it.
    out = provenance._run(["python", "-c", "print('hello')"])
    assert out == "hello"


def test_capture_git_info_reports_real_repo_state():
    info = provenance.capture_git_info()
    assert info["available"] is True
    assert len(info["commit"]) == 40  # full SHA
    # Cross-check against git directly, so this test fails if the
    # function's git invocation is ever wrong, not just "returns a string".
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
    ).stdout.strip()
    assert info["commit"] == expected
    assert isinstance(info["dirty"], bool)


def test_capture_git_info_unavailable_outside_a_repo(tmp_path):
    info = provenance.capture_git_info(repo_root=tmp_path)
    assert info == {"available": False}


def test_capture_blas_thread_env_has_expected_keys():
    env = provenance.capture_blas_thread_env()
    assert set(env.keys()) == {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"}


def test_capture_cpu_info_has_a_core_count():
    info = provenance.capture_cpu_info()
    assert info["logical_cores"] >= 1
    assert info["system"]


def test_capture_gpu_info_structure():
    info = provenance.capture_gpu_info()
    assert isinstance(info["available"], bool)
    if info["available"]:
        assert len(info["gpus"]) >= 1
        assert info["gpus"][0]["name"]
        assert info["gpus"][0]["driver_version"]


def test_capture_package_versions_matches_whats_actually_imported():
    import numpy as np
    import scipy
    import torch
    versions = provenance.capture_package_versions()
    assert versions["numpy"] == np.__version__
    assert versions["scipy"] == scipy.__version__
    assert versions["torch"] == torch.__version__


def test_capture_environment_text_contains_both_show_configs():
    text = provenance.capture_environment_text()
    assert "numpy.show_config()" in text
    assert "scipy.show_config()" in text
    assert len(text) > 100  # not an empty/truncated capture


def test_save_provenance_writes_both_files(tmp_path):
    provenance.save_provenance(tmp_path)

    hardware_path = tmp_path / "hardware.json"
    env_path = tmp_path / "environment.txt"
    assert hardware_path.exists()
    assert env_path.exists()

    hardware = json.loads(hardware_path.read_text())
    assert set(hardware.keys()) == {"cpu", "gpu", "blas_thread_env", "packages", "git"}
    assert env_path.read_text()  # non-empty
