"""Tests for lilq.provenance -- the hardware/environment/git capture
utility implementing Computational_Package_1_v2.md Section 2's
hardware.json / environment.txt requirement (Codebase_v3_Proposal.md S2.5).
"""

import json
import subprocess
from pathlib import Path

import pytest

from lilq import provenance


# The cluster upload bundle ships without .git (scripts/make_hprc_bundle.py):
# tests that read git history skip there instead of failing.
requires_git_checkout = pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / ".git").exists(),
    reason="needs a git checkout",
)


def test_run_returns_none_for_missing_executable():
    assert provenance._run(["this-executable-does-not-exist-anywhere"]) is None


def test_run_returns_stdout_for_a_real_command():
    # python is guaranteed available -- this process is running under it.
    out = provenance._run(["python", "-c", "print('hello')"])
    assert out == "hello"


@requires_git_checkout
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


def test_capture_git_info_falls_back_to_bundle_provenance_file(tmp_path):
    record = {"commit": "a" * 40, "branch": "v3-dev", "dirty": False, "diff": None}
    (tmp_path / provenance.BUNDLE_PROVENANCE_FILE).write_text(json.dumps(record))

    info = provenance.capture_git_info(repo_root=tmp_path)

    assert info["available"] is True
    assert info["source"] == "bundle"
    assert info["commit"] == "a" * 40
    assert info["branch"] == "v3-dev"


@requires_git_checkout
def test_real_bundle_extracts_with_correct_provenance(tmp_path):
    """End-to-end: build the actual upload bundle, extract it (no .git),
    and confirm provenance capture inside it reports this repo's HEAD."""
    import sys
    import tarfile
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "scripts"))
    try:
        import make_hprc_bundle
    finally:
        sys.path.pop(0)

    bundle = tmp_path / "bundle.tar.gz"
    make_hprc_bundle.build_bundle(bundle)
    with tarfile.open(bundle) as tar:
        names = tar.getnames()
        assert all(n.startswith("lilq-pinn/") for n in names)
        assert "lilq-pinn/PROVENANCE.json" in names
        assert "lilq-pinn/tests/test_provenance.py" in names
        assert not any(n.startswith("lilq-pinn/reference_results/") for n in names)
        assert not any(n.startswith("lilq-pinn/.git/") for n in names)
        # No carriage returns in text files (bash on the cluster rejects them).
        for n in names:
            if n.endswith((".py", ".sh", ".slurm", ".md", ".toml", ".txt")):
                assert b"\r\n" not in tar.extractfile(n).read(), n
        tar.extractall(tmp_path / "x", filter="data")

    extracted = tmp_path / "x" / "lilq-pinn"
    info = provenance.capture_git_info(repo_root=extracted)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                          capture_output=True, text=True).stdout.strip()
    assert info["source"] == "bundle"
    assert info["commit"] == head


def test_capture_scheduler_info_outside_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert provenance.capture_scheduler_info() == {"slurm": False}


def test_capture_scheduler_info_inside_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    info = provenance.capture_scheduler_info()
    assert info["slurm"] is True
    assert info["SLURM_JOB_ID"] == "12345"
    assert info["SLURM_CPUS_PER_TASK"] == "16"


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
    assert set(hardware.keys()) == {"cpu", "gpu", "scheduler", "blas_thread_env", "threads", "packages", "git"}
    assert env_path.read_text()  # non-empty


def test_torch_threads_pinned_to_the_blas_allocation(monkeypatch):
    import torch
    from lilq.blas_threads import pin_torch
    old = torch.get_num_threads()
    try:
        monkeypatch.setenv("OMP_NUM_THREADS", "3")
        assert pin_torch() == 3 and torch.get_num_threads() == 3
        from lilq.provenance import capture_thread_info
        assert capture_thread_info()["torch_num_threads"] == 3
    finally:
        torch.set_num_threads(old)


def test_scheduler_info_records_node_type_and_exclusivity(monkeypatch):
    import lilq.provenance as prov
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURMD_NODENAME", "g001")
    outputs = {
        ("scontrol", "show", "node", "g001"):
            "NodeName=g001 Arch=x86_64 CPUTot=48 AvailableFeatures=a100,gpu ActiveFeatures=a100,gpu "
            "Gres=gpu:a100:2 RealMemory=376000 Partitions=gpu",
        ("scontrol", "show", "job", "123"):
            "JobId=123 Account=132698954494 Partition=gpu NumCPUs=48 OverSubscribe=NO TRES=cpu=48,gres/gpu=2",
    }
    monkeypatch.setattr(prov, "_run", lambda args, **kw: outputs.get(tuple(args)))
    info = prov.capture_scheduler_info()
    assert info["node"]["AvailableFeatures"] == "a100,gpu" and info["node"]["Gres"] == "gpu:a100:2"
    assert info["job"]["Account"] == "132698954494" and info["exclusive"] is True
