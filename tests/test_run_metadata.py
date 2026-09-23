import json

import numpy as np
import pytest

from lilq.run_metadata import (
    RUN_METADATA_FIELDS, build_run_metadata, first_stall_iteration, write_run_json,
)


def test_build_run_metadata_fills_all_fields():
    meta = build_run_metadata(N_total=100, P_total=25, device="cpu")
    assert set(meta.keys()) == set(RUN_METADATA_FIELDS)
    assert meta["N_total"] == 100
    assert meta["P_total"] == 25
    assert meta["device"] == "cpu"
    # Everything not passed is None, not omitted.
    for key in RUN_METADATA_FIELDS:
        if key not in ("N_total", "P_total", "device"):
            assert meta[key] is None


def test_build_run_metadata_rejects_unknown_field():
    with pytest.raises(ValueError, match="typo_field"):
        build_run_metadata(typo_field=1)


def test_write_run_json_round_trips(tmp_path):
    meta = build_run_metadata(
        N_total=300, N_composition={"pde": 200, "bc": 100},
        P_total=25, P_composition={"u": 25},
        row_weights={"pde": 0.1, "bc": 3.16},
        collocation_construction={"method": "random-tensor", "seed": 42},
        basis_description={"u": {"family": "fourier", "modes": 5}},
        initial_coefficients="zero",
        solver_driver="gelsy", rcond=2.220446049250313e-16,
        stopping_rule={"type": "loss_target", "value": 1e-4}, K_max=100,
        stopping_reason="target",
        first_stall_iteration=None,
        device="cpu", thread_count=24,
    )
    out_path = tmp_path / "run.json"
    write_run_json(out_path, meta)

    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded == meta


def test_write_run_json_handles_numpy_types(tmp_path):
    meta = build_run_metadata(
        N_total=np.int64(300), P_total=np.int64(25),
        rcond=np.float64(2.220446049250313e-16),
        thread_count=np.int32(8),
    )
    out_path = tmp_path / "run.json"
    write_run_json(out_path, meta)  # must not raise

    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded["N_total"] == 300
    assert isinstance(loaded["N_total"], int)
    assert loaded["rcond"] == pytest.approx(2.220446049250313e-16)


def test_first_stall_iteration_returns_first_true_not_last():
    rows = [
        {"k": 1, "stall_flag": False},
        {"k": 2, "stall_flag": False},
        {"k": 3, "stall_flag": True},
        {"k": 4, "stall_flag": True},
    ]
    assert first_stall_iteration(rows) == 3


def test_first_stall_iteration_none_when_never_stalled():
    rows = [{"k": 1, "stall_flag": False}, {"k": 2, "stall_flag": False}]
    assert first_stall_iteration(rows) is None


def test_first_stall_iteration_none_for_empty_rows():
    assert first_stall_iteration([]) is None
