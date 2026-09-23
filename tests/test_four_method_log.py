import csv
import json

import pytest

from lilq.four_method_log import (
    FOUR_METHOD_CSV_COLUMNS, FourMethodLogger,
    classify_stopping_reason, subsample_loss_history,
)


def test_classify_stopping_reason_target_when_converged():
    assert classify_stopping_reason(True, 500, 10000, 5000, 100000) == "target"


def test_classify_stopping_reason_iteration_cap():
    assert classify_stopping_reason(False, 10000, 10000, 5000, 100000) == "iteration_cap"


def test_classify_stopping_reason_line_search_cap():
    # iterations_used < iterations_cap, so only the line-search cap explains the stop.
    assert classify_stopping_reason(False, 500, 10000, 100000, 100000) == "line_search_cap"


def test_classify_stopping_reason_iteration_cap_checked_before_line_search_cap_on_tie():
    # Both caps reached simultaneously -- iteration_cap wins by documented convention.
    assert classify_stopping_reason(False, 10000, 10000, 100000, 100000) == "iteration_cap"


def test_classify_stopping_reason_failure_when_neither_cap_reached_and_not_converged():
    assert classify_stopping_reason(False, 50, 10000, 500, 100000) == "failure"


def test_subsample_loss_history_keeps_every_nth_row():
    metrics_dict = {
        "iteration": list(range(25)),
        "loss": [1.0 / (i + 1) for i in range(25)],
    }
    history = subsample_loss_history(metrics_dict, every=10)
    iterations_kept = [row[0] for row in history]
    assert iterations_kept == [0, 10, 20, 24]  # stride of 10, plus the final row


def test_subsample_loss_history_empty_when_no_data():
    assert subsample_loss_history({"iteration": [], "loss": []}) == []


def test_subsample_loss_history_does_not_duplicate_last_row_when_already_on_stride():
    metrics_dict = {"iteration": list(range(21)), "loss": [float(i) for i in range(21)]}
    history = subsample_loss_history(metrics_dict, every=10)
    iterations_kept = [row[0] for row in history]
    assert iterations_kept == [0, 10, 20]  # 20 is both on-stride and the last row


def test_logger_record_rejects_unknown_column():
    logger = FourMethodLogger()
    with pytest.raises(ValueError, match="typo_field"):
        logger.record(typo_field=1)


def test_logger_record_fills_omitted_columns_with_none():
    logger = FourMethodLogger()
    logger.record(benchmark="bratu", P=25, method="NiL-N", seed=0, device="cuda")
    row = logger.rows[0]
    assert set(row.keys()) == set(FOUR_METHOD_CSV_COLUMNS)
    assert row["final_loss"] is None
    assert row["benchmark"] == "bratu"


def test_to_csv_round_trip_and_loss_history_is_valid_json(tmp_path):
    logger = FourMethodLogger()
    logger.record(
        benchmark="bratu", P=25, method="NiL-N", seed=0, device="cuda",
        total_iterations=100, total_line_searches=500, training_time_s=1.23,
        final_loss=1e-5, converged=True, stopping_reason="target",
        iterations_cap=10000, line_searches_cap=100000,
        loss_history_every_10=[[0, 1.0], [10, 0.5], [20, 0.1]],
    )
    out_path = tmp_path / "four_method_tables.csv"
    logger.to_csv(out_path)

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["benchmark"] == "bratu"
    assert rows[0]["converged"] == "True"
    loaded_history = json.loads(rows[0]["loss_history_every_10"])
    assert loaded_history == [[0, 1.0], [10, 0.5], [20, 0.1]]


def test_to_csv_empty_field_for_none_and_nan(tmp_path):
    logger = FourMethodLogger()
    logger.record(benchmark="bratu", P=25, method="LiL-N", seed=None, device="cpu",
                  final_loss=float("nan"))
    out_path = tmp_path / "four_method_tables.csv"
    logger.to_csv(out_path)

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["seed"] == ""
    assert rows[0]["final_loss"] == ""


def test_row_key_matches_between_fresh_and_csv_rows():
    from lilq.four_method_log import row_key
    fresh = {"benchmark": "bratu", "P": 25, "method": "NiL-N", "seed": 0, "device": "cuda"}
    from_csv = {"benchmark": "bratu", "P": "25", "method": "NiL-N", "seed": "0", "device": "cuda"}
    assert row_key(fresh) == row_key(from_csv)
    assert row_key({**fresh, "seed": None}) == row_key({**from_csv, "seed": ""})
