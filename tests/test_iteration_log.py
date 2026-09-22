"""Tests for lilq.iteration_log -- the iterations.csv schema (exact
column order from Computational_Package_1_v2.md Section 6) and the
IterationLogger accumulator.
"""

import csv
import math

import pytest

from lilq.iteration_log import ITERATION_CSV_COLUMNS, IterationLogger


def test_schema_has_exactly_the_spec_columns_in_order():
    expected = (
        "k", "t_assemble_s", "t_solve_s", "t_cum_s",
        "norm_R_h", "norm_R_interior", "norm_Rlin_h", "norm_Rlin_interior",
        "norm_f_h", "norm_dbeta", "rel_dbeta", "chi", "order_obs",
        "stall_flag", "roundoff_ratio", "kappa_eps", "kappa", "kappa_method",
        "num_rank_svd", "num_rank_gelsy", "rcond",
        "eps_u", "eps_v", "eps_p", "eps_p_meanfree",
        "maxerr_u", "maxerr_v", "maxerr_p",
        "solver_path", "gpu_mem_peak_bytes",
    )
    assert ITERATION_CSV_COLUMNS == expected
    assert len(ITERATION_CSV_COLUMNS) == 30


def test_record_fills_unspecified_columns_with_none():
    logger = IterationLogger()
    logger.record(k=0, norm_R_h=1.5)
    row = logger.rows[0]
    assert row["k"] == 0
    assert row["norm_R_h"] == 1.5
    assert row["chi"] is None
    assert row["solver_path"] is None
    assert set(row.keys()) == set(ITERATION_CSV_COLUMNS)


def test_record_rejects_unknown_column():
    logger = IterationLogger()
    with pytest.raises(ValueError, match="typo_column"):
        logger.record(k=0, typo_column=1.0)


def test_len_and_rows_track_insertion_order():
    logger = IterationLogger()
    logger.record(k=0)
    logger.record(k=1)
    logger.record(k=2)
    assert len(logger) == 3
    assert [row["k"] for row in logger.rows] == [0, 1, 2]


def test_to_csv_writes_header_and_all_rows(tmp_path):
    logger = IterationLogger()
    logger.record(k=0, norm_R_h=1.0, stall_flag=False)
    logger.record(k=1, norm_R_h=0.5, stall_flag=True)

    out_path = tmp_path / "iterations.csv"
    logger.to_csv(out_path)

    with open(out_path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == list(ITERATION_CSV_COLUMNS)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["k"] == "0"
    assert rows[0]["norm_R_h"] == "1.0"
    assert rows[1]["stall_flag"] == "True"


def test_to_csv_writes_none_and_nan_as_empty_field(tmp_path):
    logger = IterationLogger()
    logger.record(k=0, norm_R_h=1.0, chi=float("nan"))  # order_obs left as None

    out_path = tmp_path / "iterations.csv"
    logger.to_csv(out_path)

    with open(out_path, newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    assert row["chi"] == ""       # NaN -> empty
    assert row["order_obs"] == ""  # None -> empty


def test_to_csv_creates_parent_directories(tmp_path):
    logger = IterationLogger()
    logger.record(k=0)
    nested = tmp_path / "a" / "b" / "c" / "iterations.csv"
    logger.to_csv(nested)
    assert nested.exists()
