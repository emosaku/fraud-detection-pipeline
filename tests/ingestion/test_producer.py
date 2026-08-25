import numpy as np
import pytest

from fraud_pipeline.ingestion.producer import (
    assign_synthetic_accounts,
    build_ground_truth_record,
    build_message,
    compute_sleep_intervals,
)


def test_assign_synthetic_accounts_deterministic():
    a = assign_synthetic_accounts(5000, 50, seed=42)
    b = assign_synthetic_accounts(5000, 50, seed=42)
    assert np.array_equal(a, b)


def test_assign_synthetic_accounts_distinct_count():
    accounts = assign_synthetic_accounts(5000, 50, seed=42)
    assert len(np.unique(accounts)) == 50


def test_assign_synthetic_accounts_not_uniform():
    accounts = assign_synthetic_accounts(5000, 50, seed=42)
    counts = np.bincount(accounts, minlength=50)
    assert counts.max() > counts.min() * 3


def test_compute_sleep_intervals_sums_to_target():
    times = np.array([0, 1, 1, 5, 20, 100, 500, 172792], dtype=float)
    target = 600.0
    intervals = compute_sleep_intervals(times, target)
    assert intervals[0] == 0.0
    assert intervals.sum() == pytest.approx(target, rel=1e-6)


def test_compute_sleep_intervals_preserves_relative_burstiness():
    times = np.array([0, 1, 2, 100], dtype=float)
    intervals = compute_sleep_intervals(times, 100.0)
    assert intervals[1] < intervals[3]
    assert intervals[2] < intervals[3]


def _make_row(class_value=1):
    row = {"Time": 12345.0, "Amount": 99.5, "account_id": "ACCT-000001", "Class": class_value}
    for i in range(1, 29):
        row[f"V{i}"] = float(i) * 0.1
    return row


def test_build_message_excludes_class():
    message = build_message(_make_row(), transaction_id=7, event_time="2026-01-01T00:00:00+00:00")
    assert "Class" not in message
    assert "class" not in message


def test_build_message_contains_expected_fields():
    message = build_message(_make_row(), transaction_id=7, event_time="2026-01-01T00:00:00+00:00")
    assert message["transaction_id"] == 7
    assert message["account_id"] == "ACCT-000001"
    assert message["original_time_offset"] == 12345.0
    assert message["amount"] == 99.5
    assert message["event_time"] == "2026-01-01T00:00:00+00:00"
    assert message["V14"] == pytest.approx(1.4)


def test_build_ground_truth_record_contains_expected_fields():
    record = build_ground_truth_record(
        transaction_id=7,
        account_id="ACCT-000001",
        class_value=1,
        event_time="2026-01-01T00:00:00+00:00",
    )
    assert record == {
        "transaction_id": 7,
        "account_id": "ACCT-000001",
        "class": 1,
        "event_time": "2026-01-01T00:00:00+00:00",
    }
