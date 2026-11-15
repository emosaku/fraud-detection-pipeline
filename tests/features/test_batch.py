import apache_beam as beam
import numpy as np
import pandas as pd
import pytest
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

from fraud_pipeline.features.batch import (
    REPLAY_START,
    AddTimeSinceLastTxnSorted,
    build_replay_records,
    write_windowed_features,
)
from fraud_pipeline.ingestion.producer import compute_sleep_intervals
from fraud_pipeline.models.windowed_dataset import check_window_coverage, load_jsonl


def _prepared(times, accounts, amounts=None):
    """Same shape as producer.load_and_prepare's output."""
    n = len(times)
    return pd.DataFrame(
        {
            "transaction_id": range(n),
            "account_id": accounts,
            "Time": times,
            "Amount": amounts if amounts is not None else [10.0] * n,
        }
    )


def _run(tmp_path, records, **kwargs):
    path = write_windowed_features(records, tmp_path / "windows.jsonl", **kwargs)
    return load_jsonl(path)


def test_replay_records_follow_the_producers_compressed_clock():
    times = np.array([0.0, 100.0, 100.0, 400.0])
    df = _prepared(times, ["ACCT-1", "ACCT-2", "ACCT-1", "ACCT-3"])
    records = build_replay_records(df, duration_seconds=40.0)

    offsets = [r["_publish_ts"] - REPLAY_START.timestamp() for r in records]
    assert offsets == pytest.approx(np.cumsum(compute_sleep_intervals(times, 40.0)))
    assert offsets == pytest.approx([0.0, 10.0, 10.0, 40.0])
    assert [r["account_id"] for r in records] == ["ACCT-1", "ACCT-2", "ACCT-1", "ACCT-3"]
    assert [r["transaction_id"] for r in records] == [0, 1, 2, 3]


def test_time_since_last_txn_is_correct_regardless_of_input_order():
    records = [
        {"transaction_id": 2, "_publish_ts": 130.0},
        {"transaction_id": 0, "_publish_ts": 100.0},
        {"transaction_id": 1, "_publish_ts": 104.0},
    ]
    with TestPipeline() as p:
        result = (
            p
            | beam.Create([("ACCT-1", records)])
            | beam.ParDo(AddTimeSinceLastTxnSorted())
            | beam.Map(lambda kv: (kv[1]["transaction_id"], kv[1]["seconds_since_last_txn"]))
        )
        assert_that(result, equal_to([(0, None), (1, 4.0), (2, 26.0)]))


def test_windows_match_hand_computed_sliding_windows(tmp_path):
    # Replay offsets 0s, 5s, 25s for one account, 30s windows every 10s. The windows that
    # contain each transaction start at -20/-10/0 (t=0, t=5) and 0/10/20 (t=25).
    df = _prepared([0.0, 5.0, 25.0], ["ACCT-1"] * 3, [1.0, 2.0, 4.0])
    windows = _run(tmp_path, build_replay_records(df, duration_seconds=25.0))

    windows["start_offset"] = (
        pd.to_datetime(windows["window_start"]) - REPLAY_START.replace(tzinfo=None)
    ).dt.total_seconds()
    by_start = windows.set_index("start_offset").sort_index()

    assert by_start.index.tolist() == [-20.0, -10.0, 0.0, 10.0, 20.0]
    assert by_start["transaction_ids"].tolist() == [[0, 1], [0, 1], [0, 1, 2], [2], [2]]
    assert by_start["total_amount"].tolist() == [3.0, 3.0, 7.0, 4.0, 4.0]
    assert by_start.loc[0.0, "min_seconds_since_last_txn"] == 5.0


def test_every_transaction_lands_in_exactly_size_over_period_windows(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    times = np.sort(rng.uniform(0, 1_000, n))
    accounts = [f"ACCT-{i}" for i in rng.integers(0, 7, n)]
    amounts = rng.uniform(1, 100, n).round(2)
    df = _prepared(times, accounts, amounts)

    windows = _run(tmp_path, build_replay_records(df, duration_seconds=200.0))

    check_window_coverage(windows, df["transaction_id"], 30, 10)
    assert windows["transaction_count"].sum() == 3 * n
    assert windows["total_amount"].sum() == pytest.approx(3 * amounts.sum(), abs=0.05 * n)


def test_coverage_holds_for_other_window_shapes(tmp_path):
    rng = np.random.default_rng(1)
    n = 120
    df = _prepared(np.sort(rng.uniform(0, 500, n)), [f"ACCT-{i}" for i in rng.integers(0, 4, n)])

    windows = _run(
        tmp_path,
        build_replay_records(df, duration_seconds=100.0),
        window_size_seconds=20,
        window_period_seconds=5,
    )

    check_window_coverage(windows, df["transaction_id"], 20, 5)
    assert windows["transaction_count"].sum() == 4 * n
