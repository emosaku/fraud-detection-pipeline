import pandas as pd
import pytest

from fraud_pipeline.models.windowed_dataset import (
    build_labeled_windowed_dataset,
    check_window_coverage,
    label_windows,
    scoring_windows,
    window_fraud_stats,
    windows_per_transaction,
)


def _window(account_id, window_start, window_end, transaction_ids, **extra):
    return {
        "account_id": account_id,
        "window_start": window_start,
        "window_end": window_end,
        "transaction_count": len(transaction_ids),
        "transaction_ids": transaction_ids,
        **extra,
    }


def _labels(classes):
    """{transaction_id: class} -> labels frame in ground_truth.jsonl's shape."""
    return pd.DataFrame({"transaction_id": list(classes), "class": list(classes.values())})


def test_window_containing_a_fraud_id_is_labeled_positive():
    windows = pd.DataFrame([_window("ACCT-1", "t0", "t30", [1, 2])])
    labels = label_windows(windows, _labels({1: 0, 2: 1}))
    assert labels.tolist() == [1]


def test_window_with_no_fraud_ids_is_labeled_negative():
    windows = pd.DataFrame([_window("ACCT-1", "t0", "t30", [1, 2])])
    labels = label_windows(windows, _labels({1: 0, 2: 0}))
    assert labels.tolist() == [0]


def test_only_the_windows_own_transactions_count():
    # Regression test for the first Phase 3b capture: transaction 3 belongs to the same
    # account and time range, but the window never aggregated it, so its fraud must not
    # leak into the label. The old time-range join labeled this window positive.
    windows = pd.DataFrame([_window("ACCT-1", "t0", "t30", [1, 2])])
    labels = label_windows(windows, _labels({1: 0, 2: 0, 3: 1}))
    assert labels.tolist() == [0]


def test_fraction_rule_needs_enough_fraud_in_the_window():
    windows = pd.DataFrame(
        [
            _window("ACCT-1", "t0", "t30", [1, 2, 3, 4]),  # 1 of 4 fraud
            _window("ACCT-2", "t0", "t30", [5, 6]),  # 1 of 2 fraud
        ]
    )
    labels_df = _labels({1: 1, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0})
    assert label_windows(windows, labels_df, rule="any").tolist() == [1, 1]
    assert label_windows(windows, labels_df, rule="fraction", min_fraud_fraction=0.5).tolist() == [
        0,
        1,
    ]


def test_unknown_label_rule_raises():
    windows = pd.DataFrame([_window("ACCT-1", "t0", "t30", [1])])
    with pytest.raises(ValueError, match="unknown label rule"):
        label_windows(windows, _labels({1: 0}), rule="majority-ish")


def test_windowed_transaction_without_a_label_raises():
    windows = pd.DataFrame([_window("ACCT-1", "t0", "t30", [1, 99])])
    with pytest.raises(ValueError, match="no label"):
        window_fraud_stats(windows, _labels({1: 0}))


def test_fraud_stats_count_and_fraction():
    windows = pd.DataFrame([_window("ACCT-1", "t0", "t30", [1, 2, 3, 4])])
    stats = window_fraud_stats(windows, _labels({1: 1, 2: 1, 3: 0, 4: 0}))
    assert stats.loc[0, "fraud_count"] == 2
    assert stats.loc[0, "fraud_fraction"] == 0.5


def test_build_labeled_windowed_dataset_preserves_feature_columns():
    windows = pd.DataFrame([_window("ACCT-1", "t0", "t30", [1, 2], total_amount=12.5)])
    result = build_labeled_windowed_dataset(windows, _labels({1: 0, 2: 1}))
    assert result.loc[0, "total_amount"] == 12.5
    assert result.loc[0, "fraud_count"] == 1
    assert result.loc[0, "label"] == 1


def test_windows_per_transaction_requires_a_whole_number():
    assert windows_per_transaction(30, 10) == 3
    with pytest.raises(ValueError, match="not a multiple"):
        windows_per_transaction(30, 7)


def _three_windows_covering(ids_by_window):
    return pd.DataFrame(
        [_window("ACCT-1", f"s{i}", f"e{i}", ids) for i, ids in enumerate(ids_by_window)]
    )


def test_coverage_check_passes_when_every_transaction_is_in_three_windows():
    windows = _three_windows_covering([[1, 2], [1, 2], [1, 2]])
    check_window_coverage(windows, [1, 2], 30, 10)


def test_coverage_check_fails_on_missing_transactions():
    windows = _three_windows_covering([[1], [1], [1]])
    with pytest.raises(ValueError, match="1 of 2 transactions are in no window"):
        check_window_coverage(windows, [1, 2], 30, 10)


def test_coverage_check_fails_on_partial_window_membership():
    windows = _three_windows_covering([[1, 2], [1, 2], [1]])
    with pytest.raises(ValueError, match="other than 3"):
        check_window_coverage(windows, [1, 2], 30, 10)


def test_coverage_check_fails_on_unexpected_ids():
    windows = _three_windows_covering([[1, 7], [1, 7], [1, 7]])
    with pytest.raises(ValueError, match="not expected"):
        check_window_coverage(windows, [1], 30, 10)


def test_coverage_check_fails_when_count_disagrees_with_ids():
    windows = _three_windows_covering([[1], [1], [1]])
    windows.loc[0, "transaction_count"] = 5
    with pytest.raises(ValueError, match="transaction_count"):
        check_window_coverage(windows, [1], 30, 10)


def test_scoring_windows_uses_the_earliest_closing_window():
    day = "2026-01-01"
    windows = pd.DataFrame(
        [
            _window("ACCT-1", f"{day}T00:00:10", f"{day}T00:00:40", [1, 2], mean_amount=5.0),
            _window("ACCT-1", f"{day}T00:00:00", f"{day}T00:00:30", [1], mean_amount=3.0),
            _window("ACCT-1", f"{day}T00:00:20", f"{day}T00:00:50", [2], mean_amount=8.0),
        ]
    )
    for column in ("total_amount", "min_seconds_since_last_txn"):
        windows[column] = 0.0

    result = scoring_windows(windows)

    assert result.loc[1, "window_end"] == "2026-01-01T00:00:30"
    assert result.loc[1, "mean_amount"] == 3.0
    assert result.loc[2, "window_end"] == "2026-01-01T00:00:40"
    assert result.loc[2, "mean_amount"] == 5.0
