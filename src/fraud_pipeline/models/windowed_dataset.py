"""Builds labeled datasets from Phase 2's windowed velocity features for Phase 3b.

Every window record lists the `transaction_ids` it aggregated, so labels are joined by
ID. The first version of this module re-derived window membership from timestamps
instead, and labeled windows using hundreds of transactions the captured features never
included (docs/phase3-decisions.md has the numbers).

Two framings are supported:
- Window-level: one row per window, labeled from the fraud among its own transactions
  (`build_labeled_windowed_dataset`, with a configurable label rule).
- Transaction-level: one row per transaction, carrying the features of the first window
  that could score it (`scoring_windows`), so the label is the transaction's own.
"""

import json
from pathlib import Path

import pandas as pd

LABEL_COLUMN = "label"
FEATURE_COLUMNS = [
    "transaction_count",
    "total_amount",
    "mean_amount",
    "min_seconds_since_last_txn",
]
LABEL_RULES = ("any", "fraction")


def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def windows_per_transaction(window_size_seconds: float, window_period_seconds: float) -> int:
    """How many sliding windows each transaction belongs to: size / period, which must be
    a whole number for that count to be the same for every transaction."""
    ratio = window_size_seconds / window_period_seconds
    if not float(ratio).is_integer():
        raise ValueError(
            f"window size {window_size_seconds}s is not a multiple of period "
            f"{window_period_seconds}s"
        )
    return int(ratio)


def _explode_ids(featurized_df: pd.DataFrame) -> pd.Series:
    """One row per (window, transaction), keeping the window's index."""
    return featurized_df["transaction_ids"].explode().astype("int64")


def check_window_coverage(
    featurized_df: pd.DataFrame,
    transaction_ids,
    window_size_seconds: float,
    window_period_seconds: float,
) -> None:
    """Raises ValueError unless every expected transaction appears in exactly
    size / period windows, no window lists a transaction outside `transaction_ids`, and
    each window's `transaction_count` matches the IDs it lists.

    This is the check that would have caught the first Phase 3b capture, where windows
    held about 0.6% of the transactions that were actually published.
    """
    expected_per_txn = windows_per_transaction(window_size_seconds, window_period_seconds)
    appearances = _explode_ids(featurized_df).value_counts()
    expected = pd.Index(transaction_ids)

    problems = []
    missing = expected.difference(appearances.index)
    if len(missing):
        problems.append(f"{len(missing):,} of {len(expected):,} transactions are in no window")
    unexpected = appearances.index.difference(expected)
    if len(unexpected):
        problems.append(f"{len(unexpected):,} windowed transaction ids are not expected")
    wrong = appearances[appearances.index.isin(expected) & (appearances != expected_per_txn)]
    if len(wrong):
        problems.append(
            f"{len(wrong):,} transactions are in a number of windows other than {expected_per_txn}"
        )
    listed = featurized_df["transaction_ids"].str.len()
    count_mismatch = featurized_df["transaction_count"] != listed
    if count_mismatch.any():
        problems.append(
            f"{int(count_mismatch.sum()):,} windows have transaction_count != len(transaction_ids)"
        )
    if problems:
        raise ValueError("window coverage check failed: " + "; ".join(problems))


def window_fraud_stats(featurized_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Per window: how many of its own transactions are fraud, and what fraction.

    `labels_df` needs `transaction_id` and `class` columns (`ground_truth.jsonl`'s shape;
    the CSV's `Class` must be renamed). Every windowed transaction must have a label.
    """
    ids = _explode_ids(featurized_df)
    classes = labels_df.set_index("transaction_id")["class"]
    window_classes = ids.map(classes)
    if window_classes.isna().any():
        n_missing = int(window_classes.isna().sum())
        raise ValueError(f"{n_missing:,} windowed transactions have no label")

    fraud_count = window_classes.groupby(level=0).sum().astype(int)
    fraud_count = fraud_count.reindex(featurized_df.index, fill_value=0)
    return pd.DataFrame(
        {
            "fraud_count": fraud_count,
            "fraud_fraction": fraud_count / featurized_df["transaction_ids"].str.len(),
        },
        index=featurized_df.index,
    )


def label_windows(
    featurized_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    rule: str = "any",
    min_fraud_fraction: float = 0.5,
) -> pd.Series:
    """Window labels under one of two rules:

    - "any": 1 if any of the window's own transactions is fraud.
    - "fraction": 1 if at least `min_fraud_fraction` of them are fraud.
    """
    if rule not in LABEL_RULES:
        raise ValueError(f"unknown label rule {rule!r}; expected one of {LABEL_RULES}")
    stats = window_fraud_stats(featurized_df, labels_df)
    if rule == "any":
        labels = stats["fraud_count"] >= 1
    else:
        labels = stats["fraud_fraction"] >= min_fraud_fraction
    return labels.astype(int).rename(LABEL_COLUMN)


def build_labeled_windowed_dataset(
    featurized_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    rule: str = "any",
    min_fraud_fraction: float = 0.5,
) -> pd.DataFrame:
    df = featurized_df.copy()
    df = df.join(window_fraud_stats(featurized_df, labels_df))
    df[LABEL_COLUMN] = label_windows(featurized_df, labels_df, rule, min_fraud_fraction)
    return df


def scoring_windows(featurized_df: pd.DataFrame) -> pd.DataFrame:
    """For each transaction, the window features available the first time the streaming
    pipeline could score it: those of the earliest-closing window that contains it.

    Indexed by `transaction_id`. That window closes up to one window period after the
    transaction, so its features can include a few later transactions from the same
    account -- the same information a live scorer would have when that window fires.
    """
    exploded = featurized_df.explode("transaction_ids").rename(
        columns={"transaction_ids": "transaction_id"}
    )
    exploded["transaction_id"] = exploded["transaction_id"].astype("int64")
    exploded["_window_end"] = pd.to_datetime(exploded["window_end"], format="ISO8601")
    earliest = exploded.sort_values(["transaction_id", "_window_end"]).drop_duplicates(
        "transaction_id"
    )
    return earliest.set_index("transaction_id")[
        ["account_id", "window_start", "window_end", *FEATURE_COLUMNS]
    ]
