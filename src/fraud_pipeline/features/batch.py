"""Batch builder for Phase 2's windowed velocity features -- the source of Phase 3b's
training set.

Phase 3b's first dataset was captured from the live streaming pipeline and turned out to
cover only ~0.6% of the transactions its labels counted (see docs/phase3-decisions.md).
This module runs the same windowing and aggregation (`WindowedVelocityFeatures`) over the
CSV as a bounded input instead, so every transaction lands in every window it belongs to
and a rerun produces the same output.

Time base: the streaming pipeline windows on Pub/Sub publish time, i.e. the producer's
compressed replay clock. Batch reproduces that clock exactly -- each transaction's
timestamp is its cumulative offset from the producer's own `compute_sleep_intervals` --
so a 30-second window here covers the same slice of the dataset that a 30-second window
covers in a live replay of the same `--duration-minutes`.
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import apache_beam as beam
import numpy as np
import pandas as pd
from apache_beam.transforms import window

from fraud_pipeline.features.pipeline import (
    DEFAULT_WINDOW_PERIOD_SECONDS,
    DEFAULT_WINDOW_SIZE_SECONDS,
    WindowedVelocityFeatures,
)
from fraud_pipeline.ingestion.loader import RAW_DATA_PATH
from fraud_pipeline.ingestion.producer import compute_sleep_intervals, load_and_prepare

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "windowed_features_batch.jsonl"

# Arbitrary fixed replay start, so output is identical run to run. Aligned to a whole
# minute so window boundaries fall on round numbers relative to the start of the replay.
REPLAY_START = datetime(2026, 1, 1, tzinfo=UTC)

log = logging.getLogger("features.batch")


def build_replay_records(
    df: pd.DataFrame, duration_seconds: float, start: datetime = REPLAY_START
) -> list[dict]:
    """One record per transaction, timestamped on the producer's replay clock.

    `df` must come from `producer.load_and_prepare` -- sorted by `Time`, with the same
    `transaction_id` and synthetic `account_id` a live replay with the same seed publishes.
    `_publish_ts` mirrors the field `ParseMessage` sets in the streaming pipeline.
    """
    offsets = np.cumsum(compute_sleep_intervals(df["Time"].to_numpy(), duration_seconds))
    start_ts = start.timestamp()
    return [
        {
            "transaction_id": int(transaction_id),
            "account_id": account_id,
            "amount": float(amount),
            "_publish_ts": start_ts + float(offset),
        }
        for transaction_id, account_id, amount, offset in zip(
            df["transaction_id"], df["account_id"], df["Amount"], offsets, strict=True
        )
    ]


class AddTimeSinceLastTxnSorted(beam.DoFn):
    """Batch equivalent of `ComputeTimeSinceLastTxn`.

    The streaming version keeps the previous timestamp in per-key state, which relies on
    elements arriving roughly in time order. Beam guarantees no order in batch, so this
    takes each account's full history (one GroupByKey in the global window), sorts it, and
    emits each record with its gap and its event timestamp restored. Ties on timestamp
    break by `transaction_id`, matching the producer's stable sort.
    """

    def process(self, element):
        account_id, records = element
        previous_ts = None
        for record in sorted(records, key=lambda r: (r["_publish_ts"], r["transaction_id"])):
            current_ts = record["_publish_ts"]
            gap = None if previous_ts is None else current_ts - previous_ts
            previous_ts = current_ts
            yield window.TimestampedValue(
                (account_id, {**record, "seconds_since_last_txn": gap}), current_ts
            )


def build_batch_pipeline(
    pipeline: beam.Pipeline,
    records: list[dict],
    window_size_seconds: float = DEFAULT_WINDOW_SIZE_SECONDS,
    window_period_seconds: float = DEFAULT_WINDOW_PERIOD_SECONDS,
) -> beam.PCollection:
    return (
        pipeline
        | "CreateRecords" >> beam.Create(records)
        | "KeyByAccount" >> beam.Map(lambda r: (r["account_id"], r))
        | "GroupAccountHistory" >> beam.GroupByKey()
        | "AddTimeSinceLastTxn" >> beam.ParDo(AddTimeSinceLastTxnSorted())
        | "WindowedVelocityFeatures"
        >> WindowedVelocityFeatures(window_size_seconds, window_period_seconds)
    )


def write_windowed_features(
    records: list[dict],
    output_path: Path,
    window_size_seconds: float = DEFAULT_WINDOW_SIZE_SECONDS,
    window_period_seconds: float = DEFAULT_WINDOW_PERIOD_SECONDS,
) -> Path:
    """Runs the batch pipeline locally (DirectRunner) and writes one JSON window per line."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with beam.Pipeline(runner="DirectRunner") as pipeline:
        (
            build_batch_pipeline(pipeline, records, window_size_seconds, window_period_seconds)
            | "FormatJSON" >> beam.Map(json.dumps)
            | "WriteJSONL" >> beam.io.WriteToText(str(output_path), shard_name_template="")
        )
    return output_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build windowed velocity features from the CSV in batch."
    )
    parser.add_argument("--input", default=str(RAW_DATA_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=10.0,
        help="Replay duration to reproduce; match the producer's --duration-minutes.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-accounts", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size-seconds", type=float, default=DEFAULT_WINDOW_SIZE_SECONDS)
    parser.add_argument(
        "--window-period-seconds", type=float, default=DEFAULT_WINDOW_PERIOD_SECONDS
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    df = load_and_prepare(args.input, args.limit, args.num_accounts, args.seed)
    records = build_replay_records(df, args.duration_minutes * 60)
    log.info(
        "Built %d replay records (%.1f-minute replay clock)", len(records), args.duration_minutes
    )

    output_path = write_windowed_features(
        records, Path(args.output), args.window_size_seconds, args.window_period_seconds
    )
    log.info("Wrote windowed features to %s", output_path)


if __name__ == "__main__":
    main()
