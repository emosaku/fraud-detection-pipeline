"""Replays data/raw/creditcard.csv as a simulated real-time transaction stream to Pub/Sub."""

import argparse
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from fraud_pipeline.ingestion.loader import RAW_DATA_PATH

GROUND_TRUTH_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "ground_truth.jsonl"
)

ZIPF_EXPONENT = 1.2
LOG_EVERY = 1000

log = logging.getLogger("producer")


def assign_synthetic_accounts(n_rows: int, num_accounts: int, seed: int) -> np.ndarray:
    """Assigns each row to one of `num_accounts` synthetic accounts via a Zipf-like distribution."""
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, num_accounts + 1)
    weights = 1.0 / ranks**ZIPF_EXPONENT
    weights /= weights.sum()
    return rng.choice(num_accounts, size=n_rows, p=weights)


def format_account_id(index: int) -> str:
    return f"ACCT-{index:06d}"


def compute_sleep_intervals(times: np.ndarray, target_duration_seconds: float) -> np.ndarray:
    """Compresses gaps between sorted `times` so they sum to `target_duration_seconds`."""
    times = np.asarray(times, dtype=float)
    span = times[-1] - times[0]
    if span <= 0:
        return np.zeros(len(times))
    compression = target_duration_seconds / span
    deltas = np.diff(times, prepend=times[0])
    return deltas * compression


def build_message(row: dict, transaction_id: int, event_time: str) -> dict:
    """Builds the Pub/Sub payload for one transaction. Never includes `Class`."""
    message = {
        "transaction_id": int(transaction_id),
        "account_id": row["account_id"],
        "event_time": event_time,
        "original_time_offset": float(row["Time"]),
        "amount": float(row["Amount"]),
    }
    for i in range(1, 29):
        key = f"V{i}"
        message[key] = float(row[key])
    return message


def build_ground_truth_record(
    transaction_id: int, account_id: str, class_value: int, event_time: str
) -> dict:
    return {
        "transaction_id": int(transaction_id),
        "account_id": account_id,
        "class": int(class_value),
        "event_time": event_time,
    }


def ensure_topic(publisher, project_id: str, topic_id: str) -> str:
    from google.api_core.exceptions import NotFound

    topic_path = publisher.topic_path(project_id, topic_id)
    try:
        publisher.get_topic(request={"topic": topic_path})
    except NotFound:
        publisher.create_topic(request={"name": topic_path})
    return topic_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay creditcard.csv as a simulated transaction stream."
    )
    parser.add_argument("--input", default=str(RAW_DATA_PATH))
    parser.add_argument("--duration-minutes", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-accounts", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_and_prepare(
    input_path: str, limit: int | None, num_accounts: int, seed: int
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if limit is not None:
        df = df.head(limit)

    df = df.reset_index(drop=False).rename(columns={"index": "transaction_id"})

    account_indices = assign_synthetic_accounts(len(df), num_accounts, seed)
    df["account_id"] = [format_account_id(i) for i in account_indices]

    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    df = load_and_prepare(args.input, args.limit, args.num_accounts, args.seed)
    sleep_intervals = compute_sleep_intervals(df["Time"].to_numpy(), args.duration_minutes * 60)

    publisher = None
    topic_path = None
    if args.dry_run:
        log.info("Dry run — messages will be printed, not published to Pub/Sub")
    else:
        from google.cloud import pubsub_v1

        project_id = os.environ["GCP_PROJECT_ID"]
        topic_id = os.environ["PUBSUB_TOPIC"]
        publisher = pubsub_v1.PublisherClient()
        topic_path = ensure_topic(publisher, project_id, topic_id)
        log.info("Publishing to %s", topic_path)

    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)

    published = 0
    fraud_seen = 0
    start = time.monotonic()

    try:
        with GROUND_TRUTH_PATH.open("a") as gt_file:
            for i, row in enumerate(df.itertuples(index=False)):
                if i > 0:
                    time.sleep(max(float(sleep_intervals[i]), 0.0))

                row_dict = row._asdict()
                event_time = datetime.now(UTC).isoformat()
                message = build_message(row_dict, row_dict["transaction_id"], event_time)

                payload = json.dumps(message).encode("utf-8")
                if args.dry_run:
                    print(payload.decode("utf-8"))
                else:
                    publisher.publish(topic_path, payload)

                gt_record = build_ground_truth_record(
                    row_dict["transaction_id"],
                    row_dict["account_id"],
                    row_dict["Class"],
                    event_time,
                )
                gt_file.write(json.dumps(gt_record) + "\n")
                gt_file.flush()

                published += 1
                fraud_seen += int(row_dict["Class"])

                if published % LOG_EVERY == 0:
                    elapsed = time.monotonic() - start
                    log.info(
                        "published=%d fraud_seen=%d elapsed=%.1fs", published, fraud_seen, elapsed
                    )
    except KeyboardInterrupt:
        elapsed = time.monotonic() - start
        log.info(
            "Interrupted. published=%d fraud_seen=%d elapsed=%.1fs", published, fraud_seen, elapsed
        )
        return

    elapsed = time.monotonic() - start
    log.info("Done. published=%d fraud_seen=%d elapsed=%.1fs", published, fraud_seen, elapsed)


if __name__ == "__main__":
    main()
