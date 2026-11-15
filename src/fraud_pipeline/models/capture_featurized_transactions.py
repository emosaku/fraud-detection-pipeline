"""One-off capture utility for Phase 3b: pulls messages from the featurized-transactions
Pub/Sub subscription and appends each to a local JSONL file, so the feature pipeline's
live output can be joined against ground_truth.jsonl afterward to build a labeled
training dataset. Nothing currently persists this output otherwise -- run this
concurrently with the feature pipeline (launched first) and the producer.
"""

import argparse
import concurrent.futures
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "featurized_transactions.jsonl"
LOG_EVERY = 100

log = logging.getLogger("capture_featurized_transactions")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture featurized-transactions messages to a local JSONL file."
    )
    parser.add_argument("--topic", default="featurized-transactions")
    parser.add_argument("--subscription", default="featurized-transactions-capture")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Stop automatically after this many seconds; omit to run until Ctrl+C.",
    )
    return parser.parse_args(argv)


def ensure_subscription(subscriber, project_id: str, topic_id: str, subscription_id: str) -> str:
    from google.api_core.exceptions import NotFound

    topic_path = subscriber.topic_path(project_id, topic_id)
    subscription_path = subscriber.subscription_path(project_id, subscription_id)
    try:
        subscriber.get_subscription(request={"subscription": subscription_path})
    except NotFound:
        subscriber.create_subscription(request={"name": subscription_path, "topic": topic_path})
    return subscription_path


def main(argv=None) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    project_id = os.environ["GCP_PROJECT_ID"]

    from google.cloud import pubsub_v1

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = ensure_subscription(subscriber, project_id, args.topic, args.subscription)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    def callback(message):
        nonlocal count
        with OUTPUT_PATH.open("a") as f:
            f.write(message.data.decode("utf-8") + "\n")
        message.ack()
        count += 1
        if count % LOG_EVERY == 0:
            log.info("captured=%d", count)

    future = subscriber.subscribe(subscription_path, callback=callback)
    log.info("Listening on %s, writing to %s", subscription_path, OUTPUT_PATH)

    try:
        future.result(timeout=args.duration_seconds)
    except concurrent.futures.TimeoutError:
        future.cancel()
        future.result()
    except KeyboardInterrupt:
        future.cancel()
        future.result()

    log.info("Done. captured=%d", count)


if __name__ == "__main__":
    main()
