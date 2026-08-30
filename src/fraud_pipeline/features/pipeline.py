"""Windowed velocity-feature engineering: reads the raw transaction stream from Pub/Sub,
computes per-account aggregates over sliding windows, and republishes to a second topic
for Phase 4 (scoring) to consume.
"""

import argparse
import json
import logging
import os
import subprocess
import time

import apache_beam as beam
from apache_beam.coders import FloatCoder
from apache_beam.options.pipeline_options import (
    GoogleCloudOptions,
    PipelineOptions,
    StandardOptions,
)
from apache_beam.transforms import window
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from dotenv import load_dotenv

DEFAULT_WINDOW_SIZE_SECONDS = 30
DEFAULT_WINDOW_PERIOD_SECONDS = 10

log = logging.getLogger("features.pipeline")


class ParseMessage(beam.DoFn):
    """Parses the JSON payload published by the Phase 1 producer, dropping anything malformed."""

    def process(self, element, timestamp=beam.DoFn.TimestampParam):
        try:
            record = json.loads(element.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning("dropping malformed message")
            return
        record["_publish_ts"] = timestamp.micros / 1_000_000
        yield record


class ComputeTimeSinceLastTxn(beam.DoFn):
    """Stateful per-account velocity feature: seconds since this account's previous transaction.

    Requires a keyed PCollection: (account_id, record). Uses Beam's per-key state API, so
    the "previous timestamp" persists across windows and across the life of the streaming job
    for each key -- this is the one feature here that windowing alone cannot produce.
    """

    LAST_TS = ReadModifyWriteStateSpec("last_ts", FloatCoder())

    def process(self, element, last_ts=beam.DoFn.StateParam(LAST_TS)):
        account_id, record = element
        current_ts = record["_publish_ts"]
        previous_ts = last_ts.read()
        record["seconds_since_last_txn"] = None if previous_ts is None else current_ts - previous_ts
        last_ts.write(current_ts)
        yield (account_id, record)


class ComputeVelocityFeatures(beam.DoFn):
    """Aggregates one account's transactions within one sliding window into a feature record."""

    def process(self, element, window=beam.DoFn.WindowParam):
        account_id, records = element
        records = list(records)
        amounts = [r["amount"] for r in records]
        count = len(amounts)
        total_amount = sum(amounts)

        yield {
            "account_id": account_id,
            "window_start": window.start.to_utc_datetime().isoformat(),
            "window_end": window.end.to_utc_datetime().isoformat(),
            "transaction_count": count,
            "total_amount": round(total_amount, 2),
            "mean_amount": round(total_amount / count, 2) if count else 0.0,
            "min_seconds_since_last_txn": _min_or_none(
                r["seconds_since_last_txn"] for r in records
            ),
        }


def _min_or_none(values):
    present = [v for v in values if v is not None]
    return round(min(present), 3) if present else None


def build_pipeline(
    pipeline: beam.Pipeline,
    input_topic: str,
    output_topic: str,
    window_size_seconds: float = DEFAULT_WINDOW_SIZE_SECONDS,
    window_period_seconds: float = DEFAULT_WINDOW_PERIOD_SECONDS,
) -> beam.PCollection:
    return (
        pipeline
        | "ReadFromPubSub" >> beam.io.ReadFromPubSub(topic=input_topic)
        | "ParseJSON" >> beam.ParDo(ParseMessage())
        | "KeyByAccount" >> beam.Map(lambda r: (r["account_id"], r))
        | "AddTimeSinceLastTxn" >> beam.ParDo(ComputeTimeSinceLastTxn())
        | "Window"
        >> beam.WindowInto(window.SlidingWindows(window_size_seconds, window_period_seconds))
        | "GroupByAccount" >> beam.GroupByKey()
        | "ComputeVelocityFeatures" >> beam.ParDo(ComputeVelocityFeatures())
        | "FormatJSON" >> beam.Map(lambda d: json.dumps(d).encode("utf-8"))
        | "WriteToPubSub" >> beam.io.WriteToPubSub(topic=output_topic)
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Windowed velocity-feature pipeline.")
    parser.add_argument("--input-topic", default=None, help="Defaults to PUBSUB_TOPIC from .env")
    parser.add_argument("--output-topic", default="featurized-transactions")
    parser.add_argument("--window-size-seconds", type=float, default=DEFAULT_WINDOW_SIZE_SECONDS)
    parser.add_argument(
        "--window-period-seconds", type=float, default=DEFAULT_WINDOW_PERIOD_SECONDS
    )
    parser.add_argument(
        "--runner", default="DataflowRunner", choices=["DataflowRunner", "DirectRunner"]
    )
    parser.add_argument("--region", default="us-central1")
    parser.add_argument(
        "--auto-stop-seconds",
        type=float,
        default=None,
        help=(
            "DataflowRunner only: wait for the job to reach RUNNING, run for this many "
            "seconds, then cancel it automatically. Omit to submit and exit immediately, "
            "leaving the job running until stopped manually."
        ),
    )
    return parser.parse_known_args(argv)


def _gcloud_job_state(job_id: str, region: str) -> str:
    result = subprocess.run(
        [
            "gcloud",
            "dataflow",
            "jobs",
            "describe",
            job_id,
            "--region",
            region,
            "--format=value(currentState)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def wait_for_job_state(
    job_id: str, region: str, target_states: set[str], timeout_seconds: float
) -> str:
    deadline = time.monotonic() + timeout_seconds
    state = ""
    while time.monotonic() < deadline:
        state = _gcloud_job_state(job_id, region)
        log.info("job %s state=%s", job_id, state)
        if state in target_states:
            return state
        time.sleep(10)
    msg = (
        f"job {job_id} did not reach {target_states} within {timeout_seconds}s (last seen: {state})"
    )
    raise TimeoutError(msg)


def cancel_job(job_id: str, region: str) -> None:
    subprocess.run(["gcloud", "dataflow", "jobs", "cancel", job_id, "--region", region], check=True)
    log.info("Cancelled job %s", job_id)


def main(argv=None) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    args, beam_args = parse_args(argv)
    project_id = os.environ["GCP_PROJECT_ID"]
    bucket = os.environ["GCS_BUCKET_NAME"]
    input_topic = args.input_topic or f"projects/{project_id}/topics/{os.environ['PUBSUB_TOPIC']}"
    output_topic = f"projects/{project_id}/topics/{args.output_topic}"

    pipeline_options = PipelineOptions(beam_args)
    pipeline_options.view_as(StandardOptions).runner = args.runner
    pipeline_options.view_as(StandardOptions).streaming = True
    gcp_options = pipeline_options.view_as(GoogleCloudOptions)
    gcp_options.project = project_id
    gcp_options.region = args.region
    gcp_options.staging_location = f"gs://{bucket}/staging"
    gcp_options.temp_location = f"gs://{bucket}/temp"
    gcp_options.job_name = "fraud-detection-feature-pipeline"

    log.info("input_topic=%s output_topic=%s runner=%s", input_topic, output_topic, args.runner)

    pipeline = beam.Pipeline(options=pipeline_options)
    build_pipeline(
        pipeline,
        input_topic,
        output_topic,
        args.window_size_seconds,
        args.window_period_seconds,
    )
    result = pipeline.run()

    if args.runner == "DirectRunner":
        # Local streaming run: blocks until interrupted (Ctrl+C), same as the producer.
        result.wait_until_finish()
        return

    job_id = result.job_id()
    console_url = f"https://console.cloud.google.com/dataflow/jobs/{args.region}/{job_id}?project={project_id}"
    log.info("Submitted Dataflow job %s", job_id)
    log.info("Console: %s", console_url)

    if args.auto_stop_seconds is None:
        log.info("No --auto-stop-seconds given -- job will keep running until stopped manually:")
        log.info("  gcloud dataflow jobs cancel %s --region %s", job_id, args.region)
        return

    wait_for_job_state(job_id, args.region, {"JOB_STATE_RUNNING"}, timeout_seconds=600)
    log.info("Job running -- letting it process for %.0fs before stopping", args.auto_stop_seconds)
    time.sleep(args.auto_stop_seconds)
    cancel_job(job_id, args.region)


if __name__ == "__main__":
    main()
