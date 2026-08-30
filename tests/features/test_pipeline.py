import json
import time

import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
from apache_beam.transforms import window as beam_window

from fraud_pipeline.features.pipeline import (
    ComputeTimeSinceLastTxn,
    ComputeVelocityFeatures,
    ParseMessage,
)


def _msg(**fields):
    return json.dumps(fields).encode("utf-8")


def test_parse_message_valid_json():
    with TestPipeline() as p:
        result = (
            p | beam.Create([_msg(account_id="ACCT-1", amount=10.0)]) | beam.ParDo(ParseMessage())
        )
        assert_that(
            result | beam.Map(lambda r: {k: v for k, v in r.items() if k != "_publish_ts"}),
            equal_to([{"account_id": "ACCT-1", "amount": 10.0}]),
        )


def test_parse_message_drops_malformed():
    with TestPipeline() as p:
        result = (
            p
            | beam.Create([b"not json", _msg(account_id="ACCT-1", amount=1.0)])
            | beam.ParDo(ParseMessage())
        )
        assert_that(result | beam.Map(lambda r: r["account_id"]), equal_to(["ACCT-1"]))


def test_compute_time_since_last_txn_first_seen_is_none():
    with TestPipeline() as p:
        records = [("ACCT-1", {"account_id": "ACCT-1", "_publish_ts": 100.0})]
        result = p | beam.Create(records) | beam.ParDo(ComputeTimeSinceLastTxn())
        assert_that(
            result | beam.Map(lambda kv: kv[1]["seconds_since_last_txn"]),
            equal_to([None]),
        )


def test_compute_velocity_features_aggregates_correctly():
    with TestPipeline() as p:
        elements = [
            ("ACCT-1", {"amount": 10.0, "seconds_since_last_txn": None}),
            ("ACCT-1", {"amount": 20.0, "seconds_since_last_txn": 5.0}),
            ("ACCT-1", {"amount": 30.0, "seconds_since_last_txn": 2.0}),
        ]
        # Route through a real window, same as the pipeline does -- ComputeVelocityFeatures
        # reads window.start/window.end, which only exist meaningfully post-windowing.
        # beam.Create() defaults to MIN_TIMESTAMP (-infinity), which -- even after
        # windowing -- is still out of Python datetime's representable range, so the test
        # elements need an explicit, realistic timestamp assigned first.
        now = time.time()
        grouped = (
            p
            | beam.Create(elements)
            | beam.Map(lambda e: beam_window.TimestampedValue(e, now))
            | beam.WindowInto(beam_window.FixedWindows(60))
            | beam.GroupByKey()
        )
        result = grouped | beam.ParDo(ComputeVelocityFeatures())

        def check(outputs):
            assert len(outputs) == 1
            record = outputs[0]
            assert record["account_id"] == "ACCT-1"
            assert record["transaction_count"] == 3
            assert record["total_amount"] == 60.0
            assert record["mean_amount"] == 20.0
            assert record["min_seconds_since_last_txn"] == 2.0

        assert_that(result, check)
