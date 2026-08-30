# Phase 2 — Engineering Decisions

Scope: `src/fraud_pipeline/features/pipeline.py` — a Beam pipeline that reads the raw
transaction stream from Pub/Sub, computes windowed per-account features, and
republishes them. Status: built, unit-tested, and confirmed against a real Dataflow
run with live data flowing through it end to end; not yet flagged complete pending
final sign-off.

## Real Dataflow deployment, not just `DirectRunner`

The pipeline defaults to `--runner DataflowRunner` (a real, billed streaming job on
GCP), with `--runner DirectRunner` available for free local development against the
same live Pub/Sub topic.

**Why:** this was an explicit choice, not a default — `DirectRunner`-only would have
been free and simpler, but wouldn't demonstrate actual Dataflow deployment (worker
provisioning, GCS staging, job lifecycle management), which is a meaningfully different
and more realistic skill than running Beam locally. The tradeoff is real ongoing cost
while a job runs, which shaped several other decisions below.

## Feature set: per-account windowed velocity features

Sliding windows (30s size / 10s period by default) compute, per account: transaction
count, total amount, mean amount, and — via a stateful `DoFn`, not windowing —
seconds since that account's previous transaction.

**Why:** these are standard, well-understood fraud signals (sudden bursts of activity,
unusually large amounts relative to an account's pattern) and, more specifically, the
"time since last transaction" feature is the one thing in this pipeline that
*requires* real per-key state rather than just windowed aggregation — a window only
sees what happened inside it, so knowing about the previous transaction (possibly in
an earlier window, or even before the job started, for an account seen once already)
needs Beam's state API (`ReadModifyWriteStateSpec`). Including it was a deliberate
choice to use — and demonstrate — a Beam capability that windowing alone can't
provide.

## Window sizes recalibrated to the replay's actual pace, not the original dataset's

Defaults are 30-second windows sliding every 10 seconds — chosen for the compressed
replay's wall-clock pace (the full dataset replays in ~10 minutes), not the original
dataset's ~2-day span.

**Why:** the initial plan sketched window sizes (1-min, 5-min, 60-min) that made sense
for the original data's real timescale, but the producer compresses that entire span
into a ~10-minute run — a 60-minute window literally wouldn't fit inside a single run.
Window sizing had to follow from how the upstream producer actually behaves, not from
the source dataset's original timescale.

## Event-time windowing on Pub/Sub publish time, not `original_time_offset`

Windowing uses each message's Pub/Sub publish timestamp (via `beam.DoFn.TimestampParam`,
which Beam attaches automatically from `ReadFromPubSub`) rather than the dataset's own
`Time` column (carried through as `original_time_offset` for debugging only).

**Why:** the producer already re-stamps every message with a real `event_time` at
publish time — that's the actual "when did this happen" for a live stream, and it's
what a real system would window on. `original_time_offset` is preserved for
debugging/traceability back to the source row, but was never meant to drive windowing
once the data is actually flowing as a stream.

## Output goes to a second Pub/Sub topic, not a database

Featurized records are published to `featurized-transactions` (JSON, same shape
convention as the producer's messages) rather than written to BigQuery or any other
store.

**Why:** BigQuery is explicitly Phase 5's concern in the roadmap, and introducing it
here would blur the phase boundary for no real benefit yet. A second Pub/Sub topic
keeps the same decoupled-producer/consumer pattern Phase 1 established — Phase 2 has
no idea what will eventually read `featurized-transactions` (Phase 4's scorer, most
likely), the same way Phase 1's producer has no idea Phase 2 exists.

## `apache-beam[gcp]` promoted to a core dependency immediately

Same pattern as `google-cloud-pubsub` in Phase 1: moved out of
`docs/future-dependencies.md` into `pyproject.toml`'s real dependencies the moment it
was actually needed, rather than staying "planned for later" while already being used.

## GCP infra: dedicated staging bucket, not a reused existing one

A new bucket (`<project-id>-fraud-dataflow`) was created specifically for Dataflow
staging/temp files, even though the shared GCP project already had several other
buckets from unrelated prior work (an old Cloud Composer environment, App Engine
defaults, an ETL bucket).

**Why:** reusing an unrelated existing bucket would couple this project's Dataflow
staging artifacts to infrastructure with a different purpose and lifecycle. A
dedicated bucket keeps this project's cloud footprint identifiable and cleanly
removable later without affecting anything else in the shared project.

## Output topic had to be created manually — `WriteToPubSub` doesn't auto-create

Unlike the producer's own `ensure_topic()` helper (Phase 1), Beam's `WriteToPubSub`
assumes its target topic already exists. This was caught before the first real run
only because it was checked proactively; `featurized-transactions` was created via
`gcloud pubsub topics create` ahead of time.

**Why it matters as a decision, not just a fact:** it's a real asymmetry between the
two halves of this pipeline worth knowing about explicitly — the producer's
convenience doesn't extend to the consumer side, and assuming otherwise would have
caused the pipeline's first real run to fail on its first write.

## Launch order: pipeline before producer

Same root cause as the debug-subscription lesson from Phase 1 (a subscription only
receives messages published after it exists) — Beam's `ReadFromPubSub` creates its own
subscription when the job starts, so the Dataflow job has to reach `RUNNING` *before*
the producer publishes anything it's meant to process.

**Why this is documented as a decision rather than left implicit:** it's an easy
mistake to make exactly once (and was, in an earlier debug-subscription check), so the
correct order is written directly into the README's "Running the feature pipeline"
section and into the launch script's own behavior (auto-stop waits for `RUNNING`
before its timer starts) rather than relying on remembering it.

## Built-in auto-stop, because a streaming job bills continuously

`--auto-stop-seconds N` submits the job, polls `gcloud dataflow jobs describe` until
it reaches `RUNNING`, waits `N` seconds, then cancels it automatically
(`gcloud dataflow jobs cancel`). Omitting the flag submits and exits immediately,
leaving the job running until stopped manually.

**Why:** a streaming Dataflow job does not stop on its own — unlike the producer,
which naturally finishes when it runs out of rows to replay, a forgotten Dataflow job
keeps costing money indefinitely. Building automatic, bounded lifecycle management into
the launch path (rather than relying on remembering to cancel it by hand) turns
"launch a real cloud job" from a small ongoing operational risk into a bounded,
self-cleaning action — which is what made repeatedly testing this phase against real
Dataflow practical in the first place.

## Testing: Beam's local test utilities, same "no real infra" boundary as Phase 1

`TestPipeline` / `assert_that` cover message parsing, the stateful time-since-last-txn
`DoFn`, and window-level aggregation — all without GCP. One real Beam edge case
surfaced while writing these tests: `GlobalWindow`'s default start timestamp is outside
Python's `datetime` representable range, which crashed a test that skipped real
windowing. The fix was to route test data through actual `WindowInto` with explicit
timestamps (matching how the pipeline genuinely runs), not to change the pipeline code
to avoid the edge case — the pipeline was never wrong; the test didn't yet resemble
production usage.

## Known nuance, documented rather than hidden: out-of-order arrival

A real run produced a negative `seconds_since_last_txn` value
(`-4.082`) — which shouldn't be possible if messages for one account are always
processed in strict publish order. Pub/Sub doesn't guarantee ordering by default, and
parallel Dataflow workers can process two messages for the same account slightly out
of true publish order. This doesn't affect the count/sum/mean features (order-
independent), only the ordering-sensitive one. It's noted here rather than silently
tolerated: the real fix, if strict ordering is ever required, is Pub/Sub's ordering
keys feature (keyed by `account_id`), not yet implemented.
