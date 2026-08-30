# Phase 1 — Engineering Decisions

Scope: `src/fraud_pipeline/ingestion/producer.py` — replays `data/raw/creditcard.csv`
as a simulated real-time transaction stream published to Pub/Sub. Status: built,
unit-tested, and confirmed against a real end-to-end Pub/Sub run (all 284,807 rows
published successfully); not yet flagged complete in the roadmap pending final
sign-off.

## Synthetic account IDs, Zipf-distributed rather than uniform

The dataset has no account/customer identifier, so each transaction is assigned one of
`N` synthetic accounts (default 2000) via a seeded random draw weighted by a Zipf-like
distribution (`weight ∝ rank^-1.2`), not uniformly.

**Why:** uniform assignment would make every synthetic account equally active, which
no real population of accounts looks like. A Zipf-shaped draw gives a small number of
very active accounts and a long tail of rarely-active ones — closer to how real
transaction volume concentrates — which matters once Phase 2's velocity features
(transaction count per account per window) are computed on top of this data. The seed
is fixed for reproducibility; the account ID format (`ACCT-000123`) is a deliberately
readable string rather than a bare integer, matching what a real system's IDs would
look like.

## No label leakage: `Class` never enters the Pub/Sub payload

The message payload includes `transaction_id`, `account_id`, `event_time`,
`original_time_offset`, `amount`, and `V1`–`V28` — explicitly excluding `Class`. The
label is written separately to `data/processed/ground_truth.jsonl`
(`transaction_id`, `account_id`, `class`, `event_time`), append-only, for a later join
against real predictions.

**Why:** this mirrors a real production constraint — a fraud-scoring system never has
the answer at prediction time. Building the leakage boundary into the producer itself
(rather than trusting a downstream consumer to ignore the label) makes it structurally
impossible for later phases to accidentally train or score on the answer. This was
tested directly: unit tests assert `Class`/`class` are absent from the message dict,
and a real run's pulled Pub/Sub messages were decoded and inspected to confirm the
field is genuinely never published, not just usually excluded.

## `transaction_id` = original CSV row index, assigned before sorting

Account IDs and `transaction_id` are attached to each row immediately after loading
the CSV (in original file order), *before* the DataFrame is sorted by `Time` for
replay. The index travels with its row through the sort.

**Why:** this keeps `transaction_id` a stable, deterministic reference back to the
source data regardless of replay order — necessary for the ground-truth join to mean
anything, and for `--limit` (smoke testing) to behave predictably.

## Compressed replay timing, computed from the dataset's own span

Sleep intervals between consecutive publishes are computed as
`(time[i] - time[i-1]) * (target_duration_seconds / total_span_seconds)` — i.e., the
gaps are scaled down uniformly, not replaced with a fixed interval.

**Why:** a fixed interval between messages would erase the original data's
burstiness (transactions that happened close together stay close together, just
faster) — which matters for anything downstream that's sensitive to transaction
timing, including Phase 2's windowed features. Scaling is computed against whatever
subset of the data is actually being replayed (respecting `--limit`), so a smoke test
with `--limit 100` fills its own `--duration-minutes` window rather than replaying a
near-instant sliver of the full 10-minute schedule.

## `--dry-run` as a first-class mode, not an afterthought

The entire message-construction and timing logic runs identically in dry-run and real
modes; only the final publish step branches (`print` vs. `publisher.publish`). GCP
credentials, project ID, and topic are only required when `--dry-run` is *not* set.

**Why:** this let the core logic (account assignment, timing, message shape) be
validated and unit-tested completely independently of any GCP setup — dry-run was used
repeatedly during development before GCP credentials and the topic even existed, and
the test suite never touches real infrastructure.

## Pub/Sub topic auto-creation (`ensure_topic`)

The producer checks whether its target topic exists (`get_topic`) and creates it
(`create_topic`) if not, rather than requiring the topic to be provisioned out of band.

**Why:** one less manual setup step for a fresh run. This convenience turned out to be
specific to how the *producer's own* client library was used — Beam's `WriteToPubSub`
in Phase 2 does **not** do the same auto-creation, which was discovered the hard way
when the `featurized-transactions` topic had to be created manually before the
feature pipeline's first real run.

## Ground truth file: append-only, flushed per write

`ground_truth.jsonl` is opened once in append mode and each record is flushed to disk
immediately after being written, rather than buffered.

**Why:** paired with the graceful `KeyboardInterrupt` handling (catch, log a summary,
exit cleanly), this means a manually interrupted run never loses ground-truth data for
transactions it already published — flushing eagerly trades a small amount of I/O
overhead for that durability guarantee.

## Testing: pure functions extracted and tested without GCP

`assign_synthetic_accounts`, `compute_sleep_intervals`, `build_message`, and
`build_ground_truth_record` are free functions with no GCP dependency, each unit
tested directly. `ensure_topic` and the publish loop itself are not unit tested — they
were instead validated with real runs against real Pub/Sub.

**Why:** the logic worth testing in isolation (distribution shape, timing math, message
shape, leakage) doesn't need real infrastructure to verify, and forcing it to would
mean either mocking GCP (fragile, tests what the mock does rather than what GCP does)
or requiring credentials just to run `pytest`. The thin GCP-interacting layer is
validated the way it actually matters — by running it for real.

## `kagglehub` as a one-time, optional dependency

The Kaggle dataset was downloaded once via `kagglehub.dataset_download(...)` and the
resulting CSV copied into `data/raw/`. `kagglehub` lives in an optional `data` extra
in `pyproject.toml`, not a core dependency, and the producer never imports it — it only
ever reads the already-downloaded local CSV.

**Why:** the download is a one-time acquisition step, unrelated to the ongoing
streaming pipeline. Making it a core dependency would mean every environment running
the producer also needs Kaggle credentials, which isn't true — this was a point of
real confusion during development (Kaggle and Pub/Sub were briefly conflated as related
systems) and keeping the dependency boundary explicit avoids that.
