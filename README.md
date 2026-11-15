# Fraud Detection Pipeline

## Overview

This is a portfolio project demonstrating how to build a real-time,
streaming fraud/anomaly detection system using statistical and ML-based
detection methods on top of a cloud-native data pipeline. It replays a
public fraud transaction dataset as a simulated live stream, ingests and
featurizes that stream in near-real-time, scores each transaction for
anomalousness, stores and surfaces the results, and monitors the deployed
model for drift over time.

**Honest scope:** the "real-time" stream is a simulated replay of a static,
public dataset — not a connection to a live payment processor — and the
infrastructure is sized and operated as a personal/portfolio project, not
a production system. The goal is to demonstrate the architecture, tradeoffs,
and engineering practices of a real-time ML system end-to-end, not to stand
up production infrastructure at scale.

## Architecture

The intended end-to-end flow:

```
[Simulated transaction producer]
        |
        v
   Pub/Sub topic  ------------------------------------> (raw event bus)
        |
        v
  Dataflow / Apache Beam  ---> windowed feature engineering
        |
        v
   Real-time scoring  ---> anomaly/fraud model (Isolation Forest /
        |                  autoencoder / XGBoost, trained offline)
        v
    BigQuery  ---> scored transactions + alerts (storage & querying)
        |
        v
  Drift monitoring  ---> triggers retraining when input/model drift detected
        |
        v
  Cloud Run + dashboard (Streamlit)  ---> serving layer for review/inspection
```

Cloud Composer is intended to orchestrate the offline/batch pieces (model
retraining, drift evaluation jobs) once those phases exist.

None of the above is wired up yet — see "Current state" below.

## Phase roadmap

| Phase | Description | Status |
|---|---|---|
| 0 | Project scaffolding, tooling, environment | **Complete** |
| 1 | Simulated transaction-stream producer replaying a public fraud dataset | **In progress** |
| 2 | Streaming ingestion + windowed feature engineering (Pub/Sub + Dataflow/Beam) | **In progress** |
| 3a | Offline model comparison on raw features (Isolation Forest, Mahalanobis, XGBoost, Autoencoder) | **Complete** |
| 3b | Retrain the winning approach on Phase 2's windowed features | **Complete** |
| 4 | Real-time scoring | Not started |
| 5 | Storage + alerting (BigQuery) | Not started |
| 6 | Drift detection & retraining triggers | Not started |
| 7 | Productionization + dashboard (Cloud Run/Streamlit) | Not started |

## Current state

Phases 0 (scaffolding), 3a (offline model comparison) and 3b (windowed-feature
retrain) are complete. Phases 1 and 2 are in progress: both have been built,
unit-tested, and executed end to end against real Pub/Sub and Dataflow, but aren't
flagged complete pending final sign-off. Open work is tracked in
[`docs/backlog.md`](docs/backlog.md). See `docs/phase0-decisions.md`, `docs/phase1-decisions.md`,
`docs/phase2-decisions.md`, `docs/phase3-decisions.md`, and `docs/phase3-overview.md`
(a narrative deep-dive on the whole of Phase 3) for the engineering decisions and
reasoning behind each.

- `data/raw/creditcard.csv` is downloaded locally (git-ignored) and explored in
  `notebooks/01_eda.ipynb` and `notebooks/02_generic_skeleton.ipynb`; see
  `docs/data-dictionary.md` and `docs/eda-strategies.md`
- `src/fraud_pipeline/ingestion/producer.py` replays the dataset as a simulated stream —
  see "Running the producer" below. A full real run (all 284,807 rows) has completed
  successfully against live Pub/Sub.
- `src/fraud_pipeline/features/pipeline.py` is a Beam pipeline computing windowed,
  per-account velocity features (transaction count/amount over sliding windows, time
  since the account's last transaction) and republishing them to a second Pub/Sub topic
  — confirmed against a real Dataflow run with live data flowing through it end to end.
  See "Running the feature pipeline" below.
- **Phase 3a** (offline model comparison, raw features) is complete:
  `src/fraud_pipeline/models/` has a shared evaluation harness plus all four originally
  planned methods — Isolation Forest, Mahalanobis distance (robust `MinCovDet`
  covariance), XGBoost, and a PyTorch autoencoder (trained on legitimate transactions
  only) — all compared against a re-scored IQR baseline in one five-row table in
  `notebooks/03_phase3a_comparison.ipynb`. Trained models save to
  `models/<name>/<date>/` (git-ignored).
- **Phase 3b** (retraining XGBoost, Phase 3a's clear winner, on Phase 2's windowed
  features) is complete. `src/fraud_pipeline/features/batch.py` rebuilds the windowed
  features from the CSV with the same transform the streaming pipeline uses, and
  `src/fraud_pipeline/models/windowed_dataset.py` joins labels by transaction ID.
  `notebooks/04_phase3b_windowed_retrain.ipynb` has the results. In this simulation the
  velocity features carry no fraud signal (synthetic accounts are assigned at random),
  so **no windowed model is kept: Phase 3a's raw-feature XGBoost is the model to serve.**
  A first pass reported a ~20% positive rate and a weaker windowed model; that dataset
  turned out to be invalid and those numbers are superseded. See
  `docs/phase3-overview.md`.
- A tuning pass for the unsupervised Phase 3a models ran on 2026-11-14
  (`src/fraud_pipeline/models/tuning.py`, results in `results/phase3a/`). On a proper
  validation split, tuning roughly doubled Isolation Forest (PR-AUC 0.04 to 0.07),
  improved the autoencoder (about 0.11 to 0.17), and confirmed Mahalanobis's 0.53. The
  ranking didn't change. See `docs/phase3a-tuning-results.md`.
- Everything from Phase 4 onward (real-time scoring, storage, drift monitoring, the
  dashboard) is **not started**

## Setup

Requires Python 3.11+.

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package with dev dependencies (pytest, ruff)
pip install -e ".[dev]"

# Copy the environment template (not needed until later phases,
# but the placeholders are documented now)
cp .env.example .env

# Confirm the test harness works
pytest
```

If you prefer `uv` or `poetry`, they'll work fine against the same
`pyproject.toml` — this project doesn't assume either one is installed.

## Running the producer

`src/fraud_pipeline/ingestion/producer.py` replays `data/raw/creditcard.csv` as a simulated
transaction stream: synthetic account IDs are assigned (Zipf-distributed, so some accounts
are busier than others), rows are replayed in original time order on a compressed timescale,
and each transaction is published as a Pub/Sub message that never includes the `Class`
(fraud) label — that label is written separately to `data/processed/ground_truth.jsonl` for
evaluation later.

**Dry run — no GCP setup required, prints messages to the console instead of publishing:**

```bash
python -m fraud_pipeline.ingestion.producer --dry-run --limit 1000 --duration-minutes 1
```

**Real Pub/Sub mode** — requires `GCP_PROJECT_ID` and `PUBSUB_TOPIC` set (in `.env` or the
environment); the topic is created automatically if it doesn't exist:

```bash
python -m fraud_pipeline.ingestion.producer --duration-minutes 10
```

Other flags: `--input` (CSV path, default `data/raw/creditcard.csv`), `--num-accounts`
(default 2000), `--seed` (default 42, for reproducible account assignment).

## Running the feature pipeline

`src/fraud_pipeline/features/pipeline.py` reads the raw transaction stream from Pub/Sub,
computes per-account velocity features over sliding windows (transaction count, total
and mean amount, time since that account's last transaction — a stateful feature that
persists across windows), and republishes the results to a second Pub/Sub topic
(`featurized-transactions` by default) for Phase 4 to consume later.

Requires `GCP_PROJECT_ID`, `PUBSUB_TOPIC`, and `GCS_BUCKET_NAME` set (`.env` or the
environment) — Dataflow needs the bucket for staging/temp files, in addition to the
Pub/Sub setup the producer needed.

```bash
python -m fraud_pipeline.features.pipeline \
  --window-size-seconds 30 --window-period-seconds 10
```

Defaults to `--runner DataflowRunner`, which deploys a **real, continuously billed**
streaming job on GCP. Pass `--runner DirectRunner` instead to run the same pipeline
locally against the real Pub/Sub topic, at no cost, for development/testing.

**Launch order matters.** Like the debug subscription earlier, `ReadFromPubSub` only
creates its subscription once the job starts, so it only sees messages published
*after* that point — launch the pipeline *before* running the producer, not after:

1. Launch the pipeline (see below for auto-stop) and wait for it to reach `RUNNING`
   (check the Dataflow console link printed at startup, or see `--auto-stop-seconds`).
2. *Then* run the producer — a small `--limit` smoke test is enough to prove the
   pipeline works end to end; no need for the full replay.
3. Check `featurized-transactions` for output (same `gcloud pubsub subscriptions
   create`/`pull` pattern as `docs/commands.md` describes for the raw topic).

**Auto-stop, so a job is never accidentally left running:**

```bash
python -m fraud_pipeline.features.pipeline --auto-stop-seconds 120
```

This submits the job, polls `gcloud dataflow jobs describe` until it reaches
`RUNNING`, lets it run for the given duration, then cancels it automatically — run the
producer during that window. Omit the flag to submit and exit immediately, leaving the
job running until you stop it yourself:

```bash
gcloud dataflow jobs cancel <job-id> --region us-central1
```

## Running the model comparison

`notebooks/03_phase3a_comparison.ipynb` trains and evaluates Isolation Forest,
Mahalanobis distance, XGBoost, and a PyTorch autoencoder on `data/raw/creditcard.csv`
(all V1-V28 + Amount, a temporal 70/30 train/test split), plus a re-scored version of
the IQR baseline from `01_eda.ipynb` so all five appear on equal footing. No GCP
required — everything here is local/offline.

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/03_phase3a_comparison.ipynb
```

Or open it in Jupyter/JupyterLab and run all cells. Trained models are saved to
`models/<name>/<date>/` (git-ignored) each time it runs — `model.pkl`+`metadata.json`
for the first three, `model.pt`+`scaler.pkl`+`metadata.json` for the autoencoder.

## Running the tuning experiment

```bash
python -m fraud_pipeline.models.tuning --stage all   # about 15 minutes on 10 cores
```

`--stage baseline` re-runs the original notebook-03 protocol and writes
`results/phase3a/baseline.json`. `--stage tuned` runs the 60/15/25 split, the grids and
the 5-seed final test scoring, and writes `results/phase3a/tuned.json`. Both files are
committed, so results stay comparable over time.

## Building the windowed training features (batch)

Phase 3b's training data comes from a bounded Beam run over the CSV, not a live capture.
No GCP needed:

```bash
python -m fraud_pipeline.features.batch --duration-minutes 10
```

This writes `data/processed/windowed_features_batch.jsonl` (git-ignored), which
`notebooks/04_phase3b_windowed_retrain.ipynb` reads. `--duration-minutes` should match
the producer replay you want to reproduce; window size/period, seed and account count
default to the same values as the streaming path.

## Tech stack

**In use now:**
- Python 3.11+
- pandas, numpy
- python-dotenv
- google-cloud-pubsub
- apache-beam[gcp] / Google Cloud Dataflow
- scikit-learn — Isolation Forest, Mahalanobis (MinCovDet), evaluation metrics
- XGBoost — supervised fraud classifier baseline
- PyTorch — autoencoder (trained on legitimate transactions only)
- pytest
- ruff (lint + format)

**Introduced in later phases** (see [`docs/future-dependencies.md`](docs/future-dependencies.md)
for exactly when and why):
- Google Cloud BigQuery — storage, querying, and alerting
- Google Cloud Composer — orchestration of offline/batch jobs
- Google Cloud Run — serving the dashboard
- Streamlit — dashboard UI

## Project layout

```
fraud-detection-pipeline/
├── src/fraud_pipeline/       # installable package
│   ├── ingestion/            # Phase 1-2: stream producer + Pub/Sub consumption
│   ├── features/             # Phase 2: windowed feature engineering
│   ├── models/                # Phase 3a-b: offline model development
│   ├── scoring/               # Phase 4: real-time scoring
│   ├── storage/                # Phase 5: BigQuery storage + alerting
│   ├── monitoring/             # Phase 6: drift detection & retraining triggers
│   └── dashboard/               # Phase 7: serving dashboard
├── tests/                     # mirrors src/fraud_pipeline/
├── notebooks/                  # exploratory analysis
├── data/
│   ├── raw/                     # git-ignored
│   └── processed/                # git-ignored
├── models/                      # trained model artifacts, git-ignored (Phase 3+)
├── docs/                        # design notes
├── pyproject.toml
├── .env.example
└── README.md
```
