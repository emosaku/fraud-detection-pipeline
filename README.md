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
| 2 | Streaming ingestion + windowed feature engineering (Pub/Sub + Dataflow/Beam) | Not started |
| 3 | Offline anomaly/fraud model development (Isolation Forest, autoencoder, XGBoost comparison) | Not started |
| 4 | Real-time scoring | Not started |
| 5 | Storage + alerting (BigQuery) | Not started |
| 6 | Drift detection & retraining triggers | Not started |
| 7 | Productionization + dashboard (Cloud Run/Streamlit) | Not started |

## Current state

Phase 0 (scaffolding) is complete. Phase 1 (the transaction-stream producer) is in
progress: the producer code exists, has been validated with `--dry-run`, and passes its
unit tests, but has **not yet been confirmed with a real end-to-end Pub/Sub run** — that
confirmation will flip Phase 1 to complete in the roadmap above.

- `data/raw/creditcard.csv` is downloaded locally (git-ignored) and explored in
  `notebooks/01_eda.ipynb` and `notebooks/02_generic_skeleton.ipynb`; see
  `docs/data-dictionary.md` and `docs/eda-strategies.md`
- `src/fraud_pipeline/ingestion/producer.py` replays the dataset as a simulated stream —
  see "Running the producer" below
- Everything from Phase 2 onward (feature engineering, model training, scoring, storage,
  drift monitoring, the dashboard) is **not started**

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

## Tech stack

**In use now:**
- Python 3.11+
- pandas, numpy
- python-dotenv
- google-cloud-pubsub
- pytest
- ruff (lint + format)

**Introduced in later phases** (see [`docs/future-dependencies.md`](docs/future-dependencies.md)
for exactly when and why):
- Apache Beam / Google Cloud Dataflow — streaming ingestion & feature engineering
- Google Cloud BigQuery — storage, querying, and alerting
- Google Cloud Composer — orchestration of offline/batch jobs
- Google Cloud Run — serving the dashboard
- scikit-learn — Isolation Forest and general model tooling
- pyod — autoencoder and other anomaly detection algorithms
- XGBoost — supervised fraud classifier, for comparison against unsupervised methods
- Streamlit — dashboard UI

## Project layout

```
fraud-detection-pipeline/
├── src/fraud_pipeline/       # installable package
│   ├── ingestion/            # Phase 1-2: stream producer + Pub/Sub consumption
│   ├── features/             # Phase 2: windowed feature engineering
│   ├── models/                # Phase 3: offline model development
│   ├── scoring/               # Phase 4: real-time scoring
│   ├── storage/                # Phase 5: BigQuery storage + alerting
│   ├── monitoring/             # Phase 6: drift detection & retraining triggers
│   └── dashboard/               # Phase 7: serving dashboard
├── tests/                     # mirrors src/fraud_pipeline/
├── notebooks/                  # exploratory analysis
├── data/
│   ├── raw/                     # git-ignored
│   └── processed/                # git-ignored
├── docs/                        # design notes
├── pyproject.toml
├── .env.example
└── README.md
```
