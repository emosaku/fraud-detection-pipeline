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
| 1 | Simulated transaction-stream producer replaying a public fraud dataset | Not started |
| 2 | Streaming ingestion + windowed feature engineering (Pub/Sub + Dataflow/Beam) | Not started |
| 3 | Offline anomaly/fraud model development (Isolation Forest, autoencoder, XGBoost comparison) | Not started |
| 4 | Real-time scoring | Not started |
| 5 | Storage + alerting (BigQuery) | Not started |
| 6 | Drift detection & retraining triggers | Not started |
| 7 | Productionization + dashboard (Cloud Run/Streamlit) | Not started |

## Current state

**This is scaffolding only.** As of Phase 0, the repository contains:

- The installable package structure (`src/fraud_pipeline/` and its
  subpackages), each currently empty apart from a docstring describing what
  it will hold and which phase introduces it
- A mirrored `tests/` tree with one placeholder test per subpackage, so
  `pytest` has something to discover
- Project tooling: `pyproject.toml`, `ruff` configuration, `.gitignore`,
  `.env.example`
- Empty `notebooks/`, `data/raw/`, `data/processed/`, and `docs/` directories

There is **no working pipeline, no data, no models, and no GCP integration
yet.** Those are built incrementally, phase by phase.

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

## Tech stack

**In use now (Phase 0):**
- Python 3.11+
- pandas, numpy
- python-dotenv
- pytest
- ruff (lint + format)

**Introduced in later phases** (see [`docs/future-dependencies.md`](docs/future-dependencies.md)
for exactly when and why):
- Apache Beam / Google Cloud Dataflow — streaming ingestion & feature engineering
- Google Cloud Pub/Sub — event streaming backbone
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
