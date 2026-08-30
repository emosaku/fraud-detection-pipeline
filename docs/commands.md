# Essential Commands

A quick reference for working in this repo. Assumes commands are run from the project
root (`fraud-detection-pipeline/`) unless noted otherwise.

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate

# Core + everything used so far (dev tools, Kaggle download, notebooks)
pip install -e ".[dev,data,notebook]"

cp .env.example .env   # then fill in GCP_PROJECT_ID, PUBSUB_TOPIC, etc.
```

## Testing & linting

```bash
pytest -q                    # run the full test suite
ruff check src/ tests/       # lint
ruff format src/ tests/      # auto-format
```

## Data setup (one-time)

Requires `KAGGLE_API_TOKEN` set in `.env` (see kaggle.com → Settings → API → Create New Token).

```bash
python -c "
import kagglehub
print(kagglehub.dataset_download('mlg-ulb/creditcardfraud'))
"
# then copy creditcard.csv from the printed path into data/raw/
```

## Notebooks

```bash
jupyter lab   # open notebooks/01_eda.ipynb, 02_generic_skeleton.ipynb, etc.
```

## GCP / Pub/Sub setup

```bash
gcloud auth login                          # CLI auth (gcloud commands)
gcloud auth application-default login       # ADC (what the Python client library uses)
gcloud config set project <PROJECT_ID>
gcloud services enable pubsub.googleapis.com
```

If either login command fails with `invalid_grant` or similar, see the `gcloud-auth`
skill (`~/.claude/skills/gcloud-auth/`) — re-running the matching login command usually
fixes it.

## Running the producer

```bash
# Dry run — no GCP setup required, prints messages instead of publishing
python -m fraud_pipeline.ingestion.producer --dry-run --limit 1000 --duration-minutes 1

# Small real-mode smoke test (creates the Pub/Sub topic if it doesn't exist)
python -m fraud_pipeline.ingestion.producer --limit 1000 --duration-minutes 1

# Full replay: all ~284,807 rows, compressed into 10 minutes
python -m fraud_pipeline.ingestion.producer --duration-minutes 10
```

Other flags: `--input` (CSV path, default `data/raw/creditcard.csv`), `--num-accounts`
(default 2000), `--seed` (default 42).

## Verifying messages actually publish

A topic with no subscription drops messages published before the subscription existed
— create the subscription first, *then* run the producer, if you want to inspect
output.

```bash
gcloud pubsub subscriptions create transactions-debug \
  --topic=<PUBSUB_TOPIC> --project=<PROJECT_ID>

gcloud pubsub subscriptions pull transactions-debug \
  --project=<PROJECT_ID> --limit=5 --format=json

# decode a pulled message's payload (paste the message.data value)
echo "<base64 data>" | base64 -d | python3 -m json.tool
```

## Running the feature pipeline (Dataflow)

Requires `GCS_BUCKET_NAME` in `.env` in addition to the Pub/Sub setup above. **Launch
before the producer** — like the debug subscription, the pipeline only sees messages
published after it starts.

```bash
gcloud services enable dataflow.googleapis.com --project=<PROJECT_ID>

# Auto-stop after N seconds -- run the producer during that window
python -m fraud_pipeline.features.pipeline --auto-stop-seconds 120

# Or submit and leave running, stop manually later:
python -m fraud_pipeline.features.pipeline
gcloud dataflow jobs cancel <job-id> --region us-central1

# Local, free, no Dataflow job -- for development
python -m fraud_pipeline.features.pipeline --runner DirectRunner
```

## Resetting ground truth between runs

`data/processed/ground_truth.jsonl` is append-only — the producer never clears it.

```bash
rm -f data/processed/ground_truth.jsonl   # start a clean run
```
