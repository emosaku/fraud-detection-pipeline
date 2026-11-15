# Backlog

The single list of open work for the fraud detection pipeline. Update it whenever a
task finishes or a decision is made: tick the box, move the item to **Done**, and date
it. Phase-level status lives in the README roadmap; the reasoning lives in the
`docs/phaseN-*.md` files.

Last updated: 2026-11-14

## Now

- [ ] **Check for a leftover Dataflow job**: `gcloud dataflow jobs list --region us-central1 --status active`.
  Phase 2 defaults to a continuously billed `DataflowRunner`.
- [ ] **Delete the invalid model artifact** `models/xgboost_windowed/2026-09-23/` (git-ignored, trained on
  the invalid first 3b capture). Kept for now; nothing should load it.
- [ ] **Sign off Phases 1 and 2.** Both are built, tested and run end to end against real Pub/Sub and
  Dataflow; only the "complete" flag is missing.

## Phase 3a follow-up

Tuning is done ([`phase3a-tuning-results.md`](phase3a-tuning-results.md)). Left over:

- [ ] Decide the alert operating point: keep matching the IQR rule's false-positive rate (now set on
  validation, 16.3%), or set a real false-positive budget. Needed for Phase 4's threshold.
- [ ] Optional: tune XGBoost on the validation split. It was only a reference in the tuning run.
- [ ] Optional: check whether Mahalanobis's near-singular direction survives retraining on a later
  time window. It's the whole signal (0.43 validation PR-AUC from that one direction).

## Phase 4: real-time scoring

Serve the Phase 3a raw-feature XGBoost on the live stream. Phase 3b changed the input: the scorer
reads the **raw** `transactions` topic (its messages already carry V1-V28 + Amount, never `Class`),
not `featurized-transactions`.

- [ ] Write `docs/phase4-decisions.md` before building: input topic, output schema, threshold source,
  separate Dataflow job vs a branch of the feature pipeline
- [ ] Pick the model artifact to serve (`models/xgboost/` has 2026-08-30 and 2026-09-03) and record why
- [ ] Choose the alert threshold on the validation split (`tuning.temporal_split(df, 0.60, 0.15)`), not the test set
- [ ] `scoring/` Beam DoFn that loads the model once per worker (`setup()`) and scores each message
- [ ] Publish scored records (transaction_id, score, flag, model version, scored_at) to a new
  `scored-transactions` topic for Phase 5
- [ ] Parity test: streaming scores match offline `score()` on the same rows exactly
- [ ] Unit tests with a tiny fixture model; DirectRunner smoke test; then one short Dataflow run with
  `--auto-stop-seconds`
- [ ] Measure end-to-end latency (publish time to scored time)
- [ ] Online evaluation: join scored output to `ground_truth.jsonl` by `transaction_id` and confirm
  PR-AUC matches offline

## Phase 5: storage and alerting (BigQuery)

- [ ] Add `google-cloud-bigquery`; set `BIGQUERY_DATASET`
- [ ] Write scored transactions and alerts to BigQuery tables
- [ ] Alert rule and an alerts table/view

## Phase 6: drift detection and retraining triggers

- [ ] Input-drift metrics on the raw features (e.g. PSI per feature) and score-distribution drift
- [ ] Retraining trigger; Cloud Composer for the batch jobs

## Phase 7: dashboard and productionization

- [ ] Streamlit dashboard on Cloud Run for flagged transactions and drift metrics

## Ideas (not scheduled)

- Producer that simulates account-level fraud (compromised accounts, bursts), so velocity features
  can be tested for real. Phase 3b found no signal only because accounts are assigned at random.
- Pub/Sub ordering keys by `account_id`, if `seconds_since_last_txn` must never go negative
  (see `phase2-decisions.md`).

## Done

- 2026-11-14: Phase 3a tuning run. Validation split, Isolation Forest grid (49 configs x 5 seeds),
  Mahalanobis covariance variants, autoencoder sweep. Test PR-AUC: Isolation Forest 0.037 to 0.070,
  autoencoder 0.124 to 0.170, Mahalanobis unchanged at 0.532 (shrinkage confirmed harmful).
  Baseline and tuned results stored in `results/phase3a/`. Autoencoder metadata now records every
  hyperparameter, and non-default bottlenecks reload correctly.
- 2026-11-14: Phase 3b commit pushed to `origin/main`.
- 2026-11-14: Phase 3b signed off complete. Docs corrected to the valid batch results (the first
  capture's 20% / 116x numbers are superseded); batch builder, ID join and notebook 04 committed.
- 2026-10-10: Phase 3b re-run on complete batch windows: velocity features add no signal; no windowed
  model kept; Phase 3a XGBoost stays the model to serve.
- 2026-10-06: Optimization plan for the unsupervised models written.
- 2026-09-02: Phase 3a complete (four models + IQR baseline compared).
