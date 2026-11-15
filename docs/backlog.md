# Backlog

The single list of open work for the fraud detection pipeline. Update it whenever a
task finishes or a decision is made: tick the box, move the item to **Done**, and date
it. Phase-level status lives in the README roadmap; the reasoning lives in the
`docs/phaseN-*.md` files.

Last updated: 2026-11-14

## Now

- [ ] **Push the Phase 3b commit** to `origin/main` (committed locally 2026-11-14).
- [ ] **Check for a leftover Dataflow job**: `gcloud dataflow jobs list --region us-central1 --status active`.
  Phase 2 defaults to a continuously billed `DataflowRunner`.
- [ ] **Delete the invalid model artifact** `models/xgboost_windowed/2026-09-23/` (git-ignored, trained on
  the invalid first 3b capture). Kept for now; nothing should load it.
- [ ] **Sign off Phases 1 and 2.** Both are built, tested and run end to end against real Pub/Sub and
  Dataflow; only the "complete" flag is missing.

## Phase 3a follow-up: tune the unsupervised models

Plan and reasoning: [`phase3-unsupervised-optimization.md`](phase3-unsupervised-optimization.md).
Doesn't block Phase 4 except for the validation split, which Phase 4's threshold also needs.

- [ ] Quick check: sweep Isolation Forest `max_samples` over 256, 1024, 4096, 0.1 and see whether its
  0.04 PR-AUC moves
- [ ] Add a time-based 60 / 15 / 25 train / validation / test split to the Phase 3a notebook; choose
  thresholds on validation
- [ ] Isolation Forest grid (`max_samples` x `n_estimators` x `max_features`, 48 configs) over 3-5 seeds
- [ ] Mahalanobis: redo shrinkage vs none on validation; compare `EmpiricalCovariance` and `LedoitWolf`;
  optional `RobustScaler`
- [ ] Autoencoder: light sweep (`bottleneck_dim` 4/7/10 x `learning_rate` 1e-3/3e-4); save every
  hyperparameter in `metadata.json`
- [ ] Score the test set once, report mean +/- std across seeds, update `phase3-overview.md`
- [ ] Decide: should the matched-FPR operating point stay tied to the IQR baseline's 14.5%, or move to
  a false-positive budget?

## Phase 4: real-time scoring

Serve the Phase 3a raw-feature XGBoost on the live stream. Phase 3b changed the input: the scorer
reads the **raw** `transactions` topic (its messages already carry V1-V28 + Amount, never `Class`),
not `featurized-transactions`.

- [ ] Write `docs/phase4-decisions.md` before building: input topic, output schema, threshold source,
  separate Dataflow job vs a branch of the feature pipeline
- [ ] Pick the model artifact to serve (`models/xgboost/` has 2026-08-30 and 2026-09-03) and record why
- [ ] Choose the alert threshold on a validation split, not the test set
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

- 2026-11-14: Phase 3b signed off complete. Docs corrected to the valid batch results (the first
  capture's 20% / 116x numbers are superseded); batch builder, ID join and notebook 04 committed.
- 2026-10-10: Phase 3b re-run on complete batch windows: velocity features add no signal; no windowed
  model kept; Phase 3a XGBoost stays the model to serve.
- 2026-10-06: Optimization plan for the unsupervised models written.
- 2026-09-02: Phase 3a complete (four models + IQR baseline compared).
