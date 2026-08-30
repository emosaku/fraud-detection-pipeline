# Future Dependencies

Phase 0 intentionally installs only what's needed to run and test an empty
skeleton: `pandas`, `numpy`, `python-dotenv`, `pytest`, `ruff`. Everything
below gets added incrementally as its corresponding phase begins, not before.

| Dependency | Introduced in | Why |
|---|---|---|
| `scikit-learn` | Phase 3 | Isolation Forest and general model utilities |
| `pyod` | Phase 3 | Autoencoder and other anomaly detection algorithms |
| `xgboost` | Phase 3 | Supervised fraud classifier for comparison against unsupervised methods |
| `google-cloud-bigquery` | Phase 5 | Storing scored transactions and alerts |
| `streamlit` | Phase 7 | Dashboard for reviewing flagged transactions and drift metrics |

When a phase begins, add its dependencies to `pyproject.toml` under
`[project.dependencies]` (or a phase-specific optional extra, e.g.
`[project.optional-dependencies.streaming]`) rather than installing them
ad hoc.

`google-cloud-pubsub` moved to `[project.dependencies]` in Phase 1 (the producer
publishes to Pub/Sub directly), earlier than originally planned above.

`apache-beam[gcp]` moved to `[project.dependencies]` in Phase 2 (the windowed
feature-engineering pipeline is built and deployed to Dataflow directly).
