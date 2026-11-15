# Fraud Pipeline: Unsupervised Detectors Explained

Oct 6, 2026

## Overview

Phase 3a compares three unsupervised fraud detectors, which never see fraud labels during training: Isolation Forest, Mahalanobis distance using a robust covariance estimate (MinCovDet), and a PyTorch autoencoder. Their results are compared with a supervised XGBoost model and a rule-based IQR baseline on the Kaggle credit-card dataset (284,807 transactions, about 0.17% fraud).

All models share the same setup, in `notebooks/03_phase3a_comparison.ipynb`:

- **Features:** all 29 columns, V1–V28 plus Amount. The EDA's six-feature Cohen's d shortlist is not used, because that ranking looked at one feature at a time and says nothing about signal in feature combinations.
- **Temporal split:** train on `Time` up to its 70th percentile and test on the rest. A random split would let the model learn from future transactions.
- **Score convention:** each model's `score()` function returns a number where **higher means more anomalous**, so one shared `evaluate()` function can score them all.
- **Training data:** Isolation Forest and Mahalanobis train on the raw training split with fraud included, as they would in production. The autoencoder trains on legitimate rows only (see its section).

## Isolation Forest

Isolation Forest scores a transaction by how quickly random splits separate it from the rest of the data. Each tree repeatedly picks a random feature and a random split value. A point in a dense, normal region takes many splits to isolate, while an outlier is often isolated in one or two. The anomaly score is the average isolation depth across all trees, where a shorter depth means more anomalous.

Code: `src/fraud_pipeline/models/isolation_forest.py`

- `fit_isolation_forest(X_train, random_state=42, **kwargs)` builds `IsolationForest(random_state=42)` and fits it on the full training split, fraud included.
- `score(model, X)` returns `-model.decision_function(X)`. sklearn's convention is that higher means more normal, so the sign is flipped.

Only `random_state` is set in the notebook. Every other parameter is sklearn's default, as recorded in `models/isolation_forest/2026-09-03/metadata.json`.

| Parameter | Value used | What it controls |
| --- | --- | --- |
| `n_estimators` | 100 (default) | Number of trees averaged into the score |
| `max_samples` | `"auto"` = 256 rows per tree | Subsample size used to build each tree |
| `max_features` | 1.0 (default) | Fraction of the 29 features each tree may split on |
| `contamination` | `"auto"` (default) | Sets `offset_`, the cut-off `predict()` uses. It does not change the score ranking, so it has no effect here: thresholds are set separately. |
| `bootstrap` | False (default) | Whether rows are sampled with replacement |
| `random_state` | 42 | Seed for subsampling and splits |

No feature scaling is applied. None is needed, because each split works on one feature at a time and only the ordering of values matters.

## Mahalanobis distance (MinCovDet)

Mahalanobis distance scores a transaction by how far it sits from the center of normal data, measured against the data's own covariance. A step along a direction where normal data varies a lot counts for little. The same step along a direction where normal data barely varies counts for a lot. Plain covariance would be pulled toward the fraud already in the training data, so `MinCovDet` (Minimum Covariance Determinant) estimates the center and covariance from the most tightly clustered subset of rows instead.

Code: `src/fraud_pipeline/models/mahalanobis.py`

1. `StandardScaler` is fit on the training split.
2. `MinCovDet(support_fraction=0.95, random_state=42)` is fit on the scaled training data, fraud included.
3. `score()` returns `model.mahalanobis(scaler.transform(X))`: the squared distance from the robust center. It already increases with anomalousness, so no sign flip is needed.

| Parameter | Value used | What it controls |
| --- | --- | --- |
| `support_fraction` | 0.95 (set explicitly) | Share of rows used for the robust estimate. The default is about 0.5, which assumes up to half the data could be outliers. |
| `random_state` | 42 | Seed for the random subsets the algorithm starts from |
| `assume_centered` | False (default) | Estimate the center rather than assume it is zero |
| `store_precision` | True (default) | Keep the inverse covariance, which `mahalanobis()` uses |
| `StandardScaler` | defaults | Rescales each feature to mean 0 and variance 1 using training-split statistics |

**Why scale first.** In exact arithmetic, Mahalanobis distance is unaffected by feature scale. Amount, though, has about 250× the variance of most V features, and MinCovDet's iterative subsetting and matrix inversion are numerically sensitive to that imbalance.

**Why `support_fraction=0.95`.** The default produced hundreds of "Determinant has increased; this should not happen" warnings, which signal that the algorithm's iterative refinement steps (C-steps) were unstable. Four values were tested: 0.501, 0.75, 0.9 and 0.95 gave 181, 869, 181 and 0 warnings. 0.95 is also statistically sound: it tolerates up to 5% outliers, against an actual fraud rate of about 0.19%.

**Near-singular covariance.** The robust covariance's smallest eigenvalue is about 8.5e-8 (condition number about 15 million), so some test scores reach about 4.4 billion. Ridge shrinkage was tested and rejected. It barely moved ROC-AUC (0.9489 to 0.9480) but cut PR-AUC from 0.53 to 0.07, which suggests the near-flat direction is what pushes many fraud cases to the top of the ranking.

## Autoencoder (PyTorch)

The autoencoder scores a transaction by how badly it can rebuild it after compressing 29 features down to 7. It is trained only on legitimate transactions, so it learns to rebuild legitimate patterns well. Transactions that look different come back with higher reconstruction error. It is the only one of the three that excludes fraud from training. A neural network has no built-in protection against contamination, and with enough capacity it would learn to reconstruct fraud too.

Code: `src/fraud_pipeline/models/autoencoder.py`

- **Architecture:** 29 → 20 → 14 → 7 → 14 → 20 → 29, with ReLU on every hidden layer and a linear output layer.
- **Data:** legitimate training rows, sorted by time. The most recent 15% is held out for early stopping, and the test split is never touched.
- **Scaling:** `StandardScaler` is fit on the fitting rows and saved as `scaler.pkl` next to the weights, so scoring uses exactly the same transform.
- **Training:** Adam optimizer with MSE loss. It stops when validation loss hasn't improved for `patience` epochs, then restores the best weights. The saved run stopped after 65 epochs with validation loss 0.577.
- **Score:** mean squared reconstruction error per row, in scaled feature space.

| Parameter | Value used |
| --- | --- |
| `bottleneck_dim` | 7 |
| `learning_rate` | 1e-3 |
| `batch_size` | 256 |
| `max_epochs` | 200 |
| `patience` | 10 |
| `validation_fraction` | 0.15 |
| `random_state` | 42 |

The module sets `KMP_DUPLICATE_LIB_OK` and `OMP_NUM_THREADS=1` before importing torch. On macOS, without them, training hangs when XGBoost and PyTorch run in the same process.

## Evaluation and results

On PR-AUC, Mahalanobis is the strongest unsupervised model by a wide margin, at 0.53 against 0.11 for the autoencoder and 0.04 for Isolation Forest. At the matched operating point, all three catch 86–89% of fraud.

The shared harness is `src/fraud_pipeline/models/evaluation.py`:

- `evaluate()` computes PR-AUC and ROC-AUC from the continuous scores, and recall, precision and false-positive rate from 0/1 predictions.
- `threshold_for_target_fpr()` picks each model's threshold so its false-positive rate matches the IQR baseline's (about 14.5%). Recall is then compared at the same false-alarm budget. That threshold is chosen using test-set labels, which is acceptable for a comparison table but not for picking a deployment threshold.

| Method | Recall | FPR | Precision | PR-AUC | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| IQR baseline | 91.67% | 14.52% | 0.79% | 0.0074 | 0.8857 |
| Isolation Forest | 88.89% | 14.55% | 0.77% | 0.0396 | 0.9456 |
| Mahalanobis (MinCovDet) | 86.11% | 14.49% | 0.75% | 0.5327 | 0.9489 |
| Autoencoder | 87.04% | 14.29% | 0.77% | 0.1063 | 0.9372 |
| XGBoost (supervised, for reference) | 93.52% | 14.51% | 0.81% | 0.7988 | 0.9825 |

Recall varies little across models, but PR-AUC varies about 100-fold. PR-AUC measures how well fraud is ranked across every threshold, so it separates the models where a single matched point cannot. Trained models and their `metadata.json` files are saved under `models/<name>/<date>/`.

Source: `docs/phase3-overview.md` and `docs/phase3-decisions.md` in the fraud-detection-pipeline repo.
