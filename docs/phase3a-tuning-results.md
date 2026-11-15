# Phase 3a Tuning: Results

Nov 14, 2026. Plan: [`phase3-unsupervised-optimization.md`](phase3-unsupervised-optimization.md).
Code: `src/fraud_pipeline/models/tuning.py`. Raw results: `results/phase3a/baseline.json` and
`results/phase3a/tuned.json`.

## Summary

Tuning helped Isolation Forest and the autoencoder a little and left Mahalanobis unchanged.
The model ranking didn't change.

- **Isolation Forest roughly doubled** its PR-AUC (0.037 to 0.070 on test). `max_samples` is the
  only setting that matters. It is still far behind Mahalanobis.
- **Mahalanobis held up.** Its 0.53 PR-AUC was suspected of depending on decisions made with test
  labels. On a proper validation split it gets the same 0.53 on new test data, and every shrinkage
  level hurts. The near-singular covariance direction is a real pattern in the data, not a fluke.
- **The autoencoder improved modestly** (0.12 to 0.17), but its results swing a lot between seeds.
  The baseline's single run (0.106) was one draw from a wide range.
- **XGBoost** (supervised, not tuned) stays far ahead at 0.78.

No tuned model is saved. The served model is still Phase 3a's XGBoost.

## How the comparison was run

Three runs, so that the effect of the new split can be separated from the effect of tuning:

| Run | Split | Settings | Threshold for recall/FPR chosen on | Seeds |
|---|---|---|---|---|
| **Baseline** (the original notebook) | 70 / 30 by time | defaults | test labels | 1 (42) |
| **Defaults, new protocol** | 60 / 15 / 25 by time | defaults | validation | 5 |
| **Tuned, new protocol** | 60 / 15 / 25 by time | best mean validation PR-AUC | validation | 5 |

The baseline re-run reproduced the notebook's original table exactly. In the new protocol, every
configuration was fit on train and ranked on validation only. The test split was then scored once,
for the default and the selected configuration of each model.

New split: train 170,888 rows (360 fraud), validation 42,717 (38 fraud), test 71,202 (94 fraud).

## Results on the test set

PR-AUC is the main metric. At 0.13% fraud on test, ROC-AUC hardly separates the models. Recall and
FPR are at a threshold set on validation to match the IQR rule's false-positive rate (16.3% on
validation).

| Model | Baseline PR-AUC | Defaults, new protocol | Tuned, new protocol | Tuned ROC-AUC | Tuned recall / FPR |
|---|---|---|---|---|---|
| IQR rule (reference) | 0.007 | 0.007 | — | 0.876 | 90.4% / 15.2% |
| Isolation Forest | 0.040 | 0.037 ± 0.011 | **0.070 ± 0.003** | 0.946 | 90.0% / 14.8% |
| Mahalanobis (MinCovDet) | 0.533 | 0.532 ± 0.000 | 0.532 ± 0.000 (default kept) | 0.953 | 87.2% / 15.2% |
| Autoencoder | 0.106 | 0.124 ± 0.058 | **0.170 ± 0.029** | 0.959 | 90.9% / 13.7% |
| XGBoost (supervised, untuned) | 0.799 | 0.785 | — | 0.971 | 91.5% / 14.2% |

± is the standard deviation across 5 seeds.

## What changed, model by model

### Isolation Forest: the number of rows per tree was the problem

Average validation PR-AUC over the grid for each setting:

| `max_samples` | 256 (default) | 1024 | 4096 | 0.1 (~17k rows) |
|---|---|---|---|---|
| Mean validation PR-AUC | 0.019 | 0.022 | 0.025 | **0.029** |

`n_estimators` (100 to 1000) and `max_features` (0.5 to 1.0) change PR-AUC by less than 0.001 on
average. The plan's guess was right: with 256 rows per tree, each tree sees about half a fraud case
and a very coarse sample of normal data. Bigger samples help steadily, and the best setting
(`max_samples=0.1, n_estimators=300, max_features=0.75`) also cut the variation between seeds from
±0.011 to ±0.003.

Even tuned, it reaches 0.07 against Mahalanobis's 0.53. The overview's explanation still holds:
random splits on one feature at a time can't find the pattern Mahalanobis uses, described next.

### Mahalanobis: the suspected weakness turned out to be the whole signal

Validation PR-AUC by covariance variant:

| Variant | Validation PR-AUC | Smallest eigenvalue |
|---|---|---|
| MinCovDet (current) | **0.430** | 8.2e-08 |
| MinCovDet + RobustScaler | 0.430 | 9.1e-08 |
| MinCovDet + shrinkage 0.0001 | 0.190 | 5.8e-05 |
| MinCovDet + shrinkage 0.001 | 0.109 | 5.7e-04 |
| MinCovDet + shrinkage 0.01 | 0.058 | 5.7e-03 |
| MinCovDet + shrinkage 0.05 | 0.032 | 2.9e-02 |
| MinCovDet + shrinkage 0.1 | 0.026 | 5.7e-02 |
| Ledoit-Wolf | 0.018 | 5.8e-02 |
| Plain covariance | 0.017 | 3.1e-02 |

Every bit of shrinkage hurts, so the "no regularization" decision is now confirmed on validation
rather than test.

The explanation is a single direction in the data. MinCovDet fits its covariance on the 95% most
typical transactions. Within that group the features are almost exactly linearly dependent:
V1-V28 predict Amount with R² 0.995, against 0.939 on all transactions. That leaves one fixed
combination of the features (mostly V17, V7, V10, V12, V3 and V14) that hardly varies at all for
normal transactions. On validation, the median legitimate transaction sits 0.00007 from the center
along it and the median fraud sits 7.5 away. **Scored on that one direction alone, validation
PR-AUC is 0.433**, which is the whole model's result.

Mahalanobis divides by the variance in each direction, so a direction with almost no normal
variance gets an enormous weight. That is exactly what shrinkage removes. The plain covariance and
Ledoit-Wolf include the outliers, which widens that direction and loses the signal too.

This is real structure in this dataset: V1-V28 come from a PCA of hidden original features, and
fraud breaks a relationship that normal transactions keep. It may not carry over to other data.
Two smaller findings:
- MinCovDet gave the same answer for all 5 seeds.
- RobustScaler changed nothing, as expected, because Mahalanobis distance doesn't depend on how
  each feature is scaled.

### Autoencoder: some gain, mostly noise

| Config (`bottleneck_dim`, `learning_rate`) | Validation PR-AUC |
|---|---|
| 10, 1e-3 (selected) | 0.067 ± 0.009 |
| 7, 3e-4 | 0.065 ± 0.020 |
| 10, 3e-4 | 0.063 ± 0.017 |
| 4, 1e-3 | 0.055 ± 0.010 |
| 4, 3e-4 | 0.054 ± 0.012 |
| 7, 1e-3 (default) | 0.047 ± 0.021 |

The default turned out to be the worst and least stable configuration. The top three are within
each other's spread, so "bottleneck 10" is a reasonable pick but not a clear winner. On test the
tuned model gains about 0.05 PR-AUC and halves the spread between seeds.

All autoencoder hyperparameters are now saved in `metadata.json`. Loading reads the bottleneck
width from there, so a model with a non-default width can be reloaded.

## Why the split alone moved some numbers

- **The test set is different.** It's the last 25% of the data instead of the last 30%, with 94
  fraud cases instead of 108. Training also uses 60% instead of 70%. That's why XGBoost moved from
  0.799 to 0.785 without any tuning.
- **Thresholds now come from validation.** The FPR target (16.3%) is the IQR rule's rate on
  validation. Applied to test, those thresholds give 13.7% to 15.2% FPR, not exactly the target,
  because the two time periods differ a little. The baseline's thresholds were set on the test
  labels, so its recall/FPR figures were slightly optimistic. The PR-AUC figures were not affected.
- **One seed hides a lot.** The autoencoder's spread across seeds (±0.058 with defaults) is larger
  than most of the differences being compared.

## Caveats

- Validation has only 38 fraud cases, so validation PR-AUC is noisy. Rankings among configurations
  within about 0.005 of each other (most of the Isolation Forest grid, the top three autoencoders)
  shouldn't be trusted.
- XGBoost was not tuned. It was included only as a reference.

## Answers to the plan's open questions

- *If tuning narrows the gap between Isolation Forest and Mahalanobis, does the overview's
  explanation still hold?* The gap barely narrowed (0.07 vs 0.53), and the investigation above
  pinned down what Isolation Forest misses: one combined direction, not any single feature. The
  explanation holds.
- *Should the matched-FPR operating point stay tied to the IQR rule?* Still open. It's now set on
  validation, but it is still the IQR rule's rate rather than a real false-positive budget. This
  matters for Phase 4's threshold.

## Reproduce

```bash
python -m fraud_pipeline.models.tuning --stage all   # about 15 minutes on 10 cores
```
