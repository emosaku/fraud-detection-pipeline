# Fraud Pipeline: Optimizing the Unsupervised Detectors

Oct 6, 2026

## Summary

Yes, these models need optimization. Isolation Forest is the top priority, and Mahalanobis's "no regularization" decision needs re-checking.

- **Isolation Forest** runs entirely on sklearn defaults with one seed. Its PR-AUC (0.04) is about 13× worse than Mahalanobis's, even though their ROC-AUCs are almost the same. The defaults, especially 256 rows per tree, are a likely cause and haven't been tested.
- **Mahalanobis (MinCovDet)** has its one key parameter, `support_fraction=0.95`, set carefully. The decision not to regularize the covariance, however, was made by comparing PR-AUC on the **test set**.
- **Autoencoder** uses proper early stopping, but its learning rate, batch size and bottleneck width were never varied, and most of them aren't saved in its metadata.
- **Evaluation** has no validation split, so all tuning and threshold choices so far rely on test-set labels.

**Update, Nov 14, 2026:** this plan has been run. Results: [`phase3a-tuning-results.md`](phase3a-tuning-results.md). Isolation Forest roughly doubled (0.04 to 0.07), the autoencoder improved modestly, and the Mahalanobis result held up on a proper validation split.

## How well each parameter is applied

Only the Mahalanobis model's parameters were chosen deliberately. The other two models mostly use starting values that were never tested.

| Model | Parameter | Current value | Assessment |
| --- | --- | --- | --- |
| Isolation Forest | `max_samples` | `"auto"` = 256 rows | **Most important untested setting.** With about 199k training rows at 0.19% fraud, each tree sees on average about 0.5 fraud cases and a very coarse sample of normal data. |
| Isolation Forest | `max_features` | 1.0 | Untested. Random splits on weakly informative features may weaken the signal from the strong ones (V14, V4, V11, V12). |
| Isolation Forest | `n_estimators` | 100 | Untested. 100 trees built on 256-row samples give noisy scores, and results from a single seed can't show how noisy. |
| Isolation Forest | `contamination` | `"auto"` | Has no effect here. It only shifts `predict()`'s cut-off and doesn't change the score ranking. No action needed. |
| Mahalanobis | `support_fraction` | 0.95 | Well handled. Four values were tested, the choice is justified, and it's documented. |
| Mahalanobis | Shrinkage on the covariance | None | Decided by comparing PR-AUC on the test set (0.53 without vs 0.07 with). Needs to be repeated on a validation split. |
| Mahalanobis | Scaler | `StandardScaler` | Acceptable. `RobustScaler` would fit the robust approach better, because Amount's mean and standard deviation are skewed by outliers. Expected impact is small. |
| Autoencoder | `bottleneck_dim`, `learning_rate`, `batch_size` | 7, 1e-3, 256 | Untested. These values aren't saved in `metadata.json`, so the saved model can't be reproduced from its metadata alone. |
| Autoencoder | Early stopping | patience 10, 15% temporal validation | Well handled. The validation data never overlaps the test set. |

## Gaps in the evaluation setup

The largest gap is that there is no validation split. Every choice made so far was checked against the same test set the final numbers are reported on.

- **Decisions made on test labels.** The Mahalanobis shrinkage decision and the matched-FPR thresholds both used test-set labels. The thresholds are documented as a reporting shortcut. The shrinkage decision is a model-selection choice, so the reported 0.53 PR-AUC is probably optimistic.
- **A single seed.** Each model was run once with `random_state=42`. Isolation Forest (random splits) and MinCovDet (random starting subsets) can both shift with the seed, so differences of a few points may just be noise.
- **A fragile Mahalanobis result.** The 0.53 PR-AUC rests on a covariance direction with an eigenvalue near 8.5e-8. A small change in the data, such as a retrain or Phase 3b's features, could make it disappear. It needs to be confirmed on held-out data before the comparison relies on it.

Using labels to choose hyperparameters is standard even for unsupervised detectors. The deployed model still never sees labels; they only decide which configuration to ship.

## Recommended optimization plan

First add a validation split, then tune each model on validation PR-AUC averaged over several seeds, and score the test set only once at the end.

1. **Split by time three ways:** train 60%, validation 15%, test 25%, all cut on `Time`. Thresholds are chosen on the validation split too.
2. **Tune Isolation Forest** with a full grid search (48 configurations × 3–5 seeds):
   - `max_samples`: 256, 1024, 4096, 0.1 (about 12k rows)
   - `n_estimators`: 100, 300, 500, 1000
   - `max_features`: 0.5, 0.75, 1.0
3. **Re-check Mahalanobis:**
   - Keep `support_fraction=0.95`, and confirm it still gives no warnings on the smaller training split.
   - Redo the shrinkage-versus-none comparison on the validation split.
   - Compare against `EmpiricalCovariance` and `LedoitWolf`, to see how much of the 0.53 comes from the robust estimator and how much from the near-singular direction.
   - Optional: swap in `RobustScaler` for `StandardScaler`.
4. **Tune the autoencoder lightly:** `bottleneck_dim` 4, 7, 10 × `learning_rate` 1e-3, 3e-4. Save every hyperparameter in `metadata.json`.
5. **Report results with spread:** give the mean and standard deviation across seeds for each model's best configuration, then score the test set once.

The main metric for model selection is PR-AUC. At about 0.2% fraud, ROC-AUC is dominated by the large number of legitimate transactions and barely separates the models (0.937–0.949).

## Open questions and next steps

The quickest useful check is a sweep of `max_samples` for Isolation Forest. It takes a few minutes and shows whether the defaults explain its 0.04 PR-AUC.

- [x] Quick check: sweep Isolation Forest `max_samples` over 256, 1024, 4096 and 0.1, and see whether PR-AUC improves noticeably
- [x] Add the train / validation / test split (done in `src/fraud_pipeline/models/tuning.py`; notebook 03 stays the original baseline)
- [x] Run the Isolation Forest grid over 3–5 seeds
- [x] Redo the Mahalanobis shrinkage comparison on the validation split
- [x] Save all autoencoder hyperparameters in `metadata.json`
- [x] Re-run the comparison table on the test set once and update `docs/phase3-overview.md`

Open questions:

- Should the matched-FPR operating point stay tied to the IQR baseline's 14.5%, or move to a false-positive budget the business would actually accept?
- If tuning narrows the gap between Isolation Forest and Mahalanobis, does the overview's explanation ("random splits can't weight the important directions") still hold?
