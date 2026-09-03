# Phase 3a — Engineering Decisions

Scope: `src/fraud_pipeline/models/` — offline comparison of Isolation Forest,
Mahalanobis distance, XGBoost, and an autoencoder on the raw dataset features, plus a
shared evaluation harness. Status: built, unit-tested, and executed end to end in
`notebooks/03_phase3a_comparison.ipynb` with all five methods (including the re-scored
IQR baseline) in one comparison table; not yet flagged complete pending final sign-off.
Phase 3b (retraining on Phase 2's windowed features) is explicitly out of scope here.

## All 29 features, not the EDA's 6-feature Cohen's d ranking

Isolation Forest, Mahalanobis, and XGBoost all train on the full `V1`-`V28` + `Amount`
feature set, not the 6 features (`V14, V4, V11, V12, V10, V16`) `01_eda.ipynb`'s
Cohen's d ranking flagged as most separating.

**Why:** that ranking answered "which features separate fraud when looked at one at a
time," which is a univariate EDA question — it was never a claim that the other 23
features carry no multivariate signal, and all three of these models are built to
handle correlated, higher-dimensional input without manual feature selection. Using
the full feature set is also what makes the Mahalanobis and Isolation Forest results
meaningfully different from the IQR baseline (which *is* deliberately restricted to
a handful of features) — the comparison is partly about whether more features change
the picture, and pre-filtering to the same 6 would have quietly answered that question
before running anything.

## Temporal train/test split, not random

Split on `Time` at whatever value falls at the 70th percentile of the actual data
(`df["Time"].quantile(0.70)`), not a random shuffle or a hard-coded row count.

**Why:** a random split lets test-set transactions leak information about
"the future" into training in a way a real deployment never would — this system will
eventually score transactions that haven't happened yet, not a random 30% held out
from the past. Computing the cutoff from the data itself (rather than hard-coding a
row index) keeps the split correct if the dataset changes size.

## Unsupervised models trained on unfiltered data, fraud included

Isolation Forest and Mahalanobis both fit on the training split exactly as it comes
(~0.19% fraud), with no step to remove or down-weight the known fraud rows first.

**Why:** a real unsupervised detector never gets to see `Class` to clean its own
training data — filtering fraud out first would be training on information the model
won't have at deployment time, making the offline comparison optimistic in a way that
wouldn't hold up in production. The "robust" part of Mahalanobis's covariance estimate
exists specifically to tolerate this contamination gracefully; testing it against
already-cleaned data would defeat the point of comparing it.

## Mahalanobis: robust covariance (`MinCovDet`), standardized first

Mahalanobis distance is computed from a `MinCovDet` covariance estimate rather than a
plain one, on standardized features.

**Why `MinCovDet` over a plain covariance matrix:** a plain estimate is pulled toward
whatever fraud is present in the training data (its mean and spread directly shape the
covariance), which weakens exactly the "normal" reference point Mahalanobis distance
depends on. `MinCovDet` is built to resist that.

**Why standardize first:** Mahalanobis distance is mathematically scale-invariant in
exact arithmetic, but `MinCovDet`'s *estimation* of the covariance matrix — iterative
subsampling, repeated matrix inversion — is not immune to numerical conditioning
issues in practice, and `Amount` has roughly 250x the raw variance of most `V`
features. This is the same lesson as the standardization work in the earlier PCA
project, applied here for a different reason (numerical stability of an estimator,
not "what PCA finds").

## A real bug caught during this decision: `MinCovDet`'s default `support_fraction`

Fitting `MinCovDet` with its default `support_fraction` (~0.501, implying tolerance
for up to ~50% contamination) produced hundreds of `RuntimeWarning: Determinant has
increased; this should not happen` warnings — a genuine numerical instability in the
C-step iterations, not cosmetic noise (the algorithm is supposed to make the
determinant monotonically decrease each step).

**Investigation, not a guess:** checked the training data for duplicate rows (716
exact, 5,782 duplicate-on-features) and the full covariance's condition number (~60,
not pathological) before concluding the issue was `MinCovDet`'s internal subsampling,
not the underlying data. Tested `support_fraction` in `{0.501, 0.75, 0.9, 0.95}`
empirically (181, 869, 181, and 0 warnings respectively — not a smooth function of the
parameter, so this had to be measured, not assumed) and found 0.95 fully resolved it.
0.95 is also the statistically correct direction to move in: it tolerates far more
contamination than the ~0.19% actually present, which is exactly what MCD guidance
recommends when the true contamination rate is known to be low.

## A finding investigated and reported, not fixed away: the near-singular covariance direction

Even after the `support_fraction` fix, Mahalanobis scores on the test set ranged up to
~4.4 billion — far beyond what a chi-square(29) distribution would suggest for ~2% of
points. Root cause: the robust covariance's smallest eigenvalue is ~8.5e-8, giving a
condition number around 15 million.

The standard fix for a near-singular covariance is ridge/shrinkage regularization
before inverting. Tested at several shrinkage levels against the actual test split:
ROC-AUC barely moved (0.9489 → 0.9480), but **PR-AUC dropped sharply** (0.53 → 0.07) —
the near-singular direction was doing real work concentrating true fraud cases near
the top of the ranking, not just adding numerical noise. At the matched-FPR operating
point, regularization did nudge recall up slightly (86.1% → 89.8%), so the tradeoff
genuinely cuts both ways depending which metric matters more.

**Decision: shipped the unregularized fit**, both because the spec calls for
`MinCovDet` specifically with no mention of additional shrinkage, and because there
was no clean win from adding it — PR-AUC lost more than recall gained. This is
documented directly in the notebook, including the actual numbers, rather than
silently choosing whichever version looked more "numerically well-behaved." Worth
re-checking if this model is ever retrained on different data (Phase 3b's windowed
features, for instance) — there's no guarantee the same fragility helps rather than
hurts on a different feature distribution.

## XGBoost: class weighting, not resampling

`scale_pos_weight` is set to `(1 - positive_rate) / positive_rate` from the training
split's actual fraud rate, rather than oversampling fraud (e.g. SMOTE) or
undersampling legitimate transactions.

**Why:** class weighting changes the loss function's sensitivity to each class without
fabricating synthetic rows or discarding real ones — simpler, leaves the training
distribution's actual structure intact, and is XGBoost's standard, built-in mechanism
for exactly this situation.

## Shared evaluation harness: one function, flexible inputs

`evaluate()` accepts predicted labels, continuous scores, or both, and returns
whichever of `recall`, `precision`, `false_positive_rate`, `pr_auc`, `roc_auc` it can
actually compute from what was given — rather than requiring every caller to pre-compute
a fixed set of arguments or splitting point-metrics and curve-metrics into separate
functions callers have to remember to call together.

**Why:** every model here produces a different kind of raw output (Isolation Forest's
`decision_function`, Mahalanobis's distance, XGBoost's probability, IQR's flat 0/1
rule) — a harness that only accepted one shape would have pushed conversion logic into
every model file instead of one place. The convention "higher score = more anomalous"
is enforced by each model file's own `score()` function (Isolation Forest's sign gets
flipped, for instance), so the harness itself never needs to know which model produced
a given score.

## Re-scoring the IQR baseline under the same protocol, not reusing the old number

`01_eda.ipynb`'s IQR rule (91.7% recall / 16.0% FPR) was computed on the *full*
dataset with no train/test split. For Phase 3a, Cohen's d ranking and the IQR bounds
are both recomputed **on the training split only**, then applied to flag the test
split — the same discipline as the standardization work in the PCA project and the
temporal split above.

**Why:** comparing the three new models' test-set performance against a baseline
number computed on the *entire* dataset (including what's now the test set) wouldn't
be a fair comparison — the old number benefited from information the new models never
see. Re-deriving it under the same split makes the 91.7%/16.0% baseline directly
comparable to the rest of the table, at the cost of it coming out slightly different
(91.7%/14.5% under the temporal split) from the original EDA number.

## Matched-FPR operating point, with an explicit leakage caveat

The comparison table's point metrics (recall, precision, FPR) come from thresholding
each model's continuous score to match the re-scored IQR baseline's false-positive
rate on the test set — not each model's own default threshold.

**Why matching FPR specifically:** comparing three models at three different,
arbitrarily-chosen thresholds doesn't answer a useful question. Fixing the
false-positive "budget" and comparing recall does: given a tolerance for flagging some
fraction of legitimate transactions, which model catches the most fraud?

**The caveat, stated directly in the notebook:** the threshold is chosen using the
*test set's own labels*, which is a form of leakage if treated as a deployment
decision — it's a valid simplification for building one reporting table, not a
substitute for choosing a real decision threshold from a separate validation split.
Introducing a third validation split wasn't part of the specified design and would
have been scope creep beyond what was asked for this pass.

## Model artifacts: pickled, git-ignored, saved with metadata including git commit

Each trained model saves to `models/<name>/<ISO-date>/{model.pkl,metadata.json}` at
the project root; `metadata.json` includes the training timestamp, git commit hash (if
available), the model's own `get_params()` output as hyperparameters, and the metrics
achieved.

**Why capture the git commit:** hyperparameters and metrics alone don't tell you what
*code* produced a given model artifact — if `evaluate()`'s logic or a model's fit
function changes later, the git commit is what lets a saved model be traced back to
the exact implementation that trained it, rather than trusting the hyperparameters
dump to have captured everything that mattered.

## A real bug caught in this pass: `.gitignore`'s `models/` pattern

Adding `models/` to `.gitignore` (intended to exclude the trained-artifact output
directory at the project root) also matched `src/fraud_pipeline/models/` and
`tests/models/` — gitignore patterns without a leading `/` match at any depth, so the
actual Python package implementing this phase and its tests were being silently
excluded from git entirely.

**Caught by:** noticing `git status` didn't list files known to have just been
created, not by any tool warning. Fixed by anchoring the pattern to `/models/`
(project root only). Worth remembering for any future top-level directory name that
also happens to match something meaningful nested inside `src/` or `tests/`.

## Autoencoder: legitimate-only training, unlike the other two unsupervised methods

Isolation Forest and Mahalanobis both train on the full training split, fraud
included — they have built-in tolerance for it (Isolation Forest assumes anomalies
are rare by construction; `MinCovDet` is a robust estimator by design). The
autoencoder trains on `Class == 0` rows only.

**Why:** a neural network has no equivalent safeguard. Given enough capacity and
training time, it will happily learn to reconstruct fraud just as well as legitimate
transactions — at which point reconstruction error stops meaning anything as an
anomaly score. Excluding fraud from training isn't a matter of taste here; it's a
requirement for the method to work at all.

A validation slice for early stopping is carved from the *end* of this already-
legitimate-only, temporally-sorted training data (most recent ~15%) — never from the
real test split, which stays untouched exactly like the other three models.

## Autoencoder: standardize, save the scaler, reuse it at inference

Same reasoning as Mahalanobis: `Amount` has roughly 250x the raw variance of most `V`
features, and a network trained on unscaled inputs would spend most of its capacity
accounting for that one feature's scale rather than learning genuine structure. The
fitted `StandardScaler` is saved alongside the model (`scaler.pkl`) specifically
because scoring new data later requires transforming it through the *same* scaler the
model was trained against — a mismatched scaler would silently produce meaningless
reconstruction errors.

## A real bug caught in this pass: XGBoost + PyTorch deadlock on macOS

Running XGBoost and PyTorch in the same process — exactly what this notebook now
does, back to back — hung indefinitely partway through the autoencoder's very first
training step, with no error or traceback. `fit_xgboost()` completed and scored
normally; `fit_autoencoder()` never returned.

**Diagnosis, not a guess:** reproduced outside Jupyter first (confirmed the
autoencoder alone works fine — 23s, clean exit), then reproduced the combination in a
plain script with unbuffered, flushed print statements at each step to find the exact
hang point. That isolated it to immediately after XGBoost's fit completed and the
autoencoder's training loop began — consistent with a known class of bug: XGBoost and
PyTorch each load and initialize their own OpenMP runtime, and on macOS having both
loaded in one process can deadlock the moment the second library tries to run its own
parallel region after the first has already claimed the thread pool. Confirmed by
setting `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` before either library
loads, which resolved it cleanly both as shell environment variables and as `os.environ`
assignments from within the same Python process.

**Where the fix lives:** inside `autoencoder.py` itself, as a module-level side
effect (`os.environ.setdefault(...)`) before `import torch` — not in the notebook,
and not in `xgboost_baseline.py`. Reasoning: `autoencoder.py` is what introduces the
second OpenMP runtime into any process that imports it, so it's the natural place for
the fix to live regardless of import order, and it protects any future code (tests,
scripts, notebooks) that imports both this module and XGBoost together without
requiring each of them to remember this gotcha independently.
