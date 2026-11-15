# Phase 3: Anomaly & Fraud Detection Modeling — A Deep Dive

This is a narrative walkthrough of Phase 3 — what each method actually does, why the
comparison is structured the way it is, and what the real results (not idealized
ones) actually show. `docs/phase3-decisions.md` is the terser engineering log of
specific choices and bugs; this document is the story around them.

## Why Phase 3 is two phases, not one

Phase 3a asks a narrow question: on the raw, static dataset, how do four different
detection philosophies compare when evaluated the same way? Phase 3b asks a different
one: does retraining on Phase 2's *streaming, aggregated* features — the actual
representation a deployed system would score against — change the picture? These are
different enough experiments that conflating them would have made both harder to
reason about. Phase 3a's answer had to exist first anyway, since "retrain the winning
approach" (3b's premise) requires already knowing which approach won.

## The four methods, and what each one is actually doing

All four see the same 29 features (`V1`-`V28`, `Amount`) and the same temporal
train/test split (train on the past, test on the chronologically later 30% — see
`docs/phase3-decisions.md` for why temporal, not random). What differs is the
*mechanism* each one uses to decide "how anomalous is this."

**Isolation Forest** works by trying to isolate each point with random splits. Pick a
random feature, pick a random split value, partition the data; repeat. A point that
sits in a dense, "normal" region takes many splits to separate from everything else —
you have to keep slicing through the crowd around it. A point that's already
far from everything else gets isolated almost immediately, often in just one or two
splits. Average that isolation depth over many random trees, and you get a score:
short average depth means anomalous. It never needs labels, and it never needs to be
told what "far" means in any particular direction — it discovers that from wherever
the data happens to be sparse.

**Mahalanobis distance** asks a more geometric question: how far is this point from
the center of "normal," accounting for how the data is actually shaped? Plain
Euclidean distance treats every direction as equally spread out, which is wrong the
moment features are correlated or have different variances — exactly the situation
here, where the 29 features aren't independent. Mahalanobis distance rescales by the
data's own covariance structure first, so "far" means far relative to how much normal
variation exists in that particular direction. The `MinCovDet` robust estimator
computes that covariance in a way specifically designed not to be dragged off-center
by whatever small amount of fraud is sitting in the training data — plain covariance
would let the outliers themselves distort the very reference point used to call them
outliers.

**XGBoost** is the odd one out: it's supervised. It sees `Class` during training and
learns, directly, "what pattern of these 29 features tends to co-occur with fraud."
Everything else in this comparison has to infer anomalousness from structure alone;
XGBoost gets to be told the answer during training and simply learn to predict it.
That's why it's the ceiling in this comparison, not a peer — it has information the
other three are deliberately denied.

**The autoencoder** takes yet another angle: compression. A neural network with a
narrow bottleneck layer is forced to learn a compact internal representation of
whatever it's trained on, then reconstruct the original from that representation. If
it's trained *only* on legitimate transactions (this is the one method here where
that filtering is required — see `docs/phase3-decisions.md` for why), it becomes good
at compressing and rebuilding legitimate-looking patterns specifically. Feed it
something structurally different, and the compression loses information the decoder
can't recover — reconstruction error spikes. That error *is* the anomaly score. This
is the same idea as PCA's reconstruction error (from the earlier `pca-breast-cancer`
project), generalized: PCA can only learn a linear compression; a neural network can
learn a nonlinear one.

## Comparing four different mechanisms fairly

Four different scoring mechanisms produce four scores on four different scales — an
Isolation Forest path length isn't comparable to a Mahalanobis distance, which isn't
comparable to a reconstruction error. Two things make comparison possible anyway:

1. **A shared evaluation function** (`evaluation.py`) that takes whatever a model
   produces and turns it into the same five numbers every time: recall, false
   positive rate, precision, and the two threshold-independent curve summaries
   (PR-AUC, ROC-AUC).
2. **A matched operating point.** Comparing four models each at their own
   arbitrarily-chosen threshold answers nothing useful. Instead, each model's
   threshold is chosen so its false-positive rate matches the IQR baseline's — then
   recall becomes directly comparable: *given the same tolerance for flagging
   legitimate transactions, which model catches more fraud?*

The IQR baseline itself got the same discipline applied to it that a leakier
comparison would have skipped: it's re-derived on the training split only (which
6 features separate the classes, and where the outlier bounds sit) and only then
applied to the test split — not reused verbatim from the original EDA notebook, which
had no train/test split at all.

## Phase 3a's actual results

| Method | Recall | FPR | Precision | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| IQR baseline | 91.67% | 14.52% | 0.79% | 0.0074 | 0.8857 |
| Isolation Forest | 88.89% | 14.55% | 0.77% | 0.0396 | 0.9456 |
| Mahalanobis (MinCovDet) | 86.11% | 14.49% | 0.75% | 0.5327 | 0.9489 |
| XGBoost | 93.52% | 14.51% | 0.81% | **0.7988** | **0.9825** |
| Autoencoder | 87.04% | 14.29% | 0.77% | 0.1063 | 0.9372 |

A few things worth noticing that aren't obvious from the table alone:

- **Recall is bunched tightly (86–94%) but PR-AUC spans two orders of magnitude
  (0.007 to 0.80).** At the specific operating point everyone's matched to, the
  methods look similar. Away from that point — across the whole range of possible
  thresholds — they're not close at all. PR-AUC is the more honest summary of which
  model's underlying ranking is actually better, since it isn't anchored to one
  arbitrarily chosen point.
- **Mahalanobis's PR-AUC (0.53) is dramatically better than Isolation Forest's
  (0.04) despite similar recall.** Isolation Forest's random-split mechanism doesn't
  have a way to weight *which* directions matter most; Mahalanobis's covariance-aware
  distance does, and for this dataset's fraud pattern that turns out to matter a lot.
- **XGBoost's ceiling isn't overwhelming.** 0.98 ROC-AUC vs. Mahalanobis's 0.95 is a
  real gap, but not the kind of gap that makes the unsupervised methods look
  pointless — Mahalanobis gets most of the way there with zero access to labels.

### After tuning (Nov 2026)

The table above is a single run with defaults, and its thresholds were picked on test
labels. A follow-up pass re-ran everything on a 60/15/25 train/validation/test split,
tuned on validation and averaged over 5 seeds
([`phase3a-tuning-results.md`](phase3a-tuning-results.md)). Test PR-AUC after tuning:
Isolation Forest 0.070 (up from about 0.04), autoencoder 0.170 (up from about 0.11),
Mahalanobis 0.532 (unchanged; every shrinkage level hurt), XGBoost 0.785 (untuned).
The ranking held.

It also explained Mahalanobis's lead. Among typical transactions the features are
almost exactly linearly dependent, so one combined direction has nearly zero normal
variance, and fraud lands far out along it. That one direction alone scores 0.43
validation PR-AUC. Isolation Forest, which splits on one feature at a time, can't see
it.

## What almost went wrong: two real numerical bugs

Building Mahalanobis's `MinCovDet` fit surfaced hundreds of `RuntimeWarning:
Determinant has increased; this should not happen` messages — a real violation of the
algorithm's own guarantee that its iterative estimate should only ever improve.
Investigation (not a guess) traced it to `support_fraction`, the parameter
controlling how much contamination the robust estimator tolerates: the default
(~0.501) assumes up to ~50% of the data could be fraud, wildly mismatched against the
real ~0.19% rate. Testing several values empirically — 0.501, 0.75, 0.9, 0.95 — found
0.95 was the first to fully eliminate the warnings (not a smooth function of the
parameter, so guessing wouldn't have worked). Even after that fix, a small fraction of
test points produced Mahalanobis scores in the *billions* — traced to a near-singular
direction in the robust covariance (condition number ~15 million). The standard
fix, ridge regularization, was tested and *rejected*: it measurably hurt PR-AUC
(0.53 → 0.07) while barely moving ROC-AUC, so the "numerically scarier-looking"
unregularized version shipped instead, on the actual evidence rather than instinct.

Separately, training the autoencoder immediately after XGBoost hung indefinitely with
no error at all — both libraries load their own OpenMP runtime, and on macOS having
both loaded in one process can deadlock. Fixed with `KMP_DUPLICATE_LIB_OK` and
`OMP_NUM_THREADS` set inside `autoencoder.py` itself, since that's the module that
introduces the conflict for any process that imports it.

## Phase 3b: from single transactions to account behavior over time

Phase 3a's models all answer the same kind of question: given one transaction's
features, is it fraud? Phase 2's actual deployed pipeline doesn't produce that
representation at all — it produces *windowed* features per account (transaction
count, total/mean amount, time since the last transaction, over sliding time
windows), with `Class` deliberately absent from every message, matching the no-
leakage discipline established all the way back in Phase 1. Retraining on that
representation means answering a fundamentally different question: not "is this
transaction fraud," but "does this account's recent activity look anomalous."

### The first attempt, and why it was thrown out

The first pass captured `featurized-transactions` from a live streaming run and joined
labels from `ground_truth.jsonl` by account and time range, under an "any fraud in the
window" rule. It reported a 20% positive rate (a "116x inflation" of the 0.17% raw
rate) and a windowed model with ROC-AUC 0.79 and PR-AUC 0.60.

Those numbers were wrong, for a reason the time-range join hid. The captured windows
held only about 0.6% of the transactions their labels counted: a median of 1
transaction per window, against 37 in ground truth for the same account and time
range. The join labeled windows using transactions the features never contained. The
first results are superseded.

### Rebuilding the dataset so it can't be wrong in that way

Two changes made the dataset trustworthy:

- **Windows list their own transactions.** Every feature record now carries
  `transaction_ids`, and labels join by ID. A coverage check verifies that every one of
  the 284,807 transactions appears in exactly 3 windows (30-second windows every 10
  seconds) and that each window's count matches its IDs.
- **Batch, not live capture.** `fraud_pipeline.features.batch` runs the *same*
  windowing transform as the streaming pipeline over the CSV, on the producer's
  compressed replay clock. It is complete by construction, reproducible, and free, and
  the shared transform means training and serving features can't drift apart.

### What the valid data shows

**Window labels mostly measure window size.** Under "any fraud", 1.4% of the 69,639
windows are positive (about 8x the raw rate, not 116x). The median fraud window is 2.4%
fraud and holds 44 transactions, so its features mostly describe legitimate activity.
Ranking windows by transaction count alone, with no model at all, beats XGBoost on all
four velocity features (PR-AUC 0.378 vs 0.251). A model trained on that label learns
"busy account", not "fraud". Stricter rules leave too few positives (132 windows at a
50% threshold).

So Phase 3b switched framing: keep each transaction's own label and attach the
velocity features of the first window that could score it. That is also exactly what
Phase 4 has to do.

**Velocity features then add nothing:**

| Features (XGBoost, Phase 3a's split) | PR-AUC | ROC-AUC |
|---|---|---|
| Raw V1-V28 + Amount (Phase 3a) | 0.7988 | 0.9825 |
| Raw + velocity | 0.7922 | 0.9725 |
| Velocity only | 0.0014 | 0.5070 |

Velocity alone is at chance (a random ranking scores 0.0013). The cause is the
simulation, not the idea: the producer assigns transactions to synthetic accounts at
random, independent of fraud, so there are no compromised accounts or bursts for
velocity to detect. The fraud rate is roughly flat across account-activity levels.

**Outcome:** no windowed model is kept. The Phase 3a raw-feature XGBoost is the model
Phase 4 will serve. Testing velocity features for real would need a producer that
simulates account-level fraud behavior.

## The throughline

Every method in Phase 3a solves the same underlying problem — "how far is this from
normal" — with a different notion of distance: random-split isolation depth,
covariance-aware geometric distance, a learned discriminative boundary, or
reconstruction error through a compressed nonlinear bottleneck. None of those
notions is more "correct" in the abstract; each makes different structural
assumptions, and the data itself decides which assumption pays off (here: covariance
structure, per Mahalanobis's PR-AUC, mattered more than either isolation depth or
reconstruction error, though not as much as direct supervision).

Phase 3b's lessons are different and arguably more important for a production system.
First, **the label itself is a design decision, not a given fact**, the moment you
aggregate over a window: the "any fraud" rule quietly turned a fraud detector into a
busy-account detector. Second, **verify a join before trusting anything downstream of
it.** The first pass's numbers were internally consistent and completely wrong, and
only a check that every transaction landed in the windows it should have exposed it.
