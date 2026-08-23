# EDA Strategies — `01_eda.ipynb` vs `02_generic_skeleton.ipynb`

Both notebooks analyze the same dataset (`data/raw/creditcard.csv`) but answer different
questions. `01_eda.ipynb` asks "how do we spot fraud in *this* data?" `02_generic_skeleton.ipynb`
asks "what does a reusable EDA process look like, and what happens to it when labels aren't
available?" They're complementary, not redundant — read together, the second explains why the
first's techniques worked as well as they did.

## `01_eda.ipynb` — domain-specific anomaly analysis

Strategy: use the `Class` label throughout to build intuition for what fraud looks like in this
feature space, then formalize that intuition into a measurable baseline.

1. **Shape and structure** — `shape`, `head`, `info`, `describe` to get oriented.
2. **Class balance** — establish the 0.17% fraud rate up front, since it governs every
   evaluation choice downstream (accuracy is meaningless; recall/false-positive rate matter).
3. **Feature ranking by separation** — Cohen's d between fraud and normal transactions on each
   `V` column, to turn 28 anonymous PCA components into a short, ranked list worth inspecting.
4. **Visual confirmation** — boxplots and a 2D scatter on the top-ranked features, to see the
   separation Cohen's d implied.
5. **Rule-based baseline** — Tukey's IQR outlier rule applied to the top features, evaluated
   against `Class` (91.7% recall, 16.0% false-positive rate). The rule itself never looks at
   `Class`; only its evaluation does — mirroring how Phase 3's unsupervised models will be
   built and scored.
6. **Time/Amount checks** — quick look at whether fraud clusters by hour or transaction size.

This notebook optimizes for **finding real signal in this specific dataset**, using every piece
of information available (including labels) to get there fastest.

## `02_generic_skeleton.ipynb` — reusable skeleton, labeled vs. blind

Strategy: apply a four-phase pattern that should transfer to *any* labeled tabular problem —
**quality check → target understanding → signal-finding → naive baseline** — and, at the last
two phases, run two tracks side by side:

- **Labeled path** — the same Cohen's d ranking and IQR baseline as `01_eda.ipynb`.
- **Blind path** — `Class` is dropped before signal-finding even starts. Feature ranking falls
  back to raw variance (the standard unsupervised proxy), and the same IQR rule is built on
  *those* features instead. `Class` is only reintroduced afterward, to measure — not to build —
  the blind detector.

Key results from running both tracks:

| | features selected | recall | false-positive rate |
|---|---|---|---|
| labeled | `V14, V4, V11, V12, V10, V16` | 91.7% | 16.0% |
| blind | `V1, V2, V3, V4, V5, V6` | 78.7% | 15.5% |

The blind path's feature choice isn't wrong by accident — `V1`–`V28` are PCA components, which
are variance-ordered by construction, so ranking by variance just re-derives `V1, V2, V3...`
regardless of fraud relevance. At nearly identical cost (flag rate, false-positive rate), the
blind detector misses 13 points more fraud — and in a genuinely blind pipeline, there'd be no
`Class` left to reveal that gap. You could ship the weaker detector with no way to know it's
weaker.

This notebook optimizes for **understanding which parts of the process depend on having
labels**, using the same dataset as a controlled comparison rather than a target to solve.

## When to reach for which

- Analyzing a **new dataset you'll actually model** → follow `01_eda.ipynb`'s pattern: let the
  label drive feature selection and validation from the start.
- Approaching a **new problem where labels are missing, delayed, or uncertain** → follow
  `02_generic_skeleton.ipynb`'s blind track: rely on variance/structure-based signal-finding,
  and treat any evaluation number as provisional until real ground truth exists.
