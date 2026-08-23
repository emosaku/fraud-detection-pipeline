# Data Dictionary — `creditcard.csv`

Source: [`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(Kaggle). Stored locally at `data/raw/creditcard.csv` (git-ignored).

## Columns

| Column | Type | Description |
|---|---|---|
| `Time` | float | Seconds elapsed between this transaction and the first transaction in the dataset. Not a wall-clock timestamp. |
| `V1`–`V28` | float | PCA-transformed features. The original transaction features were anonymized for confidentiality by the dataset provider via PCA, so these components can't be interpreted individually (e.g. "merchant category" or "device type") — only their statistical behavior is available to us. |
| `Amount` | float | Transaction amount. |
| `Class` | int (0/1) | Fraud label. `1` = fraud, `0` = legitimate. |

## Class balance

284,807 total transactions:

- **Legitimate (`Class` = 0):** 284,315 (99.8273%)
- **Fraud (`Class` = 1):** 492 (0.1727%)

Severely imbalanced — accuracy is not a meaningful metric on this dataset; use
recall / false-positive rate / precision-recall curves instead.

No missing values in any column.

## No entity identifier

This dataset has **no account, customer, or card identifier column** — each row is a
standalone transaction with no way to group transactions by the entity that made them.
This is expected and confirmed by `notebooks/explore_creditcard.py`.

This matters for later phases: windowed/velocity features (e.g. "transactions per
account in the last 10 minutes") require an entity to window over. Since this dataset
doesn't provide one, we'll simulate synthetic account IDs in a later phase to support
that feature engineering.
