"""One-off data validation script for data/raw/creditcard.csv.

Not part of the installable package — run directly with:
    python notebooks/explore_creditcard.py
"""

from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "creditcard.csv"

ACCOUNT_ID_KEYWORDS = ("account", "customer", "card", "user", "id")


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    section("Row count")
    print(f"{len(df):,} rows")

    section("Columns and dtypes")
    print(df.dtypes.to_string())

    section("Class balance")
    counts = df["Class"].value_counts().sort_index()
    pcts = df["Class"].value_counts(normalize=True).sort_index() * 100
    print(f"Class 0 (legitimate): {counts[0]:,} ({pcts[0]:.4f}%)")
    print(f"Class 1 (fraud):      {counts[1]:,} ({pcts[1]:.4f}%)")

    section("Summary statistics — Time")
    print(df["Time"].describe().to_string())

    section("Summary statistics — Amount")
    print(df["Amount"].describe().to_string())

    section("Missing values by column")
    missing = df.isna().sum()
    if missing.sum() == 0:
        print("None — no missing values in any column.")
    else:
        print(missing[missing > 0].to_string())

    section("Account/customer identifier check")
    suspect_cols = [
        col
        for col in df.columns
        if any(keyword in col.lower() for keyword in ACCOUNT_ID_KEYWORDS)
    ]
    if suspect_cols:
        print(f"Found column(s) that look like identifiers: {suspect_cols}")
    else:
        print(
            "No account/customer/card identifier column found, as expected — "
            "this dataset has no entity ID."
        )


if __name__ == "__main__":
    main()
