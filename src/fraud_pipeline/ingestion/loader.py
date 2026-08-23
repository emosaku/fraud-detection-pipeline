"""Loads the raw Kaggle credit card fraud CSV into a pandas DataFrame."""

from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "creditcard.csv"


def load_transactions(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)
