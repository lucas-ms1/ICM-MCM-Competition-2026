"""I/O utilities for loading and saving data."""

import pandas as pd
from pathlib import Path


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV. Score columns are kept as strings so 'N/A' is not coerced before melt."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def save_long(df: pd.DataFrame, path: str) -> None:
    """Write long-format table to CSV (e.g. data/ or reports/tables/)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_contestant_week(df: pd.DataFrame, path: str) -> None:
    """Write contestant-week table to CSV (e.g. data/ or reports/tables/)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
