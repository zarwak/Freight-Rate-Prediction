"""Phase 4 — Time-based split.

Why not a random shuffle split: train-test.csv covers Jan-Oct 2025, and the
real validation.csv covers Nov-Dec 2025 (the future relative to training).
A random split would mix future-like and past-like rows together, making
the self-check score too optimistic compared to the real task. Splitting
by date instead mimics the real situation: train on the past, validate on
the most recent slice, exactly like predicting what comes next.
"""

import pandas as pd

# Data runs 2025-01-01 to 2025-10-31 (10 months). Holding out the last two
# months (Sep-Oct) as validation gives a reasonably sized holdout (~20% of
# rows) while still leaving 8 months to train on.
CUTOFF_DATE = "2025-09-01"


def time_based_split(df: pd.DataFrame, cutoff: str = CUTOFF_DATE):
    """Split a features dataframe (must still have a usable date signal).

    Expects df to have either a `date` column (raw) — this module is meant
    to run on data_prepared/train_test_features.csv, which no longer has
    `date` after Phase 3. So we split BEFORE dropping the date column,
    or re-attach dates by load_id. Simplest: split happens on the raw
    cleaned dataframe (before build_feature_table drops `date`), see
    run_phase4_5.py for the actual order of operations.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    train = df[df["date"] < cutoff_ts].reset_index(drop=True)
    holdout = df[df["date"] >= cutoff_ts].reset_index(drop=True)
    return train, holdout


if __name__ == "__main__":
    from src.clean import clean
    from src.data import load_train_test

    df = clean(load_train_test())
    train, holdout = time_based_split(df)
    print(f"Train rows:   {len(train):,}  ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Holdout rows: {len(holdout):,}  ({holdout['date'].min().date()} to {holdout['date'].max().date()})")
