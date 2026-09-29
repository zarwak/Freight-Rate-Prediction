"""Phase 2 — Cleaning.

Only two columns have missing values (found in Phase 1 EDA):
  - weight        (~0.6% missing)
  - market_index  (~0.8% missing)

Why fill instead of drop: dropping ~650 rows out of 48,000 would lose real
data for no benefit — filling with a sensible estimate keeps the row usable.

This module must work on BOTH train-test.csv and validation.csv, so it never
touches posted_rate (validation.csv doesn't have that column at all).
"""

import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Track which rows were imputed BEFORE filling — this becomes a real
    # feature in Phase 3 (a model can learn "imputed rows behave differently").
    df["weight_was_missing"] = df["weight"].isna().astype(int)
    df["market_index_was_missing"] = df["market_index"].isna().astype(int)

    # Why median-by-equipment for weight: different truck types carry very
    # different typical loads (e.g. Flatbed vs Reefer), so a single global
    # median would be wrong for some rows. Grouping keeps the fill realistic.
    df["weight"] = df.groupby("equipment")["weight"].transform(
        lambda s: s.fillna(s.median())
    )

    # Why median-by-month for market_index: it's described as a market
    # signal, so its typical level likely drifts month to month. A month-local
    # median is closer to the truth than one global number for the whole year.
    month = df["date"].dt.to_period("M")
    df["market_index"] = df.groupby(month)["market_index"].transform(
        lambda s: s.fillna(s.median())
    )

    # Safety net: if an entire group had no values to compute a median from,
    # fall back to the overall column median so nothing is left blank.
    df["weight"] = df["weight"].fillna(df["weight"].median())
    df["market_index"] = df["market_index"].fillna(df["market_index"].median())

    return df


if __name__ == "__main__":
    from src.data import load_train_test, load_validation

    for name, loader in [("train-test", load_train_test), ("validation", load_validation)]:
        df = clean(loader())
        n_missing = df.isna().sum().sum()
        print(f"{name}: remaining missing values after clean() = {n_missing}")
