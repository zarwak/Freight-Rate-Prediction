"""Phase 3 — Feature engineering.

Turns cleaned raw columns into a model-ready numeric table.

Golden rule: never build a feature from posted_rate. This module never
looks at that column, so it works identically on train-test.csv and
validation.csv.
"""

import pandas as pd

# Columns kept as-is because they're already numeric and meaningful.
NUMERIC_PASSTHROUGH = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "weight_was_missing",
    "market_index_was_missing",
]


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    # Why: freight demand/pricing often has a weekly and monthly rhythm
    # (e.g. weekend pickups, month-end demand). A raw date can't be used
    # by a model directly, so it's split into small numeric signals.
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Monday .. 6=Sunday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_derived_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    # Why: raw weight alone doesn't say much; weight relative to distance
    # hints at load density, which can influence price.
    df["weight_per_mile"] = df["weight"] / df["distance"].replace(0, pd.NA)
    df["weight_per_mile"] = df["weight_per_mile"].fillna(df["weight_per_mile"].median())
    return df


def encode_equipment(df: pd.DataFrame) -> pd.DataFrame:
    # Why one-hot: equipment only has 3 categories (Dry Van, Flatbed, Reefer)
    # with no natural order, so one-hot avoids implying a false ranking.
    dummies = pd.get_dummies(df["equipment"], prefix="equipment")
    return pd.concat([df, dummies], axis=1)


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature steps and return the final numeric-only table.

    pickup/delivery city names are dropped here: their location is already
    captured numerically by pickup_lat/lon and delivery_lat/lon, and the
    city pair alone is too sparse (64 cities -> 4,000+ combinations across
    48,000 rows) to be a useful categorical feature on its own.
    """
    df = add_date_features(df)
    df = add_derived_numeric_features(df)
    df = encode_equipment(df)

    equipment_cols = [c for c in df.columns if c.startswith("equipment_")]
    geo_cols = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
    date_cols = ["month", "day_of_week", "is_weekend"]

    feature_cols = (
        ["load_id"]
        + NUMERIC_PASSTHROUGH
        + ["weight_per_mile"]
        + geo_cols
        + date_cols
        + equipment_cols
    )
    keep_cols = [c for c in feature_cols if c in df.columns]

    # posted_rate isn't a feature, but keep it alongside if present (train-test.csv)
    # so downstream training code has target + features in one table.
    if "posted_rate" in df.columns:
        keep_cols = keep_cols + ["posted_rate"]

    return df[keep_cols]


if __name__ == "__main__":
    from src.clean import clean
    from src.data import load_train_test, load_validation

    train_features = build_feature_table(clean(load_train_test()))
    val_features = build_feature_table(clean(load_validation()))

    print("train-test features shape:", train_features.shape)
    print("validation features shape:", val_features.shape)
    print("\nColumns:", list(train_features.columns))
    print("\nSample rows:")
    print(train_features.head(3))
