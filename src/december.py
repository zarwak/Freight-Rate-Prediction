"""Phase 6 (December file) — build model-ready features for
december-chart-inputs.csv, which only has:
pickup, delivery, distance, equipment, weight, date, predicted_rate(blank)

Missing vs. our normal feature table: pickup_lat/lon, delivery_lat/lon,
market_index, quote_signal. These are filled using src/geo.py lookups
built from real historical data, not guessed.
"""

import pandas as pd

from src.features import add_date_features, add_derived_numeric_features
from src.geo import build_city_lookup, daily_market_signal


def prepare_december_features(
    december_raw: pd.DataFrame, train_test_raw: pd.DataFrame, validation_raw: pd.DataFrame
) -> pd.DataFrame:
    df = december_raw.copy()
    df["date"] = pd.to_datetime(df["date"])

    city_lookup = build_city_lookup(train_test_raw)
    df["pickup_lat"] = df["pickup"].map(city_lookup["lat"])
    df["pickup_lon"] = df["pickup"].map(city_lookup["lon"])
    df["delivery_lat"] = df["delivery"].map(city_lookup["lat"])
    df["delivery_lon"] = df["delivery"].map(city_lookup["lon"])

    market = daily_market_signal(validation_raw)
    df["market_index"] = df["date"].map(market["market_index"])
    df["quote_signal"] = df["date"].map(market["quote_signal"])

    # These two signals aren't part of the original December scenario spec —
    # they're our best estimate, so flag them as imputed (matches src/clean.py
    # semantics: the model has already learned this flag can matter).
    df["weight_was_missing"] = 0
    df["market_index_was_missing"] = 1

    df = add_date_features(df)
    df = add_derived_numeric_features(df)
    df = pd.concat([df, pd.get_dummies(df["equipment"], prefix="equipment")], axis=1)

    return df


def align_to_training_columns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Add any one-hot columns missing because December only has Dry Van,
    and put columns in the exact order the trained model expects."""
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]
