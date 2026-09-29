"""Helper for december-chart-inputs.csv, which gives city NAMES but no
lat/lon and no market_index/quote_signal (columns our model needs).

Why this is safe: every city's lat/lon is constant everywhere it appears in
train-test.csv (it's real geocoding, not noisy), so a name -> (lat, lon)
lookup table built from train-test.csv is exact, not an approximation.
"""

import pandas as pd


def build_city_lookup(train_test_raw: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by city name with columns lat, lon."""
    pickups = train_test_raw[["pickup", "pickup_lat", "pickup_lon"]].rename(
        columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
    )
    deliveries = train_test_raw[["delivery", "delivery_lat", "delivery_lon"]].rename(
        columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
    )
    all_cities = pd.concat([pickups, deliveries], ignore_index=True).drop_duplicates("city")
    return all_cities.set_index("city")[["lat", "lon"]]


def daily_market_signal(validation_raw: pd.DataFrame) -> pd.DataFrame:
    """Median market_index/quote_signal per calendar date, across all routes.

    Why: those two columns aren't given for the December scenario, and
    they're not a fixed number either (they vary row to row on the same
    date). validation.csv already contains real observed values for every
    day in November-December 2025, so the daily median across all loads
    that day is the best available stand-in for "typical market conditions
    on this date."
    """
    grouped = validation_raw.groupby(validation_raw["date"].dt.date)[
        ["market_index", "quote_signal"]
    ].median()
    grouped.index = pd.to_datetime(grouped.index)
    return grouped
