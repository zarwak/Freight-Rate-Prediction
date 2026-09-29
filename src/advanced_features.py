"""#6 Extra feature engineering, tried as a later experiment (not part of the
default build_feature_table pipeline, since distance already carries 97% of
the model's feature importance — these are diminishing-returns additions).
"""

import pandas as pd
from sklearn.cluster import KMeans


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance_sq"] = df["distance"] ** 2
    for col in ("equipment_Dry Van", "equipment_Flatbed", "equipment_Reefer"):
        if col in df.columns:
            df[f"distance_x_{col}"] = df["distance"] * df[col]
    return df


def fit_region_clusters(train_df: pd.DataFrame, n_clusters: int = 8) -> tuple[KMeans, KMeans]:
    """Fit on TRAIN coordinates only, reused (not refit) on holdout/validation
    so cluster ids mean the same region everywhere — avoids leakage."""
    pickup_km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pickup_km.fit(train_df[["pickup_lat", "pickup_lon"]])

    delivery_km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    delivery_km.fit(train_df[["delivery_lat", "delivery_lon"]])

    return pickup_km, delivery_km


def add_region_clusters(df: pd.DataFrame, pickup_km: KMeans, delivery_km: KMeans) -> pd.DataFrame:
    df = df.copy()
    df["pickup_region"] = pickup_km.predict(df[["pickup_lat", "pickup_lon"]])
    df["delivery_region"] = delivery_km.predict(df[["delivery_lat", "delivery_lon"]])
    return df
