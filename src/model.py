"""Phase 5 — Models, tuning, and evaluation metrics.

v2 additions over the original:
  #1 Log-target training   -> LogTargetRegressor
  #2 Hyperparameter tuning -> tune_gradient_boosting()
  #3 XGBoost                -> get_model("xgboost" / "xgboost_log")
  #4 Sample weighting      -> fit(..., sample_weight=...)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Columns that are identifiers/targets, never fed into the model as inputs.
NON_FEATURE_COLS = {"load_id", "posted_rate"}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


class LogTargetRegressor(BaseEstimator, RegressorMixin):
    """Wraps any regressor to train on log1p(y) and predict back in the
    original dollar scale.

    Why (#1): posted_rate is right-skewed ($57 to $25,533) and the model's
    biggest errors were all on the expensive, long-haul tail (found while
    digging into the holdout misses). Training in log-space makes the
    model minimize something closer to *relative* error instead of raw
    dollar error, so it stops treating a $10,000 mistake on a $18,000 load
    the same as a $10,000 mistake on a $500 load.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y, sample_weight=None):
        self.base_estimator_ = clone(self.base_estimator)
        if sample_weight is None:
            self.base_estimator_.fit(X, np.log1p(y))
        else:
            self.base_estimator_.fit(X, np.log1p(y), sample_weight=sample_weight)
        return self

    def predict(self, X):
        return np.expm1(self.base_estimator_.predict(X))

    @property
    def feature_importances_(self):
        return self.base_estimator_.feature_importances_


def get_model(name: str, params: dict | None = None):
    """Why these: a linear model as a simple baseline, two tree ensembles
    for non-linear relationships, and log-target/tuned variants of the
    winning gradient boosting model from run_phase4_5.py.
    """
    if name == "linear":
        return LinearRegression()
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=42, **(params or {}))
    if name == "gradient_boosting_log":
        base = GradientBoostingRegressor(random_state=42, **(params or {}))
        return LogTargetRegressor(base)
    if name == "xgboost":
        return xgb.XGBRegressor(random_state=42, n_jobs=-1, **(params or {}))
    if name == "xgboost_log":
        base = xgb.XGBRegressor(random_state=42, n_jobs=-1, **(params or {}))
        return LogTargetRegressor(base)
    raise ValueError(f"Unknown model name: {name}")


def sample_weight_by_rate(y: pd.Series) -> np.ndarray:
    """#4 Sample weighting: weight each row proportionally to its own rate,
    normalized so the average weight is 1.0 (keeps the overall loss scale
    comparable to the unweighted case).

    Why: the model's biggest dollar errors are on expensive, rare, long-haul
    loads (found during error analysis) — those rows are a small fraction of
    the data, so an unweighted loss lets the model mostly ignore them. Weighting
    by rate forces getting the expensive loads right to matter as much as
    getting the cheap, common ones right.
    """
    y = np.asarray(y, dtype=float)
    return y / y.mean()


# --- #2 Hyperparameter tuning -------------------------------------------------

PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 400],
    "learning_rate": [0.03, 0.05, 0.1, 0.15],
    "max_depth": [2, 3, 4, 5],
    "subsample": [0.7, 0.85, 1.0],
    "min_samples_leaf": [1, 5, 10, 20],
}


def tune_gradient_boosting(
    X_train: pd.DataFrame, y_train: pd.Series, log_target: bool = True, n_iter: int = 25
) -> dict:
    """Random search over gradient boosting hyperparameters using
    TimeSeriesSplit (not plain KFold) so every validation fold is always
    LATER in time than its training fold — consistent with the whole
    project's time-based validation approach, avoiding leakage from
    the future into the past during tuning.
    """
    base = GradientBoostingRegressor(random_state=42)
    estimator = LogTargetRegressor(base) if log_target else base
    param_grid = (
        {f"base_estimator__{k}": v for k, v in PARAM_DISTRIBUTIONS.items()}
        if log_target
        else PARAM_DISTRIBUTIONS
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=4),
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_params_, search.best_score_


XGB_PARAM_DISTRIBUTIONS = {
    "n_estimators": [200, 300, 400, 600],
    "learning_rate": [0.02, 0.05, 0.08, 0.1],
    "max_depth": [3, 4, 5, 6],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 5, 10, 20],
}


def tune_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, log_target: bool = True, n_iter: int = 25
) -> tuple[dict, float]:
    """Same idea as tune_gradient_boosting: RandomizedSearchCV + TimeSeriesSplit
    so tuning never validates on data from before its own training fold."""
    base = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    estimator = LogTargetRegressor(base) if log_target else base
    param_grid = (
        {f"base_estimator__{k}": v for k, v in XGB_PARAM_DISTRIBUTIONS.items()}
        if log_target
        else XGB_PARAM_DISTRIBUTIONS
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=4),
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_params_, search.best_score_


# --- #5 Time-based (rolling) cross-validation ---------------------------------


def rolling_time_series_cv(
    model_factory, df_sorted_by_date: pd.DataFrame, feature_cols: list[str], n_splits: int = 5
) -> pd.DataFrame:
    """Evaluate a model across several rolling train/holdout windows instead
    of a single Sep-Oct split.

    Why: one holdout split could be lucky or unlucky by chance. Repeating the
    same time-respecting split multiple times (each fold's holdout is always
    later than its training data) and averaging gives a trustworthy accuracy
    number instead of a single roll of the dice.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    X = df_sorted_by_date[feature_cols]
    y = df_sorted_by_date["posted_rate"]
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        metrics = evaluate(y.iloc[val_idx], preds)
        metrics["fold"] = fold
        metrics["train_rows"] = len(train_idx)
        metrics["val_rows"] = len(val_idx)
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("fold")


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "R2": r2}
