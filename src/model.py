"""Phase 5 — Models, tuning, and evaluation metrics.

v2 additions over the original:
  #1 Log-target training  -> LogTargetRegressor
  #2 Hyperparameter tuning -> tune_gradient_boosting()
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
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

    def fit(self, X, y):
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X, np.log1p(y))
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
    raise ValueError(f"Unknown model name: {name}")


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


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}
