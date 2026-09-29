"""Try improvements #3-#6 one at a time, log the accuracy trend, pick a winner.

All comparisons use the SAME Sep-Oct 2025 holdout (train_feat/holdout_feat)
as every earlier step in this project, so numbers are directly comparable
to run_phase4_5.py's table.

Run with: python experiments.py
"""

import pandas as pd

from src.advanced_features import (
    add_interaction_features,
    add_region_clusters,
    fit_region_clusters,
)
from src.clean import clean
from src.data import load_train_test
from src.features import build_feature_table
from src.model import (
    evaluate,
    get_feature_columns,
    get_model,
    rolling_time_series_cv,
    sample_weight_by_rate,
    tune_gradient_boosting,
    tune_xgboost,
)
from src.split import time_based_split


def log_result(trend: list, label: str, metrics: dict) -> None:
    metrics = dict(metrics)
    metrics["step"] = label
    trend.append(metrics)
    print(
        f"  MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  "
        f"MAPE={metrics['MAPE_%']:.2f}%  R2={metrics['R2']:.4f}"
    )


def main() -> None:
    trend = []

    print("Loading + cleaning + splitting (Jan-Aug train / Sep-Oct holdout)...")
    df = clean(load_train_test())
    train_raw, holdout_raw = time_based_split(df)
    train_feat = build_feature_table(train_raw)
    holdout_feat = build_feature_table(holdout_raw)
    feature_cols = get_feature_columns(train_feat)
    X_train, y_train = train_feat[feature_cols], train_feat["posted_rate"]
    X_holdout, y_holdout = holdout_feat[feature_cols], holdout_feat["posted_rate"]

    # --- Step 0: current champion from run_phase4_5.py (log target + tuned GB) ---
    print("\n[Step 0] Reference: gradient_boosting + log target + tuning (previous best)")
    gb_params = {
        "subsample": 1.0, "n_estimators": 200, "min_samples_leaf": 20,
        "max_depth": 3, "learning_rate": 0.1,
    }
    gb_model = get_model("gradient_boosting_log", params=gb_params)
    gb_model.fit(X_train, y_train)
    preds = gb_model.predict(X_holdout)
    log_result(trend, "0_gb_log_tuned (reference)", evaluate(y_holdout, preds))

    # --- Step 1 (#3): XGBoost, log target, tuned ---
    print("\n[Step 1] #3 XGBoost + log target, default params first...")
    xgb_default = get_model("xgboost_log")
    xgb_default.fit(X_train, y_train)
    preds = xgb_default.predict(X_holdout)
    log_result(trend, "1a_xgboost_log_default", evaluate(y_holdout, preds))

    print("  Tuning XGBoost (RandomizedSearchCV + TimeSeriesSplit)...")
    xgb_best_params_raw, xgb_cv_score = tune_xgboost(X_train, y_train, log_target=True, n_iter=25)
    xgb_best_params = {k.replace("base_estimator__", ""): v for k, v in xgb_best_params_raw.items()}
    print(f"  Best XGBoost params: {xgb_best_params}  (CV MAE={-xgb_cv_score:.2f})")
    xgb_tuned = get_model("xgboost_log", params=xgb_best_params)
    xgb_tuned.fit(X_train, y_train)
    preds = xgb_tuned.predict(X_holdout)
    log_result(trend, "1b_xgboost_log_tuned", evaluate(y_holdout, preds))

    # Track whichever model (GB or XGB) is winning so far, to build on it.
    trend_df_so_far = pd.DataFrame(trend).set_index("step")
    champion_step = trend_df_so_far["MAE"].idxmin()
    use_xgb = champion_step.startswith("1")
    print(f"\n  Champion so far: {champion_step}  (using {'XGBoost' if use_xgb else 'GradientBoosting'} going forward)")
    champion_name = "xgboost_log" if use_xgb else "gradient_boosting_log"
    champion_params = xgb_best_params if use_xgb else gb_params

    # --- Step 2 (#4): + sample weighting on top of the champion ---
    print("\n[Step 2] #4 + sample weighting (weight rows by their own rate)...")
    weights = sample_weight_by_rate(y_train)
    weighted_model = get_model(champion_name, params=champion_params)
    weighted_model.fit(X_train, y_train, sample_weight=weights)
    preds = weighted_model.predict(X_holdout)
    log_result(trend, "2_champion+sample_weight", evaluate(y_holdout, preds))

    # --- Step 3 (#5): rolling time-based CV of the champion (robustness check) ---
    print("\n[Step 3] #5 Rolling time-based CV (5 folds) of the champion, on ALL of train-test.csv...")
    full_feat = build_feature_table(df)  # whole labeled dataset, still date-sorted
    full_cols = get_feature_columns(full_feat)

    def champion_factory():
        return get_model(champion_name, params=champion_params)

    cv_results = rolling_time_series_cv(champion_factory, full_feat, full_cols, n_splits=5)
    print(cv_results.round(2))
    print(f"  Mean MAE={cv_results['MAE'].mean():.2f}  Mean R2={cv_results['R2'].mean():.4f}  (std MAE={cv_results['MAE'].std():.2f})")

    # --- Step 4 (#6): + extra engineered features on top of the champion ---
    print("\n[Step 4] #6 + interaction features (distance^2, distance x equipment) + region clusters...")
    train_ext = add_interaction_features(train_feat)
    holdout_ext = add_interaction_features(holdout_feat)

    pickup_km, delivery_km = fit_region_clusters(train_raw)  # fit on TRAIN coords only
    train_ext = add_region_clusters(train_ext, pickup_km, delivery_km)
    holdout_ext = add_region_clusters(holdout_ext, pickup_km, delivery_km)

    ext_cols = get_feature_columns(train_ext)
    ext_model = get_model(champion_name, params=champion_params)
    ext_model.fit(train_ext[ext_cols], y_train, sample_weight=weights)
    preds = ext_model.predict(holdout_ext[ext_cols])
    log_result(trend, "4_champion+sample_weight+extra_features", evaluate(y_holdout, preds))

    # --- Trend summary ---
    trend_df = pd.DataFrame(trend).set_index("step")[["MAE", "RMSE", "MAPE_%", "R2"]]
    print("\n" + "=" * 70)
    print("ACCURACY TREND (lower MAE/RMSE/MAPE better, higher R2 better):")
    print(trend_df.round(4))

    best_step = trend_df["R2"].idxmax()
    print(f"\nBest step by R2: {best_step}  (R2={trend_df.loc[best_step, 'R2']:.4f})")


if __name__ == "__main__":
    main()
