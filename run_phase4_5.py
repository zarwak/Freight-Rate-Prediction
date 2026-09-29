"""Phase 4 + 5 (v2): time-based split, then compare:
  baseline          - original gradient_boosting (v1's winner, default params)
  + log target      - change #1 (train on log1p(posted_rate))
  + tuned           - change #2 (hyperparameter search, still log-target)

Run with: python run_phase4_5.py
"""

import pandas as pd

from src.clean import clean
from src.data import load_train_test
from src.features import build_feature_table
from src.model import evaluate, get_feature_columns, get_model, tune_gradient_boosting
from src.split import time_based_split


def main() -> None:
    print("Loading + cleaning train-test.csv...")
    df = clean(load_train_test())

    print("Splitting by date (train: Jan-Aug 2025, holdout: Sep-Oct 2025)...")
    train_raw, holdout_raw = time_based_split(df)

    print("Building features separately for each split...")
    train_feat = build_feature_table(train_raw)
    holdout_feat = build_feature_table(holdout_raw)

    feature_cols = get_feature_columns(train_feat)
    X_train, y_train = train_feat[feature_cols], train_feat["posted_rate"]
    X_holdout, y_holdout = holdout_feat[feature_cols], holdout_feat["posted_rate"]

    print(f"Train rows: {len(X_train):,}  Holdout rows: {len(X_holdout):,}  Features: {len(feature_cols)}")

    results = []

    print("\n[1/3] Baseline gradient_boosting (v1 winner, default params)...")
    baseline = get_model("gradient_boosting")
    baseline.fit(X_train, y_train)
    preds = baseline.predict(X_holdout)
    m = evaluate(y_holdout, preds)
    m["model"] = "baseline"
    results.append(m)
    print(f"  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  MAPE={m['MAPE_%']:.2f}%")

    print("\n[2/3] + log target (change #1), default hyperparams...")
    log_model = get_model("gradient_boosting_log")
    log_model.fit(X_train, y_train)
    preds = log_model.predict(X_holdout)
    m = evaluate(y_holdout, preds)
    m["model"] = "+log_target"
    results.append(m)
    print(f"  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  MAPE={m['MAPE_%']:.2f}%")

    print("\n[3/3] + hyperparameter tuning (change #2), on top of log target...")
    print("  Running RandomizedSearchCV with TimeSeriesSplit(4) on the training slice...")
    best_params_raw, best_cv_score = tune_gradient_boosting(X_train, y_train, log_target=True, n_iter=25)
    best_params = {k.replace("base_estimator__", ""): v for k, v in best_params_raw.items()}
    print(f"  Best params: {best_params}")
    print(f"  Best CV MAE (log-space search score): {-best_cv_score:.4f}")

    tuned_model = get_model("gradient_boosting_log", params=best_params)
    tuned_model.fit(X_train, y_train)
    preds = tuned_model.predict(X_holdout)
    m = evaluate(y_holdout, preds)
    m["model"] = "+log_target+tuned"
    results.append(m)
    print(f"  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  MAPE={m['MAPE_%']:.2f}%")

    results_df = pd.DataFrame(results).set_index("model")[["MAE", "RMSE", "MAPE_%"]]
    print("\n" + "=" * 55)
    print("Comparison (lower is better):")
    print(results_df.round(2))

    print(f"\nBest tuned params to reuse in run_phase6_7.py: {best_params}")


if __name__ == "__main__":
    main()
