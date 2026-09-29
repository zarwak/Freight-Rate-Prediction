"""Phase 6 + 7: train final model on ALL labeled data, predict the two
required output files, then run score.py to validate + build the chart.

Run with: python run_phase6_7.py

Why retrain on everything now (not just Jan-Aug): Phase 4/5 held out
Sep-Oct only to HONESTLY measure how good the model is. Once we trust the
model (gradient_boosting won on MAE/RMSE/MAPE in run_phase4_5.py), the
holdout rows are just more real data being thrown away if we don't use
them for the real, final predictions.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.clean import clean
from src.data import load_train_test, load_validation
from src.december import align_to_training_columns, prepare_december_features
from src.features import build_feature_table
from src.model import get_feature_columns, get_model

REPO_ROOT = Path(__file__).resolve().parent
FINAL_MODEL_NAME = "gradient_boosting_log"  # v2 winner: log target + tuned params
FINAL_MODEL_PARAMS = {
    # from run_phase4_5.py's RandomizedSearchCV (best CV MAE 114.90 in log-space)
    "subsample": 1.0,
    "n_estimators": 200,
    "min_samples_leaf": 20,
    "max_depth": 3,
    "learning_rate": 0.1,
}


def train_final_model():
    print("Loading + cleaning + featurizing FULL train-test.csv...")
    train_feat = build_feature_table(clean(load_train_test()))
    feature_cols = get_feature_columns(train_feat)

    print(f"Training final model ({FINAL_MODEL_NAME}, params={FINAL_MODEL_PARAMS}) on {len(train_feat):,} rows...")
    model = get_model(FINAL_MODEL_NAME, params=FINAL_MODEL_PARAMS)
    model.fit(train_feat[feature_cols], train_feat["posted_rate"])
    return model, feature_cols


def predict_validation_csv(model, feature_cols) -> None:
    print("\nPredicting validation.csv (12,000 loads)...")
    validation_feat = build_feature_table(clean(load_validation()))
    preds = model.predict(validation_feat[feature_cols])

    predictions = pd.DataFrame(
        {"load_id": validation_feat["load_id"], "predicted_rate": preds}
    )

    template = pd.read_csv(REPO_ROOT / "validation-predictions-template.csv")
    filled = template[["load_id"]].merge(predictions, on="load_id", how="left")

    assert filled["predicted_rate"].isna().sum() == 0, "some load_ids didn't get a prediction"
    assert len(filled) == len(template) == 12_000

    out_path = REPO_ROOT / "validation_predictions.csv"
    filled.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(filled):,} rows)")


def predict_december_csv(model, feature_cols) -> None:
    print("\nPredicting december-chart-inputs.csv (31 days)...")
    december_raw = pd.read_csv(REPO_ROOT / "december-chart-inputs.csv")
    train_test_raw = load_train_test()
    validation_raw = load_validation()

    december_feat = prepare_december_features(december_raw, train_test_raw, validation_raw)
    december_feat = align_to_training_columns(december_feat, feature_cols)

    preds = model.predict(december_feat)

    result = december_raw.copy()
    result["predicted_rate"] = preds

    out_path = REPO_ROOT / "december-chart-inputs.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(result)} rows)")
    print(result[["date", "predicted_rate"]].to_string(index=False))


def run_scorer() -> None:
    print("\nRunning score.py to validate outputs and build the chart...")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "score.py"),
            "--predictions",
            str(REPO_ROOT / "validation_predictions.csv"),
            "--december-predictions",
            str(REPO_ROOT / "december-chart-inputs.csv"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("score.py failed — see output above.")


def main() -> None:
    model, feature_cols = train_final_model()
    predict_validation_csv(model, feature_cols)
    predict_december_csv(model, feature_cols)
    run_scorer()


if __name__ == "__main__":
    main()
