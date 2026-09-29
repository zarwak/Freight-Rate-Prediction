"""Phases 0-3 end to end: load -> clean -> engineer features -> save.

Run with: python run_phase0_3.py

Produces:
  data_prepared/train_test_features.csv  (has posted_rate — for training)
  data_prepared/validation_features.csv  (no posted_rate — for final predictions)

These are what Phase 4 (splitting) and Phase 5 (modeling) will load next,
so cleaning/feature logic only has to be written once.
"""

from pathlib import Path

from src.clean import clean
from src.data import load_train_test, load_validation
from src.features import build_feature_table

OUT_DIR = Path(__file__).resolve().parent / "data_prepared"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading raw data...")
    train_test_raw = load_train_test()
    validation_raw = load_validation()

    print("Cleaning (filling missing weight/market_index)...")
    train_test_clean = clean(train_test_raw)
    validation_clean = clean(validation_raw)

    print("Engineering features...")
    train_test_features = build_feature_table(train_test_clean)
    validation_features = build_feature_table(validation_clean)

    train_path = OUT_DIR / "train_test_features.csv"
    val_path = OUT_DIR / "validation_features.csv"
    train_test_features.to_csv(train_path, index=False)
    validation_features.to_csv(val_path, index=False)

    print(f"\nSaved: {train_path}  shape={train_test_features.shape}")
    print(f"Saved: {val_path}  shape={validation_features.shape}")
    print("\nFinal columns:", list(train_test_features.columns))

    assert train_test_features.isna().sum().sum() == 0, "train features still have NaNs"
    assert validation_features.isna().sum().sum() == 0, "validation features still have NaNs"
    print("\nNo missing values remain. Ready for Phase 4 (splitting).")


if __name__ == "__main__":
    main()
