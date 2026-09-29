"""Phase 0 — loading the raw CSV files.

Kept separate from cleaning/features so every later script loads data
the exact same way (one source of truth for file paths and dtypes).
"""

from pathlib import Path

import pandas as pd

# Repo root = one level up from this file (src/data.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent

TRAIN_TEST_PATH = REPO_ROOT / "train-test.csv"
VALIDATION_PATH = REPO_ROOT / "validation.csv"


def load_train_test() -> pd.DataFrame:
    """Load the labeled data (has posted_rate). Used to train + self-validate."""
    df = pd.read_csv(TRAIN_TEST_PATH, parse_dates=["date"])
    return df


def load_validation() -> pd.DataFrame:
    """Load the unlabeled data (no posted_rate). This is what we must predict."""
    df = pd.read_csv(VALIDATION_PATH, parse_dates=["date"])
    return df


if __name__ == "__main__":
    # Quick manual check: run `python src/data.py` to confirm both files load.
    train_test = load_train_test()
    validation = load_validation()
    print("train-test.csv:", train_test.shape)
    print("validation.csv:", validation.shape)
    print(train_test.dtypes)
