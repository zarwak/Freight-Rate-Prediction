"""Phase 1 — Explore the training data (EDA = Exploratory Data Analysis).

Goal: look at the data BEFORE touching it, so cleaning/feature decisions
in Phase 2/3 are based on evidence, not guesses.

Run with: python eda.py
Outputs: printed stats in the terminal + PNG charts saved to eda_outputs/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file instead of popping a window
import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_train_test

OUT_DIR = Path(__file__).resolve().parent / "eda_outputs"


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def basic_shape_and_types(df: pd.DataFrame) -> None:
    section("1. Shape and column types")
    print(f"Rows: {len(df):,}  Columns: {df.shape[1]}")
    print(df.dtypes)


def missing_values(df: pd.DataFrame) -> None:
    section("2. Missing values per column")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values.")
    else:
        pct = (missing / len(df) * 100).round(2)
        report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
        print(report)


def target_summary(df: pd.DataFrame) -> None:
    section("3. Target column: posted_rate")
    print(df["posted_rate"].describe())


def categorical_summary(df: pd.DataFrame) -> None:
    section("4. Equipment types (categorical column)")
    print(df["equipment"].value_counts())
    section("5. Average posted_rate by equipment")
    print(df.groupby("equipment")["posted_rate"].mean().round(2))


def date_range(df: pd.DataFrame) -> None:
    section("6. Date range")
    print(f"Earliest date: {df['date'].min().date()}")
    print(f"Latest date:   {df['date'].max().date()}")


def save_plots(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # Why: a histogram shows if posted_rate is skewed (it is — long right tail),
    # which affects which model/metric choices make sense later.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df["posted_rate"].hist(bins=60, ax=ax, color="#064A56")
    ax.set_title("Distribution of posted_rate")
    ax.set_xlabel("posted_rate ($)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "posted_rate_histogram.png", dpi=150)
    plt.close(fig)

    # Why: distance is the most obvious price driver — confirm the relationship
    # actually exists and see how noisy it is.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(df["distance"], df["posted_rate"], s=4, alpha=0.15, color="#064A56")
    ax.set_title("distance vs posted_rate")
    ax.set_xlabel("distance (miles)")
    ax.set_ylabel("posted_rate ($)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "distance_vs_rate.png", dpi=150)
    plt.close(fig)

    # Why: check if average price moves over time (seasonality/trend) —
    # if it does, date-based features and a time-based split both matter more.
    monthly = df.set_index("date")["posted_rate"].resample("W").mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    monthly.plot(ax=ax, color="#064A56", marker="o", markersize=3)
    ax.set_title("Average posted_rate by week")
    ax.set_ylabel("posted_rate ($)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rate_over_time.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved 3 charts to {OUT_DIR}")


def main() -> None:
    df = load_train_test()
    basic_shape_and_types(df)
    missing_values(df)
    target_summary(df)
    categorical_summary(df)
    date_range(df)
    save_plots(df)


if __name__ == "__main__":
    main()
