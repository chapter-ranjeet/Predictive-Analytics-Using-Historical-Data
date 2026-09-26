"""
Data Preprocessing Module
=========================
Handles data loading, cleaning, validation, and preparation
for the predictive sales forecasting pipeline.
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")


def load_data(filepath):
    """Load the raw CSV dataset and perform initial type inference."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def inspect_data(df):
    """Print a comprehensive summary of the dataset structure."""
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print(f"\nShape: {df.shape}")
    print(f"\nColumn Names:\n{list(df.columns)}")

    print(f"\nData Types:")
    print(df.dtypes)

    print(f"\nBasic Statistics:")
    print(df.describe())

    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    print("=" * 60)


def clean_data(df):
    """
    Clean and preprocess the raw dataset.

    Steps:
        1. Convert Date to datetime
        2. Remove duplicate rows
        3. Handle missing values
        4. Remove invalid/negative sales
        5. Sort chronologically
        6. Reset index

    Returns:
        Cleaned DataFrame and a log of changes made.
    """
    changes_log = []
    initial_rows = len(df)

    # 1. Convert Date column
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    invalid_dates = df["Date"].isnull().sum()
    if invalid_dates > 0:
        df = df.dropna(subset=["Date"])
        changes_log.append(f"Removed {invalid_dates} rows with invalid dates")

    # 2. Remove duplicates
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        df = df.drop_duplicates()
        changes_log.append(f"Removed {n_duplicates} duplicate rows")

    # 3. Handle missing values in numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            changes_log.append(
                f"Filled {n_missing} missing values in '{col}' with median ({median_val:.2f})"
            )

    # Handle missing values in categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            changes_log.append(
                f"Filled {n_missing} missing values in '{col}' with mode ('{mode_val}')"
            )

    # 4. Remove invalid sales values (negative or zero revenue)
    if "Sales" in df.columns:
        invalid_sales = (df["Sales"] <= 0).sum()
        if invalid_sales > 0:
            df = df[df["Sales"] > 0]
            changes_log.append(f"Removed {invalid_sales} rows with non-positive Sales")

    if "Quantity" in df.columns:
        invalid_qty = (df["Quantity"] <= 0).sum()
        if invalid_qty > 0:
            df = df[df["Quantity"] > 0]
            changes_log.append(
                f"Removed {invalid_qty} rows with non-positive Quantity"
            )

    # 5. Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)
    changes_log.append("Sorted data chronologically by Date")

    final_rows = len(df)
    changes_log.append(
        f"Final dataset: {final_rows} rows "
        f"(removed {initial_rows - final_rows} rows total, "
        f"{(initial_rows - final_rows) / initial_rows * 100:.1f}%)"
    )

    return df, changes_log


def save_processed_data(df, filepath):
    """Save the cleaned DataFrame to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Processed data saved to: {filepath}")


def get_date_range(df):
    """Return the date range of the dataset."""
    return df["Date"].min(), df["Date"].max()


def get_data_summary(df):
    """Generate a concise summary dictionary of the dataset."""
    date_min, date_max = get_date_range(df)
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "date_range": f"{date_min.date()} to {date_max.date()}",
        "duration_days": (date_max - date_min).days,
        "numerical_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(df.select_dtypes(include=["object"]).columns),
    }
    if "Sales" in df.columns:
        summary["total_sales"] = df["Sales"].sum()
        summary["avg_daily_sales"] = df.groupby("Date")["Sales"].sum().mean()
    return summary
