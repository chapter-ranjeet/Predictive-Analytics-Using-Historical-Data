"""
Feature Engineering Module
===========================
Creates meaningful predictive features from the cleaned dataset
for use in machine learning models.

Features are designed to avoid data leakage — no future information
is used to create features for historical predictions.
"""

import pandas as pd
import numpy as np


def create_date_features(df):
    """
    Extract calendar-based features from the Date column.

    These features capture temporal patterns like seasonality,
    day-of-week effects, and periodic cycles.
    """
    df = df.copy()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["day"] = df["Date"].dt.day
    df["day_of_week"] = df["Date"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"] = df["Date"].dt.dayofyear

    # Cyclical encoding for month (captures the circular nature of months)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Cyclical encoding for day of week
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    print(f"  Created {9 + 4} date-based features")
    return df


def create_lag_features(df, target_col="Sales", lags=None):
    """
    Create lag features from the target variable.

    Lag features use past values to predict future values.
    This is essential for time-series prediction as recent
    history is often the best predictor of near-future values.

    IMPORTANT: Lag features are created AFTER sorting by date
    to prevent data leakage.

    Args:
        df: DataFrame sorted chronologically
        target_col: Column to create lags from
        lags: List of lag periods (default: [1, 7, 14, 30])
    """
    df = df.copy()
    if lags is None:
        lags = [1, 7, 14, 30]

    for lag in lags:
        col_name = f"lag_{lag}"
        df[col_name] = df[target_col].shift(lag)

    print(f"  Created {len(lags)} lag features: {lags}")
    return df


def create_rolling_features(df, target_col="Sales", windows=None):
    """
    Create rolling window statistics from the target variable.

    Rolling features smooth out noise and capture local trends.
    The window looks backward only (shift(1)) to prevent leakage.

    Args:
        df: DataFrame sorted chronologically
        target_col: Column to compute rolling stats from
        windows: List of window sizes (default: [7, 14, 30])
    """
    df = df.copy()
    if windows is None:
        windows = [7, 14, 30]

    for window in windows:
        # Shift by 1 to exclude the current row (prevent leakage)
        shifted = df[target_col].shift(1)
        df[f"rolling_mean_{window}"] = shifted.rolling(
            window=window, min_periods=1
        ).mean()
        df[f"rolling_std_{window}"] = shifted.rolling(
            window=window, min_periods=1
        ).std()

    print(f"  Created {len(windows) * 2} rolling features for windows: {windows}")
    return df


def create_category_features(df):
    """
    Encode categorical variables for ML models.

    Uses one-hot encoding for categories to preserve
    the non-ordinal nature of product categories.
    """
    df = df.copy()
    cat_cols = ["Category", "Product", "Region", "Store"]
    encoded_cols = 0

    for col in cat_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            encoded_cols += len(dummies.columns)

    if encoded_cols > 0:
        print(f"  Created {encoded_cols} one-hot encoded features")
    return df


def engineer_features(df, target_col="Sales", include_lags=True,
                      include_rolling=True, include_categories=True):
    """
    Main feature engineering pipeline.

    Orchestrates all feature creation steps and handles NaN values
    introduced by lag/rolling operations.

    Args:
        df: Cleaned, chronologically sorted DataFrame
        target_col: Target variable column name
        include_lags: Whether to create lag features
        include_rolling: Whether to create rolling features
        include_categories: Whether to encode categorical variables

    Returns:
        Feature-engineered DataFrame
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # Step 1: Date features (always created)
    print("\n1. Date Features:")
    df = create_date_features(df)

    # Step 2: Lag features
    if include_lags:
        print("\n2. Lag Features:")
        df = create_lag_features(df, target_col)

    # Step 3: Rolling features
    if include_rolling:
        print("\n3. Rolling Features:")
        df = create_rolling_features(df, target_col)

    # Step 4: Category encoding
    if include_categories:
        print("\n4. Category Encoding:")
        df = create_category_features(df)

    # Drop rows with NaN from lag/rolling (first N rows)
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"\n  Dropped {dropped} rows with NaN values from lag/rolling features")

    print(f"\n  Final feature set: {df.shape[1]} columns, {df.shape[0]} rows")
    print("✓ Feature Engineering Complete")
    return df


def get_feature_names(df, exclude_cols=None):
    """
    Get the list of feature column names, excluding non-feature columns.

    Args:
        df: Feature-engineered DataFrame
        exclude_cols: Additional columns to exclude

    Returns:
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = []

    # Default columns to exclude (target, identifiers, raw categoricals)
    always_exclude = [
        "Date", "Sales", "Product", "Category", "Region",
        "Store", "YearMonth"
    ]
    all_exclude = set(always_exclude + exclude_cols)

    features = [col for col in df.columns if col not in all_exclude]
    return features
