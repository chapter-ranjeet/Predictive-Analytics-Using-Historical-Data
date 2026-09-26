"""
Model Training Module
======================
Handles training of baseline, ML, and time-series models
for the sales forecasting pipeline.

Supports:
    - Baseline models (naive, moving average)
    - Linear Regression
    - Random Forest Regressor
    - Gradient Boosting (HistGradientBoosting)
    - SARIMA time-series model
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def time_series_split(df, train_ratio=0.8):
    """
    Split data chronologically for time-series.

    NEVER shuffles the data — preserves temporal order.
    The most recent portion is used as the test set.

    Args:
        df: Chronologically sorted DataFrame
        train_ratio: Proportion used for training (default 0.8)

    Returns:
        train_df, test_df
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Chronological Train/Test Split:")
    print(f"  Training: {len(train_df)} rows "
          f"({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"  Testing:  {len(test_df)} rows "
          f"({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")
    return train_df, test_df


def prepare_xy(df, feature_cols, target_col="Sales"):
    """Extract feature matrix X and target vector y."""
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y


def train_baseline_naive(train_df, test_df, target_col="Sales"):
    """
    Baseline Model 1: Previous-period prediction.

    Predicts each value as the immediately preceding value.
    This establishes the minimum bar any ML model should beat.
    """
    # Use the last training value as prediction for first test point,
    # then shift forward through test
    preds = np.empty(len(test_df))
    last_known = train_df[target_col].iloc[-1]
    preds[0] = last_known
    actual = test_df[target_col].values
    for i in range(1, len(preds)):
        preds[i] = actual[i - 1]

    print("  Trained: Naive Baseline (previous-value prediction)")
    return preds


def train_baseline_moving_avg(train_df, test_df, target_col="Sales", window=7):
    """
    Baseline Model 2: Moving average prediction.

    Uses the trailing N-day average as the forecast.
    More stable than naive but still simple.
    """
    combined = pd.concat([train_df[[target_col]], test_df[[target_col]]])
    rolling = combined[target_col].rolling(window=window, min_periods=1).mean()
    # Shift by 1 to avoid using current value
    preds = rolling.shift(1).iloc[len(train_df):].values

    print(f"  Trained: Moving Average Baseline (window={window})")
    return preds


def train_linear_regression(X_train, y_train, X_test):
    """Train a Linear Regression model."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    print("  Trained: Linear Regression")
    return model, scaler, preds


def train_random_forest(X_train, y_train, X_test):
    """
    Train a Random Forest Regressor.

    Random Forest handles non-linear relationships and interactions
    well, and provides feature importance rankings.
    """
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("  Trained: Random Forest Regressor")
    return model, preds


def train_gradient_boosting(X_train, y_train, X_test):
    """
    Train a HistGradientBoosting Regressor.

    HistGradientBoosting is efficient and handles large datasets
    well. It natively supports missing values and is faster
    than standard GradientBoosting.
    """
    model = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("  Trained: HistGradient Boosting Regressor")
    return model, preds


def train_sarima(train_series, test_len, order=(1, 1, 1),
                 seasonal_order=(1, 1, 1, 7)):
    """
    Train a SARIMA (Seasonal ARIMA) model.

    SARIMA is a statistical time-series model that captures:
        - Autoregressive (AR) patterns
        - Integrated (differencing for stationarity)
        - Moving Average (MA) of residuals
        - Seasonal components

    Unlike ML regression, SARIMA explicitly models temporal
    autocorrelation and does not require external features.

    Args:
        train_series: pandas Series with DatetimeIndex
        test_len: Number of steps to forecast
        order: ARIMA(p,d,q) parameters
        seasonal_order: Seasonal (P,D,Q,s) parameters

    Returns:
        Fitted model and predictions
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=200)
    preds = fitted.forecast(steps=test_len)

    print(f"  Trained: SARIMA{order}x{seasonal_order}")
    return fitted, preds.values


def save_model(model, filename="best_model.pkl", scaler=None):
    """Save a trained model (and optionally its scaler) to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, model_path)
    print(f"  Model saved: {model_path}")

    if scaler is not None:
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler saved: {scaler_path}")


def load_model(filename="best_model.pkl"):
    """Load a saved model from disk."""
    model_path = os.path.join(MODEL_DIR, filename)
    model = joblib.load(model_path)
    print(f"  Model loaded: {model_path}")
    return model
