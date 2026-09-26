"""
Forecasting Module
===================
Generates future sales predictions using trained models.

Handles the creation of future date features and iterative
forecasting where lag/rolling features depend on prior predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


def generate_future_dates(last_date, periods=30, freq="D"):
    """Generate a DatetimeIndex of future dates starting after the last known date."""
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=periods,
        freq=freq,
    )
    return future_dates


def create_future_features(future_dates, historical_sales, feature_cols):
    """
    Create feature DataFrame for future dates.

    For lag and rolling features, uses the most recent historical
    values and iteratively updates as forecasts are generated.

    Args:
        future_dates: DatetimeIndex of dates to forecast
        historical_sales: Series of historical sales values (chronological)
        feature_cols: List of feature column names expected by the model

    Returns:
        DataFrame with future features
    """
    future_df = pd.DataFrame({"Date": future_dates})

    # Date features
    future_df["year"] = future_df["Date"].dt.year
    future_df["month"] = future_df["Date"].dt.month
    future_df["quarter"] = future_df["Date"].dt.quarter
    future_df["week"] = future_df["Date"].dt.isocalendar().week.astype(int)
    future_df["day"] = future_df["Date"].dt.day
    future_df["day_of_week"] = future_df["Date"].dt.dayofweek
    future_df["is_weekend"] = (future_df["day_of_week"] >= 5).astype(int)
    future_df["day_of_year"] = future_df["Date"].dt.dayofyear

    # Cyclical features
    future_df["month_sin"] = np.sin(2 * np.pi * future_df["month"] / 12)
    future_df["month_cos"] = np.cos(2 * np.pi * future_df["month"] / 12)
    future_df["dow_sin"] = np.sin(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["dow_cos"] = np.cos(2 * np.pi * future_df["day_of_week"] / 7)

    return future_df


def iterative_forecast(model, future_df, historical_sales, feature_cols,
                       scaler=None):
    """
    Generate forecasts iteratively, updating lag/rolling features
    as each prediction is made.

    This is necessary because lag features for future dates depend
    on predictions from earlier future dates.

    Args:
        model: Trained sklearn model
        future_df: DataFrame with date features for future dates
        historical_sales: Array/Series of recent historical sales
        feature_cols: Feature column names used during training
        scaler: Optional StandardScaler for models that need scaling

    Returns:
        Array of forecasted values
    """
    # Build a combined sales history (will be extended with predictions)
    sales_history = list(historical_sales.values[-60:])  # Keep last 60 for rolling
    forecasts = []

    for i in range(len(future_df)):
        row = future_df.iloc[[i]].copy()

        # Create lag features from available history
        lags = [1, 7, 14, 30]
        for lag in lags:
            col_name = f"lag_{lag}"
            if col_name in feature_cols:
                idx = len(sales_history) - lag
                row[col_name] = sales_history[idx] if idx >= 0 else np.mean(sales_history)

        # Create rolling features from available history
        for window in [7, 14, 30]:
            mean_col = f"rolling_mean_{window}"
            std_col = f"rolling_std_{window}"
            recent = sales_history[-window:] if len(sales_history) >= window else sales_history
            if mean_col in feature_cols:
                row[mean_col] = np.mean(recent)
            if std_col in feature_cols:
                row[std_col] = np.std(recent) if len(recent) > 1 else 0

        # Ensure all required features exist, fill missing with 0
        for col in feature_cols:
            if col not in row.columns:
                row[col] = 0

        # Predict
        X = row[feature_cols].values
        if scaler is not None:
            X = scaler.transform(X)
        pred = model.predict(X)[0]
        pred = max(pred, 0)  # Sales cannot be negative

        forecasts.append(pred)
        sales_history.append(pred)

    return np.array(forecasts)


def create_forecast_dataframe(future_dates, forecasts):
    """Create a clean DataFrame of forecasted values."""
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecasted_Sales": np.round(forecasts, 2),
    })
    return forecast_df


def save_forecast(forecast_df, filepath):
    """Save the forecast DataFrame to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    forecast_df.to_csv(filepath, index=False)
    print(f"  Forecast saved: {filepath}")


def plot_forecast(historical_df, forecast_df, target_col="Sales",
                  title="Sales Forecast", save_path=None):
    """
    Create a professional forecast visualization.

    Shows historical actual sales transitioning into the
    forecasted period with clear visual distinction.
    """
    fig, ax = plt.subplots(figsize=(16, 7))

    # Plot historical data (last 120 days for context)
    recent = historical_df.tail(120)
    ax.plot(recent["Date"], recent[target_col],
            color="#1565C0", linewidth=1.5, label="Historical Sales", alpha=0.8)

    # Plot forecast
    ax.plot(forecast_df["Date"], forecast_df["Forecasted_Sales"],
            color="#E53935", linewidth=2, label="Forecasted Sales",
            linestyle="--", marker="o", markersize=3)

    # Add confidence-like shading (±15% illustrative band)
    upper = forecast_df["Forecasted_Sales"] * 1.15
    lower = forecast_df["Forecasted_Sales"] * 0.85
    ax.fill_between(forecast_df["Date"], lower, upper,
                    alpha=0.15, color="#E53935", label="Uncertainty Band (±15%)")

    # Visual separator between historical and forecast
    split_date = historical_df["Date"].max()
    ax.axvline(split_date, color="gray", linestyle=":", linewidth=2,
               alpha=0.7, label="Forecast Start")

    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()

    return fig
