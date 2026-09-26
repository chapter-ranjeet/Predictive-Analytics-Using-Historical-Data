"""
Model Evaluation Module
========================
Provides functions to evaluate, compare, and visualize
model performance using standard regression metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")


def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate standard regression evaluation metrics.

    Metrics:
        MAE  - Mean Absolute Error (average magnitude of errors)
        RMSE - Root Mean Squared Error (penalizes large errors more)
        R²   - Coefficient of Determination (variance explained)
        MAPE - Mean Absolute Percentage Error (relative error)
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE: avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    metrics = {
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R²": round(r2, 4),
        "MAPE (%)": round(mape, 2),
    }
    return metrics


def compare_models(results_list):
    """
    Create a comparison table from a list of metric dictionaries.

    Args:
        results_list: List of dicts from calculate_metrics()

    Returns:
        DataFrame with model comparison
    """
    df = pd.DataFrame(results_list)
    df = df.set_index("Model")

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)

    # Identify best model by RMSE (lower is better)
    best_model = df["RMSE"].idxmin()
    print(f"\n★ Best Model (lowest RMSE): {best_model}")
    print(f"  RMSE: {df.loc[best_model, 'RMSE']:.2f}")
    print(f"  R²:   {df.loc[best_model, 'R²']:.4f}")

    return df, best_model


def plot_actual_vs_predicted(dates, y_true, y_pred, model_name="Model",
                             save_path=None):
    """
    Create a professional actual vs predicted chart.

    This is one of the most important visualizations for demonstrating
    model performance to stakeholders.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, y_true, label="Actual Sales", color="#1976D2",
            linewidth=1.5, alpha=0.8)
    ax.plot(dates, y_pred, label=f"Predicted ({model_name})", color="#F44336",
            linewidth=1.5, alpha=0.8, linestyle="--")

    ax.fill_between(dates, y_true, y_pred, alpha=0.1, color="gray",
                    label="Prediction Error")

    ax.set_title(f"Actual vs Predicted Sales — {model_name}", fontweight="bold",
                 fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.legend(loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


def plot_model_comparison_bar(comparison_df, save_path=None):
    """Create a grouped bar chart comparing model metrics."""
    metrics_to_plot = ["MAE", "RMSE"]
    available = [m for m in metrics_to_plot if m in comparison_df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 6))
    if len(available) == 1:
        axes = [axes]

    colors = sns.color_palette("Set2", len(comparison_df))

    for ax, metric in zip(axes, available):
        bars = ax.bar(comparison_df.index, comparison_df[metric],
                      color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title(f"{metric} Comparison", fontweight="bold")
        ax.set_ylabel(metric)

        # Highlight best (lowest) value
        best_idx = comparison_df[metric].idxmin()
        for bar, name in zip(bars, comparison_df.index):
            if name == best_idx:
                bar.set_edgecolor("gold")
                bar.set_linewidth(3)

        # Add value labels
        for bar, val in zip(bars, comparison_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10)

        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Model Performance Comparison", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


def plot_residuals(y_true, y_pred, model_name="Model", save_path=None):
    """Plot residual diagnostics for model evaluation."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Residual distribution
    axes[0].hist(residuals, bins=40, color="#7B1FA2", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Residual Distribution", fontweight="bold")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")

    # Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10, color="#0097A7")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals vs Predicted", fontweight="bold")
    axes[1].set_xlabel("Predicted Sales")
    axes[1].set_ylabel("Residual")

    # QQ-like plot (sorted residuals)
    sorted_res = np.sort(residuals)
    theoretical = np.linspace(-3, 3, len(sorted_res))
    axes[2].scatter(theoretical, sorted_res, alpha=0.3, s=10, color="#E64A19")
    axes[2].set_title("Sorted Residuals", fontweight="bold")
    axes[2].set_xlabel("Theoretical Quantile")
    axes[2].set_ylabel("Residual")

    plt.suptitle(f"Residual Analysis — {model_name}", fontweight="bold", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_feature_importance(model, feature_names, top_n=15, save_path=None):
    """
    Plot feature importance for tree-based models.

    Note: Feature importance shows which features the model
    relies on most, but does NOT imply causal relationships.
    """
    if not hasattr(model, "feature_importances_"):
        print("  Model does not support feature_importances_ — skipping.")
        return None

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importance_df)))
    ax.barh(importance_df["Feature"], importance_df["Importance"],
            color=colors, edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance Score")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()

    return importance_df
