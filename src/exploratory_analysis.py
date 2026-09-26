"""
Exploratory Data Analysis Module
=================================
Provides functions for visualizing and analyzing the sales dataset
to uncover patterns, trends, and distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# Professional plot styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 100,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
})

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")


def _save_fig(fig, name):
    """Save figure to the visualizations directory."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    filepath = os.path.join(SAVE_DIR, f"{name}.png")
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    print(f"  Saved: {filepath}")


def plot_sales_trend(df, save=True):
    """Plot overall daily sales trend over time."""
    daily = df.groupby("Date")["Sales"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily["Date"], daily["Sales"], linewidth=0.8, alpha=0.5, label="Daily Sales")

    # Add 30-day moving average for clarity
    daily["MA30"] = daily["Sales"].rolling(window=30, min_periods=1).mean()
    ax.plot(daily["Date"], daily["MA30"], linewidth=2, color="crimson",
            label="30-Day Moving Average")

    ax.set_title("Overall Sales Trend", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Sales ($)")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        _save_fig(fig, "01_sales_trend")
    plt.show()


def plot_monthly_sales(df, save=True):
    """Analyze and plot monthly sales aggregation."""
    df_copy = df.copy()
    df_copy["YearMonth"] = df_copy["Date"].dt.to_period("M")
    monthly = df_copy.groupby("YearMonth")["Sales"].sum().reset_index()
    monthly["YearMonth"] = monthly["YearMonth"].astype(str)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(monthly)))
    ax.bar(range(len(monthly)), monthly["Sales"], color=colors, edgecolor="white",
           linewidth=0.5)
    ax.set_xticks(range(0, len(monthly), max(1, len(monthly) // 12)))
    ax.set_xticklabels(
        monthly["YearMonth"].iloc[::max(1, len(monthly) // 12)],
        rotation=45, ha="right"
    )
    ax.set_title("Monthly Sales Performance", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales ($)")
    plt.tight_layout()
    if save:
        _save_fig(fig, "02_monthly_sales")
    plt.show()


def plot_yearly_sales(df, save=True):
    """Compare yearly sales performance."""
    df_copy = df.copy()
    df_copy["Year"] = df_copy["Date"].dt.year
    yearly = df_copy.groupby("Year")["Sales"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(yearly["Year"].astype(str), yearly["Sales"],
                  color=sns.color_palette("coolwarm", len(yearly)),
                  edgecolor="white", linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, yearly["Sales"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + yearly["Sales"].max() * 0.01,
                f"${val:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Year-over-Year Sales Comparison", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Sales ($)")
    plt.tight_layout()
    if save:
        _save_fig(fig, "03_yearly_sales")
    plt.show()


def plot_weekly_pattern(df, save=True):
    """Analyze day-of-week sales behavior."""
    df_copy = df.copy()
    df_copy["DayOfWeek"] = df_copy["Date"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    weekly = df_copy.groupby("DayOfWeek")["Sales"].mean().reindex(day_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3" if d in ["Saturday", "Sunday"] else "#4CAF50" for d in day_order]
    bars = ax.bar(weekly.index, weekly.values, color=colors, edgecolor="white", linewidth=1.5)

    ax.set_title("Average Sales by Day of Week", fontweight="bold")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Sales ($)")

    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#4CAF50", label="Weekday"),
                       Patch(facecolor="#2196F3", label="Weekend")]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    if save:
        _save_fig(fig, "04_weekly_pattern")
    plt.show()


def plot_category_performance(df, save=True):
    """Compare sales across product categories."""
    if "Category" not in df.columns:
        print("No 'Category' column found — skipping category analysis.")
        return

    cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Horizontal bar chart
    colors = sns.color_palette("Set2", len(cat_sales))
    axes[0].barh(cat_sales.index, cat_sales.values, color=colors, edgecolor="white")
    axes[0].set_title("Total Sales by Category", fontweight="bold")
    axes[0].set_xlabel("Total Sales ($)")

    # Pie chart
    axes[1].pie(cat_sales.values, labels=cat_sales.index, autopct="%1.1f%%",
                colors=colors, startangle=140, pctdistance=0.85)
    centre_circle = plt.Circle((0, 0), 0.55, fc="white")
    axes[1].add_artist(centre_circle)
    axes[1].set_title("Category Share of Sales", fontweight="bold")

    plt.tight_layout()
    if save:
        _save_fig(fig, "05_category_performance")
    plt.show()


def plot_sales_distribution(df, save=True):
    """Plot the distribution of the Sales target variable."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(df["Sales"], bins=50, color="#3F51B5", edgecolor="white",
                 alpha=0.8, density=True)
    axes[0].set_title("Sales Distribution", fontweight="bold")
    axes[0].set_xlabel("Sales ($)")
    axes[0].set_ylabel("Density")

    # Box plot
    axes[1].boxplot(df["Sales"], vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#3F51B5", alpha=0.6),
                    medianprops=dict(color="red", linewidth=2))
    axes[1].set_title("Sales Box Plot", fontweight="bold")
    axes[1].set_ylabel("Sales ($)")

    plt.tight_layout()
    if save:
        _save_fig(fig, "06_sales_distribution")
    plt.show()


def plot_correlation_heatmap(df, save=True):
    """Create a correlation heatmap for numerical variables."""
    numerical = df.select_dtypes(include=[np.number])
    if numerical.shape[1] < 2:
        print("Not enough numerical columns for correlation analysis.")
        return

    corr = numerical.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    if save:
        _save_fig(fig, "07_correlation_heatmap")
    plt.show()


def run_full_eda(df, save=True):
    """Execute the complete EDA pipeline."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\n1. Overall Sales Trend")
    plot_sales_trend(df, save)

    print("\n2. Monthly Sales")
    plot_monthly_sales(df, save)

    print("\n3. Yearly Sales")
    plot_yearly_sales(df, save)

    print("\n4. Weekly Patterns")
    plot_weekly_pattern(df, save)

    print("\n5. Category Performance")
    plot_category_performance(df, save)

    print("\n6. Sales Distribution")
    plot_sales_distribution(df, save)

    print("\n7. Correlation Analysis")
    plot_correlation_heatmap(df, save)

    print("\n✓ EDA Complete")
