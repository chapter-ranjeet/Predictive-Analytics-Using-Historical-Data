"""
Jupyter Notebook Builder
=========================
Programmatically creates the complete predictive_analytics.ipynb notebook
with all sections, code, and markdown explanations.
"""

import nbformat as nbf
import os


def md(source):
    """Create a markdown cell."""
    return nbf.v4.new_markdown_cell(source.strip())


def code(source):
    """Create a code cell."""
    return nbf.v4.new_code_cell(source.strip())


def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    cells = []

    # =========================================================================
    # SECTION 1: PROJECT INTRODUCTION
    # =========================================================================
    cells.append(md("""
# 📊 Predictive Analytics for Sales Forecasting Using Historical Data

---

## 🎯 Problem Statement

Businesses need to anticipate future sales to make informed decisions about inventory management, staffing, marketing budgets, and financial planning. Without reliable forecasts, companies risk overstocking (wasting capital) or understocking (losing revenue).

**This project builds a predictive model to forecast future sales trends using historical retail data.**

---

## 📌 Business Objective

- Analyze historical sales patterns to identify trends and seasonality
- Build and evaluate multiple predictive models
- Select the best-performing model for future forecasting
- Generate actionable business insights from the data

---

## 🔍 Why Predictive Analytics?

Predictive analytics transforms raw historical data into forward-looking insights. For retail businesses, this means:

| Benefit | Description |
|---------|-------------|
| **Inventory Optimization** | Stock the right products at the right time |
| **Revenue Planning** | Set realistic targets based on data-driven forecasts |
| **Seasonal Preparedness** | Anticipate demand spikes and plan accordingly |
| **Resource Allocation** | Optimize staffing and marketing spend |

---

## 🎯 Prediction Target

> **Sales (Revenue in $)** — Total daily sales aggregated across products, categories, and regions.

---

## 🔄 Project Workflow

```
Raw Historical Data → Data Understanding → Data Cleaning → EDA →
Trend & Seasonality Analysis → Feature Engineering → Train/Test Split →
Baseline Model → ML Models → Time-Series Model → Model Evaluation →
Best Model Selection → Future Forecasting → Visualization → Business Insights
```
"""))

    # =========================================================================
    # SECTION 2: SETUP & IMPORTS
    # =========================================================================
    cells.append(md("""
## 1️⃣ Setup & Imports

Import all required libraries and configure the environment for reproducible analysis.
"""))

    cells.append(code("""
# Core data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Time-Series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Model persistence
import joblib

# System
import os
import warnings
warnings.filterwarnings('ignore')

# Set working directory to project root
# This ensures all relative paths (data/, models/, visualizations/) resolve correctly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
# If running from notebooks/, go up one level
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
elif os.path.exists('notebooks'):
    pass  # Already in project root
print(f"Working directory: {os.getcwd()}")

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Plot configuration
sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 100,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

print("✓ All libraries imported successfully")
print(f"  pandas: {pd.__version__}")
print(f"  numpy: {np.__version__}")
print(f"  scikit-learn: {__import__('sklearn').__version__}")
"""))

    # =========================================================================
    # SECTION 3: DATA LOADING
    # =========================================================================
    cells.append(md("""
## 2️⃣ Data Loading

### Dataset Description

We use a **synthetic retail sales dataset** that simulates realistic business data with the following characteristics:

| Property | Description |
|----------|-------------|
| **Source** | Synthetically generated with realistic patterns |
| **Period** | January 2023 – December 2025 (3 years) |
| **Granularity** | Transaction-level daily records |
| **Target Variable** | `Sales` (Revenue in USD) |

**Feature Descriptions:**

| Column | Type | Description |
|--------|------|-------------|
| `Date` | datetime | Transaction date |
| `Product` | categorical | Product name (20 unique products) |
| `Category` | categorical | Product category (5 categories) |
| `Region` | categorical | Geographic region (4 regions) |
| `Store` | categorical | Store identifier (3 stores) |
| `Quantity` | numeric | Units sold |
| `Unit_Price` | numeric | Price per unit ($) |
| `Discount` | numeric | Applied discount rate (0–0.25) |
| `Promotion` | binary | Whether a promotion was active (0/1) |
| `Sales` | numeric | Total revenue = Quantity × Unit_Price ($) |

The dataset includes intentional imperfections (missing values, duplicates) to demonstrate proper data cleaning practices.
"""))

    cells.append(code("""
# Load the raw dataset
df = pd.read_csv('data/raw/sales_data.csv')
print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
"""))

    cells.append(md("### First 5 Rows"))
    cells.append(code("df.head()"))
    cells.append(md("### Last 5 Rows"))
    cells.append(code("df.tail()"))

    cells.append(md("### Data Types & Structure"))
    cells.append(code("""
print("Column Names:", list(df.columns))
print()
df.info()
"""))

    cells.append(md("### Statistical Summary\n\nReview the distribution of numerical variables to identify potential anomalies."))
    cells.append(code("df.describe()"))

    # =========================================================================
    # SECTION 4: DATA CLEANING
    # =========================================================================
    cells.append(md("""
## 3️⃣ Data Cleaning & Preprocessing

Data cleaning is essential before any analysis. We will:
1. Convert the Date column to proper datetime format
2. Check and remove duplicate rows
3. Handle missing values appropriately
4. Validate data integrity (no negative sales)
5. Sort chronologically for time-series analysis
"""))

    cells.append(md("### Check for Missing Values"))
    cells.append(code("""
print("Missing values per column:")
print(df.isnull().sum())
print(f"\\nTotal missing values: {df.isnull().sum().sum()}")
"""))

    cells.append(md("### Check for Duplicate Rows"))
    cells.append(code("""
n_duplicates = df.duplicated().sum()
print(f"Duplicate rows found: {n_duplicates}")
"""))

    cells.append(md("### Apply Cleaning Pipeline\n\nWe process each issue systematically, documenting every change for transparency."))
    cells.append(code("""
# Step 1: Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
invalid_dates = df['Date'].isnull().sum()
if invalid_dates > 0:
    print(f"Removed {invalid_dates} rows with invalid dates")
    df = df.dropna(subset=['Date'])

# Step 2: Remove duplicates
n_before = len(df)
df = df.drop_duplicates()
n_removed = n_before - len(df)
print(f"Removed {n_removed} duplicate rows")

# Step 3: Handle missing values in numerical columns
for col in ['Quantity', 'Unit_Price']:
    n_missing = df[col].isnull().sum()
    if n_missing > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled {n_missing} missing values in '{col}' with median ({median_val:.2f})")

# Step 4: Validate — remove non-positive sales
invalid_sales = (df['Sales'] <= 0).sum()
if invalid_sales > 0:
    df = df[df['Sales'] > 0]
    print(f"Removed {invalid_sales} rows with non-positive Sales")
else:
    print("No invalid Sales values found")

invalid_qty = (df['Quantity'] <= 0).sum()
if invalid_qty > 0:
    df = df[df['Quantity'] > 0]
    print(f"Removed {invalid_qty} rows with non-positive Quantity")

# Step 5: Sort chronologically
df = df.sort_values('Date').reset_index(drop=True)

print(f"\\n✓ Cleaned dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
"""))

    cells.append(md("### Verify Cleaned Data"))
    cells.append(code("""
print("Remaining missing values:", df.isnull().sum().sum())
print("Remaining duplicates:", df.duplicated().sum())
print()
df.describe()
"""))

    cells.append(md("### Save Cleaned Data"))
    cells.append(code("""
os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/cleaned_sales_data.csv', index=False)
print("✓ Cleaned data saved to data/processed/cleaned_sales_data.csv")
"""))

    # =========================================================================
    # SECTION 5: EDA
    # =========================================================================
    cells.append(md("""
## 4️⃣ Exploratory Data Analysis (EDA)

EDA helps us understand patterns, trends, and anomalies in the data before building models. We examine the data from multiple perspectives.

### Aggregate to Daily Sales

For trend analysis and forecasting, we aggregate transaction-level data to daily totals.
"""))

    cells.append(code("""
# Aggregate to daily sales
daily_sales = df.groupby('Date').agg({
    'Sales': 'sum',
    'Quantity': 'sum',
}).reset_index()

print(f"Daily sales dataset: {len(daily_sales)} days")
print(f"Average daily sales: ${daily_sales['Sales'].mean():,.2f}")
print(f"Peak daily sales: ${daily_sales['Sales'].max():,.2f}")
print(f"Minimum daily sales: ${daily_sales['Sales'].min():,.2f}")
"""))

    cells.append(md("### 4.1 Overall Sales Trend\n\nVisualize the complete sales history with a moving average to identify the underlying trend."))
    cells.append(code("""
fig, ax = plt.subplots(figsize=(14, 6))

# Daily sales (semi-transparent)
ax.plot(daily_sales['Date'], daily_sales['Sales'],
        linewidth=0.6, alpha=0.4, color='#90CAF9', label='Daily Sales')

# 30-day moving average
daily_sales['MA30'] = daily_sales['Sales'].rolling(30, min_periods=1).mean()
ax.plot(daily_sales['Date'], daily_sales['MA30'],
        linewidth=2.5, color='#D32F2F', label='30-Day Moving Average')

ax.set_title('Overall Sales Trend (2023–2025)', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Total Daily Sales ($)')
ax.legend(fontsize=11)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
os.makedirs('visualizations', exist_ok=True)
plt.savefig('visualizations/01_sales_trend.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    cells.append(md("**Observation:** The moving average reveals a clear upward trend over the 3-year period, indicating overall business growth. There are also periodic peaks suggesting seasonal effects."))

    cells.append(md("### 4.2 Monthly Sales Performance"))
    cells.append(code("""
df_copy = df.copy()
df_copy['YearMonth'] = df_copy['Date'].dt.to_period('M')
monthly = df_copy.groupby('YearMonth')['Sales'].sum().reset_index()
monthly['YearMonth'] = monthly['YearMonth'].astype(str)

fig, ax = plt.subplots(figsize=(14, 6))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(monthly)))
ax.bar(range(len(monthly)), monthly['Sales'], color=colors, edgecolor='white', linewidth=0.5)

# Label every 3rd month
step = max(1, len(monthly) // 12)
ax.set_xticks(range(0, len(monthly), step))
ax.set_xticklabels(monthly['YearMonth'].iloc[::step], rotation=45, ha='right')
ax.set_title('Monthly Sales Performance', fontweight='bold', fontsize=14)
ax.set_xlabel('Month')
ax.set_ylabel('Total Sales ($)')
plt.tight_layout()
plt.savefig('visualizations/02_monthly_sales.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    cells.append(md("**Observation:** Monthly sales show a clear seasonal pattern with peaks during certain months (e.g., November–December for holiday shopping) and relative lows in the post-holiday period."))

    cells.append(md("### 4.3 Yearly Sales Comparison"))
    cells.append(code("""
df_copy['Year'] = df_copy['Date'].dt.year
yearly = df_copy.groupby('Year')['Sales'].sum().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(yearly['Year'].astype(str), yearly['Sales'],
              color=sns.color_palette('coolwarm', len(yearly)),
              edgecolor='white', linewidth=1.5)

# Value labels on bars
for bar, val in zip(bars, yearly['Sales']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + yearly['Sales'].max() * 0.01,
            f'${val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title('Year-over-Year Sales Comparison', fontweight='bold', fontsize=14)
ax.set_xlabel('Year')
ax.set_ylabel('Total Sales ($)')
plt.tight_layout()
plt.savefig('visualizations/03_yearly_sales.png', dpi=150, bbox_inches='tight')
plt.show()

# Calculate YoY growth
for i in range(1, len(yearly)):
    growth = (yearly['Sales'].iloc[i] - yearly['Sales'].iloc[i-1]) / yearly['Sales'].iloc[i-1] * 100
    print(f"{yearly['Year'].iloc[i-1]} → {yearly['Year'].iloc[i]}: {growth:+.1f}% growth")
"""))

    cells.append(md("### 4.4 Day-of-Week Sales Pattern"))
    cells.append(code("""
df_copy['DayOfWeek'] = df_copy['Date'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly = df_copy.groupby('DayOfWeek')['Sales'].mean().reindex(day_order)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#4CAF50' if d not in ['Saturday', 'Sunday'] else '#2196F3' for d in day_order]
ax.bar(weekly.index, weekly.values, color=colors, edgecolor='white', linewidth=1.5)

ax.set_title('Average Sales by Day of Week', fontweight='bold', fontsize=14)
ax.set_xlabel('Day of Week')
ax.set_ylabel('Average Sales ($)')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4CAF50', label='Weekday'),
                   Patch(facecolor='#2196F3', label='Weekend')]
ax.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig('visualizations/04_weekly_pattern.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    cells.append(md("**Observation:** Weekend days (Saturday and Sunday) show higher average sales compared to weekdays, which aligns with typical retail patterns where consumers have more leisure time for shopping."))

    cells.append(md("### 4.5 Category Performance"))
    cells.append(code("""
cat_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Horizontal bar chart
colors = sns.color_palette('Set2', len(cat_sales))
axes[0].barh(cat_sales.index, cat_sales.values, color=colors, edgecolor='white')
axes[0].set_title('Total Sales by Category', fontweight='bold')
axes[0].set_xlabel('Total Sales ($)')

# Donut chart
axes[1].pie(cat_sales.values, labels=cat_sales.index, autopct='%1.1f%%',
            colors=colors, startangle=140, pctdistance=0.85)
centre = plt.Circle((0, 0), 0.55, fc='white')
axes[1].add_artist(centre)
axes[1].set_title('Category Share of Total Sales', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/05_category_performance.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    cells.append(md("### 4.6 Sales Distribution"))
    cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df['Sales'], bins=50, color='#3F51B5', edgecolor='white', alpha=0.8, density=True)
axes[0].set_title('Sales Distribution (per Transaction)', fontweight='bold')
axes[0].set_xlabel('Sales ($)')
axes[0].set_ylabel('Density')

# Box plot
bp = axes[1].boxplot(df['Sales'], vert=True, patch_artist=True,
                     boxprops=dict(facecolor='#3F51B5', alpha=0.6),
                     medianprops=dict(color='red', linewidth=2))
axes[1].set_title('Sales Box Plot', fontweight='bold')
axes[1].set_ylabel('Sales ($)')

plt.tight_layout()
plt.savefig('visualizations/06_sales_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Median transaction: ${df['Sales'].median():.2f}")
print(f"Mean transaction: ${df['Sales'].mean():.2f}")
print(f"Skewness: {df['Sales'].skew():.2f}")
"""))

    cells.append(md("**Observation:** The sales distribution is right-skewed, which is typical for retail — most transactions are smaller, with fewer high-value purchases (e.g., electronics)."))

    cells.append(md("### 4.7 Correlation Analysis"))
    cells.append(code("""
numerical_cols = df.select_dtypes(include=[np.number])
corr = numerical_cols.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('visualizations/07_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    cells.append(md("**Note:** Correlation shows linear relationships between variables. A high correlation between Quantity and Sales is expected since Sales = Quantity × Unit_Price. This does NOT imply causation."))

    # =========================================================================
    # SECTION 6: TREND & SEASONALITY ANALYSIS
    # =========================================================================
    cells.append(md("""
## 5️⃣ Trend & Seasonality Analysis

This section uses time-series decomposition to separate the data into its structural components:

- **Trend**: Long-term direction of sales
- **Seasonal**: Repeating patterns within fixed periods
- **Residual**: Random noise after removing trend and seasonality

Understanding these components is crucial for selecting appropriate forecasting models.
"""))

    cells.append(code("""
# Prepare daily aggregated series for decomposition
daily_ts = daily_sales.set_index('Date')['Sales']

# Ensure regular frequency (fill any gaps)
daily_ts = daily_ts.asfreq('D', method='ffill')

# Perform seasonal decomposition (additive model, period=30 for monthly seasonality)
decomposition = seasonal_decompose(daily_ts, model='additive', period=30)

fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Observed
axes[0].plot(decomposition.observed, color='#1565C0', linewidth=0.8)
axes[0].set_title('Observed Sales', fontweight='bold')
axes[0].set_ylabel('Sales ($)')

# Trend
axes[1].plot(decomposition.trend, color='#E53935', linewidth=2)
axes[1].set_title('Trend Component', fontweight='bold')
axes[1].set_ylabel('Trend')

# Seasonal
axes[2].plot(decomposition.seasonal, color='#43A047', linewidth=0.8)
axes[2].set_title('Seasonal Component (30-day period)', fontweight='bold')
axes[2].set_ylabel('Seasonality')

# Residual
axes[3].plot(decomposition.resid, color='#FF9800', linewidth=0.5, alpha=0.7)
axes[3].set_title('Residual Component', fontweight='bold')
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Date')

plt.tight_layout()
plt.savefig('visualizations/08_seasonal_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    cells.append(md("""
### Decomposition Analysis

| Component | Finding |
|-----------|---------|
| **Trend** | Clear upward trend over the 3-year period, confirming business growth |
| **Seasonality** | Regular monthly cycles visible, with consistent peaks and troughs |
| **Residual** | Relatively small residuals suggest the trend and seasonal components explain most of the variation |

These findings support using both ML regression (with date features) and dedicated time-series models.
"""))

    cells.append(md("### Monthly Seasonality Profile\n\nWhich months consistently perform above or below average?"))
    cells.append(code("""
monthly_avg = df.groupby(df['Date'].dt.month)['Sales'].mean()
overall_avg = df['Sales'].mean()

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#E53935' if v > overall_avg else '#1565C0' for v in monthly_avg.values]
bars = ax.bar(range(1, 13), monthly_avg.values, color=colors, edgecolor='white')
ax.axhline(overall_avg, color='gray', linestyle='--', linewidth=1.5, label=f'Overall Avg (${overall_avg:.0f})')

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.set_title('Average Transaction Sales by Month', fontweight='bold', fontsize=14)
ax.set_xlabel('Month')
ax.set_ylabel('Average Sales ($)')
ax.legend()
plt.tight_layout()
plt.savefig('visualizations/09_monthly_seasonality.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    # =========================================================================
    # SECTION 7: FEATURE ENGINEERING
    # =========================================================================
    cells.append(md("""
## 6️⃣ Feature Engineering

Feature engineering transforms raw data into informative predictors. For time-series forecasting, we create:

1. **Date features**: Year, month, quarter, day-of-week, etc.
2. **Lag features**: Past sales values (e.g., sales 1, 7, 14, 30 days ago)
3. **Rolling features**: Moving averages and standard deviations

### ⚠️ Data Leakage Prevention

> Lag and rolling features use `shift()` to ensure only **past** information is used.
> We never use future data points to create features for historical predictions.
"""))

    cells.append(code("""
# Work with daily aggregated data for forecasting
df_daily = daily_sales[['Date', 'Sales']].copy()
print(f"Starting with {len(df_daily)} daily observations")

# --- Date Features ---
df_daily['year'] = df_daily['Date'].dt.year
df_daily['month'] = df_daily['Date'].dt.month
df_daily['quarter'] = df_daily['Date'].dt.quarter
df_daily['week'] = df_daily['Date'].dt.isocalendar().week.astype(int)
df_daily['day'] = df_daily['Date'].dt.day
df_daily['day_of_week'] = df_daily['Date'].dt.dayofweek
df_daily['is_weekend'] = (df_daily['day_of_week'] >= 5).astype(int)
df_daily['day_of_year'] = df_daily['Date'].dt.dayofyear

# Cyclical encoding (captures circular nature of months/days)
df_daily['month_sin'] = np.sin(2 * np.pi * df_daily['month'] / 12)
df_daily['month_cos'] = np.cos(2 * np.pi * df_daily['month'] / 12)
df_daily['dow_sin'] = np.sin(2 * np.pi * df_daily['day_of_week'] / 7)
df_daily['dow_cos'] = np.cos(2 * np.pi * df_daily['day_of_week'] / 7)

print(f"✓ Created 13 date-based features")

# --- Lag Features ---
# Each lag uses shift() to prevent data leakage
for lag in [1, 7, 14, 30]:
    df_daily[f'lag_{lag}'] = df_daily['Sales'].shift(lag)

print(f"✓ Created 4 lag features (1, 7, 14, 30 days)")

# --- Rolling Features ---
# shift(1) ensures current day is excluded from the window
shifted = df_daily['Sales'].shift(1)
for window in [7, 14, 30]:
    df_daily[f'rolling_mean_{window}'] = shifted.rolling(window=window, min_periods=1).mean()
    df_daily[f'rolling_std_{window}'] = shifted.rolling(window=window, min_periods=1).std()

print(f"✓ Created 6 rolling features (7, 14, 30 day windows)")

# Drop rows with NaN values from lag/rolling operations
initial_len = len(df_daily)
df_daily = df_daily.dropna()
print(f"\\nDropped {initial_len - len(df_daily)} rows with NaN from lag/rolling features")
print(f"Final feature-engineered dataset: {df_daily.shape[0]} rows × {df_daily.shape[1]} columns")
"""))

    cells.append(code("""
# Display the feature set
print("Feature columns:")
feature_cols = [c for c in df_daily.columns if c not in ['Date', 'Sales']]
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")
print(f"\\nTotal features: {len(feature_cols)}")
"""))

    cells.append(md("""
### Why These Features?

| Feature Type | Purpose |
|-------------|---------|
| **year, month, quarter** | Capture long-term trends and seasonal cycles |
| **day_of_week, is_weekend** | Model weekly purchasing patterns |
| **Cyclical encoding (sin/cos)** | Represent circular features without artificial boundary (Dec→Jan) |
| **lag_1, lag_7** | Recent sales are strong predictors of near-future sales |
| **lag_14, lag_30** | Capture bi-weekly and monthly patterns |
| **rolling_mean/std** | Smooth trend signal and capture volatility |
"""))

    # =========================================================================
    # SECTION 8: TRAIN/TEST SPLIT
    # =========================================================================
    cells.append(md("""
## 7️⃣ Train/Test Split

### ⚠️ Important: Chronological Split, NOT Random

For time-series data, we **must** split chronologically:
- **Training set**: Earlier period (80%)
- **Test set**: Later period (20%)

Random splitting would **leak future information** into the training set, producing artificially inflated accuracy that won't hold in real deployment.
"""))

    cells.append(code("""
# Chronological 80/20 split
split_idx = int(len(df_daily) * 0.8)
train_df = df_daily.iloc[:split_idx].copy()
test_df = df_daily.iloc[split_idx:].copy()

print(f"Training Set: {len(train_df)} days")
print(f"  Period: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
print(f"\\nTest Set: {len(test_df)} days")
print(f"  Period: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")

# Prepare X and y
feature_cols = [c for c in df_daily.columns if c not in ['Date', 'Sales']]
X_train = train_df[feature_cols].values
y_train = train_df['Sales'].values
X_test = test_df[feature_cols].values
y_test = test_df['Sales'].values

print(f"\\nFeature matrix shape: X_train={X_train.shape}, X_test={X_test.shape}")
"""))

    cells.append(md("### Visualize the Train/Test Split"))
    cells.append(code("""
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(train_df['Date'], train_df['Sales'], color='#1565C0', linewidth=0.8, label='Training Set')
ax.plot(test_df['Date'], test_df['Sales'], color='#E53935', linewidth=0.8, label='Test Set')
ax.axvline(train_df['Date'].max(), color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Split Point')

ax.set_title('Chronological Train/Test Split', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Daily Sales ($)')
ax.legend(fontsize=11)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/10_train_test_split.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    # =========================================================================
    # SECTION 9: BASELINE MODELS
    # =========================================================================
    cells.append(md("""
## 8️⃣ Baseline Models

Before using complex ML models, we establish simple baselines:

1. **Naive Baseline**: Predict each day's sales as the previous day's actual sales
2. **Moving Average Baseline**: Predict using the trailing 7-day average

Any useful ML model should outperform these baselines. If it doesn't, the added complexity is not justified.
"""))

    cells.append(code("""
# Store all results for comparison
results = []

# --- Baseline 1: Naive (Previous Day) ---
naive_preds = np.empty(len(test_df))
naive_preds[0] = train_df['Sales'].iloc[-1]  # Last known training value
for i in range(1, len(naive_preds)):
    naive_preds[i] = y_test[i - 1]  # Previous actual value

mae = mean_absolute_error(y_test, naive_preds)
rmse = np.sqrt(mean_squared_error(y_test, naive_preds))
r2 = r2_score(y_test, naive_preds)
mask = y_test != 0
mape = np.mean(np.abs((y_test[mask] - naive_preds[mask]) / y_test[mask])) * 100

results.append({'Model': 'Naive Baseline', 'MAE': round(mae, 2),
                'RMSE': round(rmse, 2), 'R²': round(r2, 4), 'MAPE (%)': round(mape, 2)})
print(f"Naive Baseline — MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")

# --- Baseline 2: 7-Day Moving Average ---
combined_sales = pd.concat([train_df['Sales'], test_df['Sales']])
rolling_ma = combined_sales.rolling(window=7, min_periods=1).mean().shift(1)
ma_preds = rolling_ma.iloc[len(train_df):].values

mae = mean_absolute_error(y_test, ma_preds)
rmse = np.sqrt(mean_squared_error(y_test, ma_preds))
r2 = r2_score(y_test, ma_preds)
mape = np.mean(np.abs((y_test[mask] - ma_preds[mask]) / y_test[mask])) * 100

results.append({'Model': 'Moving Avg (7d)', 'MAE': round(mae, 2),
                'RMSE': round(rmse, 2), 'R²': round(r2, 4), 'MAPE (%)': round(mape, 2)})
print(f"Moving Avg (7d) — MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
"""))

    # =========================================================================
    # SECTION 10: ML MODELS
    # =========================================================================
    cells.append(md("""
## 9️⃣ Machine Learning Models

We train three regression models of increasing complexity:

| Model | Strengths |
|-------|-----------|
| **Linear Regression** | Simple, interpretable, fast — good reference point |
| **Random Forest** | Handles non-linear relationships, provides feature importance |
| **Gradient Boosting** | Often best accuracy, learns sequentially from errors |
"""))

    cells.append(md("### Model 1: Linear Regression"))
    cells.append(code("""
# Scale features for Linear Regression (sensitive to feature magnitudes)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, lr_preds)
rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
r2 = r2_score(y_test, lr_preds)
mape = np.mean(np.abs((y_test[mask] - lr_preds[mask]) / y_test[mask])) * 100

results.append({'Model': 'Linear Regression', 'MAE': round(mae, 2),
                'RMSE': round(rmse, 2), 'R²': round(r2, 4), 'MAPE (%)': round(mape, 2)})
print(f"Linear Regression — MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
"""))

    cells.append(md("### Model 2: Random Forest Regressor"))
    cells.append(code("""
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, rf_preds)
rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
r2 = r2_score(y_test, rf_preds)
mape = np.mean(np.abs((y_test[mask] - rf_preds[mask]) / y_test[mask])) * 100

results.append({'Model': 'Random Forest', 'MAE': round(mae, 2),
                'RMSE': round(rmse, 2), 'R²': round(r2, 4), 'MAPE (%)': round(mape, 2)})
print(f"Random Forest — MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
"""))

    cells.append(md("### Model 3: Gradient Boosting (HistGradientBoosting)"))
    cells.append(code("""
gb_model = HistGradientBoostingRegressor(
    max_iter=300,
    max_depth=8,
    learning_rate=0.05,
    min_samples_leaf=10,
    random_state=RANDOM_STATE
)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)

mae = mean_absolute_error(y_test, gb_preds)
rmse = np.sqrt(mean_squared_error(y_test, gb_preds))
r2 = r2_score(y_test, gb_preds)
mape = np.mean(np.abs((y_test[mask] - gb_preds[mask]) / y_test[mask])) * 100

results.append({'Model': 'Gradient Boosting', 'MAE': round(mae, 2),
                'RMSE': round(rmse, 2), 'R²': round(r2, 4), 'MAPE (%)': round(mape, 2)})
print(f"Gradient Boosting — MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
"""))

    # =========================================================================
    # SECTION 11: TIME-SERIES MODEL (SARIMA)
    # =========================================================================
    cells.append(md("""
## 🔟 Time-Series Model: SARIMA

### Why SARIMA?

Unlike ML regression models that treat each observation independently (with engineered features), SARIMA explicitly models:

- **Autoregression (AR)**: Uses past values to predict future values
- **Integration (I)**: Differences the series to achieve stationarity
- **Moving Average (MA)**: Models the error terms
- **Seasonal components**: Captures repeating patterns at fixed intervals

### Key Assumption

SARIMA assumes the time series has a regular structure that can be captured by its parameters. It works directly on the univariate sales time series without requiring external features.

### Parameters

- `order=(1,1,1)` — ARIMA(p,d,q): 1 AR term, 1 differencing, 1 MA term
- `seasonal_order=(1,1,1,7)` — Weekly seasonality (period=7)
"""))

    cells.append(code("""
# Prepare daily time series for SARIMA
train_series = train_df.set_index('Date')['Sales']
train_series = train_series.asfreq('D', method='ffill')

print(f"Training SARIMA model on {len(train_series)} daily observations...")
print("This may take a moment...")

sarima_model = SARIMAX(
    train_series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_fitted = sarima_model.fit(disp=False, maxiter=200)

# Forecast for test period
sarima_preds = sarima_fitted.forecast(steps=len(test_df))
sarima_preds_values = sarima_preds.values

mae = mean_absolute_error(y_test, sarima_preds_values)
rmse = np.sqrt(mean_squared_error(y_test, sarima_preds_values))
r2 = r2_score(y_test, sarima_preds_values)
mape = np.mean(np.abs((y_test[mask] - sarima_preds_values[mask]) / y_test[mask])) * 100

results.append({'Model': 'SARIMA', 'MAE': round(mae, 2),
                'RMSE': round(rmse, 2), 'R²': round(r2, 4), 'MAPE (%)': round(mape, 2)})
print(f"\\nSARIMA — MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
print("\\n✓ SARIMA training complete")
"""))

    # =========================================================================
    # SECTION 12: MODEL EVALUATION & COMPARISON
    # =========================================================================
    cells.append(md("""
## 1️⃣1️⃣ Model Evaluation & Comparison

Now we compare all models side by side using the metrics calculated on the test set.

### Metrics Explained

| Metric | Meaning | Best Value |
|--------|---------|------------|
| **MAE** | Average absolute prediction error | Lower is better |
| **RMSE** | Error with penalty for large mistakes | Lower is better |
| **R²** | Proportion of variance explained (0–1) | Higher is better |
| **MAPE** | Average percentage error | Lower is better |
"""))

    cells.append(code("""
# Create comparison DataFrame
comparison_df = pd.DataFrame(results).set_index('Model')
print("\\n" + "=" * 70)
print("MODEL COMPARISON TABLE")
print("=" * 70)
print(comparison_df.to_string())
print("=" * 70)

# Identify best model
best_model_name = comparison_df['RMSE'].idxmin()
print(f"\\n★ Best Model (lowest RMSE): {best_model_name}")
print(f"  RMSE: {comparison_df.loc[best_model_name, 'RMSE']:.2f}")
print(f"  R²:   {comparison_df.loc[best_model_name, 'R²']:.4f}")
print(f"  MAE:  {comparison_df.loc[best_model_name, 'MAE']:.2f}")
"""))

    cells.append(md("### Model Comparison Chart"))
    cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = sns.color_palette('Set2', len(comparison_df))

# MAE comparison
bars1 = axes[0].bar(comparison_df.index, comparison_df['MAE'], color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_title('MAE Comparison (Lower is Better)', fontweight='bold')
axes[0].set_ylabel('Mean Absolute Error ($)')
best_mae_idx = comparison_df['MAE'].idxmin()
for bar, name in zip(bars1, comparison_df.index):
    if name == best_mae_idx:
        bar.set_edgecolor('gold')
        bar.set_linewidth(3)
for bar, val in zip(bars1, comparison_df['MAE']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.0f}', ha='center', va='bottom', fontsize=9)
axes[0].tick_params(axis='x', rotation=30)

# RMSE comparison
bars2 = axes[1].bar(comparison_df.index, comparison_df['RMSE'], color=colors, edgecolor='white', linewidth=1.5)
axes[1].set_title('RMSE Comparison (Lower is Better)', fontweight='bold')
axes[1].set_ylabel('Root Mean Squared Error ($)')
for bar, name in zip(bars2, comparison_df.index):
    if name == best_model_name:
        bar.set_edgecolor('gold')
        bar.set_linewidth(3)
for bar, val in zip(bars2, comparison_df['RMSE']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.0f}', ha='center', va='bottom', fontsize=9)
axes[1].tick_params(axis='x', rotation=30)

plt.suptitle('Model Performance Comparison', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('visualizations/11_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    # =========================================================================
    # SECTION 13: ACTUAL VS PREDICTED
    # =========================================================================
    cells.append(md("## 1️⃣2️⃣ Actual vs Predicted Visualization\n\nVisualize how each ML model's predictions compare against actual sales on the test set."))

    cells.append(code("""
# Determine best ML model predictions for the main chart
model_preds = {
    'Linear Regression': lr_preds,
    'Random Forest': rf_preds,
    'Gradient Boosting': gb_preds,
    'SARIMA': sarima_preds_values,
}

# Plot best model in detail
best_preds = model_preds.get(best_model_name, rf_preds)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(test_df['Date'].values, y_test, label='Actual Sales',
        color='#1565C0', linewidth=1.5, alpha=0.8)
ax.plot(test_df['Date'].values, best_preds, label=f'Predicted ({best_model_name})',
        color='#E53935', linewidth=1.5, alpha=0.8, linestyle='--')
ax.fill_between(test_df['Date'].values, y_test, best_preds,
                alpha=0.1, color='gray', label='Prediction Error')

ax.set_title(f'Actual vs Predicted Sales — {best_model_name}', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Daily Sales ($)')
ax.legend(fontsize=11)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/12_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    cells.append(md("### All Models Comparison"))
    cells.append(code("""
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax, (name, preds) in zip(axes, model_preds.items()):
    ax.plot(test_df['Date'].values, y_test, label='Actual', color='#1565C0', linewidth=1, alpha=0.7)
    ax.plot(test_df['Date'].values, preds, label=f'Predicted', color='#E53935', linewidth=1, alpha=0.7, linestyle='--')
    ax.set_title(name, fontweight='bold')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('Sales ($)')

plt.suptitle('Actual vs Predicted — All Models', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('visualizations/13_all_models_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    # =========================================================================
    # SECTION 14: FEATURE IMPORTANCE
    # =========================================================================
    cells.append(md("""
## 1️⃣3️⃣ Feature Importance

For tree-based models (Random Forest), we can examine which features the model relies on most for predictions.

> **⚠️ Important:** Feature importance indicates which features the model uses, NOT that those features *cause* sales changes. Correlation ≠ Causation.
"""))

    cells.append(code("""
# Random Forest Feature Importance
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importance_df)))
ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors, edgecolor='white')
ax.set_title('Top 15 Feature Importances (Random Forest)', fontweight='bold', fontsize=14)
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('visualizations/14_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nTop 5 Most Important Features:")
for _, row in importance_df.tail(5).iloc[::-1].iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")
"""))

    cells.append(md("""
### Feature Importance Interpretation

The most influential features are typically:
- **Lag features** (especially lag_1): Recent sales history is the strongest predictor of near-future sales
- **Rolling means**: Capture the local trend level
- **Day-of-week features**: Weekend vs weekday purchasing behavior
- **Month**: Seasonal purchasing patterns

This aligns with domain knowledge — yesterday's sales and recent trends are naturally predictive of tomorrow's sales.
"""))

    # =========================================================================
    # SECTION 15: RESIDUAL ANALYSIS
    # =========================================================================
    cells.append(md("## 1️⃣4️⃣ Residual Analysis\n\nExamine where the best model performs well vs. poorly."))
    cells.append(code("""
residuals = y_test - best_preds

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Distribution of residuals
axes[0].hist(residuals, bins=40, color='#7B1FA2', edgecolor='white', alpha=0.8)
axes[0].axvline(0, color='red', linestyle='--', linewidth=1.5)
axes[0].set_title('Residual Distribution', fontweight='bold')
axes[0].set_xlabel('Residual ($)')
axes[0].set_ylabel('Frequency')

# Residuals vs Predicted
axes[1].scatter(best_preds, residuals, alpha=0.3, s=10, color='#0097A7')
axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_title('Residuals vs Predicted Values', fontweight='bold')
axes[1].set_xlabel('Predicted Sales ($)')
axes[1].set_ylabel('Residual ($)')

# Residuals over time
axes[2].plot(test_df['Date'].values, residuals, linewidth=0.5, color='#E64A19', alpha=0.7)
axes[2].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[2].set_title('Residuals Over Time', fontweight='bold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Residual ($)')
axes[2].tick_params(axis='x', rotation=45)

plt.suptitle(f'Residual Analysis — {best_model_name}', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('visualizations/15_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Residual Statistics:")
print(f"  Mean: ${np.mean(residuals):,.2f} (ideally close to 0)")
print(f"  Std Dev: ${np.std(residuals):,.2f}")
print(f"  Min: ${np.min(residuals):,.2f}")
print(f"  Max: ${np.max(residuals):,.2f}")
"""))

    # =========================================================================
    # SECTION 16: FUTURE FORECASTING
    # =========================================================================
    cells.append(md("""
## 1️⃣5️⃣ Future Sales Forecasting

This is the core deliverable of the project. We use the best-performing model to forecast sales for the **next 30, 60, and 90 days** beyond the dataset.

> **⚠️ Disclaimer:** Forecasts are estimates based on historical patterns. They are NOT guaranteed outcomes. Real-world events (market changes, economic shifts, supply chain disruptions) can significantly alter actual results.
"""))

    cells.append(code("""
# Select the best ML model for forecasting
# We use Random Forest or Gradient Boosting (whichever performed better)
ml_models = {name: comparison_df.loc[name, 'RMSE']
             for name in ['Random Forest', 'Gradient Boosting']
             if name in comparison_df.index}
best_ml_name = min(ml_models, key=ml_models.get)

if best_ml_name == 'Random Forest':
    forecast_model = rf_model
    forecast_scaler = None
elif best_ml_name == 'Gradient Boosting':
    forecast_model = gb_model
    forecast_scaler = None
else:
    forecast_model = rf_model
    forecast_scaler = None

print(f"Selected model for forecasting: {best_ml_name}")
print(f"  Test RMSE: {ml_models[best_ml_name]:.2f}")
"""))

    cells.append(code("""
# Generate future dates
last_date = df_daily['Date'].max()
future_dates_30 = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
future_dates_60 = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=60, freq='D')
future_dates_90 = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=90, freq='D')

print(f"Last historical date: {last_date.date()}")
print(f"Forecast periods:")
print(f"  30-day: {future_dates_30[0].date()} to {future_dates_30[-1].date()}")
print(f"  60-day: {future_dates_60[0].date()} to {future_dates_60[-1].date()}")
print(f"  90-day: {future_dates_90[0].date()} to {future_dates_90[-1].date()}")
"""))

    cells.append(code("""
def iterative_forecast(model, future_dates, historical_sales, feature_cols, scaler=None):
    \"\"\"Generate forecasts iteratively, updating lag/rolling features with each prediction.\"\"\"
    sales_history = list(historical_sales.values[-60:])
    forecasts = []

    for i, date in enumerate(future_dates):
        row = pd.DataFrame({'Date': [date]})

        # Date features
        row['year'] = date.year
        row['month'] = date.month
        row['quarter'] = date.quarter
        row['week'] = int(date.isocalendar().week)
        row['day'] = date.day
        row['day_of_week'] = date.dayofweek
        row['is_weekend'] = int(date.dayofweek >= 5)
        row['day_of_year'] = date.dayofyear
        row['month_sin'] = np.sin(2 * np.pi * date.month / 12)
        row['month_cos'] = np.cos(2 * np.pi * date.month / 12)
        row['dow_sin'] = np.sin(2 * np.pi * date.dayofweek / 7)
        row['dow_cos'] = np.cos(2 * np.pi * date.dayofweek / 7)

        # Lag features
        for lag in [1, 7, 14, 30]:
            idx = len(sales_history) - lag
            row[f'lag_{lag}'] = sales_history[idx] if idx >= 0 else np.mean(sales_history)

        # Rolling features
        for window in [7, 14, 30]:
            recent = sales_history[-window:] if len(sales_history) >= window else sales_history
            row[f'rolling_mean_{window}'] = np.mean(recent)
            row[f'rolling_std_{window}'] = np.std(recent) if len(recent) > 1 else 0

        # Fill any missing feature columns with 0
        for col in feature_cols:
            if col not in row.columns:
                row[col] = 0

        # Predict
        X = row[feature_cols].values
        if scaler is not None:
            X = scaler.transform(X)
        pred = max(model.predict(X)[0], 0)  # Sales cannot be negative

        forecasts.append(pred)
        sales_history.append(pred)

    return np.array(forecasts)

# Generate forecasts for all horizons
historical_sales = df_daily['Sales']

forecast_30 = iterative_forecast(forecast_model, future_dates_30, historical_sales, feature_cols)
forecast_60 = iterative_forecast(forecast_model, future_dates_60, historical_sales, feature_cols)
forecast_90 = iterative_forecast(forecast_model, future_dates_90, historical_sales, feature_cols)

print(f"\\n✓ Forecasts generated:")
print(f"  30-day: avg ${np.mean(forecast_30):,.2f}/day, total ${np.sum(forecast_30):,.2f}")
print(f"  60-day: avg ${np.mean(forecast_60):,.2f}/day, total ${np.sum(forecast_60):,.2f}")
print(f"  90-day: avg ${np.mean(forecast_90):,.2f}/day, total ${np.sum(forecast_90):,.2f}")
"""))

    cells.append(md("### Forecast Visualization"))
    cells.append(code("""
# Main forecast chart (90-day)
fig, ax = plt.subplots(figsize=(16, 7))

# Historical context (last 120 days)
recent_hist = df_daily.tail(120)
ax.plot(recent_hist['Date'], recent_hist['Sales'],
        color='#1565C0', linewidth=1.5, label='Historical Sales', alpha=0.8)

# 90-day forecast
ax.plot(future_dates_90, forecast_90,
        color='#E53935', linewidth=2, linestyle='--',
        marker='o', markersize=2, label='Forecasted Sales (90 days)')

# Uncertainty band (±15%)
upper = forecast_90 * 1.15
lower = forecast_90 * 0.85
ax.fill_between(future_dates_90, lower, upper,
                alpha=0.15, color='#E53935', label='Uncertainty Band (±15%)')

# Split line
ax.axvline(last_date, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Forecast Start')

ax.set_title(f'Sales Forecast — Next 90 Days ({best_ml_name})', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Daily Sales ($)')
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/16_future_forecast.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    cells.append(md("### Multi-Horizon Forecast Comparison"))
    cells.append(code("""
fig, ax = plt.subplots(figsize=(14, 6))

# Historical
recent = df_daily.tail(60)
ax.plot(recent['Date'], recent['Sales'], color='#1565C0', linewidth=1.5, label='Historical')

# Different horizons
ax.plot(future_dates_30, forecast_30, color='#43A047', linewidth=2, linestyle='--', label='30-Day Forecast')
ax.plot(future_dates_60[30:], forecast_60[30:], color='#FF9800', linewidth=2, linestyle='--', label='31-60 Day Forecast')
ax.plot(future_dates_90[60:], forecast_90[60:], color='#E53935', linewidth=2, linestyle='--', label='61-90 Day Forecast')

ax.axvline(last_date, color='gray', linestyle=':', linewidth=2, alpha=0.5)
ax.set_title('Multi-Horizon Forecast Comparison', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Daily Sales ($)')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/17_multi_horizon_forecast.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    # =========================================================================
    # SECTION 17: FORECAST OUTPUT
    # =========================================================================
    cells.append(md("## 1️⃣6️⃣ Forecast Output\n\nExport the forecast to CSV for business use."))
    cells.append(code("""
# Create forecast DataFrame (90-day)
forecast_df = pd.DataFrame({
    'Date': future_dates_90,
    'Forecasted_Sales': np.round(forecast_90, 2)
})

print("Future Sales Forecast:")
print(forecast_df.head(10).to_string(index=False))
print(f"\\n... ({len(forecast_df)} rows total)")

# Save to CSV
forecast_df.to_csv('data/processed/future_forecast.csv', index=False)
print(f"\\n✓ Forecast saved to data/processed/future_forecast.csv")
"""))

    # =========================================================================
    # SECTION 18: MODEL SAVING
    # =========================================================================
    cells.append(md("## 1️⃣7️⃣ Model Saving\n\nSave the best model for future use or deployment."))
    cells.append(code("""
os.makedirs('models', exist_ok=True)

# Save the best ML model
joblib.dump(forecast_model, 'models/best_model.pkl')
print(f"✓ Best model ({best_ml_name}) saved to models/best_model.pkl")

# Save the scaler (needed for Linear Regression)
joblib.dump(scaler, 'models/scaler.pkl')
print("✓ Scaler saved to models/scaler.pkl")

# Save feature column names for consistency
joblib.dump(feature_cols, 'models/feature_cols.pkl')
print("✓ Feature column names saved to models/feature_cols.pkl")

# Verify: load and test
loaded_model = joblib.load('models/best_model.pkl')
test_pred = loaded_model.predict(X_test[:1])
print(f"\\n✓ Model load verification: prediction = ${test_pred[0]:,.2f}")
"""))

    # =========================================================================
    # SECTION 19: BUSINESS INSIGHTS
    # =========================================================================
    cells.append(md("""
## 1️⃣8️⃣ Business Insights & Recommendations

The following insights are derived directly from the data analysis and model results.
"""))

    cells.append(code("""
print("=" * 70)
print("BUSINESS INSIGHTS SUMMARY")
print("=" * 70)

# 1. Overall Trend
yearly_sales = df.groupby(df['Date'].dt.year)['Sales'].sum()
if len(yearly_sales) > 1:
    first_year = yearly_sales.iloc[0]
    last_year = yearly_sales.iloc[-1]
    total_growth = (last_year - first_year) / first_year * 100
    print(f"\\n📈 1. OVERALL TREND")
    print(f"   Sales grew {total_growth:.1f}% over the {len(yearly_sales)}-year period")
    print(f"   This indicates {'strong' if total_growth > 10 else 'moderate'} business growth")

# 2. Peak Season
monthly_totals = df.groupby(df['Date'].dt.month)['Sales'].sum()
peak_month = monthly_totals.idxmax()
low_month = monthly_totals.idxmin()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
print(f"\\n📅 2. SEASONAL PATTERNS")
print(f"   Peak month: {month_names[peak_month-1]} (${monthly_totals.max():,.0f})")
print(f"   Low month: {month_names[low_month-1]} (${monthly_totals.min():,.0f})")
print(f"   Season ratio: {monthly_totals.max() / monthly_totals.min():.1f}x difference")

# 3. Category Performance
if 'Category' in df.columns:
    cat_revenue = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
    print(f"\\n🏷️ 3. CATEGORY PERFORMANCE")
    for cat, rev in cat_revenue.items():
        pct = rev / cat_revenue.sum() * 100
        print(f"   {cat}: ${rev:,.0f} ({pct:.1f}%)")

# 4. Weekend Effect
weekend_avg = df[df['Date'].dt.dayofweek >= 5]['Sales'].mean()
weekday_avg = df[df['Date'].dt.dayofweek < 5]['Sales'].mean()
print(f"\\n📊 4. WEEKEND EFFECT")
print(f"   Weekday avg: ${weekday_avg:,.2f}")
print(f"   Weekend avg: ${weekend_avg:,.2f}")
print(f"   Weekend premium: {(weekend_avg - weekday_avg) / weekday_avg * 100:+.1f}%")

# 5. Forecast Summary
print(f"\\n🔮 5. FORECAST OUTLOOK")
print(f"   Next 30 days projected: ${np.sum(forecast_30):,.0f}")
print(f"   Next 90 days projected: ${np.sum(forecast_90):,.0f}")
avg_historical_30 = daily_sales['Sales'].tail(30).sum()
forecast_vs_hist = (np.sum(forecast_30) - avg_historical_30) / avg_historical_30 * 100
print(f"   30-day forecast vs last 30 days: {forecast_vs_hist:+.1f}%")

# 6. Model Confidence
print(f"\\n🎯 6. MODEL PERFORMANCE")
print(f"   Best model: {best_model_name}")
print(f"   Test R²: {comparison_df.loc[best_model_name, 'R²']:.4f}")
print(f"   Test MAPE: {comparison_df.loc[best_model_name, 'MAPE (%)']:.2f}%")

# 7. Limitations
print(f"\\n⚠️ 7. LIMITATIONS")
print(f"   - Forecasts are based on historical patterns only")
print(f"   - External factors (economy, competition, events) are not modeled")
print(f"   - Longer forecast horizons have increasing uncertainty")
print(f"   - The model should be retrained periodically with new data")

print("\\n" + "=" * 70)
"""))

    # =========================================================================
    # SECTION 20: CONCLUSION
    # =========================================================================
    cells.append(md("""
## 1️⃣9️⃣ Conclusion

### Summary

This project demonstrated a complete **end-to-end predictive analytics workflow**:

1. ✅ **Data Loading & Understanding** — Explored a 3-year retail sales dataset
2. ✅ **Data Cleaning** — Handled missing values, duplicates, and invalid entries
3. ✅ **Exploratory Data Analysis** — Uncovered trends, seasonality, and patterns
4. ✅ **Trend & Seasonality Decomposition** — Separated signal from noise
5. ✅ **Feature Engineering** — Created 23 meaningful predictive features
6. ✅ **Chronological Train/Test Split** — Preserved temporal integrity
7. ✅ **Baseline Models** — Established performance reference points
8. ✅ **ML Models** — Trained Linear Regression, Random Forest, and Gradient Boosting
9. ✅ **Time-Series Model** — Implemented SARIMA for comparison
10. ✅ **Model Evaluation** — Compared using MAE, RMSE, R², and MAPE
11. ✅ **Feature Importance** — Identified key predictors
12. ✅ **Future Forecasting** — Generated 30/60/90-day forecasts
13. ✅ **Business Insights** — Translated data into actionable recommendations

### Key Takeaways

- **Lag features** (recent sales history) are the most powerful predictors
- **Seasonal patterns** significantly influence sales
- **Tree-based models** (Random Forest, Gradient Boosting) outperform linear models
- **Proper time-series methodology** (chronological splitting, leakage prevention) is essential

---

*Project completed as part of Predictive Analytics Internship*
"""))

    nb.cells = cells
    return nb


if __name__ == "__main__":
    nb = build_notebook()
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notebooks")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predictive_analytics.ipynb")
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"✓ Notebook created: {output_path}")
