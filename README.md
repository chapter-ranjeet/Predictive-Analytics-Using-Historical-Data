# 📊 Predictive Analytics for Sales Forecasting Using Historical Data

A complete, end-to-end **Data Science project** that demonstrates predictive modeling and sales forecasting using historical retail data. This project implements the full ML pipeline — from raw data through exploratory analysis, model development, evaluation, and future forecasting.

---

## 📋 Overview

This project builds a **predictive analytics system** that forecasts future retail sales using historical transaction data. It employs multiple machine learning models and time-series analysis to generate data-driven forecasts, enabling better business decisions around inventory, staffing, and revenue planning.

## ❓ Problem Statement

Retail businesses need reliable sales forecasts to optimize operations. Without data-driven predictions, companies face:
- **Overstocking** — tying up capital in unsold inventory
- **Understocking** — losing revenue from missed sales opportunities
- **Poor resource allocation** — inefficient staffing and marketing spend

This project addresses these challenges by building predictive models that learn from historical patterns to forecast future sales.

## 🎯 Objectives

1. Analyze historical sales data to uncover trends, seasonality, and patterns
2. Build and evaluate multiple predictive models (ML + time-series)
3. Select the best-performing model based on rigorous evaluation metrics
4. Generate actionable 30/60/90-day sales forecasts
5. Deliver business insights supported by data evidence

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | Synthetically generated with realistic retail patterns |
| **Period** | January 2023 – December 2025 (3 years) |
| **Records** | ~92,000 transaction-level rows |
| **Columns** | 10 features |
| **Target** | `Sales` (Revenue in USD) |
| **Categories** | Electronics, Clothing, Home & Kitchen, Books, Sports |
| **Regions** | North, South, East, West |

### Feature Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `Date` | datetime | Transaction date |
| `Product` | categorical | Product name (20 unique products) |
| `Category` | categorical | Product category (5 categories) |
| `Region` | categorical | Geographic region |
| `Store` | categorical | Store identifier |
| `Quantity` | numeric | Units sold |
| `Unit_Price` | numeric | Price per unit ($) |
| `Discount` | numeric | Applied discount rate (0–0.25) |
| `Promotion` | binary | Active promotion flag |
| `Sales` | numeric | Total revenue (Quantity × Unit_Price) |

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3 | Core programming language |
| Jupyter Notebook | Interactive analysis & documentation |
| pandas | Data manipulation & analysis |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning models & evaluation |
| Statsmodels | Time-series decomposition & SARIMA |
| Joblib | Model serialization |
| Streamlit | Interactive dashboard |

## 🏗️ Project Architecture

```
Raw Data → Cleaning → EDA → Feature Engineering → Model Training → Evaluation → Forecasting
```

### Data Processing Pipeline

1. **Data Loading** — Read CSV, inspect structure
2. **Data Cleaning** — Handle missing values, duplicates, invalid entries
3. **Aggregation** — Convert transactions to daily sales
4. **Feature Engineering** — Create 23 predictive features (date, lag, rolling)

### Modeling Pipeline

1. **Chronological Split** — 80% train / 20% test (no random shuffling)
2. **Baseline Models** — Naive and Moving Average
3. **ML Models** — Linear Regression, Random Forest, Gradient Boosting
4. **Time-Series Model** — SARIMA
5. **Evaluation** — MAE, RMSE, R², MAPE
6. **Best Model Selection** — Based on lowest RMSE

## 📈 Exploratory Data Analysis

The EDA covers 7 analysis dimensions:

1. **Overall Sales Trend** — Line chart with 30-day moving average
2. **Monthly Performance** — Seasonal bar chart
3. **Yearly Comparison** — Year-over-year growth analysis
4. **Weekly Patterns** — Day-of-week purchasing behavior
5. **Category Performance** — Revenue breakdown by product category
6. **Sales Distribution** — Histogram and box plot
7. **Correlation Analysis** — Feature correlation heatmap

## ⚙️ Feature Engineering

| Feature Type | Features Created | Purpose |
|-------------|-----------------|---------|
| Date | year, month, quarter, week, day, day_of_week, is_weekend, day_of_year | Capture temporal patterns |
| Cyclical | month_sin, month_cos, dow_sin, dow_cos | Represent circular time features |
| Lag | lag_1, lag_7, lag_14, lag_30 | Use past sales as predictors |
| Rolling | rolling_mean/std for 7, 14, 30 days | Capture local trend and volatility |

> ⚠️ All lag/rolling features use `shift()` to prevent data leakage.

## 🤖 Models

| Model | Type | Description |
|-------|------|-------------|
| Naive Baseline | Baseline | Previous-day prediction |
| Moving Average | Baseline | 7-day trailing average |
| Linear Regression | ML | Linear relationship model |
| Random Forest | ML | Ensemble of decision trees |
| Gradient Boosting | ML | Sequential error-correcting ensemble |
| SARIMA | Time-Series | Seasonal ARIMA model |

## 📏 Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **MAE** | Average absolute prediction error ($) |
| **RMSE** | Error magnitude with large-error penalty ($) |
| **R²** | Proportion of variance explained (0–1) |
| **MAPE** | Average percentage error (%) |

> All metrics are computed on the **test set only** (unseen data).

## 📊 Results

Results are generated from actual model execution. Run the notebook to see the comparison table with real computed metrics for each model.

## 🔮 Forecasting

The best-performing model generates forecasts for:
- **30 days** — Short-term operational planning
- **60 days** — Medium-term resource allocation
- **90 days** — Strategic planning horizon

Forecasts include an illustrative ±15% uncertainty band to convey that predictions are estimates, not certainties.

## 📉 Visualizations

The project generates 17+ professional visualizations saved in `visualizations/`:
- Sales trends, monthly/yearly/weekly analysis
- Category performance, distribution, correlation
- Seasonal decomposition
- Train/test split
- Model comparison
- Actual vs predicted (individual and all models)
- Feature importance
- Residual diagnostics
- Future forecast (single and multi-horizon)

## 💡 Business Insights

Insights are derived from the actual data and include:
- Overall sales trend direction and growth rate
- Peak and low seasons
- Category revenue rankings
- Weekend vs weekday purchasing patterns
- Forecast outlook with comparison to recent actuals
- Model confidence levels and limitations

## 📁 Project Structure

```
predictive-sales-forecasting/
│
├── data/
│   ├── raw/                          # Original dataset
│   │   └── sales_data.csv
│   └── processed/                    # Cleaned data & forecasts
│       ├── cleaned_sales_data.csv
│       └── future_forecast.csv
│
├── notebooks/
│   └── predictive_analytics.ipynb    # Main analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py           # Synthetic data generator
│   ├── data_preprocessing.py         # Data cleaning pipeline
│   ├── exploratory_analysis.py       # EDA visualization functions
│   ├── feature_engineering.py        # Feature creation
│   ├── train_models.py               # Model training
│   ├── evaluate_models.py            # Model evaluation & comparison
│   ├── forecasting.py                # Future prediction generation
│   └── build_notebook.py             # Notebook builder utility
│
├── models/
│   ├── best_model.pkl                # Saved best model
│   ├── scaler.pkl                    # Feature scaler
│   └── feature_cols.pkl              # Feature column names
│
├── visualizations/                   # All generated charts (PNG)
│
├── reports/
│   └── project_report.md             # Detailed project report
│
├── app/
│   └── app.py                        # Streamlit dashboard
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Installation

### Prerequisites
- Python 3.9+
- pip package manager

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd predictive-sales-forecasting

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

## ▶️ How to Run

### 1. Generate Dataset (if not present)
```bash
python src/generate_dataset.py
```

### 2. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/predictive_analytics.ipynb
```
Execute all cells sequentially (Kernel → Restart & Run All).

### 3. Launch Streamlit Dashboard
```bash
streamlit run app/app.py
```
The dashboard opens at `http://localhost:8501`.

## 📊 Streamlit Dashboard

The interactive dashboard provides:
- **KPI Cards** — Total sales, average daily sales, top category, forecast summary
- **Sales Trend** — Historical daily sales with moving average
- **Monthly Analysis** — Monthly performance bar chart
- **Category Breakdown** — Sales by product category
- **Forecast View** — Future predictions with adjustable horizon
- **Data Explorer** — Browse the raw dataset

**Controls:** Filter by category, date range, and forecast horizon via the sidebar.

## ⚠️ Limitations

1. **Synthetic Data** — Results are based on generated data and may not reflect real-world complexity
2. **External Factors** — The model does not account for market events, competition, or macroeconomic changes
3. **Forecast Uncertainty** — Longer horizons have increasing prediction uncertainty
4. **Feature Scope** — Only historical sales patterns are used; no external data sources are integrated
5. **Single Univariate Target** — Forecasts aggregate daily sales; product-level forecasting is not implemented

## 🔮 Future Improvements

- Integrate real-world external data (weather, holidays, economic indicators)
- Implement product-level and category-level forecasting
- Add automated hyperparameter tuning (e.g., Optuna)
- Deploy the model as a REST API for production use
- Implement model retraining pipeline with new data
- Add confidence intervals using quantile regression
- Explore deep learning approaches (LSTM, Transformer)

## 👤 Author

**Data Science Intern**
Predictive Analytics Internship Project

---

*This project was developed as part of a Data Science internship to demonstrate proficiency in predictive modeling, data analysis, and data-driven decision making.*
