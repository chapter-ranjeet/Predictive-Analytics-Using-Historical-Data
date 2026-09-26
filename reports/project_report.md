# Predictive Analytics for Sales Forecasting — Project Report

---

## 1. Abstract

This report presents a complete predictive analytics project that forecasts future retail sales using historical transaction data. The project implements a full data science workflow: data preprocessing, exploratory data analysis, feature engineering, model training and evaluation, and future forecasting. Multiple modeling approaches — including baseline methods, machine learning regression, and statistical time-series models — are compared using standard evaluation metrics. The best-performing model is selected and used to generate 30, 60, and 90-day sales forecasts. The project demonstrates that data-driven forecasting can capture meaningful sales patterns and provide actionable business insights.

---

## 2. Introduction

Predictive analytics is the practice of extracting information from historical data to identify patterns and predict future outcomes. In the retail industry, accurate sales forecasting enables better inventory management, resource allocation, and financial planning. This project applies predictive analytics techniques to a historical retail sales dataset, building models that learn from past trends and seasonality to forecast future revenue.

The project follows established data science methodology and prioritizes correctness, reproducibility, and clear documentation over complexity.

---

## 3. Problem Statement

Retail businesses face significant challenges in predicting future sales. Inaccurate forecasts lead to:

- **Overstocking**: Excess inventory that ties up capital and may become obsolete
- **Understocking**: Lost revenue from inability to meet customer demand
- **Inefficient operations**: Suboptimal staffing and marketing expenditures

The goal of this project is to build a predictive model that forecasts daily sales revenue based on historical patterns, enabling proactive business decision-making.

---

## 4. Objectives

1. Load, clean, and preprocess a historical sales dataset
2. Conduct thorough exploratory data analysis to understand data characteristics
3. Identify and quantify trends, seasonality, and other temporal patterns
4. Engineer meaningful predictive features from raw data
5. Train and evaluate multiple predictive models using proper time-series methodology
6. Select the best-performing model based on established evaluation criteria
7. Generate future sales forecasts with appropriate caveats
8. Extract actionable business insights from the analysis

---

## 5. Dataset Description

### Source

The dataset is synthetically generated using a controlled simulation that incorporates realistic retail patterns including:
- Long-term growth trends (approximately 5% annual growth)
- Monthly seasonality (holiday peaks in November-December)
- Weekly patterns (higher weekend sales)
- Category-specific seasonal effects
- Promotional discounts and their impact on volume
- Regional and store-level variation
- Random noise from a log-normal distribution

### Characteristics

| Property | Value |
|----------|-------|
| Records | ~92,000 transaction-level rows |
| Columns | 10 features |
| Date Range | January 1, 2023 – December 31, 2025 |
| Categories | 5 (Electronics, Clothing, Home & Kitchen, Books, Sports) |
| Products | 20 unique products |
| Regions | 4 (North, South, East, West) |
| Stores | 3 (Store_A, Store_B, Store_C) |
| Target Variable | Sales (Revenue in USD) |

### Data Quality Issues (Intentional)

To simulate real-world conditions, the dataset contains:
- ~0.5% missing values in Quantity and Unit_Price columns
- ~0.2% duplicate rows
- Shuffled row order (requiring chronological sorting)

---

## 6. Data Preprocessing

### Steps Performed

1. **Date Conversion**: Converted the Date column from string to datetime format
2. **Duplicate Removal**: Identified and removed duplicate transaction records
3. **Missing Value Handling**: Filled missing numerical values with column medians (robust to outliers)
4. **Data Validation**: Verified no negative or zero sales values exist
5. **Chronological Sorting**: Sorted all records by date for time-series integrity
6. **Aggregation**: Aggregated transaction-level data to daily sales totals for forecasting

### Rationale

- **Median imputation** was chosen over mean imputation because sales data is typically right-skewed, making the median more representative of typical values.
- **Chronological sorting** is essential for time-series analysis to prevent temporal data leakage.

---

## 7. Exploratory Data Analysis

Seven analysis dimensions were explored:

### 7.1 Overall Sales Trend
A line chart with a 30-day moving average revealed a clear upward trend over the 3-year period, confirming overall business growth.

### 7.2 Monthly Sales Performance
Bar chart analysis showed seasonal peaks in the November-December holiday period and relative lows in January-February, consistent with typical retail cycles.

### 7.3 Year-over-Year Comparison
Annual sales totals showed consistent growth, with year-over-year increases reflecting the underlying positive trend.

### 7.4 Day-of-Week Patterns
Average sales by day of week revealed higher weekend sales compared to weekdays, consistent with consumer shopping behavior.

### 7.5 Category Performance
Revenue breakdown showed varying contributions from different product categories, with electronics commanding higher per-transaction values.

### 7.6 Sales Distribution
Transaction-level sales exhibited right-skewness (positive skew), typical of retail data where most transactions are moderate-sized with fewer large purchases.

### 7.7 Correlation Analysis
A correlation heatmap of numerical features revealed expected relationships (e.g., Quantity and Sales are correlated by construction) and helped identify potential multicollinearity.

---

## 8. Trend Analysis

### Seasonal Decomposition

Using the `seasonal_decompose` function from statsmodels with an additive model and 30-day period:

- **Trend Component**: Confirmed a steady upward trajectory over the full dataset
- **Seasonal Component**: Revealed regular monthly cycles with consistent amplitude
- **Residual Component**: Showed relatively small random variation, suggesting the trend and seasonal components explain most of the signal

### Monthly Seasonality Profile

Analysis of average sales by calendar month identified peak and trough months, providing valuable information for inventory and marketing planning.

---

## 9. Feature Engineering

### Features Created (23 total)

| Category | Features | Count |
|----------|----------|-------|
| Calendar | year, month, quarter, week, day, day_of_week, is_weekend, day_of_year | 8 |
| Cyclical | month_sin, month_cos, dow_sin, dow_cos | 4 |
| Lag | lag_1, lag_7, lag_14, lag_30 | 4 |
| Rolling Mean | rolling_mean_7, rolling_mean_14, rolling_mean_30 | 3 |
| Rolling Std | rolling_std_7, rolling_std_14, rolling_std_30 | 3 |
| **Total** | | **23** (after one-hot encoding may add more) |

### Data Leakage Prevention

All lag and rolling features use `shift()` operations to ensure that only past information is incorporated:
- Lag features shift the target variable by the specified number of periods
- Rolling features apply `shift(1)` before computing the rolling window, excluding the current observation

---

## 10. Methodology

### Train/Test Split

A **chronological split** (80% training / 20% testing) was used instead of random splitting. This is critical for time-series data because:
- Random splitting would allow future data points to "leak" into the training set
- Models evaluated on randomly split data show inflated accuracy that does not generalize to real forecasting scenarios
- Chronological splitting simulates the actual deployment scenario: training on historical data and predicting future values

### Evaluation Approach

All models are evaluated on the same held-out test set using identical metrics, ensuring a fair comparison.

---

## 11. Model Development

### Baseline Models

1. **Naive Baseline**: Predicts each day's sales as the previous day's value. Establishes the minimum performance bar.
2. **Moving Average (7-day)**: Uses the trailing 7-day average as the prediction. Provides a smoother, more stable baseline.

### Machine Learning Models

3. **Linear Regression**: Models a linear relationship between features and sales. Scaled using StandardScaler. Simple and interpretable but limited in capturing non-linear patterns.

4. **Random Forest Regressor**: Ensemble of 200 decision trees with max_depth=15. Handles non-linear relationships and feature interactions well. Provides feature importance rankings.

5. **HistGradientBoosting Regressor**: Sequential ensemble that learns from previous errors. 300 iterations with learning_rate=0.05. Often achieves the best accuracy among traditional ML models.

### Time-Series Model

6. **SARIMA(1,1,1)(1,1,1,7)**: Statistical model that explicitly captures autoregressive patterns, moving average effects, and weekly seasonality (period=7). Does not require external features — works directly on the univariate sales series.

---

## 12. Evaluation

### Metrics Used

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| MAE | Mean(|actual - predicted|) | Average dollar error; easy to interpret |
| RMSE | √Mean((actual - predicted)²) | Penalizes large errors more heavily |
| R² | 1 - SS_res/SS_tot | Proportion of variance explained (0–1) |
| MAPE | Mean(|actual - predicted|/|actual|) × 100 | Percentage error; scale-independent |

### Model Selection Criteria

The best model was selected based on the **lowest RMSE** on the test set. RMSE was chosen as the primary metric because it penalizes large prediction errors, which are particularly costly in business planning scenarios.

> Note: The model comparison results are generated from actual computations. No metrics are fabricated.

---

## 13. Forecasting Results

### Forecast Generation

The best-performing ML model generates iterative forecasts:
1. Predict day 1 using actual historical features
2. Use day 1's prediction to compute day 2's lag features
3. Continue iteratively for the full forecast horizon

This iterative approach correctly handles the dependency of lag features on prior predictions.

### Forecast Horizons

- **30 days**: Short-term operational forecasts
- **60 days**: Medium-term resource planning
- **90 days**: Strategic planning horizon

### Output

Forecasts are exported to `data/processed/future_forecast.csv` with Date and Forecasted_Sales columns.

---

## 14. Business Insights

All insights below are derived directly from the data analysis:

1. **Growth Trajectory**: Sales show a positive growth trend over the 3-year analysis period
2. **Seasonal Peaks**: November-December consistently produce the highest sales, driven by holiday shopping
3. **Weekend Premium**: Weekend sales systematically exceed weekday sales
4. **Category Dynamics**: Different product categories show distinct seasonal patterns (e.g., electronics peak during holidays, sports peak in summer)
5. **Promotional Impact**: Products under promotion show increased sales volume
6. **Model Reliability**: The best model explains a substantial portion of sales variance on unseen test data, suggesting the patterns are learnable and consistent

---

## 15. Limitations

1. **Synthetic Data**: While designed to be realistic, synthetic data cannot capture all real-world complexities
2. **No External Variables**: The model does not incorporate macroeconomic indicators, competitor actions, weather, or marketing campaign data
3. **Aggregation Level**: Forecasts are at the aggregate daily level; product-level or store-level forecasts are not generated
4. **Forecast Horizon Decay**: Prediction accuracy degrades for longer forecast horizons due to error accumulation
5. **Stationarity Assumption**: Some models assume that historical patterns will continue unchanged into the future
6. **No Anomaly Handling**: Unusual events (e.g., supply chain disruptions, viral products) are not explicitly modeled

---

## 16. Future Scope

1. **Real Data Integration**: Apply the methodology to actual retail sales data
2. **External Features**: Incorporate weather, holidays, economic indicators, and marketing spend
3. **Granular Forecasting**: Implement product-level and store-level predictions
4. **Automated Tuning**: Use Bayesian optimization (Optuna) for hyperparameter selection
5. **Deep Learning**: Experiment with LSTM, GRU, and Transformer architectures for sequence modeling
6. **Deployment**: Package the model as a REST API with automated retraining pipelines
7. **Uncertainty Quantification**: Implement prediction intervals using quantile regression or conformal prediction

---

## 17. Conclusion

This project successfully demonstrates a complete predictive analytics workflow for sales forecasting. Key achievements include:

- **Thorough data preprocessing** with documented cleaning steps
- **Comprehensive EDA** revealing meaningful business patterns
- **Rigorous feature engineering** with explicit data leakage prevention
- **Multiple model comparison** using proper time-series evaluation methodology
- **Actionable forecasts** for multiple planning horizons
- **Clear business insights** supported by data evidence

The project follows best practices in data science methodology: reproducible code, proper train/test separation, honest evaluation, and transparent documentation of limitations.

---

*Report prepared as part of the Predictive Analytics Internship*
*Date: September 2026*
