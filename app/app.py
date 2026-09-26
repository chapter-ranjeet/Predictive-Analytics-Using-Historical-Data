"""
Predictive Sales Analytics Dashboard
======================================
Interactive Streamlit dashboard for exploring sales data,
model predictions, and future forecasts.

Supports:
    - Built-in project dataset
    - User-uploaded CSV / Excel files with auto-column detection

Run with:
    streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib
import os
import sys
import io

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# --- Page Configuration ---
st.set_page_config(
    page_title="Predictive Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1565C0;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-box {
        border: 2px dashed #1565C0;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        background: #f8f9ff;
        margin-bottom: 1rem;
    }
    .success-banner {
        background: linear-gradient(135deg, #43A047 0%, #66BB6A 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# DATA LOADING FUNCTIONS
# =========================================================================

@st.cache_data
def load_project_data():
    """Load the project's built-in processed dataset."""
    processed_path = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_sales_data.csv")
    raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "sales_data.csv")

    if os.path.exists(processed_path):
        return pd.read_csv(processed_path, parse_dates=["Date"])
    elif os.path.exists(raw_path):
        return pd.read_csv(raw_path, parse_dates=["Date"])
    return None


@st.cache_data
def load_project_forecast():
    """Load the project's forecast CSV if available."""
    forecast_path = os.path.join(PROJECT_ROOT, "data", "processed", "future_forecast.csv")
    if os.path.exists(forecast_path):
        return pd.read_csv(forecast_path, parse_dates=["Date"])
    return None


def parse_uploaded_file(uploaded_file):
    """
    Parse an uploaded CSV or Excel file into a DataFrame.

    Handles common formats and provides clear error messages.
    """
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload a CSV or Excel (.xlsx/.xls) file."
        return df, None
    except Exception as e:
        return None, f"Error reading file: {str(e)}"


def auto_detect_columns(df):
    """
    Automatically detect which columns map to Date, Sales, and Category.

    Returns a dict with detected column names (or None if not found).
    """
    detected = {"date": None, "sales": None, "category": None}

    # --- Detect Date column ---
    date_keywords = ["date", "datetime", "time", "timestamp", "order_date",
                     "transaction_date", "period", "day"]
    for col in df.columns:
        col_lower = col.lower().strip()
        if any(kw in col_lower for kw in date_keywords):
            detected["date"] = col
            break
    # Fallback: check for datetime-parseable columns
    if detected["date"] is None:
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    pd.to_datetime(df[col].head(20), errors="raise")
                    detected["date"] = col
                    break
                except (ValueError, TypeError):
                    continue

    # --- Detect Sales / Revenue column ---
    sales_keywords = ["sales", "revenue", "amount", "total", "price",
                      "value", "turnover", "income", "sum"]
    for col in df.columns:
        col_lower = col.lower().strip()
        if any(kw in col_lower for kw in sales_keywords):
            if pd.api.types.is_numeric_dtype(df[col]):
                detected["sales"] = col
                break
    # Fallback: pick the first numeric column that isn't quantity-like
    if detected["sales"] is None:
        for col in df.select_dtypes(include=[np.number]).columns:
            if "qty" not in col.lower() and "quantity" not in col.lower() and "id" not in col.lower():
                detected["sales"] = col
                break

    # --- Detect Category column ---
    cat_keywords = ["category", "product_category", "segment", "type",
                    "group", "department", "class"]
    for col in df.columns:
        col_lower = col.lower().strip()
        if any(kw in col_lower for kw in cat_keywords):
            detected["category"] = col
            break

    return detected


def prepare_uploaded_data(df, date_col, sales_col, category_col=None):
    """
    Clean and prepare uploaded data for analysis.

    Steps: select & rename columns, parse dates, remove invalids, sort.
    """
    # Select only the columns we need to avoid duplicate-name collisions
    cols_to_keep = [date_col, sales_col]
    if category_col and category_col in df.columns and category_col not in cols_to_keep:
        cols_to_keep.append(category_col)
    # Also keep any other numeric columns for correlation analysis
    for c in df.columns:
        if c not in cols_to_keep and pd.api.types.is_numeric_dtype(df[c]):
            cols_to_keep.append(c)

    work_df = df[cols_to_keep].copy()

    # Rename selected columns to standard names
    rename_map = {date_col: "Date", sales_col: "Sales"}
    if category_col and category_col in work_df.columns:
        rename_map[category_col] = "Category"
    work_df = work_df.rename(columns=rename_map)

    # Handle any remaining duplicate column names (keep first occurrence)
    work_df = work_df.loc[:, ~work_df.columns.duplicated()]

    # Parse dates
    work_df["Date"] = pd.to_datetime(work_df["Date"], errors="coerce")
    work_df = work_df.dropna(subset=["Date"])

    # Ensure Sales is numeric (use .iloc[:, 0] as safety if still a DataFrame)
    sales_series = work_df["Sales"]
    if isinstance(sales_series, pd.DataFrame):
        sales_series = sales_series.iloc[:, 0]
    work_df["Sales"] = pd.to_numeric(sales_series, errors="coerce")
    work_df = work_df.dropna(subset=["Sales"])
    work_df = work_df[work_df["Sales"] > 0]

    # Sort chronologically
    work_df = work_df.sort_values("Date").reset_index(drop=True)

    return work_df



def run_quick_forecast(daily_df, horizon=30):
    """
    Run a quick moving-average-based forecast on uploaded data.

    Uses a 14-day weighted moving average for simplicity and speed,
    since we can't assume the user's data has the same feature structure.
    """
    sales = daily_df["Sales"].values
    last_date = daily_df["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=horizon, freq="D")

    # Weighted moving average (recent days weighted more)
    window = min(14, len(sales))
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()

    history = list(sales[-window:])
    forecasts = []

    for i in range(horizon):
        w = weights[-len(history):]
        w = w / w.sum()
        pred = np.dot(history[-len(w):], w)

        # Add slight weekly seasonality from data
        if len(sales) >= 7:
            dow = future_dates[i].dayofweek
            dow_avg = daily_df.groupby(daily_df["Date"].dt.dayofweek)["Sales"].mean()
            overall_avg = daily_df["Sales"].mean()
            if dow in dow_avg.index and overall_avg > 0:
                seasonal_factor = dow_avg[dow] / overall_avg
                pred *= seasonal_factor

        pred = max(pred, 0)
        forecasts.append(round(pred, 2))
        history.append(pred)

    return pd.DataFrame({"Date": future_dates, "Forecasted_Sales": forecasts})


# =========================================================================
# VISUALIZATION HELPERS
# =========================================================================

def plot_trend(daily_df):
    """Plot daily sales trend with moving average."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_df["Date"], daily_df["Sales"], alpha=0.3, linewidth=0.5,
            color="#90CAF9", label="Daily")
    ma = daily_df["Sales"].rolling(min(30, len(daily_df) // 3 + 1), min_periods=1).mean()
    ax.plot(daily_df["Date"], ma, linewidth=2, color="#1565C0", label="Moving Average")
    ax.set_title("Daily Sales Trend", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_monthly(df):
    """Bar chart of monthly sales."""
    monthly = df.copy()
    monthly["YearMonth"] = monthly["Date"].dt.to_period("M").astype(str)
    agg = monthly.groupby("YearMonth")["Sales"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(agg)))
    ax.bar(range(len(agg)), agg["Sales"], color=colors)
    step = max(1, len(agg) // 12)
    ax.set_xticks(range(0, len(agg), step))
    ax.set_xticklabels(agg["YearMonth"].iloc[::step], rotation=45)
    ax.set_title("Monthly Sales", fontweight="bold")
    ax.set_ylabel("Total Sales ($)")
    plt.tight_layout()
    return fig


def plot_category(df):
    """Category breakdown charts."""
    if "Category" not in df.columns:
        return None
    cat_sales = df.groupby("Category")["Sales"].sum().sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = sns.color_palette("Set2", len(cat_sales))
    axes[0].barh(cat_sales.index, cat_sales.values, color=colors, edgecolor="white")
    axes[0].set_title("Total Sales by Category", fontweight="bold")
    axes[0].set_xlabel("Sales ($)")

    axes[1].pie(cat_sales.values, labels=cat_sales.index, autopct="%1.1f%%",
                colors=colors)
    axes[1].set_title("Sales Share", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_forecast_chart(daily_hist, forecast_df, horizon):
    """Forecast chart with historical context."""
    display = forecast_df.head(horizon)
    recent = daily_hist.tail(min(90, len(daily_hist)))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(recent["Date"], recent["Sales"], color="#1565C0",
            linewidth=1.5, label="Historical")
    ax.plot(display["Date"], display["Forecasted_Sales"],
            color="#E53935", linewidth=2, linestyle="--",
            marker="o", markersize=3, label="Forecast")

    upper = display["Forecasted_Sales"] * 1.15
    lower = display["Forecasted_Sales"] * 0.85
    ax.fill_between(display["Date"], lower, upper, alpha=0.15, color="#E53935")
    ax.axvline(daily_hist["Date"].max(), color="gray", linestyle=":",
               linewidth=2, alpha=0.5)
    ax.set_title(f"Sales Forecast — Next {horizon} Days", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


# =========================================================================
# MAIN DASHBOARD
# =========================================================================

def render_dashboard(df, forecast_df, data_source_label):
    """Render the full dashboard for a given dataset."""

    # --- Sidebar Filters ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Filters")

    # Category filter
    if "Category" in df.columns:
        categories = ["All"] + sorted(df["Category"].unique().tolist())
        selected_category = st.sidebar.selectbox("📦 Category", categories)
    else:
        selected_category = "All"

    # Date range filter
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Forecast horizon
    forecast_horizon = st.sidebar.slider("🔮 Forecast Days", 7, 90, 30)

    # --- Apply filters ---
    filtered_df = df.copy()
    if selected_category != "All" and "Category" in df.columns:
        filtered_df = filtered_df[filtered_df["Category"] == selected_category]
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df["Date"].dt.date >= start_date) &
            (filtered_df["Date"].dt.date <= end_date)
        ]

    # --- KPI Cards ---
    st.markdown("### 📈 Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    total_sales = filtered_df["Sales"].sum()
    daily_agg = filtered_df.groupby("Date")["Sales"].sum()
    avg_daily = daily_agg.mean() if len(daily_agg) > 0 else 0

    with kpi1:
        st.metric("💰 Total Sales", f"${total_sales:,.0f}")
    with kpi2:
        st.metric("📊 Avg Daily Sales", f"${avg_daily:,.0f}")
    with kpi3:
        if "Category" in filtered_df.columns and len(filtered_df) > 0:
            best_cat = filtered_df.groupby("Category")["Sales"].sum().idxmax()
            st.metric("🏆 Top Category", best_cat)
        else:
            st.metric("📦 Transactions", f"{len(filtered_df):,}")
    with kpi4:
        if forecast_df is not None:
            fc_total = forecast_df["Forecasted_Sales"].head(forecast_horizon).sum()
            st.metric(f"🔮 Forecast ({forecast_horizon}d)", f"${fc_total:,.0f}")
        else:
            n_days = (max_date - min_date).days
            st.metric("📅 Date Span", f"{n_days} days")

    st.markdown("---")

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Sales Trend", "📅 Monthly Analysis",
        "📦 Category Breakdown", "🔮 Forecast", "📊 Data Explorer"
    ])

    # Daily aggregation for charts
    daily_df = filtered_df.groupby("Date")["Sales"].sum().reset_index()

    with tab1:
        st.subheader("Historical Sales Trend")
        if len(daily_df) > 0:
            fig = plot_trend(daily_df)
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("No data available for the selected filters.")

    with tab2:
        st.subheader("Monthly Sales Performance")
        if len(filtered_df) > 0:
            fig = plot_monthly(filtered_df)
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("No data available.")

    with tab3:
        if "Category" in filtered_df.columns and len(filtered_df) > 0:
            st.subheader("Category Performance")
            fig = plot_category(filtered_df)
            if fig:
                st.pyplot(fig)
                plt.close()
        else:
            st.info("No 'Category' column available for breakdown.")

    with tab4:
        st.subheader("Future Sales Forecast")
        if forecast_df is not None and len(forecast_df) > 0:
            # Use all data (not filtered) for forecast context
            all_daily = df.groupby("Date")["Sales"].sum().reset_index()
            fig = plot_forecast_chart(all_daily, forecast_df, forecast_horizon)
            st.pyplot(fig)
            plt.close()

            # Forecast table
            st.markdown("#### Forecast Data")
            display_fc = forecast_df.head(forecast_horizon)
            st.dataframe(display_fc.style.format({
                "Forecasted_Sales": "${:,.2f}"
            }), use_container_width=True)

            # Download button for forecast
            csv_data = display_fc.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv_data,
                file_name="forecast.csv",
                mime="text/csv",
            )
        else:
            st.warning("No forecast data available.")

    with tab5:
        st.subheader("Data Explorer")
        st.markdown(f"**Source:** {data_source_label}")
        st.dataframe(filtered_df.head(1000), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Dataset Shape**")
            st.write(f"Rows: {len(filtered_df):,}")
            st.write(f"Columns: {len(filtered_df.columns)}")
        with col2:
            st.markdown("**Date Range**")
            st.write(f"From: {filtered_df['Date'].min().date()}")
            st.write(f"To: {filtered_df['Date'].max().date()}")
        with col3:
            st.markdown("**Sales Summary**")
            st.write(f"Total: ${filtered_df['Sales'].sum():,.0f}")
            st.write(f"Mean: ${filtered_df['Sales'].mean():,.2f}")

        # Download cleaned data
        csv_dl = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv_dl,
            file_name="filtered_data.csv",
            mime="text/csv",
        )


# =========================================================================
# MAIN APP
# =========================================================================

def main():
    """Main application entry point."""

    # --- Header ---
    st.markdown(
        '<div class="main-header">📊 Predictive Sales Analytics Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">'
        'Sales Forecasting Using Historical Data | Internship Project'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Sidebar: Data Source Selection ---
    st.sidebar.title("📂 Data Source")

    data_source = st.sidebar.radio(
        "Choose data source:",
        ["📁 Project Dataset", "📤 Upload Your File"],
        index=0,
    )

    # =================================================================
    # PATH 1: Built-in project dataset
    # =================================================================
    if data_source == "📁 Project Dataset":
        df = load_project_data()
        if df is None:
            st.error(
                "No project dataset found. Run the notebook first, or "
                "upload your own file using the sidebar."
            )
            st.stop()

        forecast_df = load_project_forecast()
        render_dashboard(df, forecast_df, "Built-in project dataset")

    # =================================================================
    # PATH 2: User-uploaded file
    # =================================================================
    else:
        st.sidebar.markdown("---")
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Upload a file with at least a Date column and a numeric Sales/Revenue column.",
        )

        if uploaded_file is None:
            # Show instructions when no file is uploaded yet
            st.markdown("---")

            col_l, col_c, col_r = st.columns([1, 2, 1])
            with col_c:
                st.markdown(
                    '<div class="upload-box">'
                    '<h3>📤 Upload Your Dataset</h3>'
                    '<p>Use the sidebar to upload a <strong>CSV</strong> or '
                    '<strong>Excel</strong> file.</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("#### Requirements")
            st.markdown(
                """
                Your file should contain at minimum:

                | Column | Description | Example |
                |--------|-------------|---------|
                | **Date** | A date or datetime column | `2024-01-15`, `15/01/2024` |
                | **Sales / Revenue** | A numeric column to analyze | `1500.00`, `230` |
                | **Category** *(optional)* | A categorical column for breakdown | `Electronics`, `Clothing` |

                The dashboard will **auto-detect** your column names. You can also
                override the mapping manually after upload.
                """
            )

            st.markdown("#### Supported Formats")
            c1, c2 = st.columns(2)
            with c1:
                st.success("✅ CSV (.csv)")
                st.success("✅ Excel (.xlsx, .xls)")
            with c2:
                st.info("💡 Max file size: 200 MB")
                st.info("💡 UTF-8 encoding recommended for CSV")

            st.stop()

        # --- File uploaded: parse it ---
        raw_df, error = parse_uploaded_file(uploaded_file)
        if error:
            st.error(error)
            st.stop()

        st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
        st.sidebar.write(f"Rows: {len(raw_df):,} | Cols: {len(raw_df.columns)}")

        # --- Auto-detect columns ---
        detected = auto_detect_columns(raw_df)

        st.sidebar.markdown("---")
        st.sidebar.subheader("🔗 Column Mapping")

        all_cols = list(raw_df.columns)

        # Date column selector
        date_default = all_cols.index(detected["date"]) if detected["date"] in all_cols else 0
        date_col = st.sidebar.selectbox(
            "📅 Date Column",
            all_cols,
            index=date_default,
        )

        # Sales column selector
        numeric_cols = list(raw_df.select_dtypes(include=[np.number]).columns)
        if not numeric_cols:
            st.error("No numeric columns found in your file. Cannot identify a Sales column.")
            st.stop()
        sales_default = (
            numeric_cols.index(detected["sales"])
            if detected["sales"] in numeric_cols else 0
        )
        sales_col = st.sidebar.selectbox(
            "💰 Sales / Revenue Column",
            numeric_cols,
            index=sales_default,
        )

        # Category column selector (optional)
        cat_options = ["(None)"] + [c for c in all_cols if c != date_col and c != sales_col]
        cat_default = (
            cat_options.index(detected["category"])
            if detected["category"] in cat_options else 0
        )
        category_col = st.sidebar.selectbox(
            "📦 Category Column (optional)",
            cat_options,
            index=cat_default,
        )
        if category_col == "(None)":
            category_col = None

        # --- Prepare data ---
        df = prepare_uploaded_data(raw_df, date_col, sales_col, category_col)

        if len(df) == 0:
            st.error(
                "No valid rows remain after cleaning. Check that your Date and "
                "Sales columns contain valid values."
            )
            st.stop()

        # Show success banner
        st.markdown(
            f'<div class="success-banner">'
            f'✅ <strong>{uploaded_file.name}</strong> loaded successfully — '
            f'{len(df):,} rows | '
            f'{df["Date"].min().date()} to {df["Date"].max().date()}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # --- Generate quick forecast on uploaded data ---
        daily_df = df.groupby("Date")["Sales"].sum().reset_index()
        if len(daily_df) >= 7:
            forecast_df = run_quick_forecast(daily_df, horizon=90)
        else:
            forecast_df = None
            st.warning(
                "Not enough data points for forecasting (need at least 7 days)."
            )

        render_dashboard(df, forecast_df, f"Uploaded file: {uploaded_file.name}")

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #999; font-size: 0.9rem;'>"
        "Predictive Analytics for Sales Forecasting | Internship Project"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
