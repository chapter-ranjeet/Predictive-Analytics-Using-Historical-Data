"""
Synthetic Dataset Generator
=============================
Generates a realistic historical retail sales dataset with
trends, seasonality, promotions, and noise.

This script creates a labeled synthetic dataset suitable for
demonstrating predictive analytics and time-series forecasting.

Dataset characteristics:
    - 3 years of daily transaction-level data (2023-01 to 2025-12)
    - 5 product categories with multiple products
    - Seasonal patterns (monthly, weekly)
    - Long-term growth trend
    - Promotional effects
    - Regional variation
    - Realistic noise
"""

import pandas as pd
import numpy as np
import os

# Reproducibility
np.random.seed(42)

# --- Configuration ---
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

CATEGORIES = {
    "Electronics": {
        "products": ["Laptop", "Smartphone", "Tablet", "Headphones"],
        "base_price": [800, 500, 350, 80],
        "base_qty": [3, 8, 5, 15],
    },
    "Clothing": {
        "products": ["T-Shirt", "Jeans", "Jacket", "Shoes"],
        "base_price": [25, 45, 90, 65],
        "base_qty": [20, 12, 6, 10],
    },
    "Home & Kitchen": {
        "products": ["Blender", "Cookware Set", "Vacuum", "Coffee Maker"],
        "base_price": [60, 120, 200, 85],
        "base_qty": [8, 4, 3, 7],
    },
    "Books": {
        "products": ["Fiction", "Non-Fiction", "Textbook", "Children's Book"],
        "base_price": [15, 20, 55, 12],
        "base_qty": [25, 18, 8, 20],
    },
    "Sports": {
        "products": ["Running Shoes", "Yoga Mat", "Dumbbell Set", "Water Bottle"],
        "base_price": [110, 30, 70, 15],
        "base_qty": [6, 12, 4, 25],
    },
}

REGIONS = ["North", "South", "East", "West"]
STORES = ["Store_A", "Store_B", "Store_C"]


def generate_dataset():
    """Generate the complete synthetic sales dataset."""
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    records = []

    for date in dates:
        day_of_year = date.dayofyear
        day_of_week = date.dayofweek  # 0=Monday, 6=Sunday
        month = date.month
        year = date.year

        # --- Seasonal multipliers ---
        # Monthly seasonality (higher in Nov-Dec for holidays, lower in Jan-Feb)
        monthly_seasonality = 1 + 0.3 * np.sin(2 * np.pi * (month - 4) / 12)

        # Holiday boost (Black Friday / Christmas period)
        holiday_boost = 1.0
        if month == 11 and date.day >= 20:
            holiday_boost = 1.8
        elif month == 12 and date.day <= 25:
            holiday_boost = 2.0
        elif month == 12 and date.day > 25:
            holiday_boost = 0.7  # Post-Christmas dip

        # Back-to-school (August-September)
        school_boost = 1.3 if month in [8, 9] else 1.0

        # Weekly pattern: weekends have higher sales
        weekend_mult = 1.25 if day_of_week >= 5 else 1.0

        # Long-term growth trend (5% yearly growth)
        years_elapsed = (year - 2023) + (month - 1) / 12
        trend = 1 + 0.05 * years_elapsed

        for category, info in CATEGORIES.items():
            for i, product in enumerate(info["products"]):
                # Category-specific seasonality
                cat_season = 1.0
                if category == "Clothing":
                    # Spring and fall peaks for clothing
                    cat_season = 1 + 0.2 * np.sin(2 * np.pi * (month - 3) / 6)
                elif category == "Electronics":
                    cat_season = holiday_boost  # Electronics peak during holidays
                elif category == "Sports":
                    # Summer peak for sports
                    cat_season = 1 + 0.3 * np.sin(2 * np.pi * (month - 1) / 12)
                elif category == "Books":
                    cat_season = school_boost

                # Promotion: ~15% of days have a promotion
                is_promotion = np.random.random() < 0.15
                promo_mult = 1.4 if is_promotion else 1.0
                discount = np.random.choice([0.1, 0.15, 0.2, 0.25]) if is_promotion else 0

                for region in REGIONS:
                    for store in STORES:
                        # Regional variation
                        region_mult = {
                            "North": 1.0, "South": 1.1,
                            "East": 0.95, "West": 1.05
                        }[region]

                        # Store variation
                        store_mult = {
                            "Store_A": 1.1, "Store_B": 1.0, "Store_C": 0.9
                        }[store]

                        # Not every product sells every day in every store
                        # Probability of a sale occurring
                        sale_prob = 0.35  # ~35% chance
                        if np.random.random() > sale_prob:
                            continue

                        # Calculate quantity with all effects
                        base_q = info["base_qty"][i]
                        quantity = max(1, int(
                            base_q
                            * monthly_seasonality
                            * weekend_mult
                            * trend
                            * cat_season
                            * promo_mult
                            * region_mult
                            * store_mult
                            * np.random.lognormal(0, 0.3)  # Random noise
                        ))

                        # Calculate unit price with discount
                        unit_price = round(
                            info["base_price"][i] * (1 - discount)
                            * np.random.uniform(0.95, 1.05),  # Small price variation
                            2,
                        )

                        # Sales = Quantity × Unit Price
                        sales = round(quantity * unit_price, 2)

                        records.append({
                            "Date": date,
                            "Product": product,
                            "Category": category,
                            "Region": region,
                            "Store": store,
                            "Quantity": quantity,
                            "Unit_Price": unit_price,
                            "Discount": discount,
                            "Promotion": int(is_promotion),
                            "Sales": sales,
                        })

    df = pd.DataFrame(records)

    # Add some realistic imperfections
    # ~0.5% missing values in Quantity
    n_missing = int(len(df) * 0.005)
    missing_idx = np.random.choice(df.index, n_missing, replace=False)
    df.loc[missing_idx[:n_missing // 2], "Quantity"] = np.nan
    df.loc[missing_idx[n_missing // 2:], "Unit_Price"] = np.nan

    # ~0.2% duplicate rows
    n_dupes = int(len(df) * 0.002)
    dupe_idx = np.random.choice(df.index, n_dupes, replace=False)
    dupes = df.loc[dupe_idx].copy()
    df = pd.concat([df, dupes], ignore_index=True)

    # Shuffle slightly to simulate realistic data entry
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


if __name__ == "__main__":
    print("Generating synthetic sales dataset...")
    df = generate_dataset()

    # Save to data/raw/
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sales_data.csv")
    df.to_csv(output_path, index=False)

    print(f"\nDataset generated successfully!")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Categories: {df['Category'].nunique()}")
    print(f"  Products: {df['Product'].nunique()}")
    print(f"  Regions: {df['Region'].nunique()}")
    print(f"  Stores: {df['Store'].nunique()}")
    print(f"  Total Sales: ${df['Sales'].sum():,.2f}")
    print(f"\n  Saved to: {output_path}")
