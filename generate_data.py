"""
Generates a realistic e-commerce sales dataset (Global-Superstore-style schema)
used by the Retail Insights dashboard.

NOTE ON DATA PROVENANCE:
This environment has no internet access to third-party data hosts, so we could
not download a live proprietary retail dataset. Instead we generate a large,
internally-consistent synthetic dataset that mirrors the real-world structure,
seasonality, and business economics of the well-known "Global Superstore" /
e-commerce retail dataset family (order/ship dates, customer segments, product
categories, regions, discounts, profit margins). All downstream analytics,
KPIs, charts, ML models, and forecasts operate on this CSV exactly as they
would on a real retailer's export -- nothing in app.py is hard-coded or faked.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

N_ORDERS = 20000  # order-line rows (post-thinning yields ~9-10k realistic records)

START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2024, 12, 31)
TOTAL_DAYS = (END_DATE - START_DATE).days

# ---------------------------------------------------------------------------
# Reference dimensions
# ---------------------------------------------------------------------------
REGIONS_STATES = {
    "West": ["California", "Washington", "Oregon", "Nevada", "Arizona", "Colorado"],
    "East": ["New York", "New Jersey", "Pennsylvania", "Massachusetts", "Virginia"],
    "Central": ["Texas", "Illinois", "Ohio", "Michigan", "Minnesota", "Missouri"],
    "South": ["Florida", "Georgia", "North Carolina", "Tennessee", "Alabama"],
}

STATE_ABBR = {
    "California": "CA", "Washington": "WA", "Oregon": "OR", "Nevada": "NV", "Arizona": "AZ",
    "Colorado": "CO", "New York": "NY", "New Jersey": "NJ", "Pennsylvania": "PA",
    "Massachusetts": "MA", "Virginia": "VA", "Texas": "TX", "Illinois": "IL", "Ohio": "OH",
    "Michigan": "MI", "Minnesota": "MN", "Missouri": "MO", "Florida": "FL", "Georgia": "GA",
    "North Carolina": "NC", "Tennessee": "TN", "Alabama": "AL",
}

STATE_CITY = {
    "California": ["Los Angeles", "San Francisco", "San Diego", "Sacramento"],
    "Washington": ["Seattle", "Spokane"],
    "Oregon": ["Portland", "Eugene"],
    "Nevada": ["Las Vegas", "Reno"],
    "Arizona": ["Phoenix", "Tucson"],
    "Colorado": ["Denver", "Boulder"],
    "New York": ["New York City", "Buffalo", "Albany"],
    "New Jersey": ["Newark", "Jersey City"],
    "Pennsylvania": ["Philadelphia", "Pittsburgh"],
    "Massachusetts": ["Boston", "Worcester"],
    "Virginia": ["Richmond", "Norfolk"],
    "Texas": ["Houston", "Dallas", "Austin", "San Antonio"],
    "Illinois": ["Chicago", "Springfield"],
    "Ohio": ["Columbus", "Cleveland"],
    "Michigan": ["Detroit", "Ann Arbor"],
    "Minnesota": ["Minneapolis", "St. Paul"],
    "Missouri": ["Kansas City", "St. Louis"],
    "Florida": ["Miami", "Orlando", "Tampa"],
    "Georgia": ["Atlanta", "Savannah"],
    "North Carolina": ["Charlotte", "Raleigh"],
    "Tennessee": ["Nashville", "Memphis"],
    "Alabama": ["Birmingham", "Montgomery"],
}

CATEGORY_TREE = {
    "Technology": {
        "Phones": (120, 900, 0.18),
        "Laptops": (450, 2200, 0.14),
        "Accessories": (10, 120, 0.32),
        "Cameras": (150, 1400, 0.16),
    },
    "Furniture": {
        "Chairs": (60, 650, 0.11),
        "Tables": (90, 900, 0.09),
        "Bookcases": (70, 700, 0.10),
        "Furnishings": (15, 200, 0.15),
    },
    "Office Supplies": {
        "Storage": (10, 150, 0.20),
        "Binders": (2, 40, 0.28),
        "Paper": (3, 30, 0.30),
        "Art": (5, 60, 0.25),
        "Labels": (1, 15, 0.27),
    },
    "Clothing": {
        "Men's Apparel": (15, 150, 0.22),
        "Women's Apparel": (15, 180, 0.24),
        "Footwear": (25, 220, 0.20),
        "Kids' Wear": (10, 90, 0.23),
    },
    "Home & Kitchen": {
        "Cookware": (20, 300, 0.18),
        "Appliances": (40, 800, 0.13),
        "Decor": (10, 150, 0.21),
    },
}

SEGMENTS = ["Consumer", "Corporate", "Home Office"]
SHIP_MODES = ["Standard Class", "Second Class", "First Class", "Same Day"]
SHIP_MODE_DAYS = {"Standard Class": (4, 7), "Second Class": (2, 4), "First Class": (1, 2), "Same Day": (0, 1)}

FIRST_NAMES = ["James", "Mary", "Robert", "Priya", "Wei", "Ahmed", "Sofia", "Liam", "Emma",
               "Noah", "Olivia", "Arjun", "Fatima", "Lucas", "Mia", "Ethan", "Ananya", "Carlos",
               "Grace", "Daniel", "Chen", "Isabella", "Omar", "Sara", "Ravi", "Elena", "Jack",
               "Ava", "Ken", "Nina"]
LAST_NAMES = ["Smith", "Johnson", "Patel", "Garcia", "Lee", "Khan", "Rossi", "Brown", "Davis",
              "Wilson", "Kumar", "Martinez", "Anderson", "Taylor", "Chang", "Nguyen", "Clark",
              "Lewis", "Walker", "Young"]

# ---------------------------------------------------------------------------
# Build a customer base (with repeat purchase behaviour)
# ---------------------------------------------------------------------------
N_CUSTOMERS = 1400
customer_ids = [f"CUST-{10000 + i}" for i in range(N_CUSTOMERS)]
customer_names = [f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for _ in range(N_CUSTOMERS)]
customer_segment = rng.choice(SEGMENTS, size=N_CUSTOMERS, p=[0.55, 0.30, 0.15])
# Pareto-like purchase propensity so some customers are frequent repeat buyers
customer_weight = rng.pareto(a=2.2, size=N_CUSTOMERS) + 0.1
customer_weight = customer_weight / customer_weight.sum()

rows = []
order_counter = 1

categories = list(CATEGORY_TREE.keys())
# category base popularity + a slow growth/decline trend per category (business narrative)
category_trend = {"Technology": 0.35, "Furniture": -0.05, "Office Supplies": 0.05,
                   "Clothing": 0.20, "Home & Kitchen": 0.10}
category_weight = {"Technology": 0.27, "Furniture": 0.16, "Office Supplies": 0.24,
                    "Clothing": 0.20, "Home & Kitchen": 0.13}

for i in range(N_ORDERS):
    cust_idx = rng.choice(N_CUSTOMERS, p=customer_weight)
    customer_id = customer_ids[cust_idx]
    customer_name = customer_names[cust_idx]
    segment = customer_segment[cust_idx]

    # Order date with mild upward trend + strong seasonality (Nov/Dec peak, summer dip)
    day_offset = rng.integers(0, TOTAL_DAYS)
    order_date = START_DATE + timedelta(days=int(day_offset))
    month = order_date.month
    seasonal_factor = {11: 1.6, 12: 1.9, 1: 0.8, 7: 0.85, 6: 0.9}.get(month, 1.0)
    if rng.random() > (0.45 * seasonal_factor):
        continue  # thin out / concentrate density to create realistic seasonality

    region = rng.choice(list(REGIONS_STATES.keys()), p=[0.30, 0.27, 0.24, 0.19])
    state = rng.choice(REGIONS_STATES[region])
    city = rng.choice(STATE_CITY[state])

    # category chosen with a time-based trend nudge (growth story for the AI assistant)
    yr_progress = (order_date - START_DATE).days / TOTAL_DAYS
    weights = np.array([category_weight[c] * (1 + category_trend[c] * yr_progress) for c in categories])
    weights = weights / weights.sum()
    category = rng.choice(categories, p=weights)
    subcats = list(CATEGORY_TREE[category].keys())
    sub_category = rng.choice(subcats)
    low, high, base_margin = CATEGORY_TREE[category][sub_category]

    product_no = rng.integers(100, 999)
    product_name = f"{sub_category[:-1] if sub_category.endswith('s') else sub_category} Model {product_no}"

    quantity = int(rng.choice([1, 2, 3, 4, 5], p=[0.45, 0.25, 0.15, 0.10, 0.05]))
    unit_price = float(rng.uniform(low, high))
    discount = float(rng.choice([0.0, 0.1, 0.15, 0.2, 0.3, 0.4], p=[0.35, 0.2, 0.15, 0.15, 0.1, 0.05]))
    sales = round(unit_price * quantity * (1 - discount), 2)

    margin_noise = rng.normal(0, 0.03)
    effective_margin = base_margin - discount * 0.6 + margin_noise
    profit = round(sales * effective_margin, 2)

    ship_mode = rng.choice(SHIP_MODES, p=[0.55, 0.22, 0.15, 0.08])
    ship_days = rng.integers(SHIP_MODE_DAYS[ship_mode][0], SHIP_MODE_DAYS[ship_mode][1] + 1)
    ship_date = order_date + timedelta(days=int(ship_days))

    order_id = f"ORD-{order_date.year}-{order_counter:06d}"
    order_counter += 1

    rows.append({
        "Order ID": order_id,
        "Order Date": order_date.strftime("%Y-%m-%d"),
        "Ship Date": ship_date.strftime("%Y-%m-%d"),
        "Ship Mode": ship_mode,
        "Customer ID": customer_id,
        "Customer Name": customer_name,
        "Segment": segment,
        "Country": "United States",
        "City": city,
        "State": state,
        "State Code": STATE_ABBR[state],
        "Region": region,
        "Product ID": f"{sub_category[:3].upper()}-{product_no}",
        "Category": category,
        "Sub-Category": sub_category,
        "Product Name": product_name,
        "Sales": sales,
        "Quantity": quantity,
        "Discount": discount,
        "Profit": profit,
    })

df = pd.DataFrame(rows)
df = df.sort_values("Order Date").reset_index(drop=True)

out_path = "data/ecommerce_sales_data.csv"
df.to_csv(out_path, index=False)
print(f"Generated {len(df):,} order-line records -> {out_path}")
print(df.head(3).to_string())
print("\nDate range:", df["Order Date"].min(), "to", df["Order Date"].max())
print("Total Sales: $%.2f" % df["Sales"].sum())
print("Total Profit: $%.2f" % df["Profit"].sum())
