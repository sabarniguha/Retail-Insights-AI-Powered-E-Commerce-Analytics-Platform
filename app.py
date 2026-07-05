"""
Retail Insights – AI-Powered E-Commerce Sales Analysis
========================================================
A production-grade Business Intelligence dashboard built with Streamlit.

Author: Sabarni
License: MIT

Run locally:
    streamlit run app.py

Deploy: push to GitHub and deploy directly on Streamlit Community Cloud.
"""

from __future__ import annotations

import os
import io
import json
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dotenv import load_dotenv
from groq import Groq
from mistralai import Mistral

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# Load environment variables FIRST
load_dotenv()

# Read from .env or Streamlit Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") or st.secrets.get("MISTRAL_API_KEY", "")

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


# =============================================================================
# APP CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Retail Insights | AI-Powered E-Commerce Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "data/ecommerce_sales_data.csv"

CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}

DEFAULT_SETTINGS = {
    "theme": "Dark",
    "ai_provider": "Auto",
    "ai_model": "auto",
    "forecast_horizon": 3,
    "currency": "USD",
    "date_format": "%Y-%m-%d",
}

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_SEQUENCE = ["#6366F1", "#22D3EE", "#F472B6", "#FBBF24", "#34D399", "#A78BFA", "#FB923C"]
ACCENT = "#6366F1"
ACCENT_2 = "#22D3EE"
BG_CARD = "rgba(255,255,255,0.04)"


# =============================================================================
# GLOBAL STYLES
# =============================================================================
def inject_css() -> None:
    """Injects the premium SaaS-style dark theme CSS used across all pages."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .main {
            background: radial-gradient(circle at 10% 0%, #161a2b 0%, #0e1117 45%, #0b0d13 100%);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #12142080 0%, #0b0d1380 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        /* Gradient header */
        .gradient-header {
            padding: 1.6rem 2rem;
            border-radius: 18px;
            background: linear-gradient(120deg, #6366F1 0%, #8B5CF6 45%, #22D3EE 100%);
            box-shadow: 0 8px 30px rgba(99,102,241,0.35);
            margin-bottom: 1.4rem;
        }
        .gradient-header h1 {
            color: white; font-weight: 800; font-size: 1.9rem; margin: 0;
            letter-spacing: -0.02em;
        }
        .gradient-header p {
            color: rgba(255,255,255,0.88); margin: 0.35rem 0 0 0; font-size: 0.98rem;
        }

        /* KPI cards */
        .kpi-card {
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1.1rem 1.3rem;
            transition: all 0.25s ease;
            height: 100%;
        }
        .kpi-card:hover {
            transform: translateY(-3px);
            border-color: rgba(99,102,241,0.55);
            box-shadow: 0 10px 28px rgba(99,102,241,0.18);
        }
        .kpi-label {
            font-size: 0.78rem; color: #9CA3AF; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 1.65rem; font-weight: 800; color: #F3F4F6; line-height: 1.1;
        }
        .kpi-delta-up { color: #34D399; font-size: 0.85rem; font-weight: 600; }
        .kpi-delta-down { color: #F87171; font-size: 0.85rem; font-weight: 600; }
        .kpi-icon {
            font-size: 1.4rem; opacity: 0.85; float: right;
        }

        /* Section card */
        .section-card {
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
        }
        .section-title {
            font-size: 1.05rem; font-weight: 700; color: #E5E7EB; margin-bottom: 0.6rem;
        }

        .insight-pill {
            display: inline-block;
            background: rgba(99,102,241,0.15);
            border: 1px solid rgba(99,102,241,0.35);
            color: #C7D2FE;
            border-radius: 999px;
            padding: 0.45rem 0.9rem;
            font-size: 0.85rem;
            margin: 0.2rem 0.3rem 0.2rem 0;
        }

        .badge-good { color: #34D399; font-weight: 700; }
        .badge-bad { color: #F87171; font-weight: 700; }
        .badge-neutral { color: #FBBF24; font-weight: 700; }

        div[data-testid="stMetricValue"] { font-weight: 800; }

        .stButton>button {
            border-radius: 10px; font-weight: 600; border: 1px solid rgba(99,102,241,0.4);
        }

        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.04); border-radius: 10px; padding: 8px 16px;
        }

        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_gradient_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="gradient-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float, currency: str = "USD") -> str:
    """Formats a numeric value as currency using the selected currency symbol."""
    symbol = CURRENCY_SYMBOLS.get(currency, "$")
    if abs(value) >= 1_000_000:
        return f"{symbol}{value/1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{symbol}{value/1_000:,.1f}K"
    return f"{symbol}{value:,.2f}"


def kpi_card(label: str, value: str, icon: str, delta: Optional[str] = None, delta_positive: bool = True) -> str:
    delta_html = ""
    if delta is not None:
        cls = "kpi-delta-up" if delta_positive else "kpi-delta-down"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    return f"""
    <div class="kpi-card">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


# =============================================================================
# DATA LAYER
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """
    Loads and validates the e-commerce transactions CSV.

    Raises a friendly error via st.error and returns an empty DataFrame if the
    file is missing, unreadable, or malformed.
    """
    required_cols = {
        "Order ID", "Order Date", "Ship Date", "Customer ID", "Customer Name",
        "Segment", "City", "State", "Region", "Category", "Sub-Category",
        "Product Name", "Sales", "Quantity", "Discount", "Profit",
    }
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    missing = required_cols - set(df.columns)
    if missing:
        return pd.DataFrame()

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")
    df = df.dropna(subset=["Order Date"])

    numeric_cols = ["Sales", "Quantity", "Discount", "Profit"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["MonthName"] = df["Order Date"].dt.strftime("%b")
    df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)
    df["Weekday"] = df["Order Date"].dt.day_name()
    df["Quarter"] = df["Order Date"].dt.quarter
    df["Week"] = df["Order Date"].dt.isocalendar().week
    df["Profit Margin"] = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"], 0)
    df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

    return df


def apply_filters(
    df: pd.DataFrame,
    date_range,
    categories: list,
    sub_categories: list,
    regions: list,
    states: list,
) -> pd.DataFrame:
    """Applies the global sidebar filters to the raw dataframe."""
    if df.empty:
        return df

    filtered = df.copy()

    if date_range and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = filtered[(filtered["Order Date"] >= start) & (filtered["Order Date"] <= end)]

    if categories:
        filtered = filtered[filtered["Category"].isin(categories)]
    if sub_categories:
        filtered = filtered[filtered["Sub-Category"].isin(sub_categories)]
    if regions:
        filtered = filtered[filtered["Region"].isin(regions)]
    if states:
        filtered = filtered[filtered["State"].isin(states)]

    return filtered


@st.cache_data(show_spinner=False)
def compute_kpis(df: pd.DataFrame) -> dict:
    """Computes headline KPI figures for the dashboard."""
    if df.empty:
        return {k: 0 for k in [
            "total_revenue", "total_profit", "total_orders", "total_customers",
            "avg_order_value", "profit_margin", "total_quantity",
        ]}

    total_revenue = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    total_orders = df["Order ID"].nunique()
    total_customers = df["Customer ID"].nunique()
    avg_order_value = total_revenue / total_orders if total_orders else 0
    profit_margin = (total_profit / total_revenue * 100) if total_revenue else 0
    total_quantity = df["Quantity"].sum()

    return {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "total_orders": total_orders,
        "total_customers": total_customers,
        "avg_order_value": avg_order_value,
        "profit_margin": profit_margin,
        "total_quantity": total_quantity,
    }


def period_over_period_delta(df: pd.DataFrame, metric: str) -> Optional[float]:
    """Computes % change of `metric` between the most recent and prior month present in df."""
    if df.empty or "YearMonth" not in df.columns:
        return None
    monthly = df.groupby("YearMonth")[metric].sum().sort_index()
    if len(monthly) < 2:
        return None
    latest, prior = monthly.iloc[-1], monthly.iloc[-2]
    if prior == 0:
        return None
    return (latest - prior) / abs(prior) * 100


# =============================================================================
# SIDEBAR: NAVIGATION + GLOBAL FILTERS
# =============================================================================
NAV_ITEMS = [
    ("Dashboard", "🏠"),
    ("Sales Analytics", "📈"),
    ("Product Analytics", "🛍"),
    ("Customer Analytics", "👥"),
    ("Regional Analytics", "🌍"),
    ("Inventory Insights", "📦"),
    ("Sales Forecasting", "📉"),
    ("AI Business Assistant", "🤖"),
    ("Reports", "📊"),
    ("Settings", "⚙"),
]


def render_sidebar(df: pd.DataFrame):
    """Renders the sidebar navigation and global data filters. Returns (page, filtered_df)."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 0.6rem 0 1rem 0;">
                <div style="font-size:1.6rem;">📊</div>
                <div style="font-weight:800; font-size:1.15rem; color:#F3F4F6;">Retail Insights</div>
                <div style="font-size:0.75rem; color:#9CA3AF;">AI-Powered E-Commerce Analytics</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        labels = [f"{icon}  {name}" for name, icon in NAV_ITEMS]
        choice = st.radio("Navigation", labels, label_visibility="collapsed")
        page = choice.split("  ", 1)[1]

        st.markdown("---")
        st.markdown("**🔎 Global Filters**")

        if df.empty:
            st.info("No data loaded yet.")
            return page, df

        min_date, max_date = df["Order Date"].min().date(), df["Order Date"].max().date()
        date_range = st.date_input(
            "Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date,
        )

        categories = st.multiselect("Category", sorted(df["Category"].unique()))
        available_subs = df[df["Category"].isin(categories)] if categories else df
        sub_categories = st.multiselect("Sub-Category", sorted(available_subs["Sub-Category"].unique()))

        regions = st.multiselect("Region", sorted(df["Region"].unique()))
        available_states = df[df["Region"].isin(regions)] if regions else df
        states = st.multiselect("State", sorted(available_states["State"].unique()))

        if st.button("🔄 Reset Filters", use_container_width=True):
            st.rerun()

        filtered = apply_filters(df, date_range, categories, sub_categories, regions, states)

        st.markdown("---")
        st.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** records")
        st.caption("Built with Streamlit • Plotly • Scikit-learn")

    return page, filtered


# =============================================================================
# BUSINESS INSIGHTS ENGINE (rule-based, always available offline)
# =============================================================================
def generate_business_insights(df: pd.DataFrame) -> list:
    """Automatically derives plain-English business insights from the filtered data."""
    insights = []
    if df.empty or len(df) < 5:
        return ["Not enough data in the current filter selection to generate insights."]

    monthly = df.groupby("YearMonth")["Sales"].sum().sort_index()
    if len(monthly) >= 2:
        change = (monthly.iloc[-1] - monthly.iloc[-2]) / abs(monthly.iloc[-2]) * 100 if monthly.iloc[-2] else 0
        direction = "increased" if change >= 0 else "decreased"
        insights.append(f"Revenue {direction} by {abs(change):.1f}% vs. the previous month.")

    cat_perf = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    if not cat_perf.empty:
        insights.append(f"**{cat_perf.index[0]}** is the best performing category, generating {format_currency(cat_perf.iloc[0])} in sales.")

    region_perf = df.groupby("Region")["Profit"].sum().sort_values()
    if not region_perf.empty:
        insights.append(f"**{region_perf.index[0]}** is currently the lowest performing region by profit ({format_currency(region_perf.iloc[0])}).")

    prod_profit = df.groupby("Product Name")["Profit"].sum().sort_values(ascending=False)
    if not prod_profit.empty:
        insights.append(f"**{prod_profit.index[0]}** is the highest profit-generating product ({format_currency(prod_profit.iloc[0])}).")

    if "MonthName" in df.columns:
        seasonal = df.groupby("Month")["Sales"].sum()
        if not seasonal.empty:
            peak_month = pd.Timestamp(2020, int(seasonal.idxmax()), 1).strftime("%B")
            insights.append(f"Sales peak seasonally in **{peak_month}**, suggesting inventory should be scaled ahead of this period.")

    # category growth trend (first half vs second half of the date range)
    if len(monthly) >= 4:
        half = len(monthly) // 2
        first_half, second_half = monthly.iloc[:half].mean(), monthly.iloc[half:].mean()
        growing_cats = []
        for cat in df["Category"].unique():
            cat_monthly = df[df["Category"] == cat].groupby("YearMonth")["Sales"].sum().sort_index()
            if len(cat_monthly) >= 4:
                h = len(cat_monthly) // 2
                g = (cat_monthly.iloc[h:].mean() - cat_monthly.iloc[:h].mean())
                growing_cats.append((cat, g))
        if growing_cats:
            best_growth = max(growing_cats, key=lambda x: x[1])
            if best_growth[1] > 0:
                insights.append(f"**{best_growth[0]}** shows the strongest upward growth trend across the selected period.")

    avg_discount = df["Discount"].mean()
    if avg_discount > 0.2:
        insights.append(f"Average discount rate is high at {avg_discount*100:.1f}%, which may be compressing margins.")

    return insights


def render_insights_pills(insights: list) -> None:
    html = "".join(f'<span class="insight-pill">💡 {i}</span>' for i in insights)
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# PAGE 1: DASHBOARD
# =============================================================================
def page_dashboard(df: pd.DataFrame, settings: dict) -> None:
    render_gradient_header("🏠 Executive Dashboard", "Real-time overview of sales, profit, and customer performance")

    if df.empty:
        st.warning("No data available for the selected filters. Try widening your date range or clearing filters.")
        return

    currency = settings["currency"]
    kpis = compute_kpis(df)
    rev_delta = period_over_period_delta(df, "Sales")
    profit_delta = period_over_period_delta(df, "Profit")

    cols = st.columns(4)
    kpi_defs = [
        ("Total Revenue", format_currency(kpis["total_revenue"], currency), "💰", rev_delta),
        ("Total Profit", format_currency(kpis["total_profit"], currency), "📈", profit_delta),
        ("Total Orders", f"{kpis['total_orders']:,}", "🧾", None),
        ("Total Customers", f"{kpis['total_customers']:,}", "👥", None),
    ]
    for c, (label, value, icon, delta) in zip(cols, kpi_defs):
        delta_str = f"{delta:.1f}% MoM" if delta is not None else None
        c.markdown(kpi_card(label, value, icon, delta_str, (delta or 0) >= 0), unsafe_allow_html=True)

    cols2 = st.columns(3)
    kpi_defs2 = [
        ("Avg Order Value", format_currency(kpis["avg_order_value"], currency), "🛒"),
        ("Profit Margin", f"{kpis['profit_margin']:.1f}%", "📊"),
        ("Total Quantity Sold", f"{int(kpis['total_quantity']):,}", "📦"),
    ]
    for c, (label, value, icon) in zip(cols2, kpi_defs2):
        c.markdown(kpi_card(label, value, icon), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Trend charts ---
    monthly = df.groupby("YearMonth").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-card"><div class="section-title">Monthly Sales Trend</div>', unsafe_allow_html=True)
        fig = px.area(monthly, x="YearMonth", y="Sales", template=PLOTLY_TEMPLATE,
                       color_discrete_sequence=[ACCENT])
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title="Sales")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card"><div class="section-title">Monthly Profit Trend</div>', unsafe_allow_html=True)
        fig = px.area(monthly, x="YearMonth", y="Profit", template=PLOTLY_TEMPLATE,
                       color_discrete_sequence=[ACCENT_2])
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title="Profit")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-card"><div class="section-title">Revenue vs Profit</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly["YearMonth"], y=monthly["Sales"], name="Revenue", marker_color=ACCENT))
        fig.add_trace(go.Bar(x=monthly["YearMonth"], y=monthly["Profit"], name="Profit", marker_color=ACCENT_2))
        fig.update_layout(template=PLOTLY_TEMPLATE, barmode="group", height=320,
                           margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="section-card"><div class="section-title">Top Categories</div>', unsafe_allow_html=True)
        cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(cat_sales, x="Sales", y="Category", orientation="h", template=PLOTLY_TEMPLATE,
                     color="Category", color_discrete_sequence=COLOR_SEQUENCE)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<div class="section-card"><div class="section-title">Top 10 Products</div>', unsafe_allow_html=True)
        top_products = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top_products, x="Sales", y="Product Name", orientation="h", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), yaxis={"categoryorder": "total ascending"}, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c6:
        st.markdown('<div class="section-card"><div class="section-title">Sales by Region</div>', unsafe_allow_html=True)
        region_sales = df.groupby("Region")["Sales"].sum().reset_index()
        fig = px.pie(region_sales, names="Region", values="Sales", hole=0.55, template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=COLOR_SEQUENCE)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">Customer Distribution by Segment</div>', unsafe_allow_html=True)
    seg = df.groupby("Segment")["Customer ID"].nunique().reset_index(name="Customers")
    fig = px.bar(seg, x="Segment", y="Customers", template=PLOTLY_TEMPLATE, color="Segment",
                 color_discrete_sequence=COLOR_SEQUENCE, text="Customers")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">🧠 Automated Business Insights</div>', unsafe_allow_html=True)
    render_insights_pills(generate_business_insights(df))
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 2: SALES ANALYTICS
# =============================================================================
def page_sales_analytics(df: pd.DataFrame, settings: dict) -> None:
    render_gradient_header("📈 Sales Analytics", "Deep-dive into revenue, profit, discount, and shipping performance")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    granularity = st.radio("Granularity", ["Daily", "Weekly", "Monthly", "Yearly"], horizontal=True)

    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Yearly": "YE"}
    ts = df.set_index("Order Date").resample(freq_map[granularity]).agg(
        Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "nunique")
    ).reset_index()

    tab1, tab2, tab3, tab4 = st.tabs(["📉 Revenue Trend", "💹 Profit Trend", "🏷 Discount Analysis", "🚚 Shipping Analysis"])

    with tab1:
        fig = px.line(ts, x="Order Date", y="Sales", template=PLOTLY_TEMPLATE, markers=True,
                       color_discrete_sequence=[ACCENT])
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-card"><div class="section-title">Sales Heatmap (Day of Week vs Month)</div>', unsafe_allow_html=True)
            heat = df.copy()
            heat_pivot = heat.pivot_table(index="Weekday", columns="MonthName", values="Sales", aggfunc="sum", fill_value=0)
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            heat_pivot = heat_pivot.reindex(index=[d for d in weekday_order if d in heat_pivot.index],
                                             columns=[m for m in month_order if m in heat_pivot.columns])
            fig = px.imshow(heat_pivot, template=PLOTLY_TEMPLATE, color_continuous_scale="Purples", aspect="auto")
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="section-card"><div class="section-title">Orders Over Time</div>', unsafe_allow_html=True)
            fig = px.bar(ts, x="Order Date", y="Orders", template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT_2])
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        fig = px.area(ts, x="Order Date", y="Profit", template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT_2])
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

        margin_ts = df.set_index("Order Date").resample(freq_map[granularity]).apply(
            lambda x: (x["Profit"].sum() / x["Sales"].sum() * 100) if x["Sales"].sum() else 0
        ).reset_index(name="Profit Margin %")
        st.markdown('<div class="section-card"><div class="section-title">Profit Margin % Over Time</div>', unsafe_allow_html=True)
        fig = px.line(margin_ts, x="Order Date", y="Profit Margin %", template=PLOTLY_TEMPLATE, markers=True,
                       color_discrete_sequence=["#34D399"])
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-card"><div class="section-title">Discount Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(df, x="Discount", nbins=20, template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT])
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="section-card"><div class="section-title">Discount vs Profit Margin</div>', unsafe_allow_html=True)
            sample = df.sample(min(2000, len(df)), random_state=42)
            fig = px.scatter(sample, x="Discount", y="Profit Margin", color="Category", template=PLOTLY_TEMPLATE,
                              color_discrete_sequence=COLOR_SEQUENCE, opacity=0.65)
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        avg_disc_by_cat = df.groupby("Category")["Discount"].mean().sort_values(ascending=False).reset_index()
        st.markdown('<div class="section-card"><div class="section-title">Average Discount by Category</div>', unsafe_allow_html=True)
        fig = px.bar(avg_disc_by_cat, x="Category", y="Discount", template=PLOTLY_TEMPLATE, color="Category",
                     color_discrete_sequence=COLOR_SEQUENCE, text_auto=".1%")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-card"><div class="section-title">Shipping Mode Breakdown</div>', unsafe_allow_html=True)
            ship_counts = df["Ship Mode"].value_counts().reset_index()
            ship_counts.columns = ["Ship Mode", "Orders"]
            fig = px.pie(ship_counts, names="Ship Mode", values="Orders", template=PLOTLY_TEMPLATE, hole=0.5,
                         color_discrete_sequence=COLOR_SEQUENCE)
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="section-card"><div class="section-title">Avg Shipping Days by Mode</div>', unsafe_allow_html=True)
            ship_days = df.groupby("Ship Mode")["Shipping Days"].mean().sort_values().reset_index()
            fig = px.bar(ship_days, x="Ship Mode", y="Shipping Days", template=PLOTLY_TEMPLATE, color="Ship Mode",
                         color_discrete_sequence=COLOR_SEQUENCE, text_auto=".1f")
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 3: PRODUCT ANALYTICS
# =============================================================================
def page_product_analytics(df: pd.DataFrame, settings: dict) -> None:
    render_gradient_header("🛍 Product Analytics", "Identify winning products, categories, and profit drivers")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    product_summary = df.groupby("Product Name").agg(
        Revenue=("Sales", "sum"), Profit=("Profit", "sum"), Quantity=("Quantity", "sum"),
        Category=("Category", "first"),
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-card"><div class="section-title">🏆 Best Selling Products (by Revenue)</div>', unsafe_allow_html=True)
        best = product_summary.sort_values("Revenue", ascending=False).head(10)
        fig = px.bar(best, x="Revenue", y="Product Name", orientation="h", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), yaxis={"categoryorder": "total ascending"}, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card"><div class="section-title">🐌 Worst Selling Products (by Revenue)</div>', unsafe_allow_html=True)
        worst = product_summary.sort_values("Revenue", ascending=True).head(10)
        fig = px.bar(worst, x="Revenue", y="Product Name", orientation="h", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=["#F87171"])
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), yaxis={"categoryorder": "total descending"}, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">📦 Category Performance (Treemap)</div>', unsafe_allow_html=True)
    tree_data = df.groupby(["Category", "Sub-Category"]).agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()
    fig = px.treemap(tree_data, path=["Category", "Sub-Category"], values="Revenue", color="Profit",
                      color_continuous_scale="RdYlGn", template=PLOTLY_TEMPLATE)
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-card"><div class="section-title">Product Revenue Share</div>', unsafe_allow_html=True)
        cat_rev = df.groupby("Category")["Sales"].sum().reset_index()
        fig = px.pie(cat_rev, names="Category", values="Sales", template=PLOTLY_TEMPLATE, color_discrete_sequence=COLOR_SEQUENCE)
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="section-card"><div class="section-title">Product Profit by Category</div>', unsafe_allow_html=True)
        cat_profit = df.groupby("Category")["Profit"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(cat_profit, x="Category", y="Profit", template=PLOTLY_TEMPLATE, color="Category",
                     color_discrete_sequence=COLOR_SEQUENCE)
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">📋 Product Quantity Sold — Detail Table</div>', unsafe_allow_html=True)
    display_df = product_summary.sort_values("Quantity", ascending=False).head(25).copy()
    display_df["Revenue"] = display_df["Revenue"].map(lambda x: format_currency(x, settings["currency"]))
    display_df["Profit"] = display_df["Profit"].map(lambda x: format_currency(x, settings["currency"]))
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 4: CUSTOMER ANALYTICS (with K-Means segmentation)
# =============================================================================
@st.cache_data(show_spinner=False)
def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds an RFM-style feature table per customer for segmentation."""
    if df.empty:
        return pd.DataFrame()

    snapshot_date = df["Order Date"].max() + timedelta(days=1)
    cust = df.groupby("Customer ID").agg(
        Customer_Name=("Customer Name", "first"),
        Recency=("Order Date", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Order ID", "nunique"),
        Monetary=("Sales", "sum"),
        Avg_Order_Value=("Sales", "mean"),
        Total_Profit=("Profit", "sum"),
    ).reset_index()
    return cust


@st.cache_data(show_spinner=False)
def run_customer_segmentation(cust_features: pd.DataFrame, n_clusters: int = 4):
    """Runs K-Means clustering on Recency/Frequency/Monetary features."""
    features = cust_features[["Recency", "Frequency", "Monetary"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    n_clusters = min(n_clusters, max(2, len(cust_features) // 5)) if len(cust_features) >= 10 else 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    result = cust_features.copy()
    result["Cluster"] = labels

    # Label clusters by business meaning using average monetary value rank
    cluster_rank = result.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False)
    tier_names = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Hibernating"]
    rename_map = {cluster: tier_names[i] if i < len(tier_names) else f"Segment {i}"
                  for i, cluster in enumerate(cluster_rank.index)}
    result["Segment Label"] = result["Cluster"].map(rename_map)

    return result


def page_customer_analytics(df: pd.DataFrame, settings: dict) -> None:
    render_gradient_header("👥 Customer Analytics", "Customer value, loyalty, and AI-powered segmentation")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    total_customers = df["Customer ID"].nunique()
    orders_per_customer = df.groupby("Customer ID")["Order ID"].nunique()
    repeat_customers = (orders_per_customer > 1).sum()
    clv = df.groupby("Customer ID")["Sales"].sum().mean()
    avg_spend = df["Sales"].sum() / total_customers if total_customers else 0

    cols = st.columns(4)
    metrics = [
        ("Total Customers", f"{total_customers:,}", "👥"),
        ("Repeat Customers", f"{repeat_customers:,} ({repeat_customers/total_customers*100:.1f}%)" if total_customers else "0", "🔁"),
        ("Avg Customer Lifetime Value", format_currency(clv, settings["currency"]), "💎"),
        ("Avg Spending / Customer", format_currency(avg_spend, settings["currency"]), "🛒"),
    ]
    for c, (label, value, icon) in zip(cols, metrics):
        c.markdown(kpi_card(label, value, icon), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">🧩 AI Customer Segmentation (K-Means Clustering)</div>', unsafe_allow_html=True)
    st.caption("Segments customers using Recency, Frequency, and Monetary (RFM) value via unsupervised K-Means clustering.")

    n_clusters = st.slider("Number of segments", min_value=2, max_value=6, value=4)
    cust_features = build_customer_features(df)

    if len(cust_features) < 10:
        st.info("Not enough distinct customers in the current filter to run reliable clustering (need at least 10).")
    else:
        segmented = run_customer_segmentation(cust_features, n_clusters)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.scatter(
                segmented, x="Recency", y="Monetary", size="Frequency", color="Segment Label",
                hover_data=["Customer_Name", "Frequency"], template=PLOTLY_TEMPLATE,
                color_discrete_sequence=COLOR_SEQUENCE, opacity=0.75,
                labels={"Recency": "Recency (days since last order)", "Monetary": "Total Spend"},
            )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            seg_counts = segmented["Segment Label"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Customers"]
            fig = px.pie(seg_counts, names="Segment", values="Customers", hole=0.5, template=PLOTLY_TEMPLATE,
                         color_discrete_sequence=COLOR_SEQUENCE)
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Segment Profiles")
        profile = segmented.groupby("Segment Label").agg(
            Customers=("Customer ID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
        ).round(1).reset_index().sort_values("Avg_Monetary", ascending=False)
        st.dataframe(profile, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-card"><div class="section-title">Customers by Segment (Business Type)</div>', unsafe_allow_html=True)
        seg = df.groupby("Segment")["Customer ID"].nunique().reset_index(name="Customers")
        fig = px.bar(seg, x="Segment", y="Customers", template=PLOTLY_TEMPLATE, color="Segment",
                     color_discrete_sequence=COLOR_SEQUENCE, text="Customers")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="section-card"><div class="section-title">Top 10 Customers by Revenue</div>', unsafe_allow_html=True)
        top_cust = df.groupby("Customer Name")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top_cust, x="Sales", y="Customer Name", orientation="h", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis={"categoryorder": "total ascending"}, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 5: REGIONAL ANALYTICS
# =============================================================================
def page_regional_analytics(df: pd.DataFrame, settings: dict) -> None:
    render_gradient_header("🌍 Regional Analytics", "Geographic performance across regions and states")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-card"><div class="section-title">Sales by Region</div>', unsafe_allow_html=True)
        region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(region_sales, x="Region", y="Sales", template=PLOTLY_TEMPLATE, color="Region",
                     color_discrete_sequence=COLOR_SEQUENCE, text_auto=".2s")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card"><div class="section-title">Profit by Region</div>', unsafe_allow_html=True)
        region_profit = df.groupby("Region")["Profit"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(region_profit, x="Region", y="Profit", template=PLOTLY_TEMPLATE, color="Region",
                     color_discrete_sequence=COLOR_SEQUENCE, text_auto=".2s")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">🗺 State-Wise Sales — Choropleth Map</div>', unsafe_allow_html=True)
    state_sales = df.groupby(["State", "State Code"]).agg(
        Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "nunique")
    ).reset_index()
    fig = px.choropleth(
        state_sales, locations="State Code", locationmode="USA-states", color="Sales",
        scope="usa", color_continuous_scale="Blues", hover_name="State",
        hover_data={"State Code": False, "Sales": ":.2f", "Profit": ":.2f", "Orders": True},
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10), geo=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">🫧 State Performance — Bubble Map</div>', unsafe_allow_html=True)
    fig = px.scatter_geo(
        state_sales, locations="State Code", locationmode="USA-states", size="Sales", color="Profit",
        scope="usa", hover_name="State", color_continuous_scale="Viridis", template=PLOTLY_TEMPLATE,
        size_max=40,
    )
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10), geo=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">📋 State-wise Performance Table</div>', unsafe_allow_html=True)
    table = state_sales.sort_values("Sales", ascending=False).copy()
    table["Sales"] = table["Sales"].map(lambda x: format_currency(x, settings["currency"]))
    table["Profit"] = table["Profit"].map(lambda x: format_currency(x, settings["currency"]))
    st.dataframe(table.drop(columns=["State Code"]), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 6: INVENTORY INSIGHTS
# =============================================================================
def page_inventory_insights(df: pd.DataFrame, settings: dict) -> None:
    render_gradient_header("📦 Inventory Insights", "Automated fast/slow-mover detection and restocking recommendations")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    # Recent velocity window = last 90 days present in the filtered data
    max_date = df["Order Date"].max()
    recent_window = df[df["Order Date"] >= max_date - timedelta(days=90)]
    prior_window = df[(df["Order Date"] < max_date - timedelta(days=90)) & (df["Order Date"] >= max_date - timedelta(days=180))]

    recent_velocity = recent_window.groupby("Product Name")["Quantity"].sum()
    prior_velocity = prior_window.groupby("Product Name")["Quantity"].sum()

    velocity = pd.DataFrame({"Recent_Qty": recent_velocity}).join(
        pd.DataFrame({"Prior_Qty": prior_velocity}), how="outer"
    ).fillna(0)
    velocity["Growth"] = velocity["Recent_Qty"] - velocity["Prior_Qty"]
    velocity = velocity.reset_index().rename(columns={"index": "Product Name"})

    total_qty = df.groupby("Product Name")["Quantity"].sum()
    total_rev = df.groupby("Product Name")["Sales"].sum()
    velocity = velocity.merge(total_qty.rename("Total_Qty"), on="Product Name", how="left")
    velocity = velocity.merge(total_rev.rename("Total_Revenue"), on="Product Name", how="left")

    fast_movers = velocity.sort_values("Growth", ascending=False).head(10)
    slow_movers = velocity.sort_values("Growth", ascending=True).head(10)
    high_demand = velocity.sort_values("Total_Qty", ascending=False).head(10)
    low_performers = velocity[velocity["Total_Qty"] > 0].sort_values("Total_Revenue", ascending=True).head(10)

    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Fast Moving", "🐢 Slow Moving", "⚠️ Low Performing", "🔥 High Demand"])

    with tab1:
        st.caption("Products with the strongest increase in unit velocity (last 90 days vs. prior 90 days).")
        fig = px.bar(fast_movers, x="Growth", y="Product Name", orientation="h", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=["#34D399"])
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), yaxis={"categoryorder": "total ascending"}, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.caption("Products with declining or stagnant unit velocity — candidates for reduced restocking.")
        fig = px.bar(slow_movers, x="Growth", y="Product Name", orientation="h", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=["#F87171"])
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), yaxis={"categoryorder": "total descending"}, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.caption("Lowest revenue-generating active products — review pricing, bundling, or discontinuation.")
        fig = px.bar(low_performers, x="Total_Revenue", y="Product Name", orientation="h", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=["#FBBF24"])
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), yaxis={"categoryorder": "total descending"}, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.caption("Highest total unit volume across the full selected period — prioritize stock availability.")
        fig = px.bar(high_demand, x="Total_Qty", y="Product Name", orientation="h", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), yaxis={"categoryorder": "total ascending"}, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-card"><div class="section-title">🧠 Inventory Recommendations</div>', unsafe_allow_html=True)
    recs = []
    if not fast_movers.empty and fast_movers.iloc[0]["Growth"] > 0:
        recs.append(f"Increase stock for **{fast_movers.iloc[0]['Product Name']}** — demand grew by {fast_movers.iloc[0]['Growth']:.0f} units recently.")
    if not slow_movers.empty and slow_movers.iloc[0]["Growth"] < 0:
        recs.append(f"Reduce reorder quantity for **{slow_movers.iloc[0]['Product Name']}** — demand fell by {abs(slow_movers.iloc[0]['Growth']):.0f} units.")
    if not high_demand.empty:
        recs.append(f"Ensure continuous availability of **{high_demand.iloc[0]['Product Name']}**, your highest-volume product.")
    if not low_performers.empty:
        recs.append(f"Reassess pricing or promotion for **{low_performers.iloc[0]['Product Name']}**, currently the weakest revenue contributor.")
    if not recs:
        recs.append("Not enough historical depth in the current filter to generate velocity-based recommendations.")
    render_insights_pills(recs)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 7: SALES FORECASTING
# =============================================================================
@st.cache_data(show_spinner=False)
def prepare_forecast_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates transactions into a monthly time series with engineered features."""
    monthly = df.set_index("Order Date").resample("ME").agg(
        Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "nunique")
    ).reset_index()
    monthly["t"] = np.arange(len(monthly))
    monthly["month_num"] = monthly["Order Date"].dt.month
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["month_num"] / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["month_num"] / 12)
    monthly["lag_1"] = monthly["Sales"].shift(1)
    monthly["lag_2"] = monthly["Sales"].shift(2)
    monthly["rolling_mean_3"] = monthly["Sales"].shift(1).rolling(3).mean()
    return monthly


def train_forecast_model(monthly: pd.DataFrame, target: str, model_name: str):
    """Trains a regression model to predict the target series and returns model + metrics + feature cols."""
    data = monthly.dropna().copy()
    feature_cols = ["t", "month_sin", "month_cos", "lag_1", "lag_2", "rolling_mean_3"]
    X = data[feature_cols]
    y = data[target] if target == "Sales" else data[target]

    split_idx = max(int(len(data) * 0.8), len(data) - 6)
    split_idx = min(split_idx, len(data) - 1) if len(data) > 3 else len(data)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if model_name == "XGBoost" and XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
    elif model_name == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)

    model.fit(X_train, y_train)

    metrics = {"mae": None, "rmse": None, "r2": None}
    preds_test = None
    if len(X_test) > 0:
        preds_test = model.predict(X_test)
        metrics["mae"] = mean_absolute_error(y_test, preds_test)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_test, preds_test))
        metrics["r2"] = r2_score(y_test, preds_test) if len(y_test) > 1 else None

    return model, feature_cols, metrics, (X_test.index, preds_test)


def forecast_future(model, monthly: pd.DataFrame, feature_cols: list, target: str, horizon: int) -> pd.DataFrame:
    """Iteratively forecasts `horizon` months ahead using the trained model."""
    history = monthly[["Order Date", target]].copy().rename(columns={target: "value"})
    last_t = monthly["t"].max()
    future_rows = []

    values = list(history["value"])
    last_date = monthly["Order Date"].max()

    for step in range(1, horizon + 1):
        future_date = (last_date + pd.DateOffset(months=step))
        t = last_t + step
        month_num = future_date.month
        month_sin = np.sin(2 * np.pi * month_num / 12)
        month_cos = np.cos(2 * np.pi * month_num / 12)
        lag_1 = values[-1]
        lag_2 = values[-2] if len(values) >= 2 else values[-1]
        rolling_mean_3 = np.mean(values[-3:]) if len(values) >= 3 else np.mean(values)

        X_future = pd.DataFrame([{
            "t": t, "month_sin": month_sin, "month_cos": month_cos,
            "lag_1": lag_1, "lag_2": lag_2, "rolling_mean_3": rolling_mean_3,
        }])[feature_cols]

        pred = float(model.predict(X_future)[0])
        pred = max(pred, 0)
        values.append(pred)
        future_rows.append({"Order Date": future_date, "value": pred})

    return pd.DataFrame(future_rows)


def page_sales_forecasting(df: pd.DataFrame, settings: dict) -> None:
    render_gradient_header("📉 Sales Forecasting", "Machine learning-powered revenue, profit, and order forecasts")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    monthly = prepare_forecast_dataset(df)
    if len(monthly.dropna()) < 6:
        st.info("At least 6 months of historical data (after feature engineering) are needed for reliable forecasting. Widen your date range.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        target_label = st.selectbox("Forecast Target", ["Next Month Sales (Revenue)", "Future Orders"])
        target = "Sales" if "Revenue" in target_label else "Orders"
    with c2:
        model_options = ["Random Forest", "Linear Regression"] + (["XGBoost"] if XGBOOST_AVAILABLE else [])
        model_name = st.selectbox("Model", model_options)
    with c3:
        horizon = st.slider("Forecast Horizon (months)", min_value=1, max_value=12, value=settings.get("forecast_horizon", 3))

    model, feature_cols, metrics, (test_idx, test_preds) = train_forecast_model(monthly, target, model_name)
    future = forecast_future(model, monthly, feature_cols, target, horizon)

    st.markdown('<div class="section-card"><div class="section-title">📈 Forecast Graph</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["Order Date"], y=monthly[target], mode="lines+markers",
                              name="Historical", line=dict(color=ACCENT)))
    fig.add_trace(go.Scatter(x=future["Order Date"], y=future["value"], mode="lines+markers",
                              name="Forecast", line=dict(color=ACCENT_2, dash="dash")))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=420, margin=dict(l=10, r=10, t=10, b=10),
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    c4, c5 = st.columns([2, 1])
    with c4:
        st.markdown('<div class="section-card"><div class="section-title">Actual vs Predicted (Hold-out Test Period)</div>', unsafe_allow_html=True)
        if test_preds is not None and len(test_preds) > 0:
            test_dates = monthly.dropna().loc[test_idx, "Order Date"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_dates, y=monthly.dropna().loc[test_idx, target], name="Actual",
                                      mode="lines+markers", line=dict(color=ACCENT)))
            fig.add_trace(go.Scatter(x=test_dates, y=test_preds, name="Predicted",
                                      mode="lines+markers", line=dict(color="#FBBF24", dash="dot")))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to reserve a hold-out test period; showing in-sample fit only.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c5:
        st.markdown('<div class="section-card"><div class="section-title">Model Performance</div>', unsafe_allow_html=True)
        if metrics["mae"] is not None:
            st.metric("MAE", f"{metrics['mae']:,.2f}")
            st.metric("RMSE", f"{metrics['rmse']:,.2f}")
            st.metric("R² Score", f"{metrics['r2']:.3f}" if metrics["r2"] is not None else "N/A")
        else:
            st.info("Hold-out set too small to compute metrics reliably.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">🔮 Forecast Table</div>', unsafe_allow_html=True)
    display_future = future.copy()
    display_future["Order Date"] = display_future["Order Date"].dt.strftime("%B %Y")
    if target == "Sales":
        display_future["value"] = display_future["value"].map(lambda x: format_currency(x, settings["currency"]))
    else:
        display_future["value"] = display_future["value"].round(0).astype(int)
    display_future = display_future.rename(columns={"value": target_label})
    st.dataframe(display_future, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 8: AI BUSINESS ASSISTANT
# =============================================================================
def build_data_context_summary(df: pd.DataFrame) -> str:
    """Builds a compact statistical summary of the filtered dataset to ground the AI's answers."""
    if df.empty:
        return "No data is currently available for the selected filters."

    kpis = compute_kpis(df)
    cat_perf = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    region_perf = df.groupby("Region")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False)
    monthly = df.groupby("YearMonth")["Sales"].sum().sort_index()
    top_products = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(5)
    low_products = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=True).head(5)

    lines = [
        f"Total Revenue: {kpis['total_revenue']:.2f}",
        f"Total Profit: {kpis['total_profit']:.2f}",
        f"Total Orders: {kpis['total_orders']}",
        f"Total Customers: {kpis['total_customers']}",
        f"Avg Order Value: {kpis['avg_order_value']:.2f}",
        f"Profit Margin: {kpis['profit_margin']:.2f}%",
        "",
        "Category revenue (desc): " + ", ".join(f"{k}={v:.0f}" for k, v in cat_perf.items()),
        "Region revenue & profit: " + ", ".join(f"{k}(sales={v['Sales']:.0f}, profit={v['Profit']:.0f})" for k, v in region_perf.iterrows()),
        "Top 5 products by revenue: " + ", ".join(f"{k}={v:.0f}" for k, v in top_products.items()),
        "Bottom 5 products by revenue: " + ", ".join(f"{k}={v:.0f}" for k, v in low_products.items()),
    ]
    if len(monthly) >= 2:
        lines.append(f"Latest month revenue: {monthly.iloc[-1]:.0f}, previous month: {monthly.iloc[-2]:.0f}")
    if len(monthly) >= 6:
        lines.append("Last 6 months revenue trend: " + ", ".join(f"{k}={v:.0f}" for k, v in monthly.tail(6).items()))

    return "\n".join(lines)


def call_groq(prompt: str, context: str, model: str) -> str:
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model or "llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": (
                "You are a retail business intelligence analyst. Answer the user's question using ONLY "
                "the statistics provided in the data context. Be specific, cite numbers, and give "
                "concise, actionable business recommendations. Never invent data not present in the context."
            )},
            {"role": "user", "content": f"DATA CONTEXT:\n{context}\n\nQUESTION: {prompt}"},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    return resp.choices[0].message.content


def call_mistral(prompt: str, context: str, model: str) -> str:
    from mistralai import Mistral
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set")
    client = Mistral(api_key=api_key)
    resp = client.chat.complete(
        model=model or "mistral-large-latest",
        messages=[
            {"role": "system", "content": (
                "You are a retail business intelligence analyst. Answer the user's question using ONLY "
                "the statistics provided in the data context. Be specific, cite numbers, and give "
                "concise, actionable business recommendations. Never invent data not present in the context."
            )},
            {"role": "user", "content": f"DATA CONTEXT:\n{context}\n\nQUESTION: {prompt}"},
        ],
        temperature=0.3,
        max_tokens=800,
    )
    return resp.choices[0].message.content


def rule_based_fallback_answer(prompt: str, df: pd.DataFrame) -> str:
    """Offline, deterministic fallback answer generator used when no AI provider is available or all fail."""
    prompt_lower = prompt.lower()
    insights = generate_business_insights(df)

    if "decrease" in prompt_lower or "declin" in prompt_lower or "down" in prompt_lower:
        monthly = df.groupby("YearMonth")["Sales"].sum().sort_index()
        if len(monthly) >= 2 and monthly.iloc[-1] < monthly.iloc[-2]:
            change = (monthly.iloc[-1] - monthly.iloc[-2]) / abs(monthly.iloc[-2]) * 100
            return (f"Sales decreased by {abs(change):.1f}% compared to the previous month. "
                    f"This is often driven by seasonality, reduced marketing spend, stockouts, or increased "
                    f"competition. Review the Sales Analytics page's discount and category breakdowns to "
                    f"isolate which category or region drove the decline.")
        return "Based on the current filtered data, sales have not declined month-over-month; they are stable or growing."

    if "grow" in prompt_lower or "increas" in prompt_lower:
        cat_growth = []
        for cat in df["Category"].unique():
            cat_monthly = df[df["Category"] == cat].groupby("YearMonth")["Sales"].sum().sort_index()
            if len(cat_monthly) >= 4:
                h = len(cat_monthly) // 2
                g = cat_monthly.iloc[h:].mean() - cat_monthly.iloc[:h].mean()
                cat_growth.append((cat, g))
        if cat_growth:
            best = max(cat_growth, key=lambda x: x[1])
            return f"**{best[0]}** is currently the fastest-growing category based on average monthly sales trend."
        return "Not enough historical months in the current filter to determine category growth reliably."

    if "restock" in prompt_lower or "inventory" in prompt_lower:
        max_date = df["Order Date"].max()
        recent = df[df["Order Date"] >= max_date - timedelta(days=90)]
        top = recent.groupby("Product Name")["Quantity"].sum().sort_values(ascending=False).head(3)
        if not top.empty:
            names = ", ".join(top.index.tolist())
            return f"Based on the last 90 days of demand, prioritize restocking: **{names}**."
        return "Not enough recent order history to generate a restocking recommendation."

    return "Here's what the data shows:\n\n" + "\n".join(f"• {i}" for i in insights)


def page_ai_assistant(df: pd.DataFrame, settings: dict) -> None:
    render_gradient_header("🤖 AI Business Assistant", "Ask natural-language questions about your sales data")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    groq_key = bool(os.environ.get("GROQ_API_KEY"))
    mistral_key = bool(os.environ.get("MISTRAL_API_KEY"))

    status_cols = st.columns(3)
    status_cols[0].markdown(f"**Groq:** {'🟢 Connected' if groq_key else '⚪ No API key'}")
    status_cols[1].markdown(f"**Mistral:** {'🟢 Connected' if mistral_key else '⚪ No API key'}")
    status_cols[2].markdown(f"**Provider:** {settings['ai_provider']}")

    if not groq_key and not mistral_key:
        st.info(
            "No `GROQ_API_KEY` or `MISTRAL_API_KEY` found in the environment. The assistant will still work "
            "using a built-in rule-based analysis engine grounded in your live data. Add API keys in your "
            "`.env` file or Streamlit secrets to enable full LLM-powered answers."
        )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### 💬 Ask a question about your business")
    example_cols = st.columns(3)
    example_qs = ["Why did sales decrease?", "Which category is growing?", "What products should be restocked?"]
    clicked_example = None
    for col, q in zip(example_cols, example_qs):
        if col.button(q, use_container_width=True):
            clicked_example = q

    user_prompt = st.text_area("Your question", value=clicked_example or "", placeholder="e.g. Why did sales decrease last month?")
    ask = st.button("🚀 Ask AI Assistant", type="primary")

    if ask and user_prompt.strip():
        context = build_data_context_summary(df)
        provider_order = []
        if settings["ai_provider"] == "Groq":
            provider_order = ["groq"]
        elif settings["ai_provider"] == "Mistral":
            provider_order = ["mistral"]
        else:
            provider_order = ["groq", "mistral"]

        answer, used_provider, error_msg = None, None, None
        with st.spinner("Analyzing your data..."):
            for provider in provider_order:
                try:
                    if provider == "groq" and groq_key:
                        answer = call_groq(user_prompt, context, settings.get("ai_model", "auto"))
                        used_provider = "Groq"
                        break
                    if provider == "mistral" and mistral_key:
                        answer = call_mistral(user_prompt, context, settings.get("ai_model", "auto"))
                        used_provider = "Mistral"
                        break
                except Exception as e:  # noqa: BLE001 - graceful multi-provider fallback
                    error_msg = str(e)
                    continue

            if answer is None:
                answer = rule_based_fallback_answer(user_prompt, df)
                used_provider = "Built-in Rule-Based Engine (offline fallback)"

        st.markdown(f"**Answered by:** `{used_provider}`")
        st.markdown(answer)
        if error_msg and used_provider == "Built-in Rule-Based Engine (offline fallback)":
            st.caption(f"⚠️ AI provider call failed, used fallback. Details: {error_msg}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">🧠 Auto-Generated Insights (always available)</div>', unsafe_allow_html=True)
    render_insights_pills(generate_business_insights(df))
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 9: REPORTS
# =============================================================================
def generate_pdf_report(df: pd.DataFrame, kpis: dict, insights: list, currency: str) -> bytes:
    """Generates a simple executive summary PDF report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Retail Insights - Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Key Performance Indicators", ln=True)
    pdf.set_font("Helvetica", "", 11)
    kpi_lines = [
        f"Total Revenue: {format_currency(kpis['total_revenue'], currency)}",
        f"Total Profit: {format_currency(kpis['total_profit'], currency)}",
        f"Total Orders: {kpis['total_orders']:,}",
        f"Total Customers: {kpis['total_customers']:,}",
        f"Average Order Value: {format_currency(kpis['avg_order_value'], currency)}",
        f"Profit Margin: {kpis['profit_margin']:.1f}%",
        f"Total Quantity Sold: {int(kpis['total_quantity']):,}",
    ]
    for line in kpi_lines:
        pdf.cell(0, 8, line, ln=True)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Automated Business Insights", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for insight in insights:
        clean = insight.replace("**", "")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, f"- {clean}")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Top 5 Products by Revenue", ln=True)
    pdf.set_font("Helvetica", "", 11)
    top5 = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(5)
    for name, val in top5.items():
        pdf.cell(0, 8, f"{name}: {format_currency(val, currency)}", ln=True)

    return bytes(pdf.output(dest="S"))


def generate_excel_report(df: pd.DataFrame, kpis: dict) -> bytes:
    """Generates a multi-sheet Excel workbook with raw data and summary tables."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame([kpis]).to_excel(writer, sheet_name="KPI Summary", index=False)
        df.groupby("Category").agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum")).reset_index().to_excel(
            writer, sheet_name="Category Performance", index=False
        )
        df.groupby("Region").agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum")).reset_index().to_excel(
            writer, sheet_name="Regional Performance", index=False
        )
        df.to_excel(writer, sheet_name="Raw Data", index=False)
    return buffer.getvalue()


def page_reports(df: pd.DataFrame, settings: dict) -> None:
    render_gradient_header("📊 Reports", "Export data and generate executive summary reports")

    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    kpis = compute_kpis(df)
    insights = generate_business_insights(df)

    st.markdown('<div class="section-card"><div class="section-title">📄 Executive Summary Preview</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Total Revenue", format_currency(kpis["total_revenue"], settings["currency"]))
    cols[1].metric("Total Profit", format_currency(kpis["total_profit"], settings["currency"]))
    cols[2].metric("Total Orders", f"{kpis['total_orders']:,}")
    cols[3].metric("Profit Margin", f"{kpis['profit_margin']:.1f}%")
    render_insights_pills(insights)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">⬇️ Export Options</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv_bytes, file_name="retail_insights_data.csv",
                            mime="text/csv", use_container_width=True)

    with c2:
        try:
            excel_bytes = generate_excel_report(df, kpis)
            st.download_button("📥 Download Excel Report", data=excel_bytes, file_name="retail_insights_report.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Excel export failed: {e}")

    with c3:
        if FPDF_AVAILABLE:
            try:
                pdf_bytes = generate_pdf_report(df, kpis, insights, settings["currency"])
                st.download_button("📥 Download PDF Summary", data=pdf_bytes, file_name="retail_insights_summary.pdf",
                                    mime="application/pdf", use_container_width=True)
            except Exception as e:  # noqa: BLE001
                st.error(f"PDF export failed: {e}")
        else:
            st.info("Install `fpdf2` to enable PDF export.")
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 10: SETTINGS
# =============================================================================
def page_settings(settings: dict) -> dict:
    render_gradient_header("⚙ Settings", "Configure your dashboard preferences")

    st.markdown('<div class="section-card"><div class="section-title">🎨 Appearance</div>', unsafe_allow_html=True)
    theme = st.selectbox("Theme", ["Dark", "Light"], index=["Dark", "Light"].index(settings["theme"]))
    if theme == "Light":
        st.caption("ℹ️ This dashboard is optimized for Dark mode. Light mode support is basic.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">🤖 AI Assistant Configuration</div>', unsafe_allow_html=True)
    ai_provider = st.selectbox("AI Provider", ["Auto", "Groq", "Mistral"],
                                index=["Auto", "Groq", "Mistral"].index(settings["ai_provider"]))
    ai_model = st.text_input("AI Model (optional override)", value=settings.get("ai_model", "auto"),
                              help="e.g. llama-3.1-8b-instant for Groq, mistral-large-latest for Mistral")
    st.caption("API keys are read from environment variables `GROQ_API_KEY` and `MISTRAL_API_KEY` (never entered here).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">📉 Forecasting</div>', unsafe_allow_html=True)
    forecast_horizon = st.slider("Default Forecast Horizon (months)", 1, 12, settings["forecast_horizon"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">🌐 Regional Preferences</div>', unsafe_allow_html=True)
    currency = st.selectbox("Currency", list(CURRENCY_SYMBOLS.keys()), index=list(CURRENCY_SYMBOLS.keys()).index(settings["currency"]))
    date_format = st.selectbox("Date Format", ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"],
                                index=["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"].index(settings["date_format"]))
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("💾 Save Settings", type="primary"):
        st.session_state["settings"] = {
            "theme": theme, "ai_provider": ai_provider, "ai_model": ai_model,
            "forecast_horizon": forecast_horizon, "currency": currency, "date_format": date_format,
        }
        st.success("Settings saved for this session.")
        st.rerun()

    return settings


# =============================================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================================
def main() -> None:
    inject_css()

    if "settings" not in st.session_state:
        st.session_state["settings"] = DEFAULT_SETTINGS.copy()
    settings = st.session_state["settings"]

    try:
        raw_df = load_data(DATA_PATH)
    except Exception as e:  # noqa: BLE001 - top-level safety net
        st.error(f"⚠️ Unexpected error while loading data: {e}")
        raw_df = pd.DataFrame()

    if raw_df.empty:
        st.markdown(
            """
            <div class="gradient-header">
                <h1>📊 Retail Insights</h1>
                <p>AI-Powered E-Commerce Sales Analysis</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.error(
            f"⚠️ Could not load a valid dataset from `{DATA_PATH}`. "
            "Please ensure the CSV file exists, is not empty, and contains the expected columns "
            "(Order ID, Order Date, Sales, Profit, Category, Region, etc.)."
        )
        st.info("If you're setting this up fresh, run `python generate_data.py` to create the sample dataset.")
        return

    page, filtered_df = render_sidebar(raw_df)

    try:
        if page == "Dashboard":
            page_dashboard(filtered_df, settings)
        elif page == "Sales Analytics":
            page_sales_analytics(filtered_df, settings)
        elif page == "Product Analytics":
            page_product_analytics(filtered_df, settings)
        elif page == "Customer Analytics":
            page_customer_analytics(filtered_df, settings)
        elif page == "Regional Analytics":
            page_regional_analytics(filtered_df, settings)
        elif page == "Inventory Insights":
            page_inventory_insights(filtered_df, settings)
        elif page == "Sales Forecasting":
            page_sales_forecasting(filtered_df, settings)
        elif page == "AI Business Assistant":
            page_ai_assistant(filtered_df, settings)
        elif page == "Reports":
            page_reports(filtered_df, settings)
        elif page == "Settings":
            st.session_state["settings"] = page_settings(settings)
    except Exception as e:  # noqa: BLE001 - keep the app alive on unexpected page errors
        st.error(f"⚠️ Something went wrong while rendering this page: {e}")
        st.info("Try adjusting your filters or reloading the page. If the issue persists, check your data file for anomalies.")


if __name__ == "__main__":
    main()
