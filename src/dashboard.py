"""
Streamlit Executive Decision Cockpit.
Provides interactive visual simulations for cohort retention,
econometric price elasticity curves, customer equity portfolios, and marketing attribution.
"""

import streamlit as st
import pandas as pd
import numpy as np

from data_generator import generate_ecommerce_dataset
from cohort_retention import compute_cohort_retention, compute_churn_hazard
from price_elasticity import estimate_price_elasticity, simulate_price_shift_margin_impact
from clv_model import compute_rfm_metrics, get_segment_waterfall_summary
from marketing_attribution import compute_markov_attribution

st.set_page_config(page_title="Executive Decision Analytics Engine", layout="wide")

st.title("Decision Analytics Engine | B-School Strategic Suite")
st.markdown("**Production Econometric Modeling, Customer Equity Portfolios, and Marginal Return Attribution**")

@st.cache_data
def load_data():
    tx_df, journeys_df = generate_ecommerce_dataset(num_customers=1200, num_days=300, random_seed=42)
    return tx_df, journeys_df

tx_df, journeys_df = load_data()

# Top KPI Metric Cards
total_rev = tx_df["gross_revenue"].sum()
total_margin = tx_df["gross_margin"].sum()
overall_margin_pct = (total_margin / total_rev) * 100.0 if total_rev > 0 else 0.0
total_customers = tx_df["customer_id"].nunique()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Gross Revenue", f"${total_rev:,.2f}")
col2.metric("Gross Margin", f"${total_margin:,.2f}")
col3.metric("Gross Margin %", f"{overall_margin_pct:.1f}%")
col4.metric("Active Customers", f"{total_customers:,}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Cohort Retention & Churn", 
    "Price Elasticity & Margins", 
    "Customer Lifetime Equity", 
    "Multi-Touch Attribution"
])

with tab1:
    st.subheader("Monthly Customer Retention Matrix")
    user_ret, rev_ret = compute_cohort_retention(tx_df)
    st.dataframe(user_ret.style.background_gradient(cmap="Blues", axis=None, vmin=0, vmax=100))

    st.subheader("Period-over-Period Churn Hazard Rate")
    hazard_df = compute_churn_hazard(user_ret)
    st.line_chart(hazard_df.set_index("period")[["survival_rate_pct", "hazard_rate_pct"]])

with tab2:
    st.subheader("Econometric Price Elasticity Estimations (Log-Linear OLS)")
    elasticity_dict = estimate_price_elasticity(tx_df)
    elasticity_df = pd.DataFrame(elasticity_dict.values())
    st.dataframe(elasticity_df)

    st.markdown("### Strategic Price Shock Simulation")
    selected_sku = st.selectbox("Select SKU to simulate:", list(elasticity_dict.keys()))
    sku_info = elasticity_dict[selected_sku]
    
    price_slider = st.slider("Price Change (%)", min_value=-30, max_value=30, value=5, step=1)
    sim = simulate_price_shift_margin_impact(
        current_price=sku_info["avg_selling_price"],
        current_volume=1000,
        unit_cost=sku_info["unit_cost_cogs"],
        elasticity=sku_info["price_elasticity"],
        price_change_pct=price_slider / 100.0,
    )
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("New Price", f"${sim['new_price']}")
    c2.metric("Projected Volume", f"{sim['new_volume']}")
    c3.metric("New Gross Margin", f"${sim['new_gross_margin']:,.2f}")
    c4.metric("Margin Lift (%)", f"{sim['margin_lift_pct']:+.1f}%")

with tab3:
    st.subheader("RFM Portfolio Segmentation & 12-Month Forward CLV Equity")
    rfm = compute_rfm_metrics(tx_df)
    summary = get_segment_waterfall_summary(rfm)
    st.dataframe(summary)
    st.bar_chart(summary.set_index("segment")["forward_12m_expected_margin"])

with tab4:
    st.subheader("First-Order Markov Chain Multi-Touch Attribution vs Heuristics")
    comp_df, _ = compute_markov_attribution(journeys_df)
    st.dataframe(comp_df)
    st.bar_chart(comp_df.set_index("channel")[["markov_attributed_rev", "last_touch_rev", "first_touch_rev"]])
