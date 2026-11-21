"""
Streamlit Executive Decision Cockpit.
Provides interactive visual simulations for:
1. Cohort Retention & Kaplan-Meier Churn Hazard
2. Econometric & Causal DML Price Elasticity
3. Customer Lifetime Value (CLV) Equity Waterfall
4. Multi-Touch Markov & Spectral Graph Attribution
5. Multi-Echelon Inventory Optimization (MEIO) & Safety Stock
6. Multivariate Mahalanobis Anomaly & Margin Leak Detection
7. Cash Flow at Risk (CFaR) Correlated Monte Carlo Stress Testing
8. Dynamic Multi-Armed Bandit (MAB) Experimentation & Product Cannibalization
"""

import streamlit as st
import pandas as pd
import numpy as np

from data_generator import generate_ecommerce_dataset
from cohort_retention import compute_cohort_retention, compute_churn_hazard
from price_elasticity import estimate_price_elasticity, simulate_price_shift_margin_impact
from clv_model import compute_rfm_metrics, get_segment_waterfall_summary
from marketing_attribution import compute_markov_attribution
from supply_chain_meio import compute_stochastic_safety_stock, optimize_sku_inventory_policy
from multivariate_anomaly import detect_operational_anomalies, summarize_margin_leakage
from cash_flow_at_risk_cfar import CashFlowAtRiskSimulator
from bandit_experimentation import BanditExperimentEngine
from mixed_logit_cannibalization import MixedLogitCannibalizationEngine

st.set_page_config(page_title="Executive Decision Analytics Engine", layout="wide")

st.title("Enterprise Decision Analytics Engine | C-Suite Decision Cockpit")
st.markdown("**Production Econometric Modeling, Operations Research Intelligence, and Stochastic Risk Architecture**")

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
col4.metric("Active Accounts", f"{total_customers:,}")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "1. Cohort Retention", 
    "2. Price Elasticity", 
    "3. CLV Equity", 
    "4. Attribution",
    "5. Inventory MEIO",
    "6. Anomaly Leaks",
    "7. Cash Flow at Risk",
    "8. Bandits & Cannibalization",
])

# ----------------------------------------------------
# Tab 1: Cohort Retention
# ----------------------------------------------------
with tab1:
    st.subheader("Monthly Customer Retention Matrix")
    user_ret, rev_ret = compute_cohort_retention(tx_df)
    st.dataframe(user_ret.style.background_gradient(cmap="Blues", axis=None, vmin=0, vmax=100))

    st.subheader("Period-over-Period Churn Hazard Rate")
    hazard_df = compute_churn_hazard(user_ret)
    st.line_chart(hazard_df.set_index("period")[["survival_rate_pct", "hazard_rate_pct"]])

# ----------------------------------------------------
# Tab 2: Price Elasticity
# ----------------------------------------------------
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

# ----------------------------------------------------
# Tab 3: Customer Lifetime Value (CLV)
# ----------------------------------------------------
with tab3:
    st.subheader("RFM Portfolio Segmentation & 12-Month Forward CLV Equity")
    rfm = compute_rfm_metrics(tx_df)
    summary = get_segment_waterfall_summary(rfm)
    st.dataframe(summary)
    st.bar_chart(summary.set_index("segment")["forward_12m_expected_margin"])

# ----------------------------------------------------
# Tab 4: Marketing Attribution
# ----------------------------------------------------
with tab4:
    st.subheader("First-Order Markov Chain Multi-Touch Attribution vs Heuristics")
    comp_df, _ = compute_markov_attribution(journeys_df)
    st.dataframe(comp_df)
    st.bar_chart(comp_df.set_index("channel")[["markov_attributed_rev", "last_touch_rev", "first_touch_rev"]])

# ----------------------------------------------------
# Tab 5: Multi-Echelon Inventory Optimization (MEIO)
# ----------------------------------------------------
with tab5:
    st.subheader("Multi-Echelon Inventory & Dynamic Stochastic Safety Stock")
    st.markdown("Calculates optimal safety stock and reorder points under joint demand and lead-time stochasticity.")
    
    col_a, col_b, col_c = st.columns(3)
    annual_dem = col_a.number_input("Annual Demand (Units)", min_value=100, max_value=500000, value=15000, step=500)
    unit_cost_val = col_b.number_input("Unit Cost ($)", min_value=1.0, max_value=5000.0, value=45.0, step=5.0)
    service_lvl = col_c.slider("Cycle Service Level", min_value=0.80, max_value=0.999, value=0.95, step=0.01)

    inv_policy = optimize_sku_inventory_policy(
        annual_demand=annual_dem,
        unit_cost=unit_cost_val,
        daily_demand_mean=annual_dem / 365.0,
        daily_demand_std=max(1.0, (annual_dem / 365.0) * 0.30),
        lead_time_days_mean=14.0,
        lead_time_days_std=3.0,
        target_service_level=service_lvl,
    )
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("EOQ (Reorder Batch)", f"{inv_policy['economic_order_quantity']:,.0f} units")
    m2.metric("Safety Stock", f"{inv_policy['safety_stock']:,.0f} units")
    m3.metric("Reorder Point (ROP)", f"{inv_policy['reorder_point']:,.0f} units")
    m4.metric("Item Fill Rate", f"{inv_policy['fill_rate_type2']*100:.2f}%")
    
    st.metric("Committed Working Capital in Inventory", f"${inv_policy['working_capital_committed']:,.2f}")

# ----------------------------------------------------
# Tab 6: Multivariate Anomaly Detection
# ----------------------------------------------------
with tab6:
    st.subheader("Multivariate Mahalanobis Anomaly & Margin Leakage Engine")
    st.markdown("Identifies statistically significant multidimensional outliers (Discount x Shipping x Quantity x Margin).")
    
    alpha_threshold = st.slider("Significance Level (Alpha)", min_value=0.001, max_value=0.05, value=0.01, step=0.005)
    anom_df = detect_operational_anomalies(tx_df, alpha=alpha_threshold)
    leak_summary = summarize_margin_leakage(anom_df)
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Anomalous Orders", f"{leak_summary['anomalous_transactions_count']} ({leak_summary['anomaly_rate_pct']}%)")
    k2.metric("Revenue at Risk", f"${leak_summary['anomalous_revenue_exposure']:,.2f}")
    k3.metric("Negative Margin Leakage", f"${abs(leak_summary['pure_margin_leakage_dollars']):,.2f}")
    
    st.dataframe(anom_df[anom_df["is_operational_anomaly"]][["transaction_id", "customer_id", "gross_revenue", "gross_margin", "discount_pct", "mahalanobis_dist_sq", "anomaly_classification"]].head(20))

# ----------------------------------------------------
# Tab 7: Cash Flow at Risk (CFaR) Monte Carlo
# ----------------------------------------------------
with tab7:
    st.subheader("Cash Flow at Risk (CFaR) Correlated Monte Carlo Balance Sheet Stress Testing")
    st.markdown("Simulates 5,000 multi-variable macro shock trajectories to estimate Value-at-Risk on operating cash flow.")
    
    c_rev_vol = st.slider("Revenue Volatility (%)", min_value=5, max_value=35, value=15, step=5) / 100.0
    c_cogs_vol = st.slider("COGS Inflation Volatility (%)", min_value=2, max_value=20, value=8, step=2) / 100.0
    
    cfar_engine = CashFlowAtRiskSimulator(
        base_annual_revenue=total_rev,
        base_gross_margin_pct=overall_margin_pct / 100.0,
        initial_cash_reserves=500000.0,
        debt_service_annual=150000.0,
        minimum_liquidity_covenant=200000.0,
    )
    cfar_results = cfar_engine.simulate_cfar(
        num_simulations=2000,
        revenue_volatility=c_rev_vol,
        cogs_inflation_volatility=c_cogs_vol,
        random_seed=42,
    )
    
    f1, f2, f3 = st.columns(3)
    f1.metric("95% VaR Cash Flow Floor", f"${cfar_results['var_95_cash_flow_floor']:,.2f}")
    f2.metric("Cash Flow at Risk (CFaR 95%)", f"${cfar_results['cfar_95_dollars']:,.2f}")
    f3.metric("Covenant Breach Probability", f"{cfar_results['covenant_breach_probability_pct']}%")
    
    st.dataframe(cfar_results["distribution_table"])

# ----------------------------------------------------
# Tab 8: Dynamic Bandits & Cannibalization
# ----------------------------------------------------
with tab8:
    st.subheader("Dynamic Multi-Armed Bandits & Mixed Logit Cannibalization")
    st.markdown("Compares Bayesian Thompson Sampling against static A/B testing, and evaluates internal product cannibalization.")
    
    st.markdown("#### Dynamic Multi-Armed Bandit Experimentation")
    bandit = BanditExperimentEngine({"Creative_A (4% baseline)": 0.04, "Creative_B (11% offer)": 0.11, "Creative_C (7% social)": 0.07})
    b_res = bandit.run_simulation(total_trials=3000, random_seed=42)
    st.dataframe(b_res["comparison_summary"])
    st.success(f"Thompson Sampling reduced lost conversions (opportunity regret) by {b_res['regret_reduction_vs_ab_pct']}% compared to a static A/B split.")
    
    st.markdown("#### Mixed Logit Product Cannibalization Matrix")
    portfolio_skus = [
        {"name": "Standard Tier ($99)", "price": 99.0, "base_quality": 8.0, "cogs": 45.0},
        {"name": "Premium Tier ($179)", "price": 179.0, "base_quality": 12.5, "cogs": 75.0},
        {"name": "Entry Tier ($59)", "price": 59.0, "base_quality": 5.2, "cogs": 28.0},
    ]
    cannibal_engine = MixedLogitCannibalizationEngine(portfolio_skus, random_seed=42)
    cannibal_res = cannibal_engine.compute_cannibalization_and_elasticity(price_shock_pct=0.05)
    st.dataframe(cannibal_res["portfolio_summary"])
    st.markdown("**Diversion Ratio Matrix (% of lost volume diverted):**")
    st.dataframe(cannibal_res["diversion_ratio_matrix_pct"])
