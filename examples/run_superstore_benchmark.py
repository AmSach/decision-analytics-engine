"""
Real-World Benchmark Execution on Global Superstore Dataset (9,426 Transactions).
Executes the full Decision Analytics Suite on live enterprise retail data:
1. Cohort Retention & Kaplan-Meier Churn Hazard Curves
2. Double/Debiased Machine Learning (DML) Causal Price Elasticity
3. Customer Lifetime Value (CLV) Equity Waterfall
4. DuPont 3-Tier ROE & Stochastic Negative Working Capital Float
5. Multi-Echelon Inventory Optimization (MEIO) & Dynamic Stochastic Safety Stock
6. Multivariate Operational Anomaly & Margin Leakage Detection
7. Enterprise Supply Chain & Customer Dependency Network Graph
8. Synthetic Control Quasi-Experiment (Regional Pilot Lift)
9. Markowitz Efficient Frontier Capital Allocation
Exports all executive CSV deliverables into ./examples/outputs/
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.excel_universal_ingestor import UniversalExcelIngestor
from src.cohort_retention import compute_cohort_retention, compute_churn_hazard
from src.causal_elasticity_dml import estimate_causal_elasticity_dml
from src.clv_model import compute_rfm_metrics, get_segment_waterfall_summary
from src.stochastic_float_engine import compute_dupont_3tier_decomposition, simulate_negative_working_capital_float
from src.supply_chain_meio import optimize_multi_echelon_portfolio
from src.multivariate_anomaly import detect_operational_anomalies, summarize_margin_leakage
from src.supply_network_graph import EnterpriseNetworkGraph
from src.synthetic_control import SyntheticControlEngine
from src.portfolio_markowitz import MarkowitzCapitalOptimizer


def run_real_world_superstore_benchmark():
    out_dir = Path("examples/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path("examples/raw_online_data/superstore_online_retail_raw.csv")

    print("=" * 80)
    print("EXECUTING REAL-WORLD BENCHMARK ON GLOBAL SUPERSTORE (9,426 TRANSACTIONS)")
    print("=" * 80)

    # 1. Ingestion & Schema Harmonization
    print("\n[1/9] Ingesting and harmonizing raw Global Superstore data...")
    ingestor = UniversalExcelIngestor()
    clean_df = ingestor.ingest_file(raw_path)
    print(f"      Harmonized {len(clean_df)} real transactions across {clean_df['customer_id'].nunique()} enterprise accounts.")

    # 2. Cohort Retention & Churn Hazard
    print("\n[2/9] Computing longitudinal cohort survival & Kaplan-Meier churn hazard...")
    user_ret, rev_ret = compute_cohort_retention(clean_df)
    hazard_df = compute_churn_hazard(user_ret)
    user_ret.to_csv(out_dir / "superstore_cohort_user_retention.csv")
    hazard_df.to_csv(out_dir / "superstore_churn_hazard.csv", index=False)
    print("      Top Cohort Survival (Snapshot):")
    print(user_ret.iloc[:3, :5])

    # 3. DML Causal Price Elasticity
    print("\n[3/9] Estimating unconfounded causal price elasticity via DML...")
    dml_res = estimate_causal_elasticity_dml(clean_df, n_splits=5, random_seed=42)
    pd.DataFrame([dml_res]).to_csv(out_dir / "superstore_dml_elasticity.csv", index=False)
    print(f"      DML Causal Elasticity: {dml_res['causal_elasticity_dml']} (SE: {dml_res['standard_error']}, 95% CI: {dml_res['ci_95']})")

    # 4. RFM CLV Equity Waterfall
    print("\n[4/9] Evaluating Customer Lifetime Value (CLV) & forward equity waterfall...")
    rfm_df = compute_rfm_metrics(clean_df)
    clv_summary = get_segment_waterfall_summary(rfm_df)
    clv_summary.to_csv(out_dir / "superstore_clv_waterfall.csv", index=False)
    print("      Customer Equity Segments:")
    for _, row in clv_summary.iterrows():
        print(f"      {row['segment']}: {row['customer_count']} accounts -> 12M Forward Equity: ${row['forward_12m_expected_margin']:,.2f}")

    # 5. DuPont ROE & Stochastic Cash Float
    print("\n[5/9] Calculating DuPont 3-Tier ROE Decomposition & Working Capital Float...")
    total_rev = clean_df["gross_revenue"].sum()
    total_margin = clean_df["gross_margin"].sum()
    dupont = compute_dupont_3tier_decomposition(
        revenue=total_rev,
        net_income=total_margin * 0.35,
        total_assets=total_rev * 0.40,
        shareholder_equity=total_rev * 0.20,
    )
    pd.DataFrame([dupont]).to_csv(out_dir / "superstore_dupont_roe.csv", index=False)
    float_summary, _ = simulate_negative_working_capital_float(annual_revenue=total_rev)
    pd.DataFrame([float_summary]).to_csv(out_dir / "superstore_working_capital_float.csv", index=False)
    print(f"      DuPont ROE: {dupont['return_on_equity_pct']}% | Annual Cash Float Treasury Yield: ${float_summary['total_annual_treasury_yield']:,.2f}")

    # 6. Multi-Echelon Inventory Optimization (MEIO) & Safety Stock
    print("\n[6/9] Optimizing Multi-Echelon Inventory Policy across Product Categories...")
    meio_df = optimize_multi_echelon_portfolio(clean_df, lead_time_days_mean=14.0, lead_time_days_std=3.0, service_level=0.95)
    meio_df.to_csv(out_dir / "superstore_inventory_meio.csv", index=False)
    total_wc_committed = meio_df["working_capital_committed"].sum()
    print(f"      Optimized {len(meio_df)} operational categories. Total Working Capital Committed: ${total_wc_committed:,.2f}")
    for _, r in meio_df.iterrows():
        cat_name = r.get("category", r.get("sku", "Unknown"))
        print(f"      {cat_name}: Safety Stock: {r['safety_stock']:,.0f} units | ROP: {r['reorder_point']:,.0f} units | Fill Rate: {r['fill_rate_type2']*100:.2f}%")

    # 7. Multivariate Operational Anomaly & Margin Leakage Engine
    print("\n[7/9] Running Multivariate Mahalanobis Anomaly & Margin Leak Detection...")
    anomaly_df = detect_operational_anomalies(clean_df, alpha=0.01)
    leak_summary = summarize_margin_leakage(anomaly_df)
    pd.DataFrame([leak_summary]).to_csv(out_dir / "superstore_anomaly_leakage_summary.csv", index=False)
    # Filter anomalies and export
    flagged_anomalies = anomaly_df[anomaly_df["is_operational_anomaly"]].copy()
    flagged_anomalies.to_csv(out_dir / "superstore_flagged_anomalies.csv", index=False)
    print(f"      Detected {leak_summary['anomalous_transactions_count']} anomalous transactions ({leak_summary['anomaly_rate_pct']}%)")
    print(f"      Identified ${abs(leak_summary['pure_margin_leakage_dollars']):,.2f} in severe negative margin leaks.")

    # 8. Enterprise Supply & Customer Network Graph
    print("\n[8/9] Computing Network Graph Centrality and Systemic Vulnerability...")
    target_node = "Department" if "Department" in clean_df.columns else ("category" if "category" in clean_df.columns else "sku")
    graph = EnterpriseNetworkGraph(clean_df, source_col="customer_id", target_col=target_node, weight_col="gross_revenue")
    fragility_df = graph.evaluate_systemic_fragility(top_k=10)
    fragility_df.to_csv(out_dir / "superstore_network_vulnerability.csv", index=False)
    print(f"      Top 3 Systemic Vulnerability Nodes:")
    for _, r in fragility_df.head(3).iterrows():
        print(f"      {r['entity_node']}: PageRank={r['pagerank_score']:.5f} | Flow Volume=${r['total_flow_volume']:,.2f} | Fragility Index={r['systemic_fragility_index']:.2f}")

    # 9. Synthetic Control Regional Pilot Lift (Aggregated by Quarter across 16 Quarters)
    print("\n[9/9] Evaluating Synthetic Control Causal Lift on Regional Expansion...")
    if "Region" in clean_df.columns:
        clean_df["order_quarter"] = pd.to_datetime(clean_df["order_date"]).dt.to_period("Q").astype(str)
        region_panel = clean_df.groupby(["order_quarter", "Region"])["gross_revenue"].sum().reset_index()
        quarters = sorted(region_panel["order_quarter"].unique().tolist())
        if len(quarters) >= 8:
            sc_engine = SyntheticControlEngine()
            split_quarter = quarters[len(quarters) // 2]
            treated_reg = "Central" if "Central" in region_panel["Region"].values else region_panel["Region"].iloc[0]
            sc_res = sc_engine.fit_synthetic_control(
                panel_df=region_panel,
                unit_col="Region",
                time_col="order_quarter",
                outcome_col="gross_revenue",
                treated_unit=treated_reg,
                intervention_time=split_quarter,
            )
            sc_res["trajectory_df"].to_csv(out_dir / "superstore_synthetic_control_lift.csv", index=False)
            print(f"      Synthetic Control Lift for {treated_reg}: ${sc_res['post_intervention_avg_lift']:,.2f}/quarter")
            print(f"      Pre-Intervention Fit RMSE: ${sc_res['pre_intervention_rmse']:,.2f}")

    # Bonus: Markowitz Portfolio Optimization across Departments
    print("\n[Bonus] Markowitz Mean-Variance Capital Allocation across Product Departments...")
    if "Department" in clean_df.columns:
        dept_panel = clean_df.groupby([pd.to_datetime(clean_df["order_date"]).dt.to_period("M"), "Department"])["gross_revenue"].sum().unstack().fillna(0.0)
        dept_roas = dept_panel.pct_change().dropna() + 1.0
        opt = MarkowitzCapitalOptimizer(risk_aversion=1.0, max_channel_cap=0.50)
        alloc_res = opt.fit_optimal_allocation(dept_roas, total_budget=500000.0)
        alloc_res["allocation_df"].to_csv(out_dir / "superstore_markowitz_allocation.csv", index=False)
        print(f"      Markowitz Optimal Portfolio Allocation (Budget: $500,000):")
        for _, r in alloc_res["allocation_df"].iterrows():
            print(f"      {r['channel_or_asset']}: {r['optimal_weight_pct']}% -> ${r['allocated_budget']:,.2f}")

    print("\n" + "=" * 80)
    print(f"REAL-WORLD BENCHMARK COMPLETE. All artifacts attached in ./{out_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    run_real_world_superstore_benchmark()
