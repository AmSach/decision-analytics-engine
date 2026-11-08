"""
Real-World Benchmark Execution on Global Superstore Dataset (9,426 Transactions).
Executes the full Decision Analytics Suite on live enterprise retail data:
1. Cohort Retention & Churn Hazard Curves
2. Double/Debiased Machine Learning (DML) Causal Price Elasticity
3. Customer Lifetime Value (CLV) Equity Waterfall
4. DuPont 3-Tier ROE & Stochastic Negative Working Capital Float
5. Palantir Enterprise Decision Ontology Risk Telemetry
6. Counterfactual Scenario Sandbox (Macro Shock War-Gaming)
7. Synthetic Control Quasi-Experiment (Regional Pilot Lift)
8. Markowitz Efficient Frontier Capital Allocation
Exports all executive CSV deliverables into ./examples/outputs/
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.excel_universal_ingestor import UniversalExcelIngestor
from src.cohort_retention import compute_cohort_retention, compute_churn_hazard
from src.causal_elasticity_dml import estimate_causal_elasticity_dml
from src.clv_model import compute_rfm_metrics, get_segment_waterfall_summary
from src.stochastic_float_engine import compute_dupont_3tier_decomposition, simulate_negative_working_capital_float
from src.enterprise_ontology import EnterpriseDecisionOntology
from src.scenario_sandbox import ScenarioSandboxEngine, ScenarioShockParameters
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
    print("\n[1/8] Ingesting and harmonizing raw Global Superstore data...")
    ingestor = UniversalExcelIngestor()
    clean_df = ingestor.ingest_file(raw_path)
    print(f"      Harmonized {len(clean_df)} real transactions across {clean_df['customer_id'].nunique()} enterprise accounts.")

    # 2. Cohort Retention & Churn Hazard
    print("\n[2/8] Computing longitudinal cohort survival & Kaplan-Meier churn hazard...")
    user_ret, rev_ret = compute_cohort_retention(clean_df)
    hazard_df = compute_churn_hazard(user_ret)
    user_ret.to_csv(out_dir / "superstore_cohort_user_retention.csv")
    hazard_df.to_csv(out_dir / "superstore_churn_hazard.csv", index=False)
    print("      Top Cohort Survival (Snapshot):")
    print(user_ret.iloc[:3, :5])

    # 3. DML Causal Price Elasticity
    print("\n[3/8] Estimating unconfounded causal price elasticity via DML...")
    dml_res = estimate_causal_elasticity_dml(clean_df, n_splits=5, random_seed=42)
    pd.DataFrame([dml_res]).to_csv(out_dir / "superstore_dml_elasticity.csv", index=False)
    print(f"      DML Causal Elasticity: {dml_res['causal_elasticity_dml']} (SE: {dml_res['standard_error']}, 95% CI: {dml_res['ci_95']})")

    # 4. RFM CLV Equity Waterfall
    print("\n[4/8] Evaluating Customer Lifetime Value (CLV) & forward equity waterfall...")
    rfm_df = compute_rfm_metrics(clean_df)
    clv_summary = get_segment_waterfall_summary(rfm_df)
    clv_summary.to_csv(out_dir / "superstore_clv_waterfall.csv", index=False)
    print("      Customer Equity Segments:")
    for _, row in clv_summary.iterrows():
        print(f"      {row['segment']}: {row['customer_count']} accounts -> 12M Forward Equity: ${row['forward_12m_expected_margin']:,.2f}")

    # 5. DuPont ROE & Stochastic Cash Float
    print("\n[5/8] Calculating DuPont 3-Tier ROE Decomposition & Working Capital Float...")
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

    # 6. Palantir Enterprise Decision Ontology
    print("\n[6/8] Synthesizing Palantir Enterprise Decision Ontology Telemetry...")
    ontology = EnterpriseDecisionOntology(clean_df)
    telemetry = ontology.evaluate_live_telemetry()
    telemetry_dict = {
        "customer_churn_risk_level": telemetry.customer_churn_risk_level,
        "price_elasticity_fragility": telemetry.price_elasticity_fragility,
        "channel_commons_depletion": telemetry.channel_commons_depletion,
        "working_capital_health": telemetry.working_capital_health,
        "recommended_board_action": telemetry.recommended_board_action,
    }
    with open(out_dir / "superstore_ontology_telemetry.json", "w") as f:
        json.dump(telemetry_dict, f, indent=2)
    print(f"      Board Action Trigger: {telemetry.recommended_board_action}")

    # 7. Palantir Counterfactual War-Gaming Simulator
    print("\n[7/8] Simulating Macro Shock Scenario Blast Radius...")
    sandbox = ScenarioSandboxEngine(clean_df, base_annual_revenue=total_rev, base_annual_cogs=clean_df["cogs"].sum())
    shocks = ScenarioShockParameters(
        competitor_price_cut_pct=-0.15,
        cogs_inflation_surge_pct=0.20,
        supplier_dpo_compression_days=-15.0,
    )
    blast_radius = sandbox.simulate_blast_radius(shocks)
    pd.DataFrame([blast_radius]).to_csv(out_dir / "superstore_war_gaming_blast_radius.csv", index=False)
    print(f"      P&L Margin Blast Radius: ${blast_radius['margin_delta']:,.2f} ({blast_radius['margin_delta_pct']}%)")
    print(f"      Remediation Protocol: {blast_radius['remediation_protocol']}")

    # 8. Synthetic Control Regional Pilot Lift (Aggregated by Quarter across 16 Quarters)
    print("\n[8/8] Evaluating Synthetic Control Causal Lift on Regional Expansion...")
    if "Region" in clean_df.columns:
        clean_df["order_quarter"] = pd.to_datetime(clean_df["order_date"]).dt.to_period("Q").astype(str)
        region_panel = clean_df.groupby(["order_quarter", "Region"])["gross_revenue"].sum().reset_index()
        quarters = sorted(region_panel["order_quarter"].unique().tolist())
        if len(quarters) >= 8:
            sc_engine = SyntheticControlEngine()
            split_quarter = quarters[len(quarters) // 2]  # Split at middle quarter
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
            print(f"      Synthetic Control Quasi-Experiment Lift for {treated_reg}: ${sc_res['post_intervention_avg_lift']:,.2f}/quarter")
            print(f"      Pre-Intervention Fit RMSE: ${sc_res['pre_intervention_rmse']:,.2f}")

    # 9. Markowitz Portfolio Optimization across Departments
    print("\n[Bonus] Markowitz Mean-Variance Capital Allocation across Product Departments...")
    if "Department" in clean_df.columns:
        dept_panel = clean_df.groupby([pd.to_datetime(clean_df["order_date"]).dt.to_period("M"), "Department"])["gross_revenue"].sum().unstack().fillna(0.0)
        dept_roas = dept_panel.pct_change().dropna() + 1.0  # Revenue velocity ratio
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
