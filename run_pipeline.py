"""
End-to-End Decision Analytics Pipeline Runner.
Executes data generation, cohort retention, econometric elasticity,
Double Machine Learning (DML), Level-k game pricing, Ostrom Commons governance,
Spectral graph attribution, DuPont float, Synthetic Control, and Markowitz portfolio allocation.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.data_generator import generate_ecommerce_dataset
from src.cohort_retention import compute_cohort_retention, compute_churn_hazard
from src.price_elasticity import estimate_price_elasticity
from src.causal_elasticity_dml import estimate_causal_elasticity_dml
from src.adversarial_game_pricing import AdversarialPricingEngine
from src.ostrom_commons_governance import OstromCommonsGovernanceEngine
from src.spectral_attribution import compute_spectral_graph_attribution
from src.stochastic_float_engine import compute_dupont_3tier_decomposition, simulate_negative_working_capital_float
from src.synthetic_control import SyntheticControlEngine
from src.portfolio_markowitz import MarkowitzCapitalOptimizer
from src.clv_model import compute_rfm_metrics, get_segment_waterfall_summary
from src.marketing_attribution import compute_markov_attribution


def run_full_pipeline(output_dir: str = "data") -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("EXECUTING ADVANCED B-SCHOOL DECISION ANALYTICS PIPELINE (FULL SUITE)")
    print("=" * 75)

    # 1. Dataset Generation
    print("\n[1/10] Generating multi-channel transaction and journey dataset...")
    tx_df, journeys_df = generate_ecommerce_dataset(num_customers=1500, num_days=365, random_seed=42)
    tx_df.to_csv(out_path / "transactions.csv", index=False)
    journeys_df.to_csv(out_path / "journeys.csv", index=False)
    print(f"       Exported {len(tx_df)} orders and {len(journeys_df)} customer conversion journeys.")

    # 2. Cohort Retention & Churn Hazard
    print("\n[2/10] Computing monthly cohort retention & Kaplan-Meier churn hazard...")
    user_ret, rev_ret = compute_cohort_retention(tx_df)
    hazard_df = compute_churn_hazard(user_ret)
    user_ret.to_csv(out_path / "cohort_user_retention.csv")
    hazard_df.to_csv(out_path / "churn_hazard.csv", index=False)
    print("       Cohort Retention Matrix (Snapshot):")
    print(user_ret.iloc[:3, :5])

    # 3. Double Machine Learning (DML) Causal Elasticity
    print("\n[3/10] Estimating unconfounded causal elasticity via Double Machine Learning...")
    dml_summary = estimate_causal_elasticity_dml(tx_df, n_splits=5, random_seed=42)
    pd.DataFrame([dml_summary]).to_csv(out_path / "causal_elasticity_dml.csv", index=False)
    print(f"       DML Causal Elasticity: {dml_summary['causal_elasticity_dml']} (SE: {dml_summary['standard_error']}, P-val: {dml_summary['p_value']})")

    # 4. Adversarial Game Theory & Quantal Response Equilibrium (QRE)
    print("\n[4/10] Solving Level-k Cognitive Hierarchy & Logit Quantal Response Equilibrium...")
    adv_engine = AdversarialPricingEngine(our_cost=38.0, rival_cost=42.0)
    level_k = adv_engine.compute_level_k_strategy()
    _, _, qre_our, qre_rival = adv_engine.solve_quantal_response_equilibrium()
    adv_records = [
        {"strategy": "Level-1 Best Response", "our_price": level_k["Level_1_Our_Price"], "expected_profit": level_k["Level_1_Expected_Profit"]},
        {"strategy": "Level-3 Counter-Flank", "our_price": level_k["Level_3_Counter_Flank_Price"], "expected_profit": level_k["Level_3_Counter_Flank_Profit"]},
        {"strategy": "Quantal Response Equilibrium (QRE)", "our_price": qre_our, "expected_profit": "Fixed Point Nash/QRE"},
    ]
    pd.DataFrame(adv_records).to_csv(out_path / "adversarial_pricing_qre.csv", index=False)
    print(f"       QRE Nash Equilibrium Price: ${qre_our} vs Rival ${qre_rival}")

    # 5. Elinor Ostrom Commons Governance Simulation
    print("\n[5/10] Simulating Elinor Ostrom Commons Governance vs Unregulated Bidding War...")
    ostrom_engine = OstromCommonsGovernanceEngine()
    ostrom_df = ostrom_engine.simulate_bidding_war_defection_spiral(rounds=8)
    ostrom_df.to_csv(out_path / "ostrom_commons_governance.csv", index=False)
    print(f"       Ostrom Kill-Gate Savings in Round 8: {ostrom_df['cac_savings_pct'].iloc[-1]}% CAC Reduction")

    # 6. Spectral Graph Diffusion & Higher-Order Attribution
    print("\n[6/10] Evaluating Higher-Order Spectral Graph Diffusion Attribution...")
    spectral_df, topo_df = compute_spectral_graph_attribution(journeys_df)
    spectral_df.to_csv(out_path / "spectral_attribution.csv", index=False)
    print("       Top Channel by Perron-Frobenius Centrality: " + str(spectral_df.iloc[0]["channel"]))

    # 7. Synthetic Control Quasi-Experiment
    print("\n[7/10] Constructing Abadie Synthetic Control for Promotion Lift...")
    # Simulate a 12-month regional promotional intervention
    regions = ["North_Region_Treated", "South_Donor", "East_Donor", "West_Donor"]
    panel_records = []
    for m in range(1, 13):
        for r in regions:
            base = 50000 + m * 2000
            lift = 12000 if (r == "North_Region_Treated" and m >= 8) else 0
            panel_records.append({"month": m, "region": r, "sales": base + lift + np.random.normal(0, 500)})
    sc_engine = SyntheticControlEngine()
    sc_res = sc_engine.fit_synthetic_control(
        panel_df=pd.DataFrame(panel_records),
        unit_col="region",
        time_col="month",
        outcome_col="sales",
        treated_unit="North_Region_Treated",
        intervention_time=8,
    )
    sc_res["trajectory_df"].to_csv(out_path / "synthetic_control_lift.csv", index=False)
    print(f"       Synthetic Control Estimated Lift: +${sc_res['post_intervention_avg_lift']:,.2f}/month (Pre-RMSE: ${sc_res['pre_intervention_rmse']:,.2f})")

    # 8. Markowitz Capital Portfolio Optimization
    print("\n[8/10] Solving Quadratic Markowitz Frontier for Ad Spend Allocation...")
    rng = np.random.default_rng(42)
    weekly_roas = pd.DataFrame({
        "Paid_Search": rng.normal(3.8, 0.4, 52),
        "Social_Ads": rng.normal(2.9, 0.5, 52),
        "Email": rng.normal(4.5, 0.3, 52),
        "Organic_SEO": rng.normal(3.2, 0.2, 52),
    })
    markowitz = MarkowitzCapitalOptimizer(risk_aversion=1.0, max_channel_cap=0.40)
    mark_res = markowitz.fit_optimal_allocation(weekly_roas, total_budget=200000.0)
    mark_res["allocation_df"].to_csv(out_path / "markowitz_ad_allocation.csv", index=False)
    print(f"       Optimal Portfolio ROAS: {mark_res['expected_portfolio_roas']}x | Sharpe: {mark_res['risk_adjusted_sharpe']} | 95% VaR: ${mark_res['value_at_risk_95_dollars']:,.2f}")

    # 9. DuPont 3-Tier & Stochastic Negative Working Capital Float
    print("\n[9/10] Computing DuPont ROE Decomposition & Negative Working Capital Float...")
    dupont = compute_dupont_3tier_decomposition(
        revenue=tx_df["gross_revenue"].sum(),
        net_income=tx_df["gross_margin"].sum() * 0.40,
        total_assets=tx_df["gross_revenue"].sum() * 0.45,
        shareholder_equity=tx_df["gross_revenue"].sum() * 0.22,
    )
    pd.DataFrame([dupont]).to_csv(out_path / "dupont_roe_summary.csv", index=False)
    float_summary, _ = simulate_negative_working_capital_float(annual_revenue=tx_df["gross_revenue"].sum())
    pd.DataFrame([float_summary]).to_csv(out_path / "working_capital_float_summary.csv", index=False)
    print(f"       DuPont ROE: {dupont['return_on_equity_pct']}% | Negative CCC Float Yield: ${float_summary['total_annual_treasury_yield']:,.2f}")

    # 10. RFM CLV Segmentation
    print("\n[10/10] Evaluating Customer Lifetime Value (CLV) Equity...")
    rfm_df = compute_rfm_metrics(tx_df)
    rfm_summary = get_segment_waterfall_summary(rfm_df)
    rfm_summary.to_csv(out_path / "clv_segment_summary.csv", index=False)
    print("        12M Customer Equity Waterfall:")
    for _, row in rfm_summary.iterrows():
        print(f"        {row['segment']}: {row['customer_count']} accounts -> Forward Equity: ${row['forward_12m_expected_margin']:,.2f}")

    print("\n" + "=" * 75)
    print(f"EXECUTION COMPLETE. All 10 strategic modules exported cleanly to ./{output_dir}/")
    print("=" * 75)


if __name__ == "__main__":
    run_full_pipeline()
