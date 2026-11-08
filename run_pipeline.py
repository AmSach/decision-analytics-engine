"""
End-to-End Decision Analytics Pipeline Runner.
Executes data generation, cohort retention, econometric elasticity,
Double Machine Learning (DML), Level-k game-theoretic pricing,
Ostrom Commons governance, Spectral graph attribution, and DuPont float simulations.
"""

from pathlib import Path
import pandas as pd

from src.data_generator import generate_ecommerce_dataset
from src.cohort_retention import compute_cohort_retention, compute_churn_hazard
from src.price_elasticity import estimate_price_elasticity
from src.causal_elasticity_dml import estimate_causal_elasticity_dml
from src.adversarial_game_pricing import AdversarialPricingEngine
from src.ostrom_commons_governance import OstromCommonsGovernanceEngine
from src.spectral_attribution import compute_spectral_graph_attribution
from src.stochastic_float_engine import compute_dupont_3tier_decomposition, simulate_negative_working_capital_float
from src.clv_model import compute_rfm_metrics, get_segment_waterfall_summary
from src.marketing_attribution import compute_markov_attribution


def run_full_pipeline(output_dir: str = "data") -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXECUTING ADVANCED B-SCHOOL DECISION ANALYTICS PIPELINE")
    print("=" * 70)

    # 1. Dataset Generation
    print("\n[1/8] Generating multi-channel transaction and journey dataset...")
    tx_df, journeys_df = generate_ecommerce_dataset(num_customers=1500, num_days=365, random_seed=42)
    tx_df.to_csv(out_path / "transactions.csv", index=False)
    journeys_df.to_csv(out_path / "journeys.csv", index=False)
    print(f"      Exported {len(tx_df)} orders and {len(journeys_df)} customer conversion journeys.")

    # 2. Cohort Retention & Churn Hazard
    print("\n[2/8] Computing monthly cohort retention & Kaplan-Meier churn hazard...")
    user_ret, rev_ret = compute_cohort_retention(tx_df)
    hazard_df = compute_churn_hazard(user_ret)
    user_ret.to_csv(out_path / "cohort_user_retention.csv")
    hazard_df.to_csv(out_path / "churn_hazard.csv", index=False)
    print("      Cohort Retention Matrix (Top 4 Cohorts, Months 0-5):")
    print(user_ret.iloc[:4, :6])

    # 3. Double Machine Learning (DML) Causal Elasticity
    print("\n[3/8] Estimating unconfounded causal elasticity via Double Machine Learning...")
    dml_summary = estimate_causal_elasticity_dml(tx_df, n_splits=5, random_seed=42)
    dml_df = pd.DataFrame([dml_summary])
    dml_df.to_csv(out_path / "causal_elasticity_dml.csv", index=False)
    print(f"      DML Causal Elasticity: {dml_summary['causal_elasticity_dml']} (SE: {dml_summary['standard_error']}, P-val: {dml_summary['p_value']})")
    print(f"      Naive OLS Elasticity: {dml_summary['naive_ols_elasticity']} (Endogeneity Bias Purged: {dml_summary['endogeneity_bias_delta']})")

    # 4. Adversarial Game Theory & Quantal Response Equilibrium (QRE)
    print("\n[4/8] Solving Level-k Cognitive Hierarchy & Logit Quantal Response Equilibrium...")
    adv_engine = AdversarialPricingEngine(our_cost=38.0, rival_cost=42.0)
    level_k = adv_engine.compute_level_k_strategy()
    _, _, qre_our, qre_rival = adv_engine.solve_quantal_response_equilibrium()
    adv_records = [
        {"strategy": "Level-1 Best Response", "our_price": level_k["Level_1_Our_Price"], "expected_profit": level_k["Level_1_Expected_Profit"]},
        {"strategy": "Level-3 Counter-Flank", "our_price": level_k["Level_3_Counter_Flank_Price"], "expected_profit": level_k["Level_3_Counter_Flank_Profit"]},
        {"strategy": "Quantal Response Equilibrium (QRE)", "our_price": qre_our, "expected_profit": "Fixed Point Nash/QRE"},
    ]
    pd.DataFrame(adv_records).to_csv(out_path / "adversarial_pricing_qre.csv", index=False)
    print(f"      Level-1 Our Price: ${level_k['Level_1_Our_Price']} | Level-3 Counter-Flank: ${level_k['Level_3_Counter_Flank_Price']}")
    print(f"      Quantal Response Equilibrium (QRE) Price: ${qre_our} vs Rival ${qre_rival}")

    # 5. Elinor Ostrom Commons Governance Simulation
    print("\n[5/8] Simulating Elinor Ostrom Commons Governance vs Unregulated Bidding War...")
    ostrom_engine = OstromCommonsGovernanceEngine()
    ostrom_df = ostrom_engine.simulate_bidding_war_defection_spiral(rounds=8)
    ostrom_df.to_csv(out_path / "ostrom_commons_governance.csv", index=False)
    print("      Ostrom Governance Results (Snapshot):")
    print(ostrom_df[["round", "unreg_market_cac", "ostrom_blended_cac", "cac_savings_pct", "ostrom_status"]].tail(3))

    # 6. Spectral Graph Diffusion & Higher-Order Attribution
    print("\n[6/8] Evaluating Higher-Order Spectral Graph Diffusion Attribution...")
    spectral_df, topo_df = compute_spectral_graph_attribution(journeys_df)
    spectral_df.to_csv(out_path / "spectral_attribution.csv", index=False)
    topo_df.to_csv(out_path / "graph_topology_summary.csv", index=False)
    print("      Spectral Graph Attribution Shares:")
    for _, row in spectral_df.iterrows():
        print(f"      {row['channel']}: Blended Spectral Share = {row['blended_spectral_share_pct']}% | Rev = ${row['spectral_attributed_revenue']:,.2f}")

    # 7. DuPont 3-Tier & Stochastic Negative Working Capital Float
    print("\n[7/8] Computing DuPont ROE Decomposition & Stochastic Float Engine...")
    dupont = compute_dupont_3tier_decomposition(
        revenue=tx_df["gross_revenue"].sum(),
        net_income=tx_df["gross_margin"].sum() * 0.40,
        total_assets=tx_df["gross_revenue"].sum() * 0.45,
        shareholder_equity=tx_df["gross_revenue"].sum() * 0.22,
    )
    pd.DataFrame([dupont]).to_csv(out_path / "dupont_roe_summary.csv", index=False)
    float_summary, _ = simulate_negative_working_capital_float(annual_revenue=tx_df["gross_revenue"].sum())
    pd.DataFrame([float_summary]).to_csv(out_path / "working_capital_float_summary.csv", index=False)
    print(f"      DuPont ROE: {dupont['return_on_equity_pct']}% (Margin: {dupont['profit_margin_pct']}% * Turnover: {dupont['asset_turnover_x']}x * Leverage: {dupont['equity_multiplier_x']}x)")
    print(f"      Negative CCC: {float_summary['cash_conversion_cycle_days']} days | Annual Treasury Yield Float: ${float_summary['total_annual_treasury_yield']:,.2f}")

    # 8. RFM CLV Segmentation
    print("\n[8/8] Evaluating Customer Lifetime Value (CLV) Equity...")
    rfm_df = compute_rfm_metrics(tx_df)
    rfm_summary = get_segment_waterfall_summary(rfm_df)
    rfm_summary.to_csv(out_path / "clv_segment_summary.csv", index=False)
    print("      12M Customer Equity Waterfall:")
    for _, row in rfm_summary.iterrows():
        print(f"      {row['segment']}: {row['customer_count']} accounts -> Forward Equity: ${row['forward_12m_expected_margin']:,.2f}")

    print("\n" + "=" * 70)
    print(f"EXECUTION COMPLETE. All strategic models exported cleanly to ./{output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
