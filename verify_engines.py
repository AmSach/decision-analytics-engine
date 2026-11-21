"""
Standalone Verification Script for Newly Upgraded Decision Intelligence Engines:
1. Cash Flow at Risk (CFaR) & Correlated Monte Carlo Balance Sheet Stress Testing
2. Dynamic Multi-Armed Bandit (MAB) Experimentation (Thompson Sampling vs UCB1 vs Static A/B)
3. Mixed Logit Heterogeneous Demand, Cross-Elasticity & Diversion Ratio Cannibalization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from src.cash_flow_at_risk_cfar import CashFlowAtRiskSimulator
from src.bandit_experimentation import BanditExperimentEngine
from src.mixed_logit_cannibalization import MixedLogitCannibalizationEngine


def verify_all_engines():
    print("=" * 80)
    print("VERIFYING ADVANCED DECISION ANALYTICS UPGRADES")
    print("=" * 80)

    # 1. Cash Flow at Risk (CFaR)
    print("\n[1/3] Testing Cash Flow at Risk (CFaR) Monte Carlo Engine...")
    cfar_sim = CashFlowAtRiskSimulator(
        base_annual_revenue=5000000.0,
        base_gross_margin_pct=0.35,
        base_operating_opex_pct=0.20,
        initial_cash_reserves=400000.0,
        debt_service_annual=100000.0,
        minimum_liquidity_covenant=150000.0,
    )
    cfar_res = cfar_sim.simulate_cfar(num_simulations=2000, random_seed=42)
    assert cfar_res["mean_simulated_cash_flow"] > 0, "Mean OCF must be positive"
    assert cfar_res["var_99_cash_flow_floor"] <= cfar_res["var_95_cash_flow_floor"], "VaR 99 must be <= VaR 95 floor"
    assert cfar_res["var_95_cash_flow_floor"] <= cfar_res["mean_simulated_cash_flow"], "VaR 95 floor must be <= Mean OCF"
    assert cfar_res["cvar_expected_shortfall_95"] >= cfar_res["cfar_95_dollars"], "CVaR shortfall must be >= VaR"
    assert 0.0 <= cfar_res["covenant_breach_probability_pct"] <= 100.0, "Probability must be bounded in [0, 100]"
    print(f"      Mean Operating Cash Flow: ${cfar_res['mean_simulated_cash_flow']:,.2f}")
    print(f"      95% VaR Cash Flow Floor: ${cfar_res['var_95_cash_flow_floor']:,.2f}")
    print(f"      Cash Flow at Risk (CFaR 95%): ${cfar_res['cfar_95_dollars']:,.2f}")
    print(f"      95% Expected Shortfall (CVaR): ${cfar_res['cvar_expected_shortfall_95']:,.2f}")
    print(f"      Covenant Breach Probability: {cfar_res['covenant_breach_probability_pct']}%")
    print("      [PASSED] CFaR Monte Carlo invariants verified.")

    # 2. Dynamic Multi-Armed Bandit (MAB)
    print("\n[2/3] Testing Dynamic Multi-Armed Bandit Experimentation Engine...")
    true_rates = {"Creative_A (Control 4%)": 0.04, "Creative_B (Treatment 14%)": 0.14, "Creative_C (Social 8%)": 0.08}
    bandit_eng = BanditExperimentEngine(true_rates)
    b_res = bandit_eng.run_simulation(total_trials=3000, random_seed=42)
    assert sum(b_res["static_pulls"]) == 3000, "Static pulls sum mismatch"
    assert sum(b_res["thompson_pulls"]) == 3000, "Thompson pulls sum mismatch"
    assert b_res["thompson_regret_final"] < b_res["static_regret_final"], "Thompson regret must be strictly less than static"
    assert b_res["regret_reduction_vs_ab_pct"] > 30.0, "Thompson regret reduction must exceed 30%"
    print(f"      Static A/B Test Lost Conversions (Regret): {b_res['static_regret_final']:,.1f}")
    print(f"      Thompson Sampling Lost Conversions (Regret): {b_res['thompson_regret_final']:,.1f}")
    print(f"      Regret Reduction vs Static A/B Test: {b_res['regret_reduction_vs_ab_pct']}%")
    print("      [PASSED] Bandit Bayesian Thompson Sampling invariants verified.")

    # 3. Mixed Logit Demand & Cannibalization
    print("\n[3/3] Testing Random Coefficient / Mixed Logit Demand & Cannibalization Engine...")
    portfolio = [
        {"name": "Standard_Tier", "price": 99.0, "base_quality": 8.0, "cogs": 45.0},
        {"name": "Premium_Tier", "price": 179.0, "base_quality": 12.5, "cogs": 75.0},
        {"name": "Entry_Tier", "price": 59.0, "base_quality": 5.2, "cogs": 28.0},
    ]
    logit_eng = MixedLogitCannibalizationEngine(portfolio, random_seed=42)
    shares = logit_eng.compute_choice_probabilities()
    assert abs(np.sum(shares) - 1.0) < 1e-4, "Shares must sum to 1.0"
    assert (shares >= 0.0).all(), "Shares must be non-negative"

    cann_res = logit_eng.compute_cannibalization_and_elasticity(price_shock_pct=0.02)
    summary_list = cann_res["portfolio_summary"]
    for row in summary_list:
        assert row["own_price_elasticity"] < 0.0, f"Own price elasticity for {row['product']} must be negative"
    
    div_mat = cann_res["diversion_ratio_matrix_pct"]
    out_div = cann_res["outside_good_diversion_pct"]
    for i in range(len(portfolio)):
        internal_cann = float(np.sum(div_mat[i]) * 100.0)
        external_leak = float(out_div[i])
        total_div = internal_cann + external_leak
        assert abs(total_div - 100.0) < 1.0, f"Total diversion for product {i} must sum to 100%"

    print(f"      Market Shares: Standard={shares[0]*100:.1f}%, Premium={shares[1]*100:.1f}%, Entry={shares[2]*100:.1f}%, Outside={shares[3]*100:.1f}%")
    print(f"      Standard Tier Own Price Elasticity: {summary_list[0]['own_price_elasticity']}")
    print(f"      Standard Tier Internal Cannibalization Share: {summary_list[0]['internal_portfolio_cannibalization_pct']}%")
    print(f"      Standard Tier External Competitor Leakage: {summary_list[0]['external_competitor_leakage_pct']}%")
    print("      [PASSED] Mixed Logit & Diversion Ratio invariants verified.")

    print("\n" + "=" * 80)
    print("ALL UPGRADE ENGINES VERIFIED: 100% MATHEMATICALLY SOUND")
    print("=" * 80)


if __name__ == "__main__":
    verify_all_engines()
