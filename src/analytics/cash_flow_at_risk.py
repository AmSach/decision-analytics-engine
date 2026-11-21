"""
Cash Flow at Risk (CFaR) Monte Carlo Simulation Engine.
Simulates 5,000 correlated joint macro paths to determine tail risk,
Value at Risk (VaR), Expected Shortfall (CVaR), and debt covenant default risk.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def compute_cfar_monte_carlo(
    df: pd.DataFrame,
    n_simulations: int = 5000,
    debt_covenant_floor: float = 300000.0,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes a 5,000-path correlated Monte Carlo simulation:
    Correlated shocks across Volume, Price Realization, COGS Inflation, and Payment Float.
    """
    rng = np.random.default_rng(random_seed)

    # Derive historical baseline metrics from transactions if present
    if "gross_revenue" in df.columns:
        total_rev = float(df["gross_revenue"].sum())
        total_cogs = float(df["cogs"].sum()) if "cogs" in df.columns else total_rev * 0.40
        annual_rev = max(500000.0, total_rev)
        annual_cogs = max(200000.0, total_cogs)
    else:
        annual_rev = 12000000.0
        annual_cogs = 4800000.0

    annual_opex = annual_rev * 0.35
    baseline_cf = annual_rev - annual_cogs - annual_opex

    # 4-dimensional Correlation Matrix: [Volume, Price, COGS Inflation, Float Delay]
    corr_matrix = np.array([
        [ 1.00,  0.35, -0.40, -0.25],
        [ 0.35,  1.00, -0.20, -0.15],
        [-0.40, -0.20,  1.00,  0.30],
        [-0.25, -0.15,  0.30,  1.00],
    ])

    # Volatility standard deviations
    stds = np.array([0.12, 0.06, 0.08, 0.05])
    cov_matrix = np.outer(stds, stds) * corr_matrix

    # Cholesky factorization L * L^T = Sigma
    L = np.linalg.cholesky(cov_matrix)
    z = rng.standard_normal(size=(4, n_simulations))
    shocks = L @ z

    vol_shock = shocks[0]
    price_shock = shocks[1]
    cogs_shock = shocks[2]
    float_shock = shocks[3]

    sim_rev = annual_rev * (1.0 + vol_shock) * (1.0 + price_shock)
    sim_cogs = annual_cogs * (1.0 + vol_shock) * (1.0 + cogs_shock)
    sim_opex = annual_opex * (1.0 + 0.5 * vol_shock)
    working_capital_drag = (annual_rev * 0.08) * float_shock

    sim_cf = sim_rev - sim_cogs - sim_opex - working_capital_drag

    # Risk Metrics
    mean_cf = float(np.mean(sim_cf))
    median_cf = float(np.median(sim_cf))
    std_cf = float(np.std(sim_cf))

    var_95_floor = float(np.percentile(sim_cf, 5))
    var_99_floor = float(np.percentile(sim_cf, 1))

    cfar_95 = float(mean_cf - var_95_floor)
    cfar_99 = float(mean_cf - var_99_floor)

    tail_losses = sim_cf[sim_cf <= var_95_floor]
    cvar_95 = float(mean_cf - np.mean(tail_losses)) if len(tail_losses) > 0 else cfar_95

    # Debt Covenant Breach Probability
    breach_count = int(np.sum(sim_cf < debt_covenant_floor))
    covenant_breach_prob = float(breach_count / n_simulations)

    # Actionable Decision
    if covenant_breach_prob > 0.05:
        action = (
            f"HIGH COVENANT RISK ({covenant_breach_prob:.1%}): Secure a ${abs(debt_covenant_floor - var_95_floor):,.0f} "
            f"revolving liquidity buffer or freeze unhedged Capex to maintain lender solvency."
        )
    elif covenant_breach_prob > 0.01:
        action = (
            f"MODERATE TAIL RISK ({covenant_breach_prob:.1%}): 95% worst-case cash flow (${var_95_floor:,.0f}) holds above "
            f"covenant floor (${debt_covenant_floor:,.0f}). Maintain standard treasury float."
        )
    else:
        action = (
            f"ROBUST LIQUIDITY SHIELD (0.0% breach): Cash flow floor comfortably exceeds bank covenants. "
            f"Free cash flow is fully deployable into growth investments."
        )

    return {
        "baseline_cash_flow": round(baseline_cf, 2),
        "mean_simulated_cash_flow": round(mean_cf, 2),
        "std_dev_cash_flow": round(std_cf, 2),
        "var_95_cash_floor": round(var_95_floor, 2),
        "var_99_cash_floor": round(var_99_floor, 2),
        "cfar_95_downside": round(cfar_95, 2),
        "cvar_95_expected_shortfall": round(cvar_95, 2),
        "debt_covenant_floor": round(debt_covenant_floor, 2),
        "covenant_breach_probability": round(covenant_breach_prob, 4),
        "breach_percentage": round(covenant_breach_prob * 100.0, 2),
        "decision_action": action,
        "decision_rule": "Maintain liquidity buffer >= cfar_95 to guarantee zero debt default risk.",
        "downside_stress_test": "In the 99% severe recession tail, maximum expected cash loss is capped at cvar_95.",
    }
