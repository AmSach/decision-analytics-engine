"""
DuPont 3-Tier ROE Decomposition and Stochastic Working Capital Float Engine.
Decomposes Return on Equity into Margin, Velocity, and Leverage.
Simulates Cash Conversion Cycle (CCC) dynamics under negative working capital regimes,
quantifying risk-free treasury yields generated from customer cash float.
"""

import numpy as np
import pandas as pd


def compute_dupont_3tier_decomposition(
    revenue: float,
    net_income: float,
    total_assets: float,
    shareholder_equity: float,
) -> dict:
    """
    Decomposes Return on Equity (ROE):
    ROE = Profit Margin * Asset Turnover * Financial Leverage
    """
    profit_margin = net_income / revenue if revenue > 0 else 0.0
    asset_turnover = revenue / total_assets if total_assets > 0 else 0.0
    equity_multiplier = total_assets / shareholder_equity if shareholder_equity > 0 else 0.0
    roe = profit_margin * asset_turnover * equity_multiplier

    return {
        "revenue": round(revenue, 2),
        "net_income": round(net_income, 2),
        "total_assets": round(total_assets, 2),
        "shareholder_equity": round(shareholder_equity, 2),
        "profit_margin_pct": round(profit_margin * 100.0, 2),
        "asset_turnover_x": round(asset_turnover, 2),
        "equity_multiplier_x": round(equity_multiplier, 2),
        "return_on_equity_pct": round(roe * 100.0, 2),
    }


def simulate_negative_working_capital_float(
    annual_revenue: float = 5000000.0,
    dio_days: float = 14.0,  # Days Sales of Inventory
    dso_days: float = 2.0,   # Days Sales Outstanding (Immediate Card/UPI checkout)
    dpo_days: float = 60.0,  # Days Payables Outstanding (Supplier terms)
    treasury_yield_rate: float = 0.065,  # 6.5% money market rate
    daily_volatility: float = 0.08,
    sim_days: int = 365,
    random_seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """
    Cash Conversion Cycle (CCC):
    CCC = DIO + DSO - DPO
    When CCC < 0, the enterprise operates on negative working capital float.
    Simulates stochastic daily float path via geometric Brownian motion with mean reversion.
    """
    rng = np.random.default_rng(random_seed)
    ccc_days = dio_days + dso_days - dpo_days
    daily_rev = annual_revenue / 365.0

    # Deterministic base float
    base_float = abs(ccc_days) * daily_rev if ccc_days < 0 else 0.0
    annual_treasury_yield = base_float * treasury_yield_rate

    # Stochastic daily path simulation
    dt = 1.0 / 365.0
    daily_floats = [base_float]
    daily_yields = [0.0]

    for d in range(1, sim_days):
        prev = daily_floats[-1]
        # Mean-reverting Ornstein-Uhlenbeck stochastic drift toward base_float
        kappa = 0.15  # Speed of mean reversion
        drift = kappa * (base_float - prev) * dt
        shock = daily_volatility * prev * np.sqrt(dt) * rng.normal()
        curr_float = max(0.0, prev + drift + shock)
        daily_floats.append(curr_float)
        daily_yields.append(curr_float * (treasury_yield_rate / 365.0))

    sim_df = pd.DataFrame({
        "day": np.arange(1, sim_days + 1),
        "daily_float_balance": np.round(daily_floats, 2),
        "cumulative_treasury_yield": np.round(np.cumsum(daily_yields), 2),
    })

    summary = {
        "annual_revenue": round(annual_revenue, 2),
        "dio_days": round(dio_days, 1),
        "dso_days": round(dso_days, 1),
        "dpo_days": round(dpo_days, 1),
        "cash_conversion_cycle_days": round(ccc_days, 1),
        "is_negative_working_capital": bool(ccc_days < 0),
        "average_liquid_float_balance": round(float(np.mean(daily_floats)), 2),
        "total_annual_treasury_yield": round(float(sim_df["cumulative_treasury_yield"].iloc[-1]), 2),
        "float_funded_cac_equivalent": int(float(sim_df["cumulative_treasury_yield"].iloc[-1]) / 25.0),
    }

    return summary, sim_df
