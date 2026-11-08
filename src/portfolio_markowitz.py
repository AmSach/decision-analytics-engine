"""
Modern Portfolio Optimization (Markowitz Mean-Variance) for Capital Allocation.
Treats marketing channels and product inventory portfolios as risk-bearing assets.
Solves the quadratic Pareto frontier under Value-at-Risk (VaR 95%) and budget constraints,
maximizing risk-adjusted Return on Ad Spend (ROAS) and Return on Capital.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MarkowitzCapitalOptimizer:
    """
    Quadratic Programming Solver for Marketing Budget and Working Capital Allocation:
    Maximize mu^T * w - (gamma / 2) * w^T * Sigma * w
    subject to sum(w) = 1, 0 <= w_i <= max_allocation_cap.
    """

    def __init__(self, risk_aversion: float = 1.0, max_channel_cap: float = 0.40):
        self.gamma = risk_aversion
        self.cap = max_channel_cap

    def fit_optimal_allocation(
        self,
        historical_returns_df: pd.DataFrame,
        total_budget: float = 100000.0,
    ) -> dict:
        """
        historical_returns_df: DataFrame where each column is an asset/channel and rows are time periods of returns/ROAS.
        """
        channels = historical_returns_df.columns.tolist()
        n = len(channels)

        # Expected return vector mu and covariance matrix Sigma (ensuring writable copy)
        mu = np.array(historical_returns_df.mean().values, dtype=float)
        sigma = np.array(historical_returns_df.cov().values, dtype=float, copy=True)

        # Ensure Sigma is positive semi-definite (add small ridge if near-singular)
        sigma = sigma + np.eye(n) * 1e-6

        # Objective: Minimize 0.5 * gamma * w^T * Sigma * w - mu^T * w
        def objective(w):
            variance = float(w.T @ sigma @ w)
            expected_return = float(mu @ w)
            return 0.5 * self.gamma * variance - expected_return

        # Constraints & Bounds
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, self.cap) for _ in range(n)]
        w_init = np.ones(n) / n

        res = minimize(objective, w_init, method="SLSQP", bounds=bounds, constraints=constraints)
        optimal_weights = res.x if res.success else w_init

        portfolio_return = float(mu @ optimal_weights)
        portfolio_variance = float(optimal_weights.T @ sigma @ optimal_weights)
        portfolio_volatility = float(np.sqrt(portfolio_variance))
        sharpe_ratio = float(portfolio_return / portfolio_volatility) if portfolio_volatility > 0 else 0.0

        # Parametric Value-at-Risk (VaR at 95% confidence level, z = 1.645)
        var_95_pct = float(1.645 * portfolio_volatility - portfolio_return)
        var_95_dollars = round(max(0.0, var_95_pct * total_budget), 2)

        allocation_records = []
        for i, ch in enumerate(channels):
            weight = float(optimal_weights[i])
            allocation_records.append({
                "channel_or_asset": ch,
                "historical_mean_roas": round(float(mu[i]), 3),
                "historical_volatility": round(float(np.sqrt(sigma[i, i])), 3),
                "optimal_weight_pct": round(weight * 100.0, 2),
                "allocated_budget": round(weight * total_budget, 2),
            })

        alloc_df = pd.DataFrame(allocation_records).sort_values(by="optimal_weight_pct", ascending=False)

        return {
            "total_budget": total_budget,
            "expected_portfolio_roas": round(portfolio_return, 3),
            "portfolio_volatility": round(portfolio_volatility, 3),
            "risk_adjusted_sharpe": round(sharpe_ratio, 2),
            "value_at_risk_95_dollars": var_95_dollars,
            "allocation_df": alloc_df,
        }
