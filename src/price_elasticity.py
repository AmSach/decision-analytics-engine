"""
Econometric Price Elasticity and Demand Optimization Module.
Fits log-linear demand curves: ln(Q) = beta_0 + beta_p * ln(P) + epsilon.
Derives point price elasticity (beta), evaluates discount sensitivity,
and calculates gross-margin-maximizing price points using pure NumPy & SciPy.
"""

import numpy as np
import pandas as pd
from scipy import stats


def fit_ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Fits Ordinary Least Squares (OLS) via QR decomposition / normal equations:
    Returns (beta, standard_errors, p_values, r_squared).
    """
    n, k = X.shape
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    y_pred = X @ beta
    e = y - y_pred
    rss = np.sum(e ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    r_squared = float(1.0 - (rss / tss)) if tss > 0 else 0.0

    df_resid = max(1, n - k)
    s2 = rss / df_resid
    try:
        inv_xtx = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diagonal(s2 * inv_xtx))
        t_stat = np.where(se > 0, beta / se, 0.0)
        p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df_resid))
    except Exception:
        se = np.zeros_like(beta)
        p_values = np.ones_like(beta)

    return beta, se, p_values, r_squared


def estimate_price_elasticity(df: pd.DataFrame) -> dict:
    """
    Fits an econometric log-linear demand regression for each SKU:
    ln(quantity) = beta_0 + beta_p * ln(effective_price) + epsilon
    """
    results = {}
    
    for sku, group in df.groupby("sku"):
        if len(group) < 30:
            continue

        sku_data = group.copy()
        sku_data = sku_data[sku_data["effective_price"] > 0]
        y = np.log(sku_data["quantity"].values)
        ln_p = np.log(sku_data["effective_price"].values)

        # Regress ln_q on [1, ln_p] to prevent artificial collinearity with discount
        ones = np.ones_like(ln_p)
        X = np.column_stack([ones, ln_p])

        beta, se, p_vals, r_squared = fit_ols(y, X)
        
        beta_p = float(beta[1])
        p_val = float(p_vals[1])
        
        avg_price = float(sku_data["effective_price"].mean())
        avg_cogs = float(sku_data["cogs"].sum() / sku_data["quantity"].sum())

        # Optimal price using classic Lerner Index rule: (P - MC) / P = -1 / E_p => P* = MC * (E_p / (1 + E_p))
        if beta_p < -1.0:
            optimal_price = avg_cogs * (beta_p / (1.0 + beta_p))
        else:
            optimal_price = avg_price

        results[sku] = {
            "sku": sku,
            "product_name": group["product_name"].iloc[0],
            "price_elasticity": round(beta_p, 3),
            "p_value": round(p_val, 4),
            "r_squared": round(r_squared, 3),
            "avg_selling_price": round(avg_price, 2),
            "unit_cost_cogs": round(avg_cogs, 2),
            "theoretical_optimal_price": round(optimal_price, 2),
            "elasticity_tier": "Highly Elastic" if beta_p < -1.2 else ("Inelastic" if beta_p > -1.0 else "Unitary Elastic"),
        }

    return results


def simulate_price_shift_margin_impact(
    current_price: float,
    current_volume: int,
    unit_cost: float,
    elasticity: float,
    price_change_pct: float,
) -> dict:
    """
    Simulates the volume and gross margin waterfall under a price change:
    delta_Q% = E_p * delta_P%
    """
    new_price = current_price * (1.0 + price_change_pct)
    pct_vol_change = elasticity * price_change_pct
    new_volume = max(0, int(current_volume * (1.0 + pct_vol_change)))

    base_revenue = current_price * current_volume
    base_margin = (current_price - unit_cost) * current_volume

    new_revenue = new_price * new_volume
    new_margin = (new_price - unit_cost) * new_volume
    delta_margin = new_margin - base_margin
    margin_lift_pct = (delta_margin / base_margin) * 100.0 if base_margin > 0 else 0.0

    return {
        "price_change_pct": round(price_change_pct * 100.0, 1),
        "new_price": round(new_price, 2),
        "new_volume": new_volume,
        "new_revenue": round(new_revenue, 2),
        "new_gross_margin": round(new_margin, 2),
        "delta_margin": round(delta_margin, 2),
        "margin_lift_pct": round(margin_lift_pct, 2),
    }
