"""
Multi-Echelon Inventory Optimization (MEIO) & Dynamic Stochastic Safety Stock Engine.
Engineered for enterprise supply chains and retail operations.
Computes mathematically optimal safety stock, reorder points (ROP), Economic Order Quantities (EOQ),
expected unit shortages via the standard Normal loss function, and working capital requirements
under joint demand and lead-time stochasticity.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def standard_normal_loss(z: float) -> float:
    """
    Computes the standard normal loss function G(z) = phi(z) - z * (1 - Phi(z)),
    which represents expected unit shortage normalized by standard deviation.
    """
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    return float(phi - z * (1.0 - Phi))


def compute_stochastic_safety_stock(
    daily_demand_mean: float,
    daily_demand_std: float,
    lead_time_days_mean: float,
    lead_time_days_std: float,
    service_level: float = 0.95,
) -> dict:
    """
    Computes safety stock and reorder point under joint demand and lead-time uncertainty.

    Variance of lead time demand:
    sigma_LTD^2 = mu_L * sigma_D^2 + mu_D^2 * sigma_L^2
    """
    if service_level <= 0.0 or service_level >= 1.0:
        raise ValueError("Service level must be strictly between 0 and 1.")

    mu_d = max(0.001, float(daily_demand_mean))
    sigma_d = max(0.0, float(daily_demand_std))
    mu_l = max(1.0, float(lead_time_days_mean))
    sigma_l = max(0.0, float(lead_time_days_std))

    # Joint variance under independent stochastic processes
    variance_ltd = (mu_l * (sigma_d ** 2)) + ((mu_d ** 2) * (sigma_l ** 2))
    sigma_ltd = float(np.sqrt(variance_ltd))

    # Standard normal quantile
    z_alpha = float(norm.ppf(service_level))
    safety_stock = z_alpha * sigma_ltd
    mean_lead_time_demand = mu_d * mu_l
    reorder_point = mean_lead_time_demand + safety_stock

    # Expected unit shortage per replenishment cycle
    expected_shortage_per_cycle = sigma_ltd * standard_normal_loss(z_alpha)

    return {
        "service_level": round(service_level, 4),
        "z_score": round(z_alpha, 3),
        "mean_lead_time_demand": round(mean_lead_time_demand, 2),
        "sigma_lead_time_demand": round(sigma_ltd, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "expected_unit_shortage_per_cycle": round(expected_shortage_per_cycle, 4),
    }


def optimize_sku_inventory_policy(
    annual_demand: float,
    unit_cost: float,
    order_setup_cost: float = 50.0,
    annual_holding_cost_rate: float = 0.20,
    daily_demand_mean: float = 10.0,
    daily_demand_std: float = 3.0,
    lead_time_days_mean: float = 14.0,
    lead_time_days_std: float = 3.0,
    target_service_level: float = 0.95,
) -> dict:
    """
    Full SKU inventory optimization combining EOQ, stochastic safety stock,
    fill rate, and working capital inventory commitment.
    """
    annual_d = max(1.0, float(annual_demand))
    c_unit = max(0.01, float(unit_cost))
    k_order = max(1.0, float(order_setup_cost))
    h_rate = max(0.01, float(annual_holding_cost_rate))
    h_unit = c_unit * h_rate

    # Economic Order Quantity
    eoq = float(np.sqrt((2.0 * annual_d * k_order) / h_unit))
    annual_orders = annual_d / eoq

    # Safety stock metrics
    ss_metrics = compute_stochastic_safety_stock(
        daily_demand_mean=daily_demand_mean,
        daily_demand_std=daily_demand_std,
        lead_time_days_mean=lead_time_days_mean,
        lead_time_days_std=lead_time_days_std,
        service_level=target_service_level,
    )

    safety_stock = max(0.0, ss_metrics["safety_stock"])
    expected_shortage = ss_metrics["expected_unit_shortage_per_cycle"]

    # Fill rate (Type-2 service level)
    fill_rate = max(0.0, min(1.0, 1.0 - (expected_shortage / eoq)))

    # Working capital tied up in inventory = (Cycle Stock / 2 + Safety Stock) * Unit Cost
    average_cycle_stock = eoq / 2.0
    working_capital_committed = (average_cycle_stock + safety_stock) * c_unit
    annual_holding_cost = (average_cycle_stock + safety_stock) * h_unit
    annual_ordering_cost = annual_orders * k_order
    total_annual_inventory_cost = annual_holding_cost + annual_ordering_cost

    return {
        "annual_demand": round(annual_d, 2),
        "unit_cost": round(c_unit, 2),
        "economic_order_quantity": round(eoq, 2),
        "annual_order_frequency": round(annual_orders, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(ss_metrics["reorder_point"], 2),
        "cycle_service_level": round(target_service_level, 4),
        "fill_rate_type2": round(fill_rate, 4),
        "working_capital_committed": round(working_capital_committed, 2),
        "annual_holding_cost": round(annual_holding_cost, 2),
        "annual_ordering_cost": round(annual_ordering_cost, 2),
        "total_annual_inventory_cost": round(total_annual_inventory_cost, 2),
    }


def optimize_multi_echelon_portfolio(
    category_df: pd.DataFrame,
    lead_time_days_mean: float = 14.0,
    lead_time_days_std: float = 3.0,
    service_level: float = 0.95,
) -> pd.DataFrame:
    """
    Optimizes inventory policy across multiple categories or SKUs from transactional history.
    Expects columns: ['category' or 'sku', 'quantity', 'gross_revenue', 'cogs'].
    """
    group_col = "category" if "category" in category_df.columns else "sku"
    results = []

    grouped = category_df.groupby(group_col)
    for name, group in grouped:
        total_qty = group["quantity"].sum()
        total_cogs = group["cogs"].sum()
        unit_cost = (total_cogs / total_qty) if total_qty > 0 else 10.0
        daily_d_mean = total_qty / 365.0
        daily_d_std = max(0.5, daily_d_mean * 0.35)

        res = optimize_sku_inventory_policy(
            annual_demand=total_qty,
            unit_cost=unit_cost,
            daily_demand_mean=daily_d_mean,
            daily_demand_std=daily_d_std,
            lead_time_days_mean=lead_time_days_mean,
            lead_time_days_std=lead_time_days_std,
            target_service_level=service_level,
        )
        res[group_col] = name
        results.append(res)

    out_df = pd.DataFrame(results)
    # Reorder columns to place group column first
    cols = [group_col] + [c for c in out_df.columns if c != group_col]
    return out_df[cols]
