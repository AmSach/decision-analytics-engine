"""
Enterprise Synthetic Dataset Generator.
Generates deterministic, highly realistic corporate transaction, inventory,
and journey data tables for immediate 1-click decision modeling.
"""

from datetime import datetime, timedelta
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def generate_enterprise_suite(
    num_orders: int = 2500,
    num_customers: int = 800,
    random_seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Generates three interconnected relational dataframes:
    1. transactions: Order-level transactional data with pricing, revenue, and margins.
    2. inventory: SKU-level stock positions, lead times, and holding costs.
    3. journeys: Multi-touch marketing sequence paths leading to conversions.
    """
    rng = np.random.default_rng(random_seed)
    base_start = datetime(2025, 1, 1)

    categories = ["Analytics Software", "Cloud Infrastructure", "Security Gateway", "Productivity Seats"]
    skus = [
        {"sku": "SKU-ANL-01", "name": "Enterprise Decision Suite", "category": "Analytics Software", "base_price": 120.0, "unit_cost": 32.0, "elasticity": -1.45},
        {"sku": "SKU-CLD-02", "name": "Compute Node Elastic Tier", "category": "Cloud Infrastructure", "base_price": 45.0, "unit_cost": 14.0, "elasticity": -0.85},
        {"sku": "SKU-SEC-03", "name": "Zero-Trust Perimeter Gate", "category": "Security Gateway", "base_price": 310.0, "unit_cost": 88.0, "elasticity": -1.90},
        {"sku": "SKU-PRD-04", "name": "Collaborative Team Seat", "category": "Productivity Seats", "base_price": 28.0, "unit_cost": 6.5, "elasticity": -0.65},
        {"sku": "SKU-ANL-05", "name": "Forecasting & Risk Engine", "category": "Analytics Software", "base_price": 220.0, "unit_cost": 55.0, "elasticity": -1.25},
    ]

    channels = ["Paid Search", "Organic Search", "Direct", "Email", "Social Ads", "Referral"]
    channel_weights = [0.28, 0.22, 0.16, 0.15, 0.12, 0.07]

    customer_tiers = ["Enterprise", "Mid-Market", "SMB"]
    tier_weights = [0.15, 0.35, 0.50]

    customers = []
    for c_id in range(1001, 1001 + num_customers):
        cohort_month = (base_start + timedelta(days=int(rng.integers(0, 180)))).strftime("%Y-%m")
        customers.append({
            "customer_id": c_id,
            "cohort_month": cohort_month,
            "tier": rng.choice(customer_tiers, p=tier_weights),
            "region": rng.choice(["North America", "EMEA", "APAC", "LATAM"], p=[0.45, 0.30, 0.18, 0.07]),
        })
    cust_df = pd.DataFrame(customers)

    transactions = []
    journeys = []

    for i in range(num_orders):
        cust = customers[rng.integers(0, len(customers))]
        prod = rng.choice(skus)
        
        # Order Date spanning 365 days
        order_date = base_start + timedelta(days=int(rng.integers(0, 365)))
        
        # Discount policy with occasional margin leak
        discount_pct = rng.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25], p=[0.45, 0.22, 0.15, 0.10, 0.05, 0.03])
        
        # Occasional data error / rogue discounting anomaly (0.5% rate)
        if rng.random() < 0.005:
            discount_pct = 0.85  # Severe anomaly / margin leak!
            
        variation = rng.uniform(0.95, 1.05)
        effective_price = round(prod["base_price"] * (1.0 - discount_pct) * variation, 2)
        
        # Econometric quantity demand function
        price_ratio = effective_price / prod["base_price"]
        expected_q = 2.4 * (price_ratio ** prod["elasticity"])
        quantity = max(1, int(rng.poisson(lam=expected_q)))
        
        gross_revenue = round(effective_price * quantity, 2)
        cogs = round(prod["unit_cost"] * quantity, 2)
        gross_margin = round(gross_revenue - cogs, 2)
        margin_pct = round((gross_margin / gross_revenue) * 100.0, 1) if gross_revenue > 0 else 0.0

        transactions.append({
            "order_id": f"ORD-{i + 10001}",
            "customer_id": cust["customer_id"],
            "tier": cust["tier"],
            "region": cust["region"],
            "order_date": order_date.strftime("%Y-%m-%d"),
            "sku": prod["sku"],
            "product_name": prod["name"],
            "category": prod["category"],
            "effective_price": effective_price,
            "quantity": quantity,
            "gross_revenue": gross_revenue,
            "cogs": cogs,
            "gross_margin": gross_margin,
            "margin_pct": margin_pct,
            "discount_pct": discount_pct,
        })

        # Journey preceding order
        path_len = rng.integers(1, 5)
        path = rng.choice(channels, size=path_len, p=channel_weights).tolist()
        journeys.append({
            "customer_id": cust["customer_id"],
            "order_id": f"ORD-{i + 10001}",
            "order_date": order_date.strftime("%Y-%m-%d"),
            "touchpoint_path": " > ".join(path),
            "conversion_value": gross_revenue,
            "converted": 1,
        })

    tx_df = pd.DataFrame(transactions)
    journeys_df = pd.DataFrame(journeys)

    # Inventory Table
    inventory_records = []
    for p in skus:
        daily_d = rng.uniform(8.0, 35.0)
        daily_std = daily_d * rng.uniform(0.20, 0.45)
        lead_time = int(rng.integers(7, 21))
        lead_std = rng.uniform(1.0, 3.5)
        
        # Calculate theoretical ROP and stock
        ss = 1.96 * np.sqrt(lead_time * (daily_std ** 2) + (daily_d ** 2) * (lead_std ** 2))
        rop = int(daily_d * lead_time + ss)
        current_stock = int(rng.uniform(rop * 0.70, rop * 1.80))
        
        inventory_records.append({
            "sku": p["sku"],
            "product_name": p["name"],
            "category": p["category"],
            "unit_cost": p["unit_cost"],
            "selling_price": p["base_price"],
            "current_stock": current_stock,
            "daily_demand_mean": round(daily_d, 1),
            "daily_demand_std": round(daily_std, 1),
            "lead_time_days": lead_time,
            "lead_time_std": round(lead_std, 1),
            "annual_holding_rate": 0.22,
        })
    inv_df = pd.DataFrame(inventory_records)

    return {
        "transactions": tx_df,
        "inventory": inv_df,
        "journeys": journeys_df,
    }
