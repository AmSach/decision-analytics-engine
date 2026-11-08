"""
Synthetic E-Commerce Transaction and Customer Journey Data Generator.
Produces realistic multi-channel transaction logs with customer cohorts,
pricing variability, marketing touchpoint sequences, and margins.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_ecommerce_dataset(
    num_customers: int = 2500,
    start_date: str = "2023-01-01",
    num_days: int = 365,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates two relational dataframes:
    1. transactions_df: Order-level records with customer IDs, items, revenues, costs, and discounts.
    2. journeys_df: Multi-touch marketing sequence paths leading to conversions or drop-offs.
    """
    rng = np.random.default_rng(random_seed)
    base_start = datetime.strptime(start_date, "%Y-%m-%d")

    channels = ["Paid Search", "Social Ads", "Email", "Organic Search", "Direct", "Referral"]
    channel_weights = [0.28, 0.24, 0.18, 0.15, 0.10, 0.05]
    
    products = [
        {"sku": "SKU-PRO-01", "name": "Executive Analytics Suite", "base_price": 120.0, "unit_cost": 38.0, "base_elasticity": -1.45},
        {"sku": "SKU-PRO-02", "name": "Team Productivity Seat", "base_price": 45.0, "unit_cost": 12.0, "base_elasticity": -0.85},
        {"sku": "SKU-ENT-03", "name": "Enterprise Security Gateway", "base_price": 350.0, "unit_cost": 110.0, "base_elasticity": -1.95},
        {"sku": "SKU-SMB-04", "name": "Cloud Connector Tier", "base_price": 25.0, "unit_cost": 6.0, "base_elasticity": -0.65},
    ]

    customers = []
    transactions = []
    journeys = []

    for cust_id in range(1001, 1001 + num_customers):
        acquisition_day = rng.integers(0, max(1, num_days - 60))
        cohort_date = base_start + timedelta(days=int(acquisition_day))
        cohort_month = cohort_date.strftime("%Y-%m")

        customer_tier = rng.choice(["SMB", "Mid-Market", "Enterprise"], p=[0.60, 0.30, 0.10])
        num_orders = rng.poisson(lam=3.2 if customer_tier == "Enterprise" else 1.8) + 1

        customers.append({
            "customer_id": cust_id,
            "cohort_month": cohort_month,
            "tier": customer_tier,
            "initial_channel": rng.choice(channels, p=channel_weights),
        })

        curr_date = cohort_date
        for order_idx in range(num_orders):
            if order_idx > 0:
                gap_days = rng.integers(20, 80)
                curr_date += timedelta(days=int(gap_days))
                if (curr_date - base_start).days >= num_days:
                    break

            prod = rng.choice(products)
            discount_pct = rng.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25], p=[0.40, 0.20, 0.15, 0.12, 0.08, 0.05])
            
            # Dynamic price variability (regional & temporal dispersion)
            price_variation = rng.uniform(0.90, 1.10)
            effective_price = round(prod["base_price"] * (1.0 - discount_pct) * price_variation, 2)

            # Econometric Demand Invariant: Q = Q_0 * (P / P_0) ^ (beta_p)
            price_ratio = effective_price / prod["base_price"]
            expected_q = 2.5 * (price_ratio ** prod["base_elasticity"])
            quantity = max(1, int(rng.poisson(lam=expected_q)))

            gross_revenue = round(effective_price * quantity, 2)
            cogs = round(prod["unit_cost"] * quantity, 2)
            gross_margin = round(gross_revenue - cogs, 2)

            transactions.append({
                "transaction_id": f"TX-{len(transactions) + 1:07d}",
                "customer_id": cust_id,
                "order_date": curr_date.strftime("%Y-%m-%d"),
                "sku": prod["sku"],
                "product_name": prod["name"],
                "base_price": prod["base_price"],
                "effective_price": effective_price,
                "discount_pct": discount_pct,
                "quantity": quantity,
                "gross_revenue": gross_revenue,
                "cogs": cogs,
                "gross_margin": gross_margin,
            })

            # Journey sequence preceding this order
            path_len = rng.integers(1, 5)
            path = rng.choice(channels, size=path_len, p=channel_weights).tolist()
            journeys.append({
                "customer_id": cust_id,
                "order_date": curr_date.strftime("%Y-%m-%d"),
                "touchpoint_path": " > ".join(path),
                "conversion_value": gross_revenue,
                "converted": 1,
            })

    tx_df = pd.DataFrame(transactions)
    cust_df = pd.DataFrame(customers)
    full_tx_df = tx_df.merge(cust_df, on="customer_id", how="left")
    journeys_df = pd.DataFrame(journeys)

    return full_tx_df, journeys_df
