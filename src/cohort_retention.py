"""
Cohort Retention and Churn Hazard Analysis Module.
Builds monthly customer cohort matrices, computes Kaplan-Meier churn hazard functions,
and analyzes net revenue retention (NRR) dynamics.
"""

import numpy as np
import pandas as pd


def compute_cohort_retention(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes user retention percentage and revenue retention percentage by cohort index.
    Expects df with ['customer_id', 'order_date', 'cohort_month', 'gross_revenue'].
    """
    data = df.copy()
    data["order_date"] = pd.to_datetime(data["order_date"], format="mixed", errors="coerce")
    data["cohort_month_dt"] = pd.to_datetime(data["cohort_month"], format="mixed", errors="coerce")
    data.dropna(subset=["order_date", "cohort_month_dt"], inplace=True)

    data["order_month"] = data["order_date"].dt.to_period("M")
    data["cohort_period"] = data["cohort_month_dt"].dt.to_period("M")

    # Calculate non-negative period offset
    data["period_number"] = (
        (data["order_month"].dt.year - data["cohort_period"].dt.year) * 12
        + (data["order_month"].dt.month - data["cohort_period"].dt.month)
    )
    data = data[data["period_number"] >= 0].copy()

    # 1. Customer Counts Matrix
    cohort_group = data.groupby(["cohort_month", "period_number"])
    user_cohort = cohort_group["customer_id"].nunique().unstack(1)
    if user_cohort.empty:
        return pd.DataFrame(), pd.DataFrame()

    cohort_sizes = user_cohort.iloc[:, 0]
    user_retention_pct = user_cohort.divide(cohort_sizes, axis=0) * 100.0

    # 2. Revenue Matrix
    rev_cohort = cohort_group["gross_revenue"].sum().unstack(1)
    initial_revenue = rev_cohort.iloc[:, 0]
    revenue_retention_pct = rev_cohort.divide(initial_revenue, axis=0) * 100.0

    return user_retention_pct.round(2), revenue_retention_pct.round(2)


def compute_churn_hazard(retention_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Kaplan-Meier style conditional survival S(t) and hazard rate h(t):
    For periods t = 0, 1, 2...
    h(t) = 1 - (Active at t / Active at t-1) among cohorts observed at both periods.
    S(t) = S(t-1) * (1 - h(t))
    Guarantees h(t) in [0, 1] and S(t) monotonically non-increasing.
    """
    if retention_matrix.empty:
        return pd.DataFrame()

    periods = sorted([c for c in retention_matrix.columns if isinstance(c, (int, np.integer)) and c >= 0])
    
    hazard_records = []
    current_survival = 1.0

    for i in range(len(periods)):
        p = periods[i]
        if i == 0:
            hazard_rate = 0.0
            current_survival = 1.0
        else:
            p_prev = periods[i - 1]
            valid_mask = retention_matrix[p_prev].notna() & retention_matrix[p].notna()
            if valid_mask.sum() > 0:
                prev_sum = retention_matrix.loc[valid_mask, p_prev].sum()
                curr_sum = retention_matrix.loc[valid_mask, p].sum()
                cond_ret = min(1.0, max(0.0, curr_sum / prev_sum)) if prev_sum > 0 else 0.0
                hazard_rate = 1.0 - cond_ret
            else:
                hazard_rate = 0.0

            current_survival *= (1.0 - hazard_rate)

        hazard_records.append({
            "period": int(p),
            "survival_rate_pct": round(float(current_survival * 100.0), 2),
            "hazard_rate_pct": round(float(hazard_rate * 100.0), 2),
        })

    return pd.DataFrame(hazard_records)
