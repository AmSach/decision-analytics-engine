"""
Customer Lifetime Value (CLV) and RFM Portfolio Segmentation Module.
Computes Recency, Frequency, Monetary value (RFM), fits expected forward transactions,
and calculates customer lifetime equity waterfalls.
"""

from datetime import datetime
import numpy as np
import pandas as pd


def compute_rfm_metrics(df: pd.DataFrame, reference_date: str | None = None) -> pd.DataFrame:
    """
    Computes RFM features for each customer:
    - Recency: Days between last order and reference date.
    - Frequency: Total count of repeat purchases (orders - 1).
    - Monetary: Average gross margin per order.
    - Tenure: Total days between first order and reference date.
    """
    data = df.copy()
    data["order_date"] = pd.to_datetime(data["order_date"], format="mixed", errors="coerce")
    data.dropna(subset=["order_date", "customer_id"], inplace=True)

    if reference_date is None:
        ref_dt = data["order_date"].max()
    else:
        ref_dt = pd.to_datetime(reference_date)

    rfm = data.groupby("customer_id").agg(
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
        order_count=("transaction_id", "nunique"),
        total_revenue=("gross_revenue", "sum"),
        total_margin=("gross_margin", "sum"),
    ).reset_index()

    rfm["recency_days"] = (ref_dt - rfm["last_order"]).dt.days.fillna(0).astype(int)
    rfm["tenure_days"] = (ref_dt - rfm["first_order"]).dt.days.fillna(0).astype(int)
    rfm["frequency"] = rfm["order_count"] - 1
    rfm["avg_order_value"] = np.where(rfm["order_count"] > 0, (rfm["total_revenue"] / rfm["order_count"]).round(2), 0.0)
    rfm["avg_margin_per_order"] = np.where(rfm["order_count"] > 0, (rfm["total_margin"] / rfm["order_count"]).round(2), 0.0)

    # Quantile-based RFM Scoring
    rfm["r_score"] = pd.qcut(rfm["recency_days"].rank(method="first"), 4, labels=[4, 3, 2, 1]).fillna(1).astype(int)
    rfm["f_score"] = pd.qcut(rfm["order_count"].rank(method="first"), 4, labels=[1, 2, 3, 4]).fillna(1).astype(int)
    rfm["m_score"] = pd.qcut(rfm["total_margin"].rank(method="first"), 4, labels=[1, 2, 3, 4]).fillna(1).astype(int)

    # Segment mapping
    def assign_segment(row):
        score = row["r_score"] * 10 + row["f_score"]
        if score in [44, 43, 34]:
            return "Champions"
        elif score in [33, 42, 32]:
            return "Loyal Customers"
        elif score in [41, 31]:
            return "Recent Acquisitions"
        elif score in [24, 23, 14, 13]:
            return "At-Risk High-Value"
        else:
            return "Dormant / Lost"

    rfm["segment"] = rfm.apply(assign_segment, axis=1)

    # 12-Month Forward Estimated Margin (Heuristic BG/NBD frequency projection)
    p_active = np.exp(-0.015 * rfm["recency_days"])
    rfm["prob_active"] = p_active.round(3)
    annual_freq = np.where(rfm["tenure_days"] > 30, (rfm["order_count"] / (rfm["tenure_days"] / 365.0)), rfm["order_count"])
    rfm["expected_forward_orders_12m"] = (annual_freq * p_active).round(2)
    rfm["forward_12m_clv_margin"] = (rfm["expected_forward_orders_12m"] * rfm["avg_margin_per_order"]).round(2)

    return rfm


def get_segment_waterfall_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes customer portfolio by segment, showing customer count,
    historical margin, and 12-month forward expected lifetime equity.
    """
    summary = rfm.groupby("segment").agg(
        customer_count=("customer_id", "count"),
        avg_recency_days=("recency_days", "mean"),
        total_historical_margin=("total_margin", "sum"),
        forward_12m_expected_margin=("forward_12m_clv_margin", "sum"),
        avg_prob_active=("prob_active", "mean"),
    ).reset_index()

    total_customers = summary["customer_count"].sum()
    summary["customer_share_pct"] = ((summary["customer_count"] / total_customers) * 100.0).round(1) if total_customers > 0 else 0.0
    summary["avg_recency_days"] = summary["avg_recency_days"].round(1)
    summary["total_historical_margin"] = summary["total_historical_margin"].round(2)
    summary["forward_12m_expected_margin"] = summary["forward_12m_expected_margin"].round(2)
    summary["avg_prob_active"] = (summary["avg_prob_active"] * 100.0).round(1)

    return summary.sort_values(by="forward_12m_expected_margin", ascending=False)
