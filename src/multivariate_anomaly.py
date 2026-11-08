"""
Multivariate Operational Anomaly and Margin Leakage Detection Engine.
Engineered for C-suite risk surveillance and financial audit quants.
Applies regularized Mahalanobis Distance and Chi-Square p-value testing
to multi-dimensional transaction vectors (Sales, Margin, Discount, Quantity, Shipping Cost).
Identifies statistically significant margin leaks, predatory pricing exploits, and fulfillment anomalies.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2


def detect_operational_anomalies(
    df: pd.DataFrame,
    feature_cols: list[str] = None,
    alpha: float = 0.01,
    shrinkage: float = 1e-4,
) -> pd.DataFrame:
    """
    Detects multivariate operational outliers using regularized Mahalanobis Distance.

    Formula:
    D_M^2 = (X - mu)^T * (Sigma + lambda * I)^(-1) * (X - mu)
    Under multivariate normality, D_M^2 ~ Chi-Square(k degrees of freedom).
    A point is an anomaly if p-value < alpha.
    """
    if feature_cols is None:
        candidate_cols = ["gross_revenue", "gross_margin", "discount_pct", "quantity", "shipping_cost"]
        feature_cols = [c for c in candidate_cols if c in df.columns]

    if len(feature_cols) < 2:
        raise ValueError(f"Need at least 2 numeric features for multivariate analysis. Found: {feature_cols}")

    # Extract numerical matrix
    sub_df = df[feature_cols].copy().fillna(0.0)
    X = sub_df.values.astype(np.float64)
    n_samples, n_features = X.shape

    if n_samples < n_features + 2:
        raise ValueError(f"Sample size {n_samples} is too small for {n_features} features.")

    # Empirical mean and covariance
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)

    # Regularization to prevent singularity
    cov_reg = cov + np.eye(n_features) * shrinkage

    try:
        inv_cov = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_reg)

    diff = X - mu
    # Compute Mahalanobis distance squared: diag(diff @ inv_cov @ diff.T)
    d_sq = np.sum((diff @ inv_cov) * diff, axis=1)
    d_sq = np.maximum(0.0, d_sq)

    # Chi-square cumulative distribution p-value
    # Upper tail survival function: sf = 1 - cdf
    p_values = chi2.sf(d_sq, df=n_features)

    res_df = df.copy()
    res_df["mahalanobis_dist_sq"] = np.round(d_sq, 3)
    res_df["anomaly_p_value"] = np.round(p_values, 6)
    res_df["is_operational_anomaly"] = p_values < alpha

    # Classify anomaly archetypes based on dominant feature deviation
    anomaly_types = []
    for i, is_anom in enumerate(res_df["is_operational_anomaly"]):
        if not is_anom:
            anomaly_types.append("Normal Operation")
            continue

        row_x = X[i]
        # Check specific economic conditions
        gm_idx = feature_cols.index("gross_margin") if "gross_margin" in feature_cols else -1
        disc_idx = feature_cols.index("discount_pct") if "discount_pct" in feature_cols else -1
        ship_idx = feature_cols.index("shipping_cost") if "shipping_cost" in feature_cols else -1

        if gm_idx != -1 and row_x[gm_idx] < 0:
            anomaly_types.append("Severe Negative Margin Leak")
        elif disc_idx != -1 and row_x[disc_idx] > 0.40:
            anomaly_types.append("Predatory Discount Exploitation")
        elif ship_idx != -1 and row_x[ship_idx] > np.percentile(X[:, ship_idx], 95):
            anomaly_types.append("Outlier Freight Fulfillment Cost")
        else:
            anomaly_types.append("Multivariate Variance Outlier")

    res_df["anomaly_classification"] = anomaly_types
    return res_df


def summarize_margin_leakage(anomaly_df: pd.DataFrame) -> dict:
    """
    Summarizes total financial leakage and lost profits attributed to anomalies.
    """
    total_tx = len(anomaly_df)
    anom_mask = anomaly_df["is_operational_anomaly"]
    anom_count = int(anom_mask.sum())

    total_revenue = float(anomaly_df["gross_revenue"].sum())
    total_margin = float(anomaly_df["gross_margin"].sum())

    anom_rev = float(anomaly_df.loc[anom_mask, "gross_revenue"].sum())
    anom_margin = float(anomaly_df.loc[anom_mask, "gross_margin"].sum())

    # Negative margin transactions
    neg_margin_mask = anom_mask & (anomaly_df["gross_margin"] < 0)
    total_negative_leakage = float(anomaly_df.loc[neg_margin_mask, "gross_margin"].sum())

    return {
        "total_transactions": total_tx,
        "anomalous_transactions_count": anom_count,
        "anomaly_rate_pct": round((anom_count / total_tx) * 100.0, 2) if total_tx > 0 else 0.0,
        "total_revenue": round(total_revenue, 2),
        "total_margin": round(total_margin, 2),
        "anomalous_revenue_exposure": round(anom_rev, 2),
        "anomalous_margin_contribution": round(anom_margin, 2),
        "pure_margin_leakage_dollars": round(total_negative_leakage, 2),
    }
