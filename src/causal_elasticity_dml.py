"""
Double / Debiased Machine Learning (DML) for Causal Price Elasticity.
Implements the Chernozhukov et al. (2018) Neyman-orthogonal score framework:
1. Cross-fit E[ln(Q) | X] and E[ln(P) | X] via non-parametric estimators.
2. Residualize: Y_tilde = Y - E[Y|X], D_tilde = D - E[D|X].
3. Estimate unconfounded causal elasticity: beta_dml = (D_tilde^T * Y_tilde) / (D_tilde^T * D_tilde).
Controls for product fixed effects, customer tier, and promotional endogeneity.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold


def estimate_causal_elasticity_dml(
    df: pd.DataFrame,
    sku: str | None = None,
    n_splits: int = 5,
    random_seed: int = 42,
) -> dict:
    """
    Fits Double/Debiased Machine Learning (DML) for causal price elasticity.
    Outcome Y: ln(quantity)
    Treatment D: ln(effective_price)
    Confounders X: product fixed effects (SKU), customer tier, discount_pct, base price ratio
    """
    data = df.copy()
    if sku is not None:
        data = data[data["sku"] == sku].copy()

    if len(data) < 50:
        return {"error": "Insufficient sample size for DML (N < 50)"}

    # Construct Outcome, Treatment, and Confounder Matrix
    Y = np.log(data["quantity"].values)
    D = np.log(data["effective_price"].values)

    # Encode confounders: Product fixed effects are essential for multi-product demand systems
    confounder_list = []
    
    if "sku" in data.columns and data["sku"].nunique() > 1:
        sku_dummies = pd.get_dummies(data["sku"], drop_first=True, dtype=float).values
        confounder_list.append(sku_dummies)

    if "tier" in data.columns:
        tier_dummies = pd.get_dummies(data["tier"], drop_first=True, dtype=float).values
        confounder_list.append(tier_dummies)

    discount = data["discount_pct"].values.reshape(-1, 1)
    confounder_list.append(discount)

    if "base_price" in data.columns:
        base_ratio = (data["effective_price"] / data["base_price"]).values.reshape(-1, 1)
        confounder_list.append(base_ratio)

    X = np.hstack(confounder_list)

    n = len(Y)
    Y_res = np.zeros(n)
    D_res = np.zeros(n)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    # Cross-fitting nuisance estimators under Neyman orthogonality
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        D_train, D_test = D[train_idx], D[test_idx]

        # Model 1: E[Y | X]
        model_y = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=random_seed)
        model_y.fit(X_train, Y_train)
        Y_res[test_idx] = Y_test - model_y.predict(X_test)

        # Model 2: E[D | X]
        model_d = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=random_seed)
        model_d.fit(X_train, D_train)
        D_res[test_idx] = D_test - model_d.predict(X_test)

    # Orthogonal score estimation: beta = (D_res^T * Y_res) / (D_res^T * D_res)
    denom = np.sum(D_res ** 2)
    if denom == 0:
        return {"error": "Degenerate treatment variation after orthogonalization"}

    beta_dml = float(np.sum(D_res * Y_res) / denom)

    # Asymptotic variance and standard error via sandwich formula
    residuals = Y_res - beta_dml * D_res
    sigma2 = np.mean((residuals * D_res) ** 2) / (np.mean(D_res ** 2) ** 2)
    se_dml = float(np.sqrt(sigma2 / n))
    t_stat = float(beta_dml / se_dml) if se_dml > 0 else 0.0
    p_val = float(2.0 * (1.0 - stats.norm.cdf(np.abs(t_stat))))

    # 95% Confidence Interval
    ci_lower = float(beta_dml - 1.96 * se_dml)
    ci_upper = float(beta_dml + 1.96 * se_dml)

    # Naive OLS comparison
    X_naive = np.column_stack([np.ones(n), D])
    beta_naive, _, _, _ = np.linalg.lstsq(X_naive, Y, rcond=None)
    naive_elasticity = float(beta_naive[1])

    return {
        "sku": sku if sku else "All_Portfolio",
        "causal_elasticity_dml": round(beta_dml, 3),
        "standard_error": round(se_dml, 3),
        "t_statistic": round(t_stat, 2),
        "p_value": round(p_val, 4),
        "ci_95": (round(ci_lower, 3), round(ci_upper, 3)),
        "naive_ols_elasticity": round(naive_elasticity, 3),
        "endogeneity_bias_delta": round(beta_dml - naive_elasticity, 3),
        "sample_size": n,
    }
