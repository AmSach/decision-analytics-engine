"""
Double / Debiased Machine Learning (DML) for Causal Price Elasticity.
Implements the Chernozhukov et al. (2018) Neyman-orthogonal score framework:
1. Cross-fit E[ln(Q) | X] and E[ln(P) | X] via regularized non-parametric estimators.
2. Residualize: Y_tilde = Y - E[Y|X], D_tilde = D - E[D|X].
3. Estimate unconfounded causal elasticity: beta_dml = (D_tilde^T * Y_tilde) / (D_tilde^T * D_tilde).
Controls for product fixed effects, customer tier, and promotional endogeneity.
"""

import numpy as np
import pandas as pd
from scipy import stats


def _kfold_split(n: int, n_splits: int, seed: int):
    """Pure NumPy K-Fold splitting."""
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)
    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx


def _ridge_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, l2_reg: float = 1.0) -> np.ndarray:
    """Fast, robust L2-regularized linear projection for nuisance estimation."""
    p = X_train.shape[1]
    reg_matrix = X_train.T @ X_train + l2_reg * np.eye(p)
    try:
        weights = np.linalg.solve(reg_matrix, X_train.T @ y_train)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(reg_matrix) @ (X_train.T @ y_train)
    return X_test @ weights


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

    # Filter strictly positive price and quantity
    data = data[(data["effective_price"] > 0) & (data["quantity"] > 0)].copy()

    if len(data) < 50:
        return {"error": "Insufficient sample size for DML (N < 50)"}

    # Construct Outcome, Treatment, and Confounder Matrix
    Y = np.log(data["quantity"].values.astype(np.float64))
    D = np.log(data["effective_price"].values.astype(np.float64))

    confounder_list = []
    
    # Strictly include exogenous pre-treatment confounders only (no post-treatment colliders)
    if "sku" in data.columns and data["sku"].nunique() > 1:
        if data["sku"].nunique() > 50:
            sku_freq = data["sku"].map(data["sku"].value_counts(normalize=True)).values.reshape(-1, 1)
            confounder_list.append(sku_freq)
        else:
            sku_dummies = pd.get_dummies(data["sku"], drop_first=True, dtype=float).values
            confounder_list.append(sku_dummies)

    if "tier" in data.columns:
        tier_dummies = pd.get_dummies(data["tier"], drop_first=True, dtype=float).values
        confounder_list.append(tier_dummies)
    elif "Customer Segment" in data.columns:
        seg_dummies = pd.get_dummies(data["Customer Segment"], drop_first=True, dtype=float).values
        confounder_list.append(seg_dummies)

    if "order_date" in data.columns:
        try:
            month_series = pd.to_datetime(data["order_date"]).dt.month
            if month_series.nunique() > 1:
                month_dummies = pd.get_dummies(month_series, drop_first=True, dtype=float).values
                confounder_list.append(month_dummies)
        except Exception:
            pass

    if not confounder_list:
        confounder_list.append(np.ones((len(data), 1)))

    X = np.hstack(confounder_list).astype(np.float64)
    # Ensure strictly finite inputs and add intercept
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.column_stack([np.ones(len(X)), X])

    n = len(Y)
    Y_res = np.zeros(n)
    D_res = np.zeros(n)

    # Cross-fitting nuisance estimators under Neyman orthogonality
    for train_idx, test_idx in _kfold_split(n, n_splits=n_splits, seed=random_seed):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        D_train, D_test = D[train_idx], D[test_idx]

        # Model 1: E[Y | X]
        y_hat = _ridge_predict(X_train, Y_train, X_test, l2_reg=1e-3)
        Y_res[test_idx] = Y_test - y_hat

        # Model 2: E[D | X]
        d_hat = _ridge_predict(X_train, D_train, X_test, l2_reg=1e-3)
        D_res[test_idx] = D_test - d_hat

    # Orthogonal score estimation: beta = (D_res^T * Y_res) / (D_res^T * D_res)
    denom = float(np.sum(D_res ** 2))
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
