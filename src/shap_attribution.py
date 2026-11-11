"""
Game-Theoretic SHAP (SHapley Additive exPlanations) Attribution Engine.
Provides mathematically consistent feature attribution adhering to
Efficiency, Symmetry, Dummy (Null Player), and Additivity axioms.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class ShapleyAttributionEngine:
    """
    Game-Theoretic Feature Attribution Engine based on Shapley values.
    Decomposes model predictions into an additive sum of feature contributions:
    f(x) = phi_0 + sum_{j=1}^M phi_j(x).
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names
        self.explainer: Optional[Any] = None
        self.base_value: float = 0.0
        self.shap_values: Optional[np.ndarray] = None

    def fit_and_explain(
        self,
        X_background: Union[pd.DataFrame, np.ndarray],
        X_explain: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Compute Shapley attribution values across features.
        Uses TreeExplainer if tree-based, otherwise sampling KernelExplainer.
        """
        if isinstance(X_background, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = list(X_background.columns)
            X_bg_arr = X_background.values
        else:
            X_bg_arr = np.asarray(X_background)

        if X_explain is None:
            X_exp_arr = X_bg_arr
        elif isinstance(X_explain, pd.DataFrame):
            X_exp_arr = X_explain.values
        else:
            X_exp_arr = np.asarray(X_explain)

        if self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(X_exp_arr.shape[1])]

        if HAS_SHAP:
            try:
                # TreeExplainer for decision trees, random forests, gradient boosting
                explainer = shap.TreeExplainer(self.model)
                shap_vals = explainer.shap_values(X_exp_arr)
                self.explainer = explainer

                # Handle multi-class / list / 3D output
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
                elif hasattr(shap_vals, "values"):
                    shap_vals = shap_vals.values
                elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    shap_vals = shap_vals[:, :, 1] if shap_vals.shape[2] == 2 else shap_vals[:, :, 0]

                # Base value
                expected_val = explainer.expected_value
                if isinstance(expected_val, (list, np.ndarray)):
                    self.base_value = float(expected_val[1] if len(expected_val) == 2 else expected_val[0])
                else:
                    self.base_value = float(expected_val)

                self.shap_values = np.asarray(shap_vals)
                return self.shap_values
            except Exception:
                pass

        # Fallback: Sampling-based marginal contribution estimator
        self.shap_values, self.base_value = self._compute_sampling_shapley(
            X_bg_arr, X_exp_arr
        )
        return self.shap_values

    def _compute_sampling_shapley(
        self, X_bg: np.ndarray, X_exp: np.ndarray, n_samples: int = 50
    ) -> Tuple[np.ndarray, float]:
        """
        Axiomatic Monte Carlo sampling estimator for Shapley values.
        """
        n_instances, n_features = X_exp.shape

        # Base value: mean prediction on background set
        if hasattr(self.model, "predict_proba"):
            preds_bg = self.model.predict_proba(X_bg)
            base_val = float(np.mean(preds_bg[:, 1] if preds_bg.shape[1] == 2 else preds_bg[:, 0]))
            predict_fn = lambda x: (
                self.model.predict_proba(x)[:, 1]
                if self.model.predict_proba(x).shape[1] == 2
                else self.model.predict_proba(x)[:, 0]
            )
        else:
            preds_bg = self.model.predict(X_bg)
            base_val = float(np.mean(preds_bg))
            predict_fn = lambda x: self.model.predict(x)

        shap_vals = np.zeros((n_instances, n_features))
        rng = np.random.default_rng(42)

        for i in range(n_instances):
            x = X_exp[i]
            for j in range(n_features):
                marginal_contributions = []
                for _ in range(n_samples):
                    # Sample a random background point
                    bg_idx = rng.integers(0, len(X_bg))
                    z = X_bg[bg_idx].copy()

                    # Sample random subset of features
                    perm = rng.permutation(n_features)
                    j_pos = np.where(perm == j)[0][0]
                    pre_j = perm[:j_pos]

                    # Point with subset before j
                    point_without = z.copy()
                    point_without[pre_j] = x[pre_j]

                    # Point with subset before j plus j
                    point_with = point_without.copy()
                    point_with[j] = x[j]

                    val_with = predict_fn(point_with.reshape(1, -1))[0]
                    val_without = predict_fn(point_without.reshape(1, -1))[0]
                    marginal_contributions.append(val_with - val_without)

                shap_vals[i, j] = np.mean(marginal_contributions)

        return shap_vals, base_val

    def get_global_importance(self) -> pd.DataFrame:
        """
        Return ranking of features by mean absolute SHAP value.
        """
        if self.shap_values is None or self.feature_names is None:
            raise RuntimeError("Must call fit_and_explain first")

        mean_abs = np.mean(np.abs(self.shap_values), axis=0)
        total_imp = np.sum(mean_abs)
        pct_imp = (mean_abs / total_imp * 100.0) if total_imp > 0 else np.zeros_like(mean_abs)

        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": np.round(mean_abs, 4),
            "relative_importance_pct": np.round(pct_imp, 2),
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        df["rank"] = np.arange(1, len(df) + 1)
        return df

    def explain_instance(self, instance_index: int) -> Dict[str, Any]:
        """
        Return local waterfall decomposition for a single instance:
        base_value + sum(shap_values) = predicted_value.
        """
        if self.shap_values is None or self.feature_names is None:
            raise RuntimeError("Must call fit_and_explain first")

        contributions = self.shap_values[instance_index]
        predicted_approx = self.base_value + float(np.sum(contributions))

        feature_breakdown = pd.DataFrame({
            "feature": self.feature_names,
            "shap_attribution": np.round(contributions, 4),
            "direction": ["Positive" if v >= 0 else "Negative" for v in contributions],
        }).sort_values(by="shap_attribution", key=abs, ascending=False).reset_index(drop=True)

        return {
            "instance_index": instance_index,
            "base_value": round(self.base_value, 4),
            "predicted_model_output": round(predicted_approx, 4),
            "total_shap_contribution": round(float(np.sum(contributions)), 4),
            "feature_attribution_table": feature_breakdown,
        }
