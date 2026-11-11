"""
Cinelli & Hazlett (2020) Econometric Causal Sensitivity Analysis Engine.
Computes partial R^2 metrics, Robustness Values (RV), and formal omitted-variable
bias bounds to test whether causal recommendations survive unobserved confounders.
Reference: Cinelli, C. and Hazlett, C. (2020), JRSS-B, "Making Sense of Sensitivity".
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


class CausalSensitivityEngine:
    """
    Omitted Variable Bias & Causal Sensitivity Engine.
    Quantifies the minimum strength of an unobserved confounder required
    to overturn an empirical treatment effect or destroy statistical significance.
    """

    def __init__(
        self,
        estimate: float,
        standard_error: float,
        degrees_of_freedom: int,
        alpha: float = 0.05,
    ):
        if standard_error <= 0:
            raise ValueError("Standard error must be strictly positive")
        if degrees_of_freedom <= 1:
            raise ValueError("Degrees of freedom must be greater than 1")

        self.estimate = float(estimate)
        self.se = float(standard_error)
        self.df = int(degrees_of_freedom)
        self.alpha = float(alpha)

        # t-statistic and partial R^2
        self.t_stat = self.estimate / self.se
        self.r2_yd_x = (self.t_stat ** 2) / (self.t_stat ** 2 + self.df)

        # Critical t-value for two-tailed test
        self.t_crit = float(stats.t.ppf(1.0 - self.alpha / 2.0, df=self.df))

        # Robustness Values
        self.rv = self._compute_rv(q=1.0)
        self.rv_alpha = self._compute_rv_alpha()

    def _compute_rv(self, q: float = 1.0) -> float:
        """
        Compute Robustness Value (RV_q).
        The percentage of residual variance that an unobserved confounder must explain
        in BOTH treatment and outcome simultaneously to shift the estimate by factor q.
        """
        f = (q * abs(self.t_stat)) / np.sqrt(self.df)
        rv = 0.5 * (np.sqrt(f ** 4 + 4.0 * (f ** 2)) - (f ** 2))
        return float(rv)

    def _compute_rv_alpha(self) -> float:
        """
        Compute Robustness Value for significance (RV_alpha).
        Strength of confounding needed to bring the estimate's t-statistic to critical threshold.
        """
        if abs(self.t_stat) <= self.t_crit:
            return 0.0
        f_alpha = (abs(self.t_stat) - self.t_crit) / np.sqrt(self.df)
        rv_alpha = 0.5 * (np.sqrt(f_alpha ** 4 + 4.0 * (f_alpha ** 2)) - (f_alpha ** 2))
        return float(rv_alpha)

    def compute_bias_and_adjusted_estimate(
        self, r2_d_z: float, r2_y_z: float
    ) -> Dict[str, float]:
        """
        Compute omitted variable bias and adjusted estimate for a hypothetical confounder Z.
        r2_d_z: Partial R^2 of confounder with treatment (R^2_{D ~ Z | X}).
        r2_y_z: Partial R^2 of confounder with outcome (R^2_{Y ~ Z | D, X}).
        """
        if not (0.0 <= r2_d_z < 1.0) or not (0.0 <= r2_y_z < 1.0):
            raise ValueError("Partial R^2 values must be in [0, 1)")

        bias = self.se * np.sqrt(self.df) * np.sqrt((r2_y_z * r2_d_z) / (1.0 - r2_d_z))
        sign = 1.0 if self.estimate >= 0 else -1.0
        adjusted_estimate = self.estimate - sign * bias
        survives = (adjusted_estimate > 0) if self.estimate > 0 else (adjusted_estimate < 0)

        return {
            "hypothetical_r2_d_z": r2_d_z,
            "hypothetical_r2_y_z": r2_y_z,
            "omitted_bias_magnitude": round(float(bias), 4),
            "adjusted_estimate": round(float(adjusted_estimate), 4),
            "original_estimate": round(self.estimate, 4),
            "retains_original_sign": bool(survives),
            "percent_effect_eroded": round(min(float(bias / abs(self.estimate)) * 100.0, 100.0), 2),
        }

    def generate_sensitivity_table(
        self, benchmark_multipliers: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Generate sensitivity grid showing confounder scenarios and effect erosion.
        """
        if benchmark_multipliers is None:
            benchmark_multipliers = [0.25, 0.50, 1.0, 1.5, 2.0, 3.0]

        rows = []
        base_strength = self.r2_yd_x

        for mult in benchmark_multipliers:
            strength = min(base_strength * mult, 0.90)
            res = self.compute_bias_and_adjusted_estimate(strength, strength)
            rows.append({
                "benchmark_scenario": f"{mult:.2f}x Observed Baseline Strength",
                "confounder_r2": round(strength * 100.0, 2),
                "adjusted_estimate": res["adjusted_estimate"],
                "bias_magnitude": res["omitted_bias_magnitude"],
                "percent_eroded": res["percent_effect_eroded"],
                "recommendation_holds": res["retains_original_sign"],
            })

        return pd.DataFrame(rows)

    def summary(self) -> Dict[str, Union[float, int, bool, str]]:
        """
        Return comprehensive diagnostic summary of causal robustness.
        """
        p_val = 2.0 * (1.0 - stats.t.cdf(abs(self.t_stat), df=self.df))
        return {
            "point_estimate": round(self.estimate, 4),
            "standard_error": round(self.se, 4),
            "t_statistic": round(self.t_stat, 3),
            "p_value": round(float(p_val), 5),
            "degrees_of_freedom": self.df,
            "partial_r2_yd_x_pct": round(self.r2_yd_x * 100.0, 2),
            "robustness_value_rv_pct": round(self.rv * 100.0, 2),
            "robustness_value_significance_rv_alpha_pct": round(self.rv_alpha * 100.0, 2),
            "is_statistically_significant": bool(abs(self.t_stat) > self.t_crit),
            "causal_robustness_assessment": (
                "Highly Robust (RV > 20%)"
                if self.rv >= 0.20
                else "Moderately Robust (10% <= RV < 20%)"
                if self.rv >= 0.10
                else "Fragile to Omitted Confounders (RV < 10%)"
            ),
        }
