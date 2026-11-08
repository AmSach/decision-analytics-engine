"""
Synthetic Control Method (Abadie et al.) for Quasi-Experimental Causal Lift.
Constructs an optimal synthetic counterfactual from an untreated donor pool
to measure the exact incremental impact of price shifts, ad campaigns, or regional pilots
when randomized A/B tests are unavailable in historical Excel data.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class SyntheticControlEngine:
    """
    Implements Abadie, Diamond, and Hainmueller (2010) Synthetic Control:
    Minimizes ||X_1 - X_0 * W||_V subject to W_i >= 0, sum(W_i) = 1.
    """

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)

    def fit_synthetic_control(
        self,
        panel_df: pd.DataFrame,
        unit_col: str,
        time_col: str,
        outcome_col: str,
        treated_unit: str,
        intervention_time: int | str,
    ) -> dict:
        """
        Fits optimal donor weights W to reconstruct the treated unit's pre-intervention trajectory.
        """
        # Pivot panel into matrix (time x units)
        pivot = panel_df.pivot(index=time_col, columns=unit_col, values=outcome_col)
        
        if treated_unit not in pivot.columns:
            raise ValueError(f"Treated unit '{treated_unit}' not found in panel columns.")

        donor_units = [u for u in pivot.columns if u != treated_unit]
        if len(donor_units) < 2:
            raise ValueError("Synthetic Control requires at least 2 donor units.")

        # Split pre- and post-intervention periods
        time_index = pivot.index.tolist()
        pre_mask = pivot.index < intervention_time
        post_mask = pivot.index >= intervention_time

        if pre_mask.sum() < 3:
            raise ValueError("At least 3 pre-intervention time periods are required.")

        y_pre_treated = pivot.loc[pre_mask, treated_unit].values
        x_pre_donors = pivot.loc[pre_mask, donor_units].values

        n_donors = len(donor_units)

        # Objective function: Minimize L2 norm ||Y_treated - X_donors * W||^2
        def objective(w):
            diff = y_pre_treated - (x_pre_donors @ w)
            return float(np.sum(diff ** 2))

        # Constraints: sum(W) = 1, W_i in [0, 1] (convex combination)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(n_donors)]
        w_init = np.ones(n_donors) / n_donors

        res = minimize(objective, w_init, method="SLSQP", bounds=bounds, constraints=constraints)
        optimal_weights = res.x if res.success else w_init

        # Project synthetic counterfactual across entire timeline
        x_full_donors = pivot[donor_units].values
        synthetic_trajectory = x_full_donors @ optimal_weights
        actual_trajectory = pivot[treated_unit].values

        treatment_effect = actual_trajectory - synthetic_trajectory
        pre_rmse = float(np.sqrt(np.mean((actual_trajectory[pre_mask] - synthetic_trajectory[pre_mask]) ** 2)))
        post_lift_mean = float(np.mean(treatment_effect[post_mask]))
        post_lift_total = float(np.sum(treatment_effect[post_mask]))

        # Compile donor weights summary
        donor_weight_records = [
            {"donor_unit": donor_units[i], "weight": round(float(optimal_weights[i]), 4)}
            for i in range(n_donors)
            if optimal_weights[i] > 1e-3
        ]
        donor_weight_records.sort(key=lambda d: d["weight"], reverse=True)

        trajectory_df = pd.DataFrame({
            time_col: pivot.index,
            "actual_outcome": np.round(actual_trajectory, 2),
            "synthetic_counterfactual": np.round(synthetic_trajectory, 2),
            "treatment_effect_lift": np.round(treatment_effect, 2),
            "is_post_intervention": post_mask,
        })

        return {
            "treated_unit": treated_unit,
            "intervention_time": intervention_time,
            "pre_intervention_rmse": round(pre_rmse, 3),
            "post_intervention_avg_lift": round(post_lift_mean, 2),
            "post_intervention_cumulative_lift": round(post_lift_total, 2),
            "top_donor_weights": donor_weight_records,
            "trajectory_df": trajectory_df,
        }
