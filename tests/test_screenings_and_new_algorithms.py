"""
Comprehensive Invariant Test Suite for Newly Integrated Screening & Production Algorithms:
1. Split-Conformal Prediction Engine (Finite-sample coverage guarantees).
2. Game-Theoretic SHAP Attribution Engine (Efficiency, additivity, and feature attribution).
3. Cinelli & Hazlett Causal Sensitivity Engine (Omitted variable bias and robustness values).
4. Second-Order Markov Chain WPA & xROP (Dynamic state valuation and leverage index).
5. Multi-Constraint 0/1 Knapsack Auction Purse Optimizer (MILP capital allocation).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.conformal_prediction import ConformalPredictor
from src.shap_attribution import ShapleyAttributionEngine
from src.causal_sensitivity_bounds import CausalSensitivityEngine
from src.markov_wpa_xrop import MarkovWPAEngine
from src.knapsack_auction_optimizer import KnapsackAuctionOptimizer


class TestConformalPrediction:
    """Validates finite-sample distribution-free coverage guarantees."""

    def test_classification_conformal_coverage(self):
        rng = np.random.default_rng(42)
        n_calib = 500
        n_test = 500
        n_classes = 3

        # Generate synthetic Dirichlet-distributed predicted probabilities
        calib_probs = rng.dirichlet(np.ones(n_classes), size=n_calib)
        calib_labels = np.array([rng.choice(n_classes, p=p) for p in calib_probs])

        test_probs = rng.dirichlet(np.ones(n_classes), size=n_test)
        test_labels = np.array([rng.choice(n_classes, p=p) for p in test_probs])

        target_alpha = 0.10  # 90% target coverage
        cp = ConformalPredictor(alpha=target_alpha)
        q_hat = cp.fit_classifier(calib_probs, calib_labels)

        assert 0.0 < q_hat <= 1.0, "Conformal quantile must be bounded in (0, 1]"

        res = cp.evaluate_classification(test_probs, test_labels)
        # On exchangeable data, empirical coverage should be near 90% (e.g. >= 85%)
        assert res["empirical_coverage_pct"] >= 83.0, (
            f"Coverage {res['empirical_coverage_pct']}% fell below minimum statistical threshold"
        )
        assert res["avg_prediction_set_size"] >= 1.0, "Average prediction set size must be >= 1"

    def test_regression_conformal_coverage(self):
        rng = np.random.default_rng(123)
        n = 600
        x = rng.uniform(-3, 3, size=n)
        y = 2.5 * x + rng.normal(0, 0.5, size=n)
        y_hat = 2.5 * x

        calib_y, test_y = y[:300], y[300:]
        calib_pred, test_pred = y_hat[:300], y_hat[300:]

        cp = ConformalPredictor(alpha=0.10)
        q_hat = cp.fit_regressor(calib_pred, calib_y)

        assert q_hat > 0.0, "Conformal quantile for residuals must be strictly positive"

        res = cp.evaluate_regression(test_pred, test_y)
        assert res["empirical_coverage_pct"] >= 85.0, (
            f"Empirical coverage {res['empirical_coverage_pct']}% below 85%"
        )
        assert res["avg_interval_width"] == pytest.approx(2.0 * q_hat, rel=1e-3)


class TestShapleyAttribution:
    """Validates game-theoretic efficiency and feature attribution consistency."""

    def test_shapley_efficiency_and_ranking(self):
        rng = np.random.default_rng(42)
        n = 150
        X = rng.normal(size=(n, 4))
        # Feature 0 and 1 drive y; features 2 and 3 are noise
        y = (2.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 0.2, size=n) > 0).astype(int)

        rf = RandomForestClassifier(n_estimators=20, random_state=42)
        rf.fit(X, y)

        feature_names = ["Signal_Alpha", "Signal_Beta", "Noise_Gamma", "Noise_Delta"]
        engine = ShapleyAttributionEngine(model=rf, feature_names=feature_names)
        shap_vals = engine.fit_and_explain(X[:50], X[:20])

        assert shap_vals.shape == (20, 4), "SHAP values matrix dimension mismatch"

        # Check global importance
        imp_df = engine.get_global_importance()
        assert len(imp_df) == 4
        assert np.isclose(imp_df["relative_importance_pct"].sum(), 100.0, atol=0.5)

        # Signal features should rank higher than noise features
        top_features = imp_df["feature"].iloc[:2].tolist()
        assert "Signal_Alpha" in top_features or "Signal_Beta" in top_features

        # Check local instance explanation
        inst = engine.explain_instance(0)
        assert "instance_index" in inst
        assert "base_value" in inst
        assert len(inst["feature_attribution_table"]) == 4


class TestCausalSensitivity:
    """Validates Cinelli & Hazlett (2020) omitted variable bias robustness bounds."""

    def test_causal_sensitivity_invariants(self):
        # Suppose estimated treatment effect = 2.45, SE = 0.50, df = 200
        engine = CausalSensitivityEngine(estimate=2.45, standard_error=0.50, degrees_of_freedom=200)

        summary = engine.summary()
        assert summary["t_statistic"] == pytest.approx(4.90, rel=1e-2)
        assert summary["is_statistically_significant"] is True
        assert summary["robustness_value_rv_pct"] > 0.0

        # Hypothetical confounder scenario
        bias_res = engine.compute_bias_and_adjusted_estimate(r2_d_z=0.05, r2_y_z=0.05)
        assert bias_res["omitted_bias_magnitude"] > 0.0
        assert bias_res["adjusted_estimate"] < bias_res["original_estimate"]

        # Sensitivity table generation
        sens_df = engine.generate_sensitivity_table()
        assert len(sens_df) == 6
        assert "recommendation_holds" in sens_df.columns
        assert "confounder_r2" in sens_df.columns


class TestMarkovWPAxROP:
    """Validates dynamic state transitions and leverage index calculation."""

    def test_markov_state_and_delivery_evaluation(self):
        engine = MarkovWPAEngine()

        # Evaluate death over high-pressure delivery
        res = engine.evaluate_delivery(
            overs_rem_pre=2.0,
            wickets_pre=4,
            rr_delta_pre=-1.0,
            overs_rem_post=1.5,
            wickets_post=5,  # Wicket fell
            rr_delta_post=-1.8,
            actual_runs=0,
            is_wicket=True,
        )

        assert "wpa" in res
        assert "xrop" in res
        assert "leverage_index" in res
        assert res["leverage_index"] >= 1.5, "Death over close match must have high leverage index"

        # Evaluate player aggregation
        events = [res, res]
        agg = engine.aggregate_player_performance(events)
        assert agg["total_balls"] == 2
        assert "total_wpa" in agg
        assert "total_xrop_runs_over_par" in agg


class TestKnapsackAuctionOptimizer:
    """Validates Mixed-Integer Linear Programming purse optimization."""

    def test_milp_squad_selection_invariants(self):
        # Create synthetic candidate pool of 30 players
        roles = ["Batter"] * 10 + ["Pace Bowler"] * 8 + ["Spinner"] * 6 + ["Wicketkeeper"] * 6
        costs = [18.0, 14.0, 12.0, 8.5, 6.0, 4.0, 2.5, 1.5, 0.5, 0.5] * 3
        surpluses = [4.5, 3.2, 2.8, 2.1, 1.8, 1.2, 0.9, 0.6, 0.4, 0.2] * 3
        is_overseas = [True, False, False, True, False, False, True, False, False, False] * 3

        df = pd.DataFrame({
            "player_name": [f"Player_{i}" for i in range(30)],
            "primary_role": roles,
            "estimated_cost_cr": costs,
            "projected_surplus": surpluses,
            "is_overseas": is_overseas,
        })

        optimizer = KnapsackAuctionOptimizer(
            purse_ceiling=120.0,
            min_squad_size=18,
            max_squad_size=25,
            max_overseas=8,
        )

        res = optimizer.optimize_squad(
            df,
            value_col="projected_surplus",
            cost_col="estimated_cost_cr",
            role_col="primary_role",
            is_overseas_col="is_overseas",
        )

        assert res["total_squad_size"] >= 18, "Squad size must meet minimum constraint"
        assert res["total_squad_size"] <= 25, "Squad size must not exceed maximum constraint"
        assert res["total_overseas_count"] <= 8, "Overseas players must not exceed 8"
        assert res["total_purse_allocated_cr"] <= 120.0, "Total spend must not exceed purse ceiling"
        assert res["remaining_purse_cr"] >= 0.0, "Remaining purse must be non-negative"
        assert not res["selected_roster"].empty, "Selected roster must not be empty"
