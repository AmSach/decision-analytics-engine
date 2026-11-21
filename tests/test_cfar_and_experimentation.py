"""
Invariant Unit Tests for Advanced Decision Analytics Engines:
1. Cash Flow at Risk (CFaR) and Correlated Monte Carlo Stress Testing
2. Dynamic Multi-Armed Bandit (MAB) Experimentation (Thompson Sampling & UCB1)
3. Mixed Logit Demand, Cross-Elasticity, and Product Cannibalization Matrix
"""

import numpy as np
import pytest

from src.cash_flow_at_risk_cfar import CashFlowAtRiskSimulator
from src.bandit_experimentation import BanditExperimentEngine
from src.mixed_logit_cannibalization import MixedLogitCannibalizationEngine


class TestCashFlowAtRiskSimulator:
    """Validates Monte Carlo Cash Flow at Risk invariants."""

    def test_cfar_simulation_bounds_and_ordering(self):
        sim = CashFlowAtRiskSimulator(
            base_annual_revenue=5000000.0,
            base_gross_margin_pct=0.35,
            base_operating_opex_pct=0.20,
            initial_cash_reserves=400000.0,
            debt_service_annual=100000.0,
            minimum_liquidity_covenant=150000.0,
        )
        res = sim.simulate_cfar(num_simulations=1000, random_seed=42)

        # Invariant 1: Mean OCF must be positive
        assert res["mean_simulated_cash_flow"] > 0
        assert res["cash_flow_standard_deviation"] > 0

        # Invariant 2: Value at Risk ordering
        # VaR_99 floor <= VaR_95 floor <= Mean OCF
        assert res["var_99_cash_flow_floor"] <= res["var_95_cash_flow_floor"] <= res["mean_simulated_cash_flow"]

        # Invariant 3: Expected Shortfall (CVaR) must be >= VaR floor
        assert res["cvar_expected_shortfall_95"] >= res["cfar_95_dollars"]

        # Invariant 4: Covenant breach probability in [0, 100]
        assert 0.0 <= res["covenant_breach_probability_pct"] <= 100.0

    def test_cfar_distribution_monotonicity(self):
        sim = CashFlowAtRiskSimulator(base_annual_revenue=2000000.0)
        res = sim.simulate_cfar(num_simulations=500, random_seed=99)
        dist_records = res["distribution_table"]

        # Invariant: Percentile cash flow values must be strictly monotonically non-decreasing
        flows = [r["operating_cash_flow"] for r in dist_records]
        for i in range(len(flows) - 1):
            assert flows[i] <= flows[i + 1]


class TestBanditExperimentEngine:
    """Validates Bayesian Multi-Armed Bandit regret and allocation invariants."""

    def test_bandit_pulls_conservation(self):
        true_rates = {"Banner_A": 0.05, "Banner_B": 0.12, "Banner_C": 0.08}
        engine = BanditExperimentEngine(true_rates)
        res = engine.run_simulation(total_trials=1000, random_seed=42)

        # Total pulls must equal total trials
        assert sum(res["static_pulls"]) == 1000
        assert sum(res["thompson_pulls"]) == 1000

    def test_thompson_sampling_outperforms_static_ab(self):
        true_rates = {"Banner_A": 0.04, "Banner_B": 0.15}
        engine = BanditExperimentEngine(true_rates)
        res = engine.run_simulation(total_trials=2000, random_seed=42)

        # Invariant: Thompson Sampling cumulative regret must be significantly lower than static split
        assert res["thompson_regret_final"] < res["static_regret_final"]
        assert res["regret_reduction_vs_ab_pct"] > 25.0


class TestMixedLogitCannibalizationEngine:
    """Validates random-coefficient demand and diversion ratio invariants."""

    @pytest.fixture
    def sample_portfolio(self):
        return [
            {"name": "Standard_Model", "price": 100.0, "base_quality": 8.0, "cogs": 50.0},
            {"name": "Premium_Model", "price": 180.0, "base_quality": 12.0, "cogs": 80.0},
            {"name": "Entry_Model", "price": 60.0, "base_quality": 5.0, "cogs": 30.0},
        ]

    def test_market_shares_sum_to_one(self, sample_portfolio):
        engine = MixedLogitCannibalizationEngine(sample_portfolio, random_seed=42)
        shares = engine.compute_choice_probabilities()

        # Invariant: Total market shares (all products + outside option) sum to 1.0
        assert abs(np.sum(shares) - 1.0) < 1e-4
        assert (shares >= 0.0).all()

    def test_own_price_elasticity_is_negative(self, sample_portfolio):
        engine = MixedLogitCannibalizationEngine(sample_portfolio, random_seed=42)
        res = engine.compute_cannibalization_and_elasticity(price_shock_pct=0.01)
        summary = res["portfolio_summary"]

        # Invariant: Law of Demand: Own price elasticity must be negative for all products
        for row in summary:
            assert row["own_price_elasticity"] < 0.0

    def test_diversion_ratio_conservation(self, sample_portfolio):
        engine = MixedLogitCannibalizationEngine(sample_portfolio, random_seed=42)
        res = engine.compute_cannibalization_and_elasticity(price_shock_pct=0.01)
        div_mat = res["diversion_ratio_matrix_pct"]
        out_div = res["outside_good_diversion_pct"]

        # Invariant: For each product price shock, internal cannibalization + external leakage = 100%
        for i in range(len(sample_portfolio)):
            internal = float(np.sum(div_mat[i]) * 100.0)
            external = float(out_div[i])
            total_diversion = internal + external
            assert abs(total_diversion - 100.0) < 1.0
