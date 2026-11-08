"""
Complex Invariant Test Suite for Advanced Strategic Algorithms:
1. Double/Debiased Machine Learning (DML) Causal Elasticity.
2. Level-k and Logit Quantal Response Equilibrium (QRE) Adversarial Pricing.
3. Elinor Ostrom Polycentric Commons Governance & Tragedy-of-Commons Defection.
4. Spectral Graph Perron-Frobenius & Katz Multi-Touch Attribution.
5. DuPont 3-Tier ROE Decomposition & Stochastic Working Capital Float Engine.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_ecommerce_dataset
from src.causal_elasticity_dml import estimate_causal_elasticity_dml
from src.adversarial_game_pricing import AdversarialPricingEngine
from src.ostrom_commons_governance import OstromCommonsGovernanceEngine
from src.spectral_attribution import compute_spectral_graph_attribution
from src.stochastic_float_engine import compute_dupont_3tier_decomposition, simulate_negative_working_capital_float


@pytest.fixture(scope="module")
def robust_dataset():
    tx_df, journeys_df = generate_ecommerce_dataset(
        num_customers=800,
        start_date="2023-01-01",
        num_days=300,
        random_seed=777,
    )
    return tx_df, journeys_df


class TestCausalElasticityDML:
    """Validates Double/Debiased Machine Learning (DML) asymptotic properties."""

    def test_dml_causal_invariants(self, robust_dataset):
        tx_df, _ = robust_dataset
        res = estimate_causal_elasticity_dml(tx_df, n_splits=3, random_seed=42)

        assert "causal_elasticity_dml" in res
        assert res["causal_elasticity_dml"] < 0, "Causal elasticity must be negative (Law of Demand)"
        assert res["standard_error"] > 0, "Standard error must be strictly positive"
        assert res["ci_95"][0] <= res["causal_elasticity_dml"] <= res["ci_95"][1], "Point estimate must be within 95% CI"
        assert res["sample_size"] == len(tx_df), "Sample size must match dataframe length"


class TestAdversarialGamePricing:
    """Validates game-theoretic pricing and QRE fixed-point convergence."""

    def test_level_k_strategy_bounds(self):
        engine = AdversarialPricingEngine(our_cost=35.0, rival_cost=40.0)
        res = engine.compute_level_k_strategy()

        assert res["Level_1_Our_Price"] > 35.0, "Our price must exceed unit cost"
        assert res["Level_2_Rival_Price"] > 40.0, "Rival price must exceed their cost"
        assert res["Level_3_Counter_Flank_Price"] > 35.0, "Counter-flank price must be profitable"
        assert res["Level_1_Expected_Profit"] > 0, "Expected profit must be positive"

    def test_quantal_response_equilibrium_convergence(self):
        engine = AdversarialPricingEngine()
        prob_our, prob_rival, exp_p1, exp_p2 = engine.solve_quantal_response_equilibrium()

        # Probabilities must be valid simplex measures
        assert np.isclose(np.sum(prob_our), 1.0, atol=1e-5), "Our QRE probabilities must sum to 1.0"
        assert np.isclose(np.sum(prob_rival), 1.0, atol=1e-5), "Rival QRE probabilities must sum to 1.0"
        assert np.all(prob_our >= 0.0), "Probabilities cannot be negative"
        assert np.all(prob_rival >= 0.0), "Probabilities cannot be negative"
        assert 50.0 <= exp_p1 <= 150.0, "Equilibrium price must lie within price grid"
        assert 50.0 <= exp_p2 <= 150.0, "Equilibrium price must lie within price grid"


class TestOstromCommonsGovernance:
    """Validates Common Pool Resource depletion and Ostrom kill gate behavior."""

    def test_congestion_monotonicity(self):
        engine = OstromCommonsGovernanceEngine(pool_capacity=100000.0, base_cac=20.0)
        cac_low, dep_low = engine.calculate_pool_cac(20000.0)
        cac_high, dep_high = engine.calculate_pool_cac(120000.0)

        assert cac_high > cac_low, "CAC must strictly increase with pool congestion"
        assert dep_high > dep_low, "Depletion index must increase with capital injection"

    def test_ostrom_flank_beats_unregulated_commons(self):
        engine = OstromCommonsGovernanceEngine()
        df = engine.simulate_bidding_war_defection_spiral(rounds=8)

        # In late rounds (depletion > threshold), Ostrom blended CAC must be strictly lower than unregulated CAC
        late_rounds = df[df["round"] >= 5]
        for _, row in late_rounds.iterrows():
            assert row["ostrom_blended_cac"] < row["unreg_market_cac"], "Ostrom governance must produce lower CAC during defection spirals"
            assert row["cac_savings_pct"] > 0, "Ostrom strategy must achieve positive savings percentage"


class TestSpectralAttribution:
    """Validates Spectral Graph Theory, Perron-Frobenius, and Laplacian connectivity."""

    def test_spectral_graph_invariants(self, robust_dataset):
        _, journeys_df = robust_dataset
        spectral_df, topo_df = compute_spectral_graph_attribution(journeys_df)

        assert not spectral_df.empty, "Spectral dataframe must not be empty"
        # Weights must sum to approximately 100%
        total_share = spectral_df["blended_spectral_share_pct"].sum()
        assert np.isclose(total_share, 100.0, atol=0.2), f"Spectral shares must sum to 100% (got {total_share})"

        # Fiedler eigenvalue of graph Laplacian must be non-negative
        fiedler = topo_df["algebraic_connectivity_fiedler"].iloc[0]
        assert fiedler >= 0.0, "Algebraic connectivity (Fiedler value) must be non-negative"

        # Attributed revenue must reconcile with total conversion revenue
        total_rev = journeys_df[journeys_df["converted"] == 1]["conversion_value"].sum()
        attributed_sum = spectral_df["spectral_attributed_revenue"].sum()
        assert np.isclose(attributed_sum, total_rev, atol=1.0), "Spectral attributed revenue must conserve total conversion revenue"


class TestStochasticFloatEngine:
    """Validates DuPont 3-tier identity and stochastic negative working capital float."""

    def test_dupont_identity(self):
        dupont = compute_dupont_3tier_decomposition(
            revenue=1000000.0,
            net_income=140000.0,
            total_assets=500000.0,
            shareholder_equity=250000.0,
        )
        # Margin = 14%, Turnover = 2.0x, Leverage = 2.0x => ROE = 14% * 2.0 * 2.0 = 56%
        assert np.isclose(dupont["profit_margin_pct"], 14.0, atol=0.01)
        assert np.isclose(dupont["asset_turnover_x"], 2.0, atol=0.01)
        assert np.isclose(dupont["equity_multiplier_x"], 2.0, atol=0.01)
        assert np.isclose(dupont["return_on_equity_pct"], 56.0, atol=0.01)

    def test_negative_working_capital_float_simulation(self):
        summary, sim_df = simulate_negative_working_capital_float(
            annual_revenue=3650000.0,  # $10,000/day
            dio_days=10.0,
            dso_days=2.0,
            dpo_days=42.0,  # CCC = 10 + 2 - 42 = -30 days
            treasury_yield_rate=0.06,
        )
        assert summary["cash_conversion_cycle_days"] == -30.0
        assert summary["is_negative_working_capital"] is True
        assert summary["average_liquid_float_balance"] > 250000.0, "Float balance must be positive under negative CCC"
        assert summary["total_annual_treasury_yield"] > 0, "Treasury yield must be positive"

        # Cumulative yield must be monotonically non-decreasing
        yields = sim_df["cumulative_treasury_yield"].values
        assert np.all(np.diff(yields) >= 0.0), "Cumulative treasury yield must be monotonically non-decreasing"
