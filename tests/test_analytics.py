"""
Rigorous Test Suite for B-School Decision Analytics Engine.
Validates econometric demand invariants, cohort retention matrix properties,
RFM segment completeness, and Markov Chain transition row-stochasticity.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_ecommerce_dataset
from src.cohort_retention import compute_cohort_retention, compute_churn_hazard
from src.price_elasticity import estimate_price_elasticity, simulate_price_shift_margin_impact
from src.clv_model import compute_rfm_metrics, get_segment_waterfall_summary
from src.marketing_attribution import compute_markov_attribution


@pytest.fixture(scope="module")
def sample_dataset():
    """Generates deterministic sample dataset for regression tests."""
    tx_df, journeys_df = generate_ecommerce_dataset(
        num_customers=600,
        start_date="2023-01-01",
        num_days=240,
        random_seed=123,
    )
    return tx_df, journeys_df


class TestDataGeneratorInvariants:
    """Validates structural data integrity and transactional consistency."""

    def test_schema_and_non_null(self, sample_dataset):
        tx_df, journeys_df = sample_dataset
        assert not tx_df.empty, "Transactions dataframe must not be empty"
        assert not journeys_df.empty, "Journeys dataframe must not be empty"

        required_cols = [
            "transaction_id", "customer_id", "order_date", "sku", 
            "effective_price", "quantity", "gross_revenue", "cogs", "gross_margin"
        ]
        for col in required_cols:
            assert col in tx_df.columns, f"Missing required column {col} in transactions"
            assert tx_df[col].isna().sum() == 0, f"Column {col} contains unexpected NaN values"

    def test_financial_reconciliation(self, sample_dataset):
        tx_df, _ = sample_dataset
        # Gross revenue = effective_price * quantity
        expected_rev = (tx_df["effective_price"] * tx_df["quantity"]).round(2)
        assert np.allclose(tx_df["gross_revenue"], expected_rev, atol=0.02)

        # Gross margin = gross_revenue - cogs
        expected_margin = (tx_df["gross_revenue"] - tx_df["cogs"]).round(2)
        assert np.allclose(tx_df["gross_margin"], expected_margin, atol=0.02)


class TestCohortRetentionInvariants:
    """Tests mathematical properties of cohort survival and hazard rates."""

    def test_period_zero_is_100_percent(self, sample_dataset):
        tx_df, _ = sample_dataset
        user_ret, rev_ret = compute_cohort_retention(tx_df)
        
        # Period 0 user retention must be exactly 100% for all valid cohorts
        assert 0 in user_ret.columns, "Period 0 must be present in cohort matrix"
        p0_values = user_ret[0].dropna()
        assert np.allclose(p0_values, 100.0, atol=0.01), "Period 0 user retention must be 100%"

    def test_retention_bounded(self, sample_dataset):
        tx_df, _ = sample_dataset
        user_ret, _ = compute_cohort_retention(tx_df)
        # Retention rates cannot be negative or exceed 100%
        valid_ret = user_ret.values[~np.isnan(user_ret.values)]
        assert np.all(valid_ret >= 0.0), "Retention rates cannot be negative"
        assert np.all(valid_ret <= 100.0), "User retention rate cannot exceed 100%"

    def test_churn_hazard_math(self, sample_dataset):
        tx_df, _ = sample_dataset
        user_ret, _ = compute_cohort_retention(tx_df)
        hazard_df = compute_churn_hazard(user_ret)
        
        assert "hazard_rate_pct" in hazard_df.columns
        assert "survival_rate_pct" in hazard_df.columns
        
        # Hazard rates must be within [0, 100]
        h_vals = hazard_df["hazard_rate_pct"].values
        assert np.all(h_vals >= 0.0), "Hazard rates cannot be negative"
        assert np.all(h_vals <= 100.0), "Hazard rates cannot exceed 100%"


class TestEconometricPriceElasticityInvariants:
    """Tests demand elasticity modeling, Lerner rule, and margin simulation."""

    def test_negative_price_elasticity(self, sample_dataset):
        tx_df, _ = sample_dataset
        results = estimate_price_elasticity(tx_df)
        assert len(results) > 0, "Should have estimated elasticity for valid SKUs"
        
        for sku, info in results.items():
            # Demand curve downward sloping
            assert info["price_elasticity"] < 0, f"Elasticity for {sku} should be negative (Law of Demand)"
            assert info["avg_selling_price"] > info["unit_cost_cogs"], "Average price must exceed unit cost"

    def test_margin_simulation_invariants(self):
        # Base case: 10% price increase on elastic good (E = -1.5)
        sim = simulate_price_shift_margin_impact(
            current_price=100.0,
            current_volume=1000,
            unit_cost=40.0,
            elasticity=-1.5,
            price_change_pct=0.10,
        )
        assert sim["new_price"] == 110.0
        # Volume drops by 15% (1.5 * 10%)
        assert sim["new_volume"] == 850
        # Check margin consistency
        expected_margin = (110.0 - 40.0) * 850
        assert sim["new_gross_margin"] == expected_margin


class TestCLVAndRFMSegmentationInvariants:
    """Tests RFM scoring partition completeness and forward CLV equity."""

    def test_mece_customer_partition(self, sample_dataset):
        tx_df, _ = sample_dataset
        rfm = compute_rfm_metrics(tx_df)
        summary = get_segment_waterfall_summary(rfm)

        # Sum of segment customers must equal total unique customers
        total_unique = tx_df["customer_id"].nunique()
        assert rfm["customer_id"].nunique() == total_unique
        assert summary["customer_count"].sum() == total_unique, "Segments must form a MECE partition"

    def test_probability_active_bounds(self, sample_dataset):
        tx_df, _ = sample_dataset
        rfm = compute_rfm_metrics(tx_df)
        
        assert np.all(rfm["prob_active"] >= 0.0), "Prob active cannot be negative"
        assert np.all(rfm["prob_active"] <= 1.0), "Prob active cannot exceed 1.0"
        assert np.all(rfm["forward_12m_clv_margin"] >= 0.0), "Forward expected margin cannot be negative"


class TestMarkovChainAttributionInvariants:
    """Tests Markov chain transition matrix stochasticity and removal attribution weights."""

    def test_transition_matrix_row_stochasticity(self, sample_dataset):
        _, journeys_df = sample_dataset
        _, trans_matrix = compute_markov_attribution(journeys_df)

        # Each row of transition matrix must sum to 1.0 (or 0 for non-visited states)
        row_sums = trans_matrix.sum(axis=1)
        for state, s in row_sums.items():
            if s > 0:
                assert np.isclose(s, 1.0, atol=1e-3), f"Row {state} in transition matrix does not sum to 1.0 (sum={s})"

    def test_attribution_weights_conservation(self, sample_dataset):
        _, journeys_df = sample_dataset
        comp_df, _ = compute_markov_attribution(journeys_df)

        total_weight = comp_df["markov_weight_pct"].sum()
        assert np.isclose(total_weight, 100.0, atol=0.2), f"Attribution weights must sum to 100% (got {total_weight})"
        assert np.all(comp_df["markov_weight_pct"] >= 0.0), "Weights cannot be negative"
