"""
Invariant Test Suite for Palantir-Grade Architecture:
1. Enterprise Decision Ontology & Digital Twin Operational Telemetry.
2. Counterfactual Scenario Sandbox & Multi-Vector Shock Blast Radius.
"""

import pandas as pd
import pytest

from src.data_generator import generate_ecommerce_dataset
from src.enterprise_ontology import EnterpriseDecisionOntology
from src.scenario_sandbox import ScenarioSandboxEngine, ScenarioShockParameters


@pytest.fixture(scope="module")
def sample_ontology_data():
    tx_df, _ = generate_ecommerce_dataset(num_customers=400, num_days=180, random_seed=999)
    return tx_df


class TestEnterpriseDecisionOntology:
    """Validates operational ontology relational binding and risk telemetry."""

    def test_ontology_node_binding(self, sample_ontology_data):
        ontology = EnterpriseDecisionOntology(sample_ontology_data)

        assert ontology.entities["total_customers"] == sample_ontology_data["customer_id"].nunique()
        assert ontology.entities["total_orders"] == len(sample_ontology_data)
        assert ontology.entities["total_skus"] == sample_ontology_data["sku"].nunique()
        assert ontology.gross_revenue == sample_ontology_data["gross_revenue"].sum()

    def test_sku_node_query(self, sample_ontology_data):
        ontology = EnterpriseDecisionOntology(sample_ontology_data)
        first_sku = sample_ontology_data["sku"].iloc[0]
        node = ontology.query_sku_node(first_sku)

        assert node["sku"] == first_sku
        assert node["unit_volume"] > 0
        assert node["total_revenue"] > 0
        assert node["margin_percentage"] > 0

    def test_telemetry_action_triggers(self, sample_ontology_data):
        ontology = EnterpriseDecisionOntology(sample_ontology_data)
        telemetry = ontology.evaluate_live_telemetry()

        assert "MONITORED" in telemetry.customer_churn_risk_level
        assert "FLOAT" in telemetry.working_capital_health
        assert len(telemetry.recommended_board_action) > 10


class TestScenarioSandboxBlastRadius:
    """Validates counterfactual shock simulations and kill-gate governance."""

    def test_multi_vector_shock_propagation(self, sample_ontology_data):
        rev = sample_ontology_data["gross_revenue"].sum()
        cogs = sample_ontology_data["cogs"].sum()
        sandbox = ScenarioSandboxEngine(sample_ontology_data, base_annual_revenue=rev, base_annual_cogs=cogs)

        # Apply acute macro shock: 15% rival price drop + 20% COGS inflation
        shocks = ScenarioShockParameters(
            competitor_price_cut_pct=-0.15,
            cogs_inflation_surge_pct=0.20,
            supplier_dpo_compression_days=-15.0,
        )
        blast_radius = sandbox.simulate_blast_radius(shocks)

        assert blast_radius["shocked_revenue"] < blast_radius["baseline_revenue"]
        assert blast_radius["margin_delta"] < 0, "Adverse macro shock must reduce gross margin"
        assert blast_radius["shocked_liquid_float"] < blast_radius["baseline_liquid_float"]
        # Severe shock must trigger critical defense protocol
        assert "CRITICAL DEFENSE GATE" in blast_radius["remediation_protocol"]
