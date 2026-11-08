"""
Unit and Invariant Tests for Enterprise Operations Research Suite:
1. Multi-Echelon Inventory Optimization (MEIO) & Dynamic Stochastic Safety Stock.
2. Multivariate Operational Anomaly & Margin Leakage Detection.
3. Supply Network Graph Centrality & Systemic Vulnerability.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_ecommerce_dataset
from src.supply_chain_meio import (
    compute_stochastic_safety_stock,
    optimize_sku_inventory_policy,
    optimize_multi_echelon_portfolio,
    standard_normal_loss,
)
from src.multivariate_anomaly import (
    detect_operational_anomalies,
    summarize_margin_leakage,
)
from src.supply_network_graph import EnterpriseNetworkGraph


@pytest.fixture(scope="module")
def benchmark_data():
    df, _ = generate_ecommerce_dataset(num_customers=300, num_days=180, random_seed=777)
    return df


class TestMultiEchelonInventoryOptimization:
    """Validates MEIO stochastic safety stock, EOQ, and working capital invariants."""

    def test_normal_loss_function(self):
        # Invariant: G(0) = 1 / sqrt(2*pi) ~= 0.3989
        loss_0 = standard_normal_loss(0.0)
        assert abs(loss_0 - 0.3989) < 0.01
        # Loss must be strictly monotonically decreasing in z
        assert standard_normal_loss(1.0) < loss_0
        assert standard_normal_loss(2.0) < standard_normal_loss(1.0)
        assert standard_normal_loss(3.0) >= 0.0

    def test_stochastic_safety_stock_invariants(self):
        # Invariant: Higher service level must yield strictly higher safety stock and ROP
        res_90 = compute_stochastic_safety_stock(10.0, 3.0, 14.0, 3.0, service_level=0.90)
        res_95 = compute_stochastic_safety_stock(10.0, 3.0, 14.0, 3.0, service_level=0.95)
        res_99 = compute_stochastic_safety_stock(10.0, 3.0, 14.0, 3.0, service_level=0.99)

        assert res_90["safety_stock"] < res_95["safety_stock"] < res_99["safety_stock"]
        assert res_90["reorder_point"] < res_95["reorder_point"] < res_99["reorder_point"]
        assert res_90["expected_unit_shortage_per_cycle"] > res_99["expected_unit_shortage_per_cycle"]

    def test_sku_policy_working_capital(self):
        policy = optimize_sku_inventory_policy(
            annual_demand=12000.0,
            unit_cost=45.0,
            target_service_level=0.95,
        )
        assert policy["economic_order_quantity"] > 0
        assert policy["fill_rate_type2"] >= 0.90
        assert policy["working_capital_committed"] > 0
        assert policy["total_annual_inventory_cost"] > 0

    def test_portfolio_meio_aggregation(self, benchmark_data):
        meio_df = optimize_multi_echelon_portfolio(benchmark_data, service_level=0.95)
        assert len(meio_df) > 0
        assert "working_capital_committed" in meio_df.columns
        assert (meio_df["safety_stock"] > 0).all()


class TestMultivariateAnomalyDetection:
    """Validates Mahalanobis distance regularized outlier scoring and margin leak isolation."""

    def test_anomaly_detection_execution(self, benchmark_data):
        res_df = detect_operational_anomalies(benchmark_data, alpha=0.01)
        assert "mahalanobis_dist_sq" in res_df.columns
        assert "anomaly_p_value" in res_df.columns
        assert "is_operational_anomaly" in res_df.columns
        assert (res_df["mahalanobis_dist_sq"] >= 0).all()
        assert (res_df["anomaly_p_value"] >= 0.0).all()
        assert (res_df["anomaly_p_value"] <= 1.0).all()

    def test_margin_leak_summary(self, benchmark_data):
        res_df = detect_operational_anomalies(benchmark_data, alpha=0.05)
        summary = summarize_margin_leakage(res_df)
        assert summary["total_transactions"] == len(benchmark_data)
        assert summary["anomalous_transactions_count"] >= 0
        assert summary["anomaly_rate_pct"] >= 0.0


class TestEnterpriseNetworkGraph:
    """Validates PageRank, degree centrality, and systemic fragility graph analytics."""

    def test_pagerank_conservation(self, benchmark_data):
        graph = EnterpriseNetworkGraph(benchmark_data, source_col="customer_id", target_col="sku")
        pr = graph.compute_pagerank()
        assert len(pr) > 0
        # Invariant: PageRank probabilities sum to 1.0
        total_pr = sum(pr.values())
        assert abs(total_pr - 1.0) < 1e-4

    def test_systemic_fragility_ranking(self, benchmark_data):
        graph = EnterpriseNetworkGraph(benchmark_data, source_col="customer_id", target_col="sku")
        frag_df = graph.evaluate_systemic_fragility(top_k=5)
        assert len(frag_df) <= 5
        assert "systemic_fragility_index" in frag_df.columns
        # Top node must have non-negative fragility score
        assert (frag_df["systemic_fragility_index"] >= 0).all()
