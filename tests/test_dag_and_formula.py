"""
DAG Pipeline and Formula Invariant Tests.
Validates topological sorting, cycle detection, reactive node execution,
formula evaluation, and audit ledger SHA-256 hash chaining.
"""

import pytest
import pandas as pd
import numpy as np

from src.engine.dag_engine import DAGEngine
from src.engine.formula_parser import evaluate_formula
from src.engine.audit_ledger import AuditLedger


class TestDAGAndFormulaInvariants:
    """Validates structural integrity of the pipeline engine."""

    def test_dag_topological_sort_and_cycle_prevention(self):
        dag = DAGEngine()
        n1 = dag.add_node("source_synthetic", "Source")
        n2 = dag.add_node("transform_filter", "Filter")
        n3 = dag.add_node("analytics_causal_elasticity", "Analytics")

        assert dag.add_edge(n1.node_id, n2.node_id) is True
        assert dag.add_edge(n2.node_id, n3.node_id) is True

        order = dag.topological_sort()
        assert order == [n1.node_id, n2.node_id, n3.node_id]

        # Prevent cycle (n3 -> n1)
        assert dag.add_edge(n3.node_id, n1.node_id) is False

    def test_formula_evaluation_math(self):
        df = pd.DataFrame({
            "gross_revenue": [100.0, 200.0, 300.0],
            "cogs": [40.0, 70.0, 110.0],
            "units": [2, 4, 6]
        })

        # Test subtraction
        s, err = evaluate_formula(df, "=gross_revenue - cogs", "margin")
        assert err is None
        assert s.tolist() == [60.0, 130.0, 190.0]

        # Test multiplication and functions
        s2, err2 = evaluate_formula(df, "ROUND(units * 1.5, 0)", "scaled")
        assert err2 is None
        assert s2.tolist() == [3.0, 6.0, 9.0]

    def test_audit_ledger_cryptographic_chain(self):
        ledger = AuditLedger()
        r1 = ledger.record_action("node_01", "INGEST", "Loaded initial CSV", 0, 1000)
        r2 = ledger.record_action("node_02", "FILTER", "Dropped outliers", 1000, 980)

        assert ledger.verify_integrity() is True
        assert r2["prev_hash"] == r1["hash"]
        assert len(ledger.get_entries()) == 2
