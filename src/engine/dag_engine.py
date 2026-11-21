"""
Directed Acyclic Graph (DAG) Pipeline Engine.
Manages nodes, connection edges, topological sorting, reactive execution,
and result caching across the node network.
"""

from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, deque
import pandas as pd

from src.engine.audit_ledger import AuditLedger
from src.engine.formula_parser import evaluate_formula
from src.data.synthetic_data import generate_enterprise_suite
from src.analytics import (
    compute_causal_elasticity,
    compute_cfar_monte_carlo,
    compute_anomaly_sweep,
    compute_inventory_meio,
    compute_cohort_retention,
    compute_markov_attribution,
    compute_mixed_logit_cannibalization,
    compute_bandit_experiment,
    compute_network_fragility,
    compute_knapsack_allocation,
)


class Node:
    """Represents an atomic computation step in the DAG pipeline."""

    def __init__(self, node_id: str, node_type: str, title: str, params: Optional[Dict[str, Any]] = None):
        self.node_id = node_id
        self.node_type = node_type
        self.title = title
        self.params = params or {}
        self.output_data: Optional[Any] = None  # DataFrame or Analysis Dict
        self.status: str = "PENDING"  # PENDING, SUCCESS, ERROR
        self.error_message: Optional[str] = None
        self.x: float = 100.0
        self.y: float = 100.0


class DAGEngine:
    """Directed Acyclic Graph engine managing reactive node execution."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Tuple[str, str]] = []  # (source_id, target_id)
        self.audit_ledger = AuditLedger()

    def add_node(self, node_type: str, title: str, params: Optional[Dict[str, Any]] = None, x: float = 100.0, y: float = 100.0) -> Node:
        """Instantiates and registers a new node."""
        node_id = f"node_{len(self.nodes) + 1:03d}_{node_type}"
        node = Node(node_id, node_type, title, params)
        node.x = x
        node.y = y
        self.nodes[node_id] = node
        return node

    def add_edge(self, source_id: str, target_id: str) -> bool:
        """Connects source output to target input, validating against cycles."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
        if source_id == target_id:
            return False
        if (source_id, target_id) in self.edges:
            return True

        # Check cycle
        self.edges.append((source_id, target_id))
        if self._has_cycle():
            self.edges.remove((source_id, target_id))
            return False

        return True

    def remove_edge(self, source_id: str, target_id: str):
        if (source_id, target_id) in self.edges:
            self.edges.remove((source_id, target_id))

    def remove_node(self, node_id: str):
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.edges = [(src, tgt) for src, tgt in self.edges if src != node_id and tgt != node_id]

    def _has_cycle(self) -> bool:
        """Detects cycles via Kahn's algorithm."""
        in_degree = {n: 0 for n in self.nodes}
        adj = defaultdict(list)
        for src, tgt in self.edges:
            adj[src].append(tgt)
            in_degree[tgt] += 1

        queue = deque([n for n, deg in in_degree.items() if deg == 0])
        visited = 0
        while queue:
            curr = queue.popleft()
            visited += 1
            for nxt in adj[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        return visited != len(self.nodes)

    def topological_sort(self) -> List[str]:
        """Returns topological ordering of nodes for execution."""
        in_degree = {n: 0 for n in self.nodes}
        adj = defaultdict(list)
        for src, tgt in self.edges:
            adj[src].append(tgt)
            in_degree[tgt] += 1

        queue = deque([n for n, deg in in_degree.items() if deg == 0])
        order = []
        while queue:
            curr = queue.popleft()
            order.append(curr)
            for nxt in adj[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        return order

    def execute_node(self, node_id: str) -> bool:
        """Executes a single node using inputs from its upstream predecessors."""
        node = self.nodes.get(node_id)
        if not node:
            return False

        # Gather inputs from incoming edges
        incoming = [src for src, tgt in self.edges if tgt == node_id]
        parent_data = None
        if incoming:
            parent_node = self.nodes.get(incoming[0])
            if parent_node:
                parent_data = parent_node.output_data

        try:
            # Source Nodes
            if node.node_type == "source_synthetic":
                table_type = node.params.get("table", "transactions")
                suite = generate_enterprise_suite()
                node.output_data = suite.get(table_type, suite["transactions"])
                node.status = "SUCCESS"

            elif node.node_type == "source_raw":
                # Raw provided dataframe in params
                node.output_data = node.params.get("dataframe", pd.DataFrame())
                node.status = "SUCCESS"

            # Transform Nodes
            elif node.node_type == "transform_filter":
                if isinstance(parent_data, pd.DataFrame):
                    query_str = node.params.get("query", "")
                    if query_str:
                        node.output_data = parent_data.query(query_str).copy()
                    else:
                        node.output_data = parent_data.copy()
                    node.status = "SUCCESS"
                else:
                    node.error_message = "Parent node did not produce a tabular DataFrame."
                    node.status = "ERROR"

            elif node.node_type == "transform_formula":
                if isinstance(parent_data, pd.DataFrame):
                    formula = node.params.get("formula", "")
                    col_name = node.params.get("new_column", "computed_col")
                    series, err = evaluate_formula(parent_data, formula, col_name)
                    if err:
                        node.error_message = err
                        node.status = "ERROR"
                    else:
                        out_df = parent_data.copy()
                        out_df[col_name] = series
                        node.output_data = out_df
                        node.status = "SUCCESS"
                else:
                    node.error_message = "Parent node must produce a tabular DataFrame."
                    node.status = "ERROR"

            elif node.node_type == "transform_dedup":
                if isinstance(parent_data, pd.DataFrame):
                    subset = node.params.get("subset", None)
                    node.output_data = parent_data.drop_duplicates(subset=subset).copy()
                    node.status = "SUCCESS"
                else:
                    node.status = "ERROR"

            elif node.node_type == "transform_groupby":
                if isinstance(parent_data, pd.DataFrame):
                    by = node.params.get("by", parent_data.columns[0])
                    agg_col = node.params.get("agg_col", parent_data.columns[1] if len(parent_data.columns) > 1 else by)
                    agg_func = node.params.get("func", "sum")
                    node.output_data = parent_data.groupby(by)[agg_col].agg(agg_func).reset_index()
                    node.status = "SUCCESS"
                else:
                    node.status = "ERROR"

            # Analytics Nodes
            elif node.node_type == "analytics_causal_elasticity":
                df = parent_data if isinstance(parent_data, pd.DataFrame) else generate_enterprise_suite()["transactions"]
                node.output_data = compute_causal_elasticity(df)
                node.status = "SUCCESS"

            elif node.node_type == "analytics_cfar":
                df = parent_data if isinstance(parent_data, pd.DataFrame) else generate_enterprise_suite()["transactions"]
                floor = float(node.params.get("covenant_floor", 300000.0))
                node.output_data = compute_cfar_monte_carlo(df, debt_covenant_floor=floor)
                node.status = "SUCCESS"

            elif node.node_type == "analytics_anomaly_sweep":
                df = parent_data if isinstance(parent_data, pd.DataFrame) else generate_enterprise_suite()["transactions"]
                node.output_data = compute_anomaly_sweep(df)
                node.status = "SUCCESS"

            elif node.node_type == "analytics_inventory_meio":
                df = parent_data if isinstance(parent_data, pd.DataFrame) else generate_enterprise_suite()["inventory"]
                node.output_data = compute_inventory_meio(df)
                node.status = "SUCCESS"

            elif node.node_type == "analytics_cohort_churn":
                df = parent_data if isinstance(parent_data, pd.DataFrame) else generate_enterprise_suite()["transactions"]
                node.output_data = compute_cohort_retention(df)
                node.status = "SUCCESS"

            elif node.node_type == "analytics_markov_attribution":
                df = parent_data if isinstance(parent_data, pd.DataFrame) else generate_enterprise_suite()["journeys"]
                node.output_data = compute_markov_attribution(df)
                node.status = "SUCCESS"

            elif node.node_type == "analytics_mixed_logit":
                df = parent_data if isinstance(parent_data, pd.DataFrame) else generate_enterprise_suite()["transactions"]
                node.output_data = compute_mixed_logit_cannibalization(df)
                node.status = "SUCCESS"

            elif node.node_type == "analytics_bandit":
                node.output_data = compute_bandit_experiment()
                node.status = "SUCCESS"

            elif node.node_type == "analytics_network_fragility":
                df = parent_data if isinstance(parent_data, pd.DataFrame) else generate_enterprise_suite()["transactions"]
                node.output_data = compute_network_fragility(df)
                node.status = "SUCCESS"

            elif node.node_type == "analytics_knapsack":
                node.output_data = compute_knapsack_allocation()
                node.status = "SUCCESS"

            else:
                node.output_data = parent_data
                node.status = "SUCCESS"

            # Record in Audit Ledger
            in_rows = len(parent_data) if isinstance(parent_data, pd.DataFrame) else 0
            out_rows = len(node.output_data) if isinstance(node.output_data, pd.DataFrame) else 0
            self.audit_ledger.record_action(
                node_id=node.node_id,
                action_type=node.node_type,
                rationale=f"Executed {node.title}",
                input_rows=in_rows,
                output_rows=out_rows,
            )
            return True

        except Exception as e:
            node.status = "ERROR"
            node.error_message = str(e)
            return False

    def execute_all(self) -> bool:
        """Executes all nodes in topological dependency sequence."""
        order = self.topological_sort()
        success = True
        for node_id in order:
            res = self.execute_node(node_id)
            if not res:
                success = False
        return success
