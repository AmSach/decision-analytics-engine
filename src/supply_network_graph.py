"""
Supply Chain & Enterprise Dependency Network Graph Engine.
Constructs bipartite and directed graph networks linking Accounts, SKUs, Categories, and Logistics Hubs.
Computes PageRank authority vectors, betweenness and degree centralities, and systemic vulnerability
to quantify single points of failure (SPOF) and margin concentration risks across the operating network.
"""

import numpy as np
import pandas as pd


class EnterpriseNetworkGraph:
    """
    Constructs an enterprise dependency graph from transaction records
    and calculates graph centrality and systemic fragility metrics.
    """

    def __init__(self, df: pd.DataFrame, source_col: str = "customer_id", target_col: str = None, weight_col: str = "gross_revenue"):
        self.df = df.copy()
        self.source_col = source_col if source_col in self.df.columns else self.df.columns[0]
        
        if target_col and target_col in self.df.columns:
            self.target_col = target_col
        elif "category" in self.df.columns:
            self.target_col = "category"
        elif "sku" in self.df.columns:
            self.target_col = "sku"
        else:
            self.target_col = self.df.columns[1]

        self.weight_col = weight_col if weight_col in self.df.columns else None
        self._build_network()

    def _build_network(self):
        """Constructs adjacency matrix and node index mappings."""
        sources = self.df[self.source_col].astype(str).unique().tolist()
        targets = self.df[self.target_col].astype(str).unique().tolist()

        self.nodes = list(set(sources + targets))
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for node, i in self.node_to_idx.items()}
        self.n_nodes = len(self.nodes)

        # Build adjacency matrix
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        for _, row in self.df.iterrows():
            u = self.node_to_idx[str(row[self.source_col])]
            v = self.node_to_idx[str(row[self.target_col])]
            w = float(row[self.weight_col]) if self.weight_col in row else 1.0
            adj[u, v] += max(0.0, w)
            adj[v, u] += max(0.0, w)  # Undirected operational linkage

        self.adjacency = adj

    def compute_pagerank(self, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> dict[str, float]:
        """
        Computes PageRank centrality vector via power iteration:
        r = (1 - d) / N * 1 + d * M * r
        """
        N = self.n_nodes
        if N == 0:
            return {}

        col_sums = self.adjacency.sum(axis=0)
        # Column-stochastic transition matrix
        M = np.zeros_like(self.adjacency)
        for j in range(N):
            if col_sums[j] > 0:
                M[:, j] = self.adjacency[:, j] / col_sums[j]
            else:
                M[:, j] = 1.0 / N

        r = np.ones(N) / N
        teleport = np.ones(N) * ((1.0 - damping) / N)

        for _ in range(max_iter):
            r_next = damping * (M @ r) + teleport
            if np.linalg.norm(r_next - r, 1) < tol:
                break
            r = r_next

        return {self.idx_to_node[i]: float(r[i]) for i in range(N)}

    def compute_degree_centrality(self) -> dict[str, dict]:
        """
        Calculates node degree, weighted throughput volume, and network connectivity share.
        """
        deg = (self.adjacency > 0).sum(axis=1)
        weighted_deg = self.adjacency.sum(axis=1)
        total_volume = weighted_deg.sum()

        results = {}
        for i in range(self.n_nodes):
            node = self.idx_to_node[i]
            results[node] = {
                "degree_connections": int(deg[i]),
                "weighted_volume": round(float(weighted_deg[i]), 2),
                "volume_share_pct": round((weighted_deg[i] / total_volume) * 100.0, 4) if total_volume > 0 else 0.0,
            }
        return results

    def evaluate_systemic_fragility(self, top_k: int = 10) -> pd.DataFrame:
        """
        Identifies top systemic vulnerability nodes based on PageRank and Volume Share,
        quantifying the revenue blast radius if a node encounters a supply/fulfillment shock.
        """
        pr = self.compute_pagerank()
        dc = self.compute_degree_centrality()

        total_system_rev = float(self.df[self.weight_col].sum()) if self.weight_col in self.df.columns else 1.0

        records = []
        for node in self.nodes:
            node_rev = dc[node]["weighted_volume"]
            pr_score = pr.get(node, 0.0)
            # Fragility index is composite of PageRank authority and volume concentration
            fragility = pr_score * (node_rev / total_system_rev if total_system_rev > 0 else 1.0) * 1000.0

            records.append({
                "entity_node": node,
                "pagerank_score": round(pr_score, 6),
                "degree_connections": dc[node]["degree_connections"],
                "total_flow_volume": node_rev,
                "volume_share_pct": dc[node]["volume_share_pct"],
                "systemic_fragility_index": round(fragility, 4),
            })

        frag_df = pd.DataFrame(records).sort_values("systemic_fragility_index", ascending=False).reset_index(drop=True)
        return frag_df.head(top_k)
