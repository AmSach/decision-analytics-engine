"""
Higher-Order Spectral Graph Diffusion and Multi-Touch Attribution Engine.
Computes Perron-Frobenius eigenvector centrality, Katz multi-step path attenuation,
and Laplacian algebraic connectivity to evaluate network synergies across marketing channels.
"""

import numpy as np
import pandas as pd


def compute_spectral_graph_attribution(
    journeys_df: pd.DataFrame,
    katz_attenuation: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates multi-touch attribution via Spectral Graph Theory:
    1. Constructs adjacency and transition matrices across all marketing channels.
    2. Computes the Perron-Frobenius stationary distribution (leading eigenvector).
    3. Calculates the Katz Attenuation Matrix: K = (I - alpha * M)^(-1) - I.
    4. Evaluates Graph Laplacian Algebraic Connectivity (Fiedler Eigenvalue lambda_2).
    """
    channels_set = set()
    transitions = []

    for _, row in journeys_df[journeys_df["converted"] == 1].iterrows():
        path = [c.strip() for c in row["touchpoint_path"].split(">")]
        for ch in path:
            channels_set.add(ch)
        for i in range(len(path) - 1):
            transitions.append((path[i], path[i + 1]))

    channels = sorted(list(channels_set))
    n = len(channels)
    ch_to_idx = {ch: i for i, ch in enumerate(channels)}

    # Adjacency matrix
    A = np.zeros((n, n))
    for src, dst in transitions:
        A[ch_to_idx[src], ch_to_idx[dst]] += 1.0

    # Row-normalize to transition probability matrix M
    row_sums = A.sum(axis=1)
    M = np.divide(A, row_sums[:, np.newaxis], out=np.zeros_like(A), where=row_sums[:, np.newaxis] != 0)

    # 1. Perron-Frobenius Leading Eigenvector (Stationary Distribution via Power Iteration)
    v = np.ones(n) / n
    for _ in range(200):
        v_next = M.T @ v
        s = np.sum(v_next)
        if s > 0:
            v_next /= s
        if np.max(np.abs(v_next - v)) < 1e-7:
            break
        v = v_next

    # 2. Katz Multi-Step Path Attenuation: K = (I - alpha * M)^(-1) - I
    I = np.eye(n)
    alpha = min(katz_attenuation, 0.90)
    try:
        K = np.linalg.inv(I - alpha * M) - I
        katz_influence = np.sum(K, axis=1)
        katz_share = (katz_influence / np.sum(katz_influence)) if np.sum(katz_influence) > 0 else np.ones(n) / n
    except Exception:
        katz_share = np.ones(n) / n

    # 3. Graph Laplacian & Algebraic Connectivity (Fiedler Eigenvalue)
    # Symmetric adjacency for topology
    A_sym = 0.5 * (A + A.T)
    degrees = np.sum(A_sym, axis=1)
    L = np.diag(degrees) - A_sym
    eigvals = np.sort(np.linalg.eigvalsh(L))
    fiedler_value = float(eigvals[1]) if len(eigvals) > 1 else 0.0

    # Compile Results
    records = []
    total_rev = float(journeys_df[journeys_df["converted"] == 1]["conversion_value"].sum())

    for i, ch in enumerate(channels):
        pf_weight = float(v[i])
        kz_weight = float(katz_share[i])
        blended_spectral_weight = 0.5 * pf_weight + 0.5 * kz_weight
        attributed_rev = blended_spectral_weight * total_rev

        records.append({
            "channel": ch,
            "perron_frobenius_centrality": round(pf_weight * 100.0, 2),
            "katz_path_influence_pct": round(kz_weight * 100.0, 2),
            "blended_spectral_share_pct": round(blended_spectral_weight * 100.0, 2),
            "spectral_attributed_revenue": round(attributed_rev, 2),
        })

    spectral_df = pd.DataFrame(records).sort_values(by="spectral_attributed_revenue", ascending=False)
    topology_summary = pd.DataFrame([{
        "algebraic_connectivity_fiedler": round(fiedler_value, 4),
        "total_active_nodes": n,
        "spectral_radius": round(float(np.max(np.abs(np.linalg.eigvals(M)))), 3),
    }])

    return spectral_df, topology_summary
