"""
Analytics & Quantitative Decision Engines Package.
"""

from src.analytics.causal_elasticity import compute_causal_elasticity
from src.analytics.cash_flow_at_risk import compute_cfar_monte_carlo
from src.analytics.anomaly_sweep import compute_anomaly_sweep
from src.analytics.inventory_meio import compute_inventory_meio
from src.analytics.cohort_churn import compute_cohort_retention
from src.analytics.attribution_markov import compute_markov_attribution
from src.analytics.mixed_logit_cannibalization import compute_mixed_logit_cannibalization
from src.analytics.bandit_experiment import compute_bandit_experiment
from src.analytics.network_fragility import compute_network_fragility
from src.analytics.knapsack_allocator import compute_knapsack_allocation

__all__ = [
    "compute_causal_elasticity",
    "compute_cfar_monte_carlo",
    "compute_anomaly_sweep",
    "compute_inventory_meio",
    "compute_cohort_retention",
    "compute_markov_attribution",
    "compute_mixed_logit_cannibalization",
    "compute_bandit_experiment",
    "compute_network_fragility",
    "compute_knapsack_allocation",
]
