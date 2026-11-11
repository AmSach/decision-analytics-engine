"""
Source package initialization.
Exposes advanced strategic analytics, econometrics, game theory, and optimization engines.
"""

from .conformal_prediction import ConformalPredictor
from .shap_attribution import ShapleyAttributionEngine
from .causal_sensitivity_bounds import CausalSensitivityEngine
from .markov_wpa_xrop import MarkovWPAEngine
from .knapsack_auction_optimizer import KnapsackAuctionOptimizer

__all__ = [
    "ConformalPredictor",
    "ShapleyAttributionEngine",
    "CausalSensitivityEngine",
    "MarkovWPAEngine",
    "KnapsackAuctionOptimizer",
]
