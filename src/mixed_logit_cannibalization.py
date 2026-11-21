"""
Random Coefficient / Mixed Logit Demand and Product Cannibalization Engine.
Engineered for C-suite portfolio managers, pricing quants, and management consultants.
Models consumer choice under taste heterogeneity, overcoming the restrictive Independence of
Irrelevant Alternatives (IIA) assumption of basic Multinomial Logit.
Computes:
1. Heterogeneous Consumer Willingness-To-Pay (WTP) distributions
2. Full N x N Cross-Price Elasticity Matrix
3. Diversion Ratios and Portfolio Cannibalization Shares:
   Diversion_Ratio(j -> k) = (dQ_k / dP_j) / |dQ_j / dP_j|
Quantifies the exact percentage of volume defecting to internal portfolio SKUs vs competitor outside goods.
"""

import numpy as np


class MixedLogitCannibalizationEngine:
    """
    Simulates consumer choice under random coefficient utility:
    U_ij = delta_j - beta_i * P_j + epsilon_ij
    where beta_i ~ Normal(mu_beta, sigma_beta^2) or LogNormal represents heterogeneous price sensitivity.
    """

    def __init__(
        self,
        products: list[dict],
        outside_good_utility: float = 0.0,
        mu_price_sensitivity: float = 0.05,
        sigma_price_sensitivity: float = 0.015,
        num_simulated_consumers: int = 2000,
        random_seed: int = 42,
    ):
        """
        products: list of dicts with keys: ['name', 'price', 'base_quality', 'cogs']
        """
        self.products = products
        self.n_products = len(products)
        self.outside_utility = outside_good_utility
        self.mu_beta = mu_price_sensitivity
        self.sigma_beta = sigma_price_sensitivity
        self.n_consumers = num_simulated_consumers
        self.seed = random_seed

        # Draw heterogeneous consumer price sensitivities: beta_i > 0
        rng = np.random.default_rng(random_seed)
        # Use log-normal to ensure strictly positive price sensitivity
        self.betas = rng.lognormal(
            mean=np.log(self.mu_beta),
            sigma=self.sigma_beta,
            size=self.n_consumers,
        )

    def compute_choice_probabilities(self, prices: np.ndarray = None) -> np.ndarray:
        """
        Computes the market share choice probability for each product averaged across consumers.
        Returns array of shape (n_products + 1,) including the outside good (index -1).
        """
        if prices is None:
            prices = np.array([p["price"] for p in self.products])

        qualities = np.array([p["base_quality"] for p in self.products])

        # Systematic utilities for consumers: (n_consumers, n_products)
        # V_ij = quality_j - beta_i * price_j
        V = qualities[None, :] - (self.betas[:, None] * prices[None, :])
        # Outside good utility V_0 = 0.0
        V_outside = np.full((self.n_consumers, 1), self.outside_utility)
        V_all = np.hstack([V, V_outside])

        # Logit probabilities per consumer: exp(V) / sum(exp(V))
        # Numerical stability via max subtraction
        V_max = np.max(V_all, axis=1, keepdims=True)
        exp_V = np.exp(V_all - V_max)
        P_consumer = exp_V / np.sum(exp_V, axis=1, keepdims=True)

        # Average market shares across consumers: shape (n_products + 1,)
        market_shares = np.mean(P_consumer, axis=0)
        return market_shares

    def compute_cannibalization_and_elasticity(self, price_shock_pct: float = 0.01) -> dict:
        """
        Calculates the complete N x N cross-price elasticity matrix and diversion ratio matrix
        via numerical finite-differencing.
        """
        base_prices = np.array([float(p["price"]) for p in self.products])
        base_shares = self.compute_choice_probabilities(base_prices)
        product_names = [p["name"] for p in self.products]

        elasticity_matrix = np.zeros((self.n_products, self.n_products))
        diversion_matrix = np.zeros((self.n_products, self.n_products))
        outside_diversion = np.zeros(self.n_products)

        for j in range(self.n_products):
            # Shock price of product j
            shocked_prices = base_prices.copy()
            delta_p = base_prices[j] * price_shock_pct
            shocked_prices[j] += delta_p

            shocked_shares = self.compute_choice_probabilities(shocked_prices)
            delta_shares = shocked_shares - base_shares

            # Own share change (should be negative)
            delta_own = delta_shares[j]

            for k in range(self.n_products):
                # Cross-price elasticity: (dShare_k / Share_k) / (dPrice_j / Price_j)
                pct_share_change = delta_shares[k] / base_shares[k] if base_shares[k] > 0 else 0.0
                elasticity_matrix[k, j] = pct_share_change / price_shock_pct

                # Diversion ratio: fraction of lost sales in j captured by product k
                if abs(delta_own) > 1e-9 and k != j:
                    diversion_matrix[j, k] = max(0.0, delta_shares[k] / abs(delta_own))

            # Fraction diverted to the outside good (competitors / non-purchase)
            if abs(delta_own) > 1e-9:
                outside_diversion[j] = max(0.0, delta_shares[-1] / abs(delta_own))

        base_metrics = []
        for i, p in enumerate(self.products):
            share = base_shares[i]
            own_elasticity = elasticity_matrix[i, i]
            internal_cann = float(np.sum(diversion_matrix[i]) * 100.0)
            base_metrics.append({
                "product": p["name"],
                "price": p["price"],
                "market_share_pct": round(share * 100.0, 2),
                "own_price_elasticity": round(own_elasticity, 3),
                "internal_portfolio_cannibalization_pct": round(internal_cann, 2),
                "external_competitor_leakage_pct": round(float(outside_diversion[i] * 100.0), 2),
            })

        return {
            "portfolio_summary": base_metrics,
            "elasticity_matrix": elasticity_matrix,
            "diversion_ratio_matrix_pct": diversion_matrix,
            "outside_good_diversion_pct": outside_diversion * 100.0,
            "outside_good_market_share_pct": round(float(base_shares[-1] * 100.0), 2),
            "product_names": product_names,
        }
