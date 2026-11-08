"""
Adversarial Game Theory Pricing Engine.
Implements Level-k Cognitive Hierarchy and Logit Quantal Response Equilibrium (QRE).
Solves for the strategic pricing fixed point against rational and boundedly-rational rivals,
preventing destructive price-war discounting spirals.
"""

import numpy as np
import pandas as pd


class AdversarialPricingEngine:
    """
    Game-theoretic duopoly/oligopoly pricing simulator.
    Solves for Nash Equilibrium, Level-k best responses, and Quantal Response Equilibria (QRE).
    """

    def __init__(
        self,
        our_cost: float = 40.0,
        rival_cost: float = 45.0,
        market_capacity: int = 10000,
        price_grid: np.ndarray | None = None,
        price_sensitivity: float = 0.05,
    ):
        self.our_cost = our_cost
        self.rival_cost = rival_cost
        self.market_capacity = market_capacity
        self.gamma = price_sensitivity

        if price_grid is None:
            self.price_grid = np.linspace(50.0, 150.0, 21)
        else:
            self.price_grid = price_grid

        self.n_prices = len(self.price_grid)

    def demand_share(self, p_our: float, p_rival: float) -> tuple[float, float]:
        """
        Multinomial logit market share function based on price differential:
        S_our = 1 / (1 + exp(gamma * (p_our - p_rival)))
        """
        diff = p_our - p_rival
        # Prevent numerical overflow
        diff_clipped = np.clip(self.gamma * diff, -20.0, 20.0)
        s_our = 1.0 / (1.0 + np.exp(diff_clipped))
        s_rival = 1.0 - s_our
        return s_our, s_rival

    def payoff_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Constructs payoff matrices (Pi_our, Pi_rival) of shape (n_prices, n_prices).
        """
        pi_our = np.zeros((self.n_prices, self.n_prices))
        pi_rival = np.zeros((self.n_prices, self.n_prices))

        for i, p1 in enumerate(self.price_grid):
            for j, p2 in enumerate(self.price_grid):
                s1, s2 = self.demand_share(p1, p2)
                q1 = self.market_capacity * s1
                q2 = self.market_capacity * s2
                pi_our[i, j] = (p1 - self.our_cost) * q1
                pi_rival[i, j] = (p2 - self.rival_cost) * q2

        return pi_our, pi_rival

    def compute_level_k_strategy(self, max_k: int = 3) -> dict:
        """
        Level-0: Rival plays uniform random distribution over prices.
        Level-1: Firm best-responds to Level-0 beliefs.
        Level-2: Rival best-responds to Level-1 firm.
        Level-3: Firm best-responds to Level-2 rival.
        """
        pi_our, pi_rival = self.payoff_matrix()
        results = {}

        # Level-0 beliefs: uniform probability
        prob_rival_l0 = np.ones(self.n_prices) / self.n_prices

        # Level-1 Our Strategy: Argmax expected payoff against Level-0
        exp_pi_l1 = pi_our @ prob_rival_l0
        best_p1_idx = int(np.argmax(exp_pi_l1))
        results["Level_1_Our_Price"] = round(float(self.price_grid[best_p1_idx]), 2)
        results["Level_1_Expected_Profit"] = round(float(exp_pi_l1[best_p1_idx]), 2)

        # Level-2 Rival Strategy: Rival best-responds to our Level-1 price
        exp_pi_rival_l2 = pi_rival[best_p1_idx, :]
        best_p2_idx = int(np.argmax(exp_pi_rival_l2))
        results["Level_2_Rival_Price"] = round(float(self.price_grid[best_p2_idx]), 2)
        results["Level_2_Rival_Profit"] = round(float(exp_pi_rival_l2[best_p2_idx]), 2)

        # Level-3 Our Strategy: Firm counter-flanks Level-2 rival
        exp_pi_l3 = pi_our[:, best_p2_idx]
        best_p3_idx = int(np.argmax(exp_pi_l3))
        results["Level_3_Counter_Flank_Price"] = round(float(self.price_grid[best_p3_idx]), 2)
        results["Level_3_Counter_Flank_Profit"] = round(float(exp_pi_l3[best_p3_idx]), 2)

        return results

    def solve_quantal_response_equilibrium(
        self,
        rationality_lambda: float = 0.0001,
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Solves for the McKelvey & Palfrey Logit Quantal Response Equilibrium (QRE).
        Iterates smoothed best response probabilities until fixed point convergence.
        """
        pi_our, pi_rival = self.payoff_matrix()

        # Initialize uniform distributions
        prob_our = np.ones(self.n_prices) / self.n_prices
        prob_rival = np.ones(self.n_prices) / self.n_prices

        for _ in range(max_iter):
            # Expected payoffs
            exp_pi_our = pi_our @ prob_rival
            exp_pi_rival = pi_rival.T @ prob_our

            # Logit choice probabilities with numerical stabilization
            u_our = rationality_lambda * (exp_pi_our - np.max(exp_pi_our))
            u_rival = rationality_lambda * (exp_pi_rival - np.max(exp_pi_rival))

            new_prob_our = np.exp(u_our) / np.sum(np.exp(u_our))
            new_prob_rival = np.exp(u_rival) / np.sum(np.exp(u_rival))

            # Check convergence
            diff = max(
                np.max(np.abs(new_prob_our - prob_our)),
                np.max(np.abs(new_prob_rival - prob_rival))
            )
            prob_our = 0.5 * prob_our + 0.5 * new_prob_our
            prob_rival = 0.5 * prob_rival + 0.5 * new_prob_rival

            if diff < tol:
                break

        expected_our_price = float(np.sum(self.price_grid * prob_our))
        expected_rival_price = float(np.sum(self.price_grid * prob_rival))

        return prob_our, prob_rival, round(expected_our_price, 2), round(expected_rival_price, 2)
