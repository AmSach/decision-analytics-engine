"""
Elinor Ostrom Polycentric Commons Governance Engine.
Models open-access keyword and customer acquisition pools prone to Hardin's Tragedy of the Commons.
Enforces Ostrom's 8 Governance Principles, detects rival bidding defection spirals,
and triggers automated kill-gate liquidity reallocations to prevent CAC blowouts.
"""

import numpy as np
import pandas as pd


class OstromCommonsGovernanceEngine:
    """
    Common Pool Resource (CPR) Simulator for Multi-Brand Paid Acquisition.
    Enforces capacity bounds, graduated sanctions, and asymmetric channel flanking.
    """

    def __init__(
        self,
        pool_capacity: float = 100000.0,
        base_cac: float = 25.0,
        depletion_elasticity: float = 1.8,
        sustainability_threshold: float = 0.75,
    ):
        self.capacity = pool_capacity
        self.base_cac = base_cac
        self.gamma = depletion_elasticity
        self.threshold = sustainability_threshold

    def calculate_pool_cac(self, total_injected_capital: float) -> tuple[float, float]:
        """
        Calculates effective market CAC under pool congestion:
        CAC(B) = CAC_0 * (1 + theta * (B / Capacity)^gamma)
        Returns (effective_cac, depletion_index).
        """
        depletion_index = total_injected_capital / self.capacity
        # Congestion penalty kicks in as pool approaches capacity
        congestion_penalty = (depletion_index ** self.gamma) if depletion_index > 0 else 0.0
        effective_cac = self.base_cac * (1.0 + 1.5 * congestion_penalty)
        return float(effective_cac), float(depletion_index)

    def simulate_bidding_war_defection_spiral(
        self,
        our_base_budget: float = 20000.0,
        num_rivals: int = 4,
        rival_aggression_factor: float = 1.35,
        rounds: int = 10,
    ) -> pd.DataFrame:
        """
        Simulates two competing governance regimes over N rounds:
        1. Unregulated Commons: All firms blindly increase bids to maintain volume, triggering CAC blowout.
        2. Ostrom Polycentric Governance: Our firm monitors depletion index, enforces kill gates,
           and executes asymmetric retreat-and-flank to high-ROI proprietary channels.
        """
        records = []
        
        # State tracking
        unreg_our_budget = our_base_budget
        unreg_rival_budgets = np.full(num_rivals, our_base_budget * 0.8)

        ostrom_our_budget = our_base_budget
        ostrom_proprietary_budget = 0.0
        ostrom_rival_budgets = np.full(num_rivals, our_base_budget * 0.8)

        for r in range(1, rounds + 1):
            # --- Scenario A: Unregulated Tragedy of Commons ---
            unreg_total = unreg_our_budget + np.sum(unreg_rival_budgets)
            unreg_cac, unreg_depletion = self.calculate_pool_cac(unreg_total)
            unreg_conversions = unreg_our_budget / unreg_cac if unreg_cac > 0 else 0.0

            # Rivals escalate bids aggressively
            unreg_rival_budgets *= rival_aggression_factor
            unreg_our_budget *= 1.15  # Naive attempt to defend share

            # --- Scenario B: Ostrom Governed Asymmetric Flank ---
            ostrom_total = ostrom_our_budget + np.sum(ostrom_rival_budgets)
            ostrom_pool_cac, ostrom_depletion = self.calculate_pool_cac(ostrom_total)

            # Ostrom Rule 4 & 5: Graduated Sanction / Kill Gate Trigger
            if ostrom_depletion > self.threshold:
                # Quota Breach: Throttle pool budget by 40% and flank to proprietary channels (fixed CAC = $18)
                reallocated = ostrom_our_budget * 0.40
                ostrom_our_budget -= reallocated
                ostrom_proprietary_budget += reallocated
                status = "KILL_GATE_ACTIVATED: Flank to Proprietary"
            else:
                status = "SUSTAINABLE_COMMONS"

            # Conversions under Ostrom governance
            pool_convs = ostrom_our_budget / ostrom_pool_cac if ostrom_pool_cac > 0 else 0.0
            prop_convs = ostrom_proprietary_budget / (self.base_cac * 0.70)  # Flank channel has low, stable CAC
            ostrom_total_convs = pool_convs + prop_convs
            ostrom_blended_cac = (ostrom_our_budget + ostrom_proprietary_budget) / ostrom_total_convs if ostrom_total_convs > 0 else ostrom_pool_cac

            # Rivals in Scenario B also defect blindly
            ostrom_rival_budgets *= rival_aggression_factor

            records.append({
                "round": r,
                "unreg_depletion_index": round(unreg_depletion, 3),
                "unreg_market_cac": round(unreg_cac, 2),
                "unreg_conversions": int(unreg_conversions),
                "ostrom_depletion_index": round(ostrom_depletion, 3),
                "ostrom_blended_cac": round(ostrom_blended_cac, 2),
                "ostrom_conversions": int(ostrom_total_convs),
                "ostrom_status": status,
                "cac_savings_pct": round(((unreg_cac - ostrom_blended_cac) / unreg_cac) * 100.0, 1),
            })

        return pd.DataFrame(records)
