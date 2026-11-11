"""
Multi-Constraint 0/1 Integer Programming Knapsack Purse & Capital Allocation Engine.
Solves discrete resource allocation under financial purse ceilings, squad size bounds,
and compositional category quotas (e.g. sports auctions, R&D capital budgets).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint


class KnapsackAuctionOptimizer:
    """
    Multi-Constraint Integer Linear Programming Optimizer.
    Maximizes total portfolio value/surplus subject to purse ceilings,
    squad size bounds, and role/overseas compositional constraints.
    """

    def __init__(
        self,
        purse_ceiling: float = 120.0,
        min_squad_size: int = 18,
        max_squad_size: int = 25,
        max_overseas: int = 8,
    ):
        self.purse_ceiling = float(purse_ceiling)
        self.min_squad_size = int(min_squad_size)
        self.max_squad_size = int(max_squad_size)
        self.max_overseas = int(max_overseas)

    def optimize_squad(
        self,
        candidates_df: pd.DataFrame,
        value_col: str = "projected_surplus",
        cost_col: str = "estimated_cost_cr",
        role_col: str = "primary_role",
        is_overseas_col: str = "is_overseas",
        role_minimums: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Solve 0/1 Knapsack optimization using Mixed-Integer Linear Programming (MILP).
        """
        df = candidates_df.copy().reset_index(drop=True)
        n = len(df)

        if n < self.min_squad_size:
            raise ValueError(
                f"Candidate pool ({n}) is smaller than min squad size ({self.min_squad_size})"
            )

        # Objective: Maximize total projected surplus (minimize -1 * value)
        c = -1.0 * df[value_col].values.astype(float)

        # Integrality: all variables binary (1 for integer)
        integrality = np.ones(n)

        # Constraints
        # 1. Total cost <= purse_ceiling
        # 2. min_squad_size <= sum(x_i) <= max_squad_size
        # 3. sum(x_overseas) <= max_overseas
        # 4. role minimums

        A_rows = []
        lhs = []
        rhs = []

        # 1. Purse limit
        A_rows.append(df[cost_col].values.astype(float))
        lhs.append(0.0)
        rhs.append(self.purse_ceiling)

        # 2. Total squad size
        A_rows.append(np.ones(n))
        lhs.append(float(self.min_squad_size))
        rhs.append(float(self.max_squad_size))

        # 3. Overseas limit
        overseas_mask = df[is_overseas_col].astype(bool).astype(float).values
        A_rows.append(overseas_mask)
        lhs.append(0.0)
        rhs.append(float(self.max_overseas))

        # 4. Role minimums
        if role_minimums:
            for role_name, min_count in role_minimums.items():
                role_mask = (df[role_col] == role_name).astype(float).values
                if np.sum(role_mask) >= min_count:
                    A_rows.append(role_mask)
                    lhs.append(float(min_count))
                    rhs.append(float(n))

        A = np.array(A_rows)
        constraints = LinearConstraint(A, lhs, rhs)

        # Binary bounds: 0 <= x_i <= 1
        bounds = [(0.0, 1.0) for _ in range(n)]
        from scipy.optimize import Bounds
        lower_bounds = np.zeros(n)
        upper_bounds = np.ones(n)
        variable_bounds = Bounds(lower_bounds, upper_bounds)

        # Solve MILP
        res = milp(
            c=c,
            integrality=integrality,
            constraints=constraints,
            bounds=variable_bounds,
        )

        if not res.success:
            # Fallback to greedy heuristic if MILP finds no feasible integer point
            return self._greedy_fallback(df, value_col, cost_col, is_overseas_col)

        selected_indices = np.where(np.round(res.x) == 1)[0]
        selected_df = df.iloc[selected_indices].copy()

        total_spent = float(selected_df[cost_col].sum())
        total_value = float(selected_df[value_col].sum())
        n_selected = len(selected_df)
        n_overseas = int(selected_df[is_overseas_col].astype(bool).sum())
        remaining_purse = float(self.purse_ceiling - total_spent)

        selected_df["surplus_per_crore"] = np.round(
            selected_df[value_col] / np.maximum(selected_df[cost_col], 0.1), 4
        )
        selected_df = selected_df.sort_values(by="surplus_per_crore", ascending=False).reset_index(drop=True)

        return {
            "status": "optimal",
            "total_squad_size": n_selected,
            "total_overseas_count": n_overseas,
            "total_purse_allocated_cr": round(total_spent, 2),
            "remaining_purse_cr": round(remaining_purse, 2),
            "purse_utilization_pct": round((total_spent / self.purse_ceiling) * 100.0, 2),
            "total_portfolio_value": round(total_value, 4),
            "avg_surplus_per_crore": round(total_value / max(total_spent, 0.01), 4),
            "selected_roster": selected_df,
        }

    def _greedy_fallback(
        self,
        df: pd.DataFrame,
        value_col: str,
        cost_col: str,
        is_overseas_col: str,
    ) -> Dict[str, Any]:
        """
        Greedy ratio ranking fallback based on Dantzig surplus-per-cost heuristic.
        """
        df_sorted = df.copy()
        df_sorted["ratio"] = df_sorted[value_col] / np.maximum(df_sorted[cost_col], 0.1)
        df_sorted = df_sorted.sort_values(by="ratio", ascending=False).reset_index(drop=True)

        selected = []
        spent = 0.0
        overseas_count = 0

        for _, row in df_sorted.iterrows():
            cost = row[cost_col]
            is_ov = bool(row[is_overseas_col])

            if len(selected) >= self.max_squad_size:
                break
            if spent + cost > self.purse_ceiling:
                continue
            if is_ov and overseas_count >= self.max_overseas:
                continue

            selected.append(row)
            spent += cost
            if is_ov:
                overseas_count += 1

        res_df = pd.DataFrame(selected)
        total_val = float(res_df[value_col].sum()) if not res_df.empty else 0.0

        return {
            "status": "greedy_heuristic",
            "total_squad_size": len(res_df),
            "total_overseas_count": overseas_count,
            "total_purse_allocated_cr": round(spent, 2),
            "remaining_purse_cr": round(self.purse_ceiling - spent, 2),
            "purse_utilization_pct": round((spent / self.purse_ceiling) * 100.0, 2),
            "total_portfolio_value": round(total_val, 4),
            "avg_surplus_per_crore": round(total_val / max(spent, 0.01), 4),
            "selected_roster": res_df,
        }
