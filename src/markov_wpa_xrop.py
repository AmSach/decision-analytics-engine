"""
Second-Order Markov Chain Win Probability Added (WPA) & Expected Runs Over Par (xROP).
Models state transition dynamics across match leverage states to evaluate
player marginal impact above baseline expectation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class MarkovWPAEngine:
    """
    Second-Order Markov Chain Engine for Dynamic State Valuation,
    Win Probability Added (WPA), and Expected Runs Over Par (xROP).
    """

    def __init__(self, n_over_bins: int = 4, n_wicket_bins: int = 4, n_rr_bins: int = 4):
        self.n_over_bins = n_over_bins
        self.n_wicket_bins = n_wicket_bins
        self.n_rr_bins = n_rr_bins
        self.run_expectancy_table: Dict[Tuple[int, int, int], float] = {}
        self.win_probability_table: Dict[Tuple[int, int, int], float] = {}

    def discretize_state(
        self, overs_remaining: float, wickets_lost: int, run_rate_delta: float
    ) -> Tuple[int, int, int]:
        """
        Discretize continuous match context into a finite Markov state triplet:
        (over_bin, wicket_bin, run_rate_delta_bin).
        """
        # Over bin: 0 (death: 0-4 overs), 1 (middle: 4-10), 2 (consolidation: 10-15), 3 (powerplay: 15-20)
        if overs_remaining <= 4.0:
            o_bin = 0
        elif overs_remaining <= 10.0:
            o_bin = 1
        elif overs_remaining <= 15.0:
            o_bin = 2
        else:
            o_bin = 3

        # Wicket bin: 0 (0-2 wickets lost), 1 (3-4), 2 (5-6), 3 (7-9)
        if wickets_lost <= 2:
            w_bin = 0
        elif wickets_lost <= 4:
            w_bin = 1
        elif wickets_lost <= 6:
            w_bin = 2
        else:
            w_bin = 3

        # Run rate delta bin: difference between current scoring rate and required/par rate
        if run_rate_delta <= -2.0:
            r_bin = 0  # Severe deficit
        elif run_rate_delta <= 0.0:
            r_bin = 1  # Mild deficit
        elif run_rate_delta <= 2.0:
            r_bin = 2  # Mild surplus
        else:
            r_bin = 3  # High surplus

        return (o_bin, w_bin, r_bin)

    def fit_from_deliveries(self, deliveries_df: pd.DataFrame) -> None:
        """
        Estimate state values (Win Probability and Run Expectancy) from delivery-level logs.
        Required columns: 'overs_remaining', 'wickets_lost', 'rr_delta', 'runs_scored', 'win_outcome'.
        """
        df = deliveries_df.copy()
        df["state"] = [
            self.discretize_state(o, w, r)
            for o, w, r in zip(df["overs_remaining"], df["wickets_lost"], df["rr_delta"])
        ]

        # Calculate empirical run expectancy and win probability per state
        grouped = df.groupby("state")
        self.run_expectancy_table = grouped["runs_scored"].mean().to_dict()
        self.win_probability_table = grouped["win_outcome"].mean().to_dict()

    def get_state_values(
        self, state: Tuple[int, int, int]
    ) -> Tuple[float, float]:
        """
        Return (run_expectancy, win_probability) for a given state triplet.
        Defaults to theoretical baseline if state was unobserved.
        """
        o_bin, w_bin, r_bin = state
        # Theoretical fallback if not empirically fitted
        base_exp = 1.30 - (w_bin * 0.20) + (r_bin * 0.15) - (o_bin * 0.05)
        base_exp = max(base_exp, 0.40)

        # Base win probability logistic heuristic
        base_wp = 1.0 / (1.0 + np.exp(-((r_bin - 1.5) * 0.8 - (w_bin - 1.5) * 0.6)))

        exp_runs = self.run_expectancy_table.get(state, base_exp)
        win_prob = self.win_probability_table.get(state, base_wp)
        return float(exp_runs), float(win_prob)

    def evaluate_delivery(
        self,
        overs_rem_pre: float,
        wickets_pre: int,
        rr_delta_pre: float,
        overs_rem_post: float,
        wickets_post: int,
        rr_delta_post: float,
        actual_runs: int,
        is_wicket: bool,
    ) -> Dict[str, float]:
        """
        Compute delivery-level WPA, xROP, and Leverage Index.
        """
        s_pre = self.discretize_state(overs_rem_pre, wickets_pre, rr_delta_pre)
        s_post = self.discretize_state(overs_rem_post, wickets_post, rr_delta_post)

        exp_runs_pre, wp_pre = self.get_state_values(s_pre)
        _, wp_post = self.get_state_values(s_post)

        # If wicket fell, post-state win prob drops further
        if is_wicket:
            wp_post = max(wp_post - 0.05, 0.0)

        wpa = wp_post - wp_pre
        xrop = actual_runs - exp_runs_pre

        # Leverage Index: proxy based on how close game is and death overs
        death_multiplier = 2.0 if overs_rem_pre <= 4.0 else 1.2 if overs_rem_pre <= 10.0 else 0.8
        closeness_multiplier = 1.8 if abs(rr_delta_pre) <= 1.5 else 1.0
        leverage_index = death_multiplier * closeness_multiplier

        return {
            "wpa": round(float(wpa), 5),
            "xrop": round(float(xrop), 3),
            "expected_runs_par": round(float(exp_runs_pre), 3),
            "actual_runs": actual_runs,
            "leverage_index": round(float(leverage_index), 2),
            "win_prob_before": round(float(wp_pre), 4),
            "win_prob_after": round(float(wp_post), 4),
        }

    def aggregate_player_performance(
        self, evaluated_events: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Aggregate event log into comprehensive player performance metrics.
        """
        if not evaluated_events:
            return {}

        total_wpa = sum(e["wpa"] for e in evaluated_events)
        total_xrop = sum(e["xrop"] for e in evaluated_events)
        total_balls = len(evaluated_events)
        avg_leverage = float(np.mean([e["leverage_index"] for e in evaluated_events]))

        high_lev_events = [e for e in evaluated_events if e["leverage_index"] >= 1.5]
        clutch_wpa = sum(e["wpa"] for e in high_lev_events) if high_lev_events else 0.0

        return {
            "total_balls": total_balls,
            "total_wpa": round(float(total_wpa), 4),
            "wpa_percentage": round(float(total_wpa * 100.0), 2),
            "total_xrop_runs_over_par": round(float(total_xrop), 2),
            "xrop_per_ball": round(float(total_xrop / total_balls), 4) if total_balls > 0 else 0.0,
            "average_leverage_index": round(avg_leverage, 2),
            "clutch_wpa_high_leverage": round(float(clutch_wpa), 4),
            "high_leverage_event_count": len(high_lev_events),
        }
