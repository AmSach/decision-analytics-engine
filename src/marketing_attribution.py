"""
Marketing Multi-Touch Attribution Module.
Computes first-order Markov Chain transition matrices, evaluates removal effects,
and compares attribution weights against First-Touch, Last-Touch, and Linear models.
"""

from collections import defaultdict
import numpy as np
import pandas as pd


def compute_markov_attribution(journeys_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes first-order Markov Chain multi-touch attribution.
    Expects journeys_df with ['touchpoint_path', 'conversion_value', 'converted'].
    Path format: "Paid Search > Social Ads > Email"
    """
    transition_counts = defaultdict(lambda: defaultdict(int))
    base_conversions = 0
    total_conversion_value = 0.0

    channels_set = set()

    for _, row in journeys_df.iterrows():
        path = [c.strip() for c in row["touchpoint_path"].split(">")]
        converted = row["converted"] == 1
        conv_val = row["conversion_value"]

        if converted:
            base_conversions += 1
            total_conversion_value += conv_val

        # Add Start and End states
        full_path = ["(Start)"] + path + (["(Conversion)"] if converted else ["(Null)"])
        for ch in path:
            channels_set.add(ch)

        for i in range(len(full_path) - 1):
            src = full_path[i]
            dst = full_path[i + 1]
            transition_counts[src][dst] += 1

    channels = sorted(list(channels_set))
    all_states = ["(Start)"] + channels + ["(Conversion)", "(Null)"]

    # Build Transition Probability Matrix
    trans_matrix = pd.DataFrame(0.0, index=all_states, columns=all_states)
    for src, dst_dict in transition_counts.items():
        total_transitions = sum(dst_dict.values())
        if total_transitions > 0:
            for dst, count in dst_dict.items():
                trans_matrix.loc[src, dst] = count / total_transitions

    # Absorbing states
    trans_matrix.loc["(Conversion)", "(Conversion)"] = 1.0
    trans_matrix.loc["(Null)", "(Null)"] = 1.0

    def calculate_conversion_prob(matrix):
        """Calculates total probability of reaching (Conversion) from (Start)."""
        curr = pd.Series(0.0, index=matrix.index)
        curr["(Start)"] = 1.0
        for _ in range(15):  # 15 steps max path
            curr = curr.dot(matrix)
        return curr.get("(Conversion)", 0.0)

    base_p_conv = calculate_conversion_prob(trans_matrix)

    # Removal Effects
    removal_effects = {}
    for ch in channels:
        # Create counterfactual matrix without channel
        cf_matrix = trans_matrix.copy()
        cf_matrix.loc[ch, :] = 0.0
        cf_matrix.loc[ch, "(Null)"] = 1.0  # Divert transitions to Null
        p_cf = calculate_conversion_prob(cf_matrix)
        removal_effect = max(0.0, (base_p_conv - p_cf) / base_p_conv if base_p_conv > 0 else 0.0)
        removal_effects[ch] = removal_effect

    total_removal = sum(removal_effects.values())
    markov_weights = {ch: (re / total_removal if total_removal > 0 else 1.0 / len(channels)) for ch, re in removal_effects.items()}

    # Heuristic Attribution Models for Comparison
    first_touch = defaultdict(float)
    last_touch = defaultdict(float)
    linear_touch = defaultdict(float)

    for _, row in journeys_df[journeys_df["converted"] == 1].iterrows():
        path = [c.strip() for c in row["touchpoint_path"].split(">")]
        val = row["conversion_value"]
        first_touch[path[0]] += val
        last_touch[path[-1]] += val
        share = val / len(path)
        for ch in path:
            linear_touch[ch] += share

    comparison_records = []
    for ch in channels:
        markov_val = markov_weights[ch] * total_conversion_value
        comparison_records.append({
            "channel": ch,
            "markov_weight_pct": round(markov_weights[ch] * 100.0, 2),
            "markov_attributed_rev": round(markov_val, 2),
            "first_touch_rev": round(first_touch[ch], 2),
            "last_touch_rev": round(last_touch[ch], 2),
            "linear_touch_rev": round(linear_touch[ch], 2),
            "removal_effect_ratio": round(removal_effects[ch], 3),
        })

    comp_df = pd.DataFrame(comparison_records).sort_values(by="markov_attributed_rev", ascending=False)
    return comp_df, trans_matrix


if __name__ == "__main__":
    from data_generator import generate_ecommerce_dataset
    _, journeys = generate_ecommerce_dataset(num_customers=500, num_days=180)
    attribution_df, _ = compute_markov_attribution(journeys)
    print("Multi-Touch Attribution Comparison:")
    print(attribution_df)
