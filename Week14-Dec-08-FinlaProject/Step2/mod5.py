import pandas as pd
import numpy as np
from collections import Counter
import random

# ============================================================
# 1. LOAD DATA
# ============================================================
def load_schedule(path="Heats_hillclimb.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Ensure Athlete ID stays intact and numeric
    df["Athlete ID"] = df["Athlete ID"].astype(int)

    return df


# ============================================================
# 2. EVALUATION FUNCTIONS
# ============================================================
def evaluate_heat(heat_df):
    score = 0

    # Lane fairness reward
    for _, row in heat_df.iterrows():
        lane = row["Lane"]
        seed = row["Seed Time"]

        if lane in [3, 4, 5, 6]:
            score += (11.5 - seed)
        else:
            score += (11.0 - seed) * 0.5

    # School diversity penalty
    counts = Counter(heat_df["School"])
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    score -= 0.5 * repeats

    return score


def evaluate_schedule(df):
    total = 0
    for heat_id, heat_df in df.groupby("Heat"):
        total += evaluate_heat(heat_df)
    return total, None


# ============================================================
# MAX MOVE 1 — Full Optimal Arrangement
# ============================================================
def max_optimal_arrangement(df):
    new_df = df.copy()

    # FIXED: Sort while preserving Athlete ID alignment
    new_df = new_df.sort_values(by=["Current Rank", "Athlete ID"])

    num_heats = new_df["Heat"].nunique()

    # FIXED: Assign heats without destroying Athlete ID
    new_df["Heat"] = (np.arange(len(new_df)) % num_heats) + 1

    lane_order = [4, 5, 3, 6, 2, 7, 1, 8]

    # Assign lanes inside each heat
    for heat_id, heat_df in new_df.groupby("Heat"):
        sorted_heat = heat_df.sort_values(by="Seed Time")

        for i, idx in enumerate(sorted_heat.index):
            new_df.at[idx, "Lane"] = lane_order[i]

    return new_df


# ============================================================
# MAX MOVE 2 — Swap lanes inside a heat
# ============================================================
def swap_within_heat(df, heat_id):
    heat_df = df[df["Heat"] == heat_id]
    indices = heat_df.index.tolist()

    if len(indices) < 2:
        return None

    i, j = random.sample(indices, 2)

    new_df = df.copy()
    new_df.loc[i, "Lane"], new_df.loc[j, "Lane"] = new_df.loc[j, "Lane"], new_df.loc[i, "Lane"]
    return new_df


# ============================================================
# MAX MOVE 3 — Swap athletes between heats
# ============================================================
def swap_between_heats(df):
    df_copy = df.copy()

    heats = df["Heat"].unique()
    h1, h2 = random.sample(list(heats), 2)

    heat1_df = df[df["Heat"] == h1]
    heat2_df = df[df["Heat"] == h2]

    i = heat1_df.sample(1).index[0]
    j = heat2_df.sample(1).index[0]

    df_copy.loc[i, ["Heat", "Lane"]], df_copy.loc[j, ["Heat", "Lane"]] = \
        df_copy.loc[j, ["Heat", "Lane"]].copy(), df_copy.loc[i, ["Heat", "Lane"]].copy()

    return df_copy


# ============================================================
# MIN = Judge (no changes)
# ============================================================
def min_judge(df):
    score, _ = evaluate_schedule(df)
    return score


# ============================================================
# 4. HYBRID ALPHA–BETA
# ============================================================
def alpha_beta(df, depth, alpha, beta, maximizing_player=True):

    if depth == 0:
        return min_judge(df), df

    if maximizing_player:
        best_value = -np.inf
        best_df = df

        candidate_moves = []

        # Move 1: Full optimal arrangement
        candidate_moves.append(max_optimal_arrangement(df))

        # Move 2: Swap within heat
        for heat_id in df["Heat"].unique():
            m = swap_within_heat(df, heat_id)
            if m is not None:
                candidate_moves.append(m)

        # Move 3: Swap between heats
        candidate_moves.append(swap_between_heats(df))

        for move_df in candidate_moves:
            value, child_df = alpha_beta(move_df, depth - 1, alpha, beta, False)

            if value > best_value:
                best_value = value
                best_df = child_df

            alpha = max(alpha, best_value)
            if alpha >= beta:
                break

        return best_value, best_df

    else:
        value = min_judge(df)
        return value, df


# ============================================================
# 5. MAIN DRIVER
# ============================================================
def run_alpha_beta():
    print("Loading schedule...")
    original = load_schedule()

    print("\nEvaluating Original Schedule:")
    init_score, _ = evaluate_schedule(original)
    print(f"Initial Fairness Score = {init_score:.4f}")

    print("\nRunning Hybrid MAX Alpha-Beta (depth=2)...")
    best_score, best_df = alpha_beta(original, depth=2, alpha=-np.inf, beta=np.inf)

    # FIXED: Do not drop or rewrite ID!
    best_df = best_df.sort_values(by=["Heat", "Lane"])

    print("\n==============================")
    print(f"Best Score Found = {best_score:.4f}")
    print(f"Improvement = {best_score - init_score:.4f}")
    print("==============================\n")

    best_df.to_csv("Heats_alpha_beta.csv", index=False)
    print("Saved optimized schedule as Heats_alpha_beta.csv (sorted by Heat & Lane)")

    return original, best_df


if __name__ == "__main__":
    original, optimized = run_alpha_beta()
