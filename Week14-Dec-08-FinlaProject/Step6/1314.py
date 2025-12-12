import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
INPUT_FILE = "prelim32.csv"

LANE_ORDER = [4, 5, 3, 6, 2, 7, 1, 8]

MU0 = 10.50
SIGMA0 = 0.30

SIGMA_SEED = 0.10
SIGMA_PRELIM = 0.06
SIGMA_SEMI = 0.05   # race-day uncertainty (Chapter 14 noise)

N_SIMS = 20000

random.seed(2025)
np.random.seed(2025)


# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()

df = df.rename(columns={
    "Initial Performance (SeedTime)": "SeedTime",
    "Performance in Prelims (SimTime)": "SimTime"
})

df["Athlete ID"] = df["Athlete ID"].astype(int)
df["SeedTime"] = df["SeedTime"].astype(float)
df["SimTime"] = df["SimTime"].astype(float)


# -------------------------------------------------------------------
# 2. CH.13 — Posterior inference (Bayesian updating)
# -------------------------------------------------------------------
def posterior_for_athlete(times, sigmas, mu0, sigma0):
    """
    Chapter 13:
      Computes posterior parameters for athlete ability
    """
    tau0 = 1.0 / (sigma0 ** 2)
    tau = tau0 + sum(1.0 / (s ** 2) for s in sigmas)
    sigma_post = math.sqrt(1.0 / tau)
    weighted = mu0 * tau0 + sum(t / (s ** 2) for t, s in zip(times, sigmas))
    mu_post = sigma_post**2 * weighted
    return mu_post, sigma_post


df["PostMean"], df["PostStd"] = zip(*[
    posterior_for_athlete(
        [row["SeedTime"], row["SimTime"]],
        [SIGMA_SEED, SIGMA_PRELIM],
        MU0, SIGMA0
    )
    for _, row in df.iterrows()
])


# -------------------------------------------------------------------
# 3. Select 16 semifinalists
# -------------------------------------------------------------------
semi_df = df.sort_values(by="PostMean").head(16).reset_index(drop=True)
semi_df["SeedRank"] = np.arange(1, 17)
semi_df["Heat"] = semi_df["SeedRank"].apply(lambda r: 1 if r % 2 == 1 else 2)


# -------------------------------------------------------------------
# 4. Assign lanes
# -------------------------------------------------------------------
def assign_lanes_for_heat(group):
    g = group.sort_values(by="PostMean").reset_index(drop=True)
    g["Lane"] = LANE_ORDER[:len(g)]
    return g


semi_df = pd.concat([
    assign_lanes_for_heat(group)
    for _, group in semi_df.groupby("Heat")
]).sort_values(["Heat", "Lane"]).reset_index(drop=True)


# -------------------------------------------------------------------
# 5. CH.14 — Bayesian Network + Monte Carlo inference
# -------------------------------------------------------------------
"""
Bayesian Network structure (Chapter 14):

Ability_i  ──▶  Time_i  ──▶  Advance_i
              ▲
          RaceNoise_i

• Ability_i ~ N(PostMean, PostStd)
• RaceNoise_i ~ N(0, SIGMA_SEMI)
• Time_i = Ability_i + RaceNoise_i
• Advance_i determined by Time_i within each Heat

Inference:
  Approximate inference using Monte Carlo sampling
  (AIMA Ch.14.5)
"""

def sample_time(post_mean, post_std):
    """
    Samples from conditional distribution:
      P(Time | Ability, RaceNoise)
    """
    total_var = post_std**2 + SIGMA_SEMI**2
    return random.gauss(post_mean, math.sqrt(total_var))


def simulate_semifinals(semi_df, n_sims):
    advance_counts = defaultdict(int)
    time_storage = defaultdict(list)

    athletes = semi_df[
        ["Athlete ID", "Heat", "PostMean", "PostStd"]
    ].to_dict("records")

    for _ in range(n_sims):

        # --- Sample joint distribution (Equation 14.1) ---
        sampled_world = []
        for a in athletes:
            t = sample_time(a["PostMean"], a["PostStd"])
            sampled_world.append({
                "Athlete ID": a["Athlete ID"],
                "Heat": a["Heat"],
                "Time": t
            })
            time_storage[a["Athlete ID"]].append(t)

        # --- Infer Advance_i (derived random variable) ---
        for heat in [1, 2]:
            heat_res = sorted(
                [r for r in sampled_world if r["Heat"] == heat],
                key=lambda x: x["Time"]
            )

            for r in heat_res[:4]:
                advance_counts[r["Athlete ID"]] += 1

    # --- Convert frequencies to probabilities ---
    advance_prob = {
        aid: advance_counts[aid] / n_sims
        for aid in semi_df["Athlete ID"]
    }

    # --- Summary statistics for posterior predictive ---
    summ = {}
    for aid, arr in time_storage.items():
        arr_sorted = sorted(arr)
        summ[aid] = {
            "PredSemiMean": float(np.mean(arr)),
            "PredSemiMedian": float(np.median(arr)),
            "PredSemiMin": float(arr_sorted[0]),
            "PredSemiMax": float(arr_sorted[-1]),
            "DisplayedSemiTime": float(arr_sorted[n_sims // 2])
        }

    return advance_prob, summ


adv_prob, stats = simulate_semifinals(semi_df, N_SIMS)

semi_df["AdvanceProb"] = semi_df["Athlete ID"].map(adv_prob)
semi_df["PredSemiMean"] = semi_df["Athlete ID"].map(lambda x: stats[x]["PredSemiMean"])
semi_df["PredSemiMedian"] = semi_df["Athlete ID"].map(lambda x: stats[x]["PredSemiMedian"])
semi_df["PredSemiMin"] = semi_df["Athlete ID"].map(lambda x: stats[x]["PredSemiMin"])
semi_df["PredSemiMax"] = semi_df["Athlete ID"].map(lambda x: stats[x]["PredSemiMax"])
semi_df["DisplayedSemiTime"] = semi_df["Athlete ID"].map(lambda x: stats[x]["DisplayedSemiTime"])

semi_df.to_csv("Semifinals_CH13_CH14.csv", index=False)


# -------------------------------------------------------------------
# 6. Choose 8 Finalists → Assign Final Lanes
# -------------------------------------------------------------------
finalists_rows = []
for heat, g in semi_df.groupby("Heat"):
    finalists_rows.append(
        g.sort_values("AdvanceProb", ascending=False).head(4)
    )

finalists_df = pd.concat(finalists_rows).reset_index(drop=True)

finalists_df = finalists_df.sort_values(
    by="DisplayedSemiTime"
).reset_index(drop=True)

FINAL_LANES = [4, 5, 3, 6, 2, 7, 1, 8]
finalists_df["FinalLane"] = FINAL_LANES[:len(finalists_df)]

finalists_export = finalists_df[
    ["Athlete ID", "Name", "School",
     "SeedTime", "SimTime", "DisplayedSemiTime", "FinalLane"]
]

finalists_export = finalists_export.sort_values(
    by="FinalLane"
).reset_index(drop=True)

finalists_export.to_csv("Finalists_CH13_CH14.csv", index=False)


# -------------------------------------------------------------------
# 7. Console output
# -------------------------------------------------------------------
print("\n=== FINALISTS WITH FINAL LANE ASSIGNMENTS ===")
print(finalists_export)

print("\n=== SEMIFINALS WRITTEN TO Semifinals_CH13_CH14.csv ===")
print(semi_df[[
    "Athlete ID","Name","Heat","Lane","PostMean","AdvanceProb",
    "PredSemiMean","PredSemiMedian","PredSemiMin","PredSemiMax",
    "DisplayedSemiTime"
]])

print("\n=== FINALISTS WRITTEN TO Finalists_CH13_CH14.csv ===")
print(finalists_df[[
    "Athlete ID","Name","Heat","Lane","PostMean","AdvanceProb"
]])
