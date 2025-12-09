"""
CH.13 + CH.14 — Probabilistic Selection of Finalists + Semifinal Time Predictions
-------------------------------------------------------------------------------

This version adds:
    - Posterior predictive semifinal time distribution:
         * PredSemiMean     : mean predicted semifinal time across simulations
         * PredSemiMedian   : median predicted time
         * PredSemiMin/Max  : realistic best/worst expected outcomes
         * DisplayedSemiTime: one sampled semifinal time (like a “real” round)
"""

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
SIGMA_SEMI = 0.05

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
# 2. CH.13 Posterior inference
# -------------------------------------------------------------------
def posterior_for_athlete(times, sigmas, mu0, sigma0):
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
# 5. CH.14 — Posterior predictive simulations for semifinal
# -------------------------------------------------------------------
def sample_semifinal(mu, sigma):
    return random.gauss(mu, math.sqrt(sigma**2 + SIGMA_SEMI**2))


def simulate_semifinals(semi_df, n_sims):
    adv_counts = defaultdict(int)
    time_storage = defaultdict(list)

    athletes = semi_df[["Athlete ID", "Heat", "PostMean", "PostStd"]].to_dict("records")

    for _ in range(n_sims):
        results = []
        for a in athletes:
            t = sample_semifinal(a["PostMean"], a["PostStd"])
            results.append({**a, "Time": t})
            time_storage[a["Athlete ID"]].append(t)

        for heat in [1, 2]:
            heat_res = sorted([r for r in results if r["Heat"] == heat], key=lambda x: x["Time"])
            for place in heat_res[:4]:
                adv_counts[place["Athlete ID"]] += 1

    # Construct final metrics
    advance_prob = {aid: adv_counts[aid] / n_sims for aid in semi_df["Athlete ID"]}

    summ = {}
    for aid, arr in time_storage.items():
        arr_sorted = sorted(arr)
        summ[aid] = {
            "PredSemiMean": float(np.mean(arr)),
            "PredSemiMedian": float(np.median(arr)),
            "PredSemiMin": float(arr_sorted[0]),
            "PredSemiMax": float(arr_sorted[-1]),
            "DisplayedSemiTime": float(arr_sorted[n_sims // 2])  # single real-like time
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
# 6. Choose 8 Finalists → Sort → Assign Final Lanes
# -------------------------------------------------------------------

# Select top 4 per heat by probability of advancing
finalists_rows = []
for heat, g in semi_df.groupby("Heat"):
    finalists_rows.append(g.sort_values("AdvanceProb", ascending=False).head(4))

finalists_df = pd.concat(finalists_rows).reset_index(drop=True)

# Re-sort fastest → slowest based on displayed semifinal performance
finalists_df = finalists_df.sort_values(by="DisplayedSemiTime").reset_index(drop=True)

# Assign final lane order: 4, 5, 3, 6, 2, 7, 1, 8
FINAL_LANES = [4, 5, 3, 6, 2, 7, 1, 8]
finalists_df["FinalLane"] = FINAL_LANES[:len(finalists_df)]

# Export ONLY required columns for submitted CSV
finalists_export = finalists_df[
    ["Athlete ID", "Name", "School", "SeedTime", "SimTime", "DisplayedSemiTime", "FinalLane"]
]

finalists_export = finalists_export.sort_values(by="FinalLane").reset_index(drop=True)

finalists_export.to_csv("Finalists_CH13_CH14.csv", index=False)

print("\n=== FINALISTS WITH FINAL LANE ASSIGNMENTS ===")
print(finalists_export)


# -------------------------------------------------------------------
# 7. Console output
# -------------------------------------------------------------------
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
