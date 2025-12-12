import random
import pandas as pd

RANDOM_SEED = 2025
random.seed(RANDOM_SEED)

INPUT_CSV = "Heats_alpha_beta.csv"
OUTPUT_PRELIMS = "resultsprelims.csv"
TIME_DECIMALS = 3


# ------------------------------------------------------------
# 1. Realistic, rank-aware sprint model
# ------------------------------------------------------------
def realistic_time_from_seed(seed: float, rank_in_heat: int) -> float:

    # Rare massive meltdown (0.5%)
    if random.random() < 0.005:
        delta = random.uniform(0.25, 0.40)
        return round(seed + delta, TIME_DECIMALS)

    r_rare = random.random()

    # Rare elite bad race
    if seed <= 10.35 and r_rare < 0.08:
        return round(seed + random.uniform(0.12, 0.21), TIME_DECIMALS)

    # Rare mid-level PR
    if 10.50 <= seed <= 10.70 and r_rare < 0.08:
        return round(seed - random.uniform(0.15, 0.22), TIME_DECIMALS)

    r = random.random()

    # Fast tier
    if 10.10 <= seed <= 10.40:
        if rank_in_heat == 1:
            p_big = 0.18; p_sol = 0.50
        elif rank_in_heat == 2:
            p_big = 0.30; p_sol = 0.48
        else:
            p_big = 0.12; p_sol = 0.60

        p_off = 1 - (p_big + p_sol)

        if r < p_big:
            delta = -random.uniform(0.08, 0.04)
        elif r < p_big + p_sol:
            delta = random.uniform(-0.03, 0.06)
        else:
            delta = random.uniform(0.07, 0.17)

    # Slow tier
    elif 10.70 <= seed <= 11.00:
        if rank_in_heat <= 3:
            p_big = 0.18; p_sol = 0.62
        else:
            p_big = 0.10; p_sol = 0.65

        p_off = 1 - (p_big + p_sol)

        if r < p_big:
            delta = -random.uniform(0.16, 0.06)
        elif r < p_big + p_sol:
            delta = random.uniform(-0.04, 0.08)
        else:
            delta = random.uniform(0.05, 0.12)

    # Middle tier
    else:
        if rank_in_heat == 1:
            p_big = 0.22; p_sol = 0.55
        elif rank_in_heat == 2:
            p_big = 0.26; p_sol = 0.52
        else:
            p_big = 0.14; p_sol = 0.60

        p_off = 1 - (p_big + p_sol)

        if r < p_big:
            delta = -random.uniform(0.13, 0.05)
        elif r < p_big + p_sol:
            delta = random.uniform(-0.03, 0.09)
        else:
            delta = random.uniform(0.06, 0.14)

    # Clamp extreme values
    delta = max(-0.40, min(0.30, delta))
    return round(seed + delta, TIME_DECIMALS)


# ------------------------------------------------------------
# 2. Simulate one heat
# ------------------------------------------------------------
def simulate_heat(heat_df: pd.DataFrame):
    heat_id = int(heat_df["Heat"].iloc[0])

    # Preserve original indices, but strip column names
    heat_df = heat_df.copy()

    # Sort by seed to assign rank 1..8
    seed_sorted = heat_df.sort_values("Seed Time").reset_index()
    rank_map = {int(r["index"]): rank for rank, (_, r) in enumerate(seed_sorted.iterrows(), start=1)}

    results = []

    for idx, row in heat_df.iterrows():
        athlete_id = int(row["Athlete ID"])
        name = str(row["Name"]).strip()
        school = str(row["School"]).strip()
        seed = float(row["Seed Time"])
        lane = int(row["Lane"])
        rank_in_heat = rank_map[int(idx)]

        sim_t = realistic_time_from_seed(seed, rank_in_heat)

        results.append({
            "Athlete ID": athlete_id,
            "Name": name,
            "School": school,
            "Heat": heat_id,
            "Lane": lane,
            "SeedTime": seed,
            "SimTime": sim_t
        })

    # Sort by performance to assign place
    results.sort(key=lambda r: r["SimTime"])
    for place, r in enumerate(results, start=1):
        r["Place"] = place

    return results


# ------------------------------------------------------------
# 3. MAIN SCRIPT
# ------------------------------------------------------------
def main():
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()  # Important cleanup

    all_results = []

    for _, heat_df in df.groupby("Heat"):
        all_results.extend(simulate_heat(heat_df))

    results_df = pd.DataFrame(
        all_results,
        columns=[
            "Athlete ID", "Name", "School",
            "Heat", "Lane",
            "SeedTime", "SimTime", "Place"
        ]
    )

    results_df.to_csv(OUTPUT_PRELIMS, index=False)


if __name__ == "__main__":
    main()
