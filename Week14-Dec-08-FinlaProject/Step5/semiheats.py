import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD & PREPARE DATA USING SIMTIME (NOW "Prelims")
# ============================================================
def load_schedule(path="Semifinals_Qualifiers_from_sim.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df["Athlete ID"] = df["Athlete ID"].astype(int)

    # Rename SimTime -> Prelims
    df = df.rename(columns={"SimTime": "Prelims"})

    # Drop SeedTime column completely if exists
    if "SeedTime" in df.columns:
        df = df.drop(columns=["SeedTime"])

    # Seed semifinals by fastest prelims times
    df = df.sort_values(by=["Prelims", "Athlete ID"]).reset_index(drop=True)

    # Assign heats evenly (4 heats, 8 per heat)
    df["Heat"] = (df.index % 4) + 1

    return df


# ============================================================
# 2. ASSIGN LANES BASED ON PRELIMS PERFORMANCE
# ============================================================
def assign_final_lanes(df):
    lane_priority = [4, 5, 3, 6, 2, 7, 1, 8]  # Best lanes first

    for heat in df["Heat"].unique():
        heat_df = df[df["Heat"] == heat].sort_values(by="Prelims")
        for i, idx in enumerate(heat_df.index):
            df.at[idx, "Lane"] = lane_priority[i]

    return df


# ============================================================
# 3. WRITE FINAL HEAT SHEET
# ============================================================
def generate_semifinals_by_prelims():
    print("Loading qualifiers (Prelims-based seeding)…")
    df = load_schedule()

    print("Assigning lanes (fastest get center lanes)…")
    df = assign_final_lanes(df)

    # Sort final output nicely
    df = df.sort_values(by=["Heat", "Lane"])

    df.to_csv("Semifinal_Heats_By_Prelims.csv", index=False)

    print("\n========================================================")
    print("SUCCESS — Semifinal_Heats_By_Prelims.csv is READY! 🏃‍♂️💨")
    print("Sorted by Prelims, unique lanes, no repeats.")
    print("========================================================\n")

    return df


if __name__ == "__main__":
    generate_semifinals_by_prelims()
