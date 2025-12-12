import pandas as pd
import random

# Step 1: Load Entries.csv
df = pd.read_csv("Entries.csv")
df.columns = df.columns.str.strip()

# Step 2: Sort by Current Rank (fastest first)
# IMPORTANT: Do NOT reset index — preserves Athlete ID
df = df.sort_values(by="Current Rank")

# Step 3: Assign heats in round-robin fashion
num_heats = 13
df["Heat"] = (df.groupby("Current Rank").cumcount() % num_heats) + 1
df["Heat"] = (df.index % num_heats) + 1  # simpler version

# Step 4: Initialize lane assignment (shuffle per heat)
df["Lane"] = df.groupby("Heat").cumcount() + 1
for heat in range(1, num_heats + 1):
    heat_indices = df[df["Heat"] == heat].index
    lanes = list(range(1, len(heat_indices) + 1))
    random.shuffle(lanes)
    df.loc[heat_indices, "Lane"] = lanes

# Lane preference scoring
lane_preference = {4:8, 5:7, 3:6, 6:5, 2:4, 7:3, 1:2, 8:1}

def fairness_score(heat_df):
    score = 0
    ranked = heat_df.sort_values(by="Current Rank")
    for _, athlete in ranked.iterrows():
        lane = athlete["Lane"]
        score += lane_preference.get(lane, 0)
    return score

# Step 5: Hill-climbing optimization
improved = True
while improved:
    improved = False
    for heat in range(1, num_heats + 1):
        heat_df = df[df["Heat"] == heat]
        current_score = fairness_score(heat_df)

        indices = heat_df.index.tolist()
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                # Try swap lanes
                a, b = indices[i], indices[j]
                df.loc[a, "Lane"], df.loc[b, "Lane"] = df.loc[b, "Lane"], df.loc[a, "Lane"]

                new_score = fairness_score(df[df["Heat"] == heat])
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    # revert swap
                    df.loc[a, "Lane"], df.loc[b, "Lane"] = df.loc[b, "Lane"], df.loc[a, "Lane"]

# Step 6: Sort for user-friendly output (Athlete ID stays tied to correct data)
df = df.sort_values(by=["Heat", "Lane"])

# Step 7: Save result
df.to_csv("Heats_hillclimb.csv", index=False)

print(" Heats_hillclimb.csv created!")
