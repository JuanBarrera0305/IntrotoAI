import pandas as pd
import random

# Step 1: Load Entries.csv
df = pd.read_csv("Entries.csv")
df.columns = df.columns.str.strip()

# Step 2: Sort by Current Rank (fastest first)
df = df.sort_values(by="Current Rank").reset_index(drop=True)

# Step 3: Assign heats in round-robin fashion
num_heats = 13
df["Heat"] = (df.index % num_heats) + 1

# Step 4: Define lane preference scoring
lane_preference = {4:8, 5:7, 3:6, 6:5, 2:4, 7:3, 1:2, 8:1}

def fairness_score(heat_df):
    """Calculate fairness score for one heat based on lane preference."""
    score = 0
    # Sort athletes by rank within the heat
    ranked = heat_df.sort_values(by="Current Rank").reset_index(drop=True)
    for i, (_, athlete) in enumerate(ranked.iterrows()):
        lane = athlete["Lane"]
        score += lane_preference.get(lane, 0)
    return score

# Step 5: Random initial lane assignment per heat
df["Lane"] = df.groupby("Heat").cumcount() + 1
for heat in range(1, num_heats+1):
    lanes = list(range(1, len(df[df["Heat"]==heat])+1))
    random.shuffle(lanes)
    df.loc[df["Heat"]==heat, "Lane"] = lanes

# Step 6: Hill-climbing optimization
improved = True
while improved:
    improved = False
    for heat in range(1, num_heats+1):
        heat_df = df[df["Heat"]==heat].copy()
        current_score = fairness_score(heat_df)
        
        # Try swapping two athletes
        indices = heat_df.index.tolist()
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                # Swap lanes
                df.loc[indices[i], "Lane"], df.loc[indices[j], "Lane"] = df.loc[indices[j], "Lane"], df.loc[indices[i], "Lane"]
                
                new_score = fairness_score(df[df["Heat"]==heat])
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Swap back if no improvement
                    df.loc[indices[i], "Lane"], df.loc[indices[j], "Lane"] = df.loc[indices[j], "Lane"], df.loc[indices[i], "Lane"]

# Step 7: Save optimized assignment
df = df.sort_values(by=["Heat", "Lane"]).reset_index(drop=True)
df.to_csv("Heats_hillclimb.csv", index=False)

print("✅ Heats_hillclimb.csv created using hill-climbing optimization!")
