import pandas as pd
from constraint import Problem, AllDifferentConstraint

# ------------------------------------------------------------
# 1. LOAD CSV (FIXED: strip column name spaces)
# ------------------------------------------------------------
def load_schedule(path="Heats_alpha_beta.csv"):
    df = pd.read_csv(path)

    # FIX: Strip whitespace from column names so "Heat " becomes "Heat"
    df.columns = df.columns.str.strip()

    return df


# ------------------------------------------------------------
# 2. BUILD + SOLVE CSP
# ------------------------------------------------------------
def build_and_solve_csp(df):
    """
    Build a CSP for assigning athletes to (heat, lane) positions
    with:
      - All-different (each position has a different athlete)
      - Fastest 13 athletes all in different heats
      - Lane fairness ordering inside each heat
    """

    problem = Problem()

    # --- Basic info ---
    heats = sorted(df["Heat"].unique())        # should now work
    lanes = sorted(df["Lane"].unique())
    all_athletes = list(df.index)              # athlete IDs

    # Maps athlete index → their seed time
    seed_time = df["Seed Time"].to_dict()

    # --------------------------------------------------------
    # Variables: one variable per (heat, lane)
    # Domain: all athlete indices (AllDifferent guarantees uniqueness)
    # --------------------------------------------------------
    position_vars = []

    for h in heats:
        for l in lanes:
            var_name = f"H{h}_L{l}"
            position_vars.append(var_name)
            problem.addVariable(var_name, all_athletes)

    # --------------------------------------------------------
    # Constraint 1: All athletes must be assigned to a unique position
    # --------------------------------------------------------
    problem.addConstraint(AllDifferentConstraint(), position_vars)

    # --------------------------------------------------------
    # Constraint 2: Top 13 fastest athletes must be in DIFFERENT heats
    # --------------------------------------------------------
    df_sorted = df.sort_values(by="Seed Time")
    top13_ids = set(df_sorted.head(13).index)

    def at_most_one_top(*athletes, top_set=top13_ids):
        return sum(1 for a in athletes if a in top_set) <= 1

    for h in heats:
        heat_vars = [f"H{h}_L{l}" for l in lanes]
        problem.addConstraint(at_most_one_top, heat_vars)

    # --------------------------------------------------------
    # Constraint 3: Lane fairness constraint inside each heat
    #
    # Lane order: fastest → slowest
    #   4, 5, 3, 6, 2, 7, 1, 8
    #
    # So SeedTime(lane4) <= SeedTime(lane5) <= SeedTime(lane3) <= ...
    # --------------------------------------------------------
    lane_order = [4, 5, 3, 6, 2, 7, 1, 8]

    def lane_fairness(*athlete_ids, seed_time_map=seed_time):
        times = [seed_time_map[a] for a in athlete_ids]
        return all(times[i] <= times[i+1] for i in range(len(times) - 1))

    for h in heats:
        ordered_vars = [f"H{h}_L{l}" for l in lane_order]
        problem.addConstraint(lane_fairness, ordered_vars)

    # --------------------------------------------------------
    # Solve CSP
    # --------------------------------------------------------
    print("⏳ Solving CSP... (this may take a moment)")

    solution = problem.getSolution()
    if solution is None:
        print("❌ No CSP solution found.")
        return None

    # --------------------------------------------------------
    # Build output DataFrame
    # --------------------------------------------------------
    new_df = df.copy()

    for h in heats:
        for l in lanes:
            var_name = f"H{h}_L{l}"
            athlete_idx = solution[var_name]
            new_df.loc[athlete_idx, "Heat"] = h
            new_df.loc[athlete_idx, "Lane"] = l

    # Sort for readability
    new_df = new_df.sort_values(by=["Heat", "Lane"]).reset_index(drop=True)
    return new_df


# ------------------------------------------------------------
# 3. DRIVER
# ------------------------------------------------------------
def run_csp_scheduler():
    print("Loading schedule (for CSP)...")
    original = load_schedule()

    print("Building & solving CSP...")
    csp_df = build_and_solve_csp(original.copy())

    if csp_df is not None:
        csp_df.to_csv("Heats_CSP.csv", index=False)
        print("✅ Saved CSP-optimized schedule as Heats_CSP.csv")
    else:
        print("⚠️ CSP was unable to find a valid solution.")

    return original, csp_df


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    original, csp_schedule = run_csp_scheduler()
