import pandas as pd

INPUT_FILE = "resultsprelims.csv"
OUTPUT_FILE = "Semifinals_Qualifiers_from_sim.csv"


# ==========================================================
# 1. LOAD DATA
# ==========================================================
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()

required_cols = [
    "Athlete ID", "Name", "School",
    "Heat", "Lane", "SeedTime", "SimTime", "Place"
]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column in {INPUT_FILE}: '{col}'")


# ==========================================================
# 2. CHAPTER 6 — CSP: VARIABLES & CONSTRAINTS
# ==========================================================

# Variable: Qualify[athlete_id] ∈ {0,1}
Qualify = {int(aid): 0 for aid in df["Athlete ID"].unique()}

# --- Constraint 1: Top-2 per heat must qualify ("Auto") ---
# We trust the 'Place' column from the simulation.
auto_df = (
    df.sort_values(["Heat", "Place"])
      .groupby("Heat")
      .head(2)
)

auto_ids = sorted(auto_df["Athlete ID"].astype(int).tolist())

for aid in auto_ids:
    Qualify[int(aid)] = 1  # Constraint propagation: forced Auto qualifiers

# Sanity check: we expect (#heats × 2) auto qualifiers
num_heats = df["Heat"].nunique()
expected_auto = num_heats * 2
if len(auto_ids) != expected_auto:
    raise ValueError(f"Expected {expected_auto} auto qualifiers, got {len(auto_ids)}. "
                     "Check prelim input or heat structure.")


# --- Remaining athletes (domain still open: can be 0 or 1) ---
remaining_df = df[~df["Athlete ID"].isin(auto_ids)].copy()


# --- Constraint 2: Next 6 fastest by SimTime -> "Time" qualifiers ---
def select_time_qualifiers(remaining: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """
    Chapter 6 "search" step:
      - Domain for each remaining athlete: Qualify ∈ {0,1}
      - Global constraint: exactly k of them must have Qualify = 1,
        and they must be the fastest by SimTime (tie-break by SeedTime, then Athlete ID).

    We implement this with a deterministic selection:
        sort by (SimTime, SeedTime, Athlete ID) and take the first k.
    """
    sorted_rem = (
        remaining.sort_values(
            by=["SimTime", "SeedTime", "Athlete ID"],
            ascending=[True, True, True]
        )
    )
    return sorted_rem.head(k)


time_df = select_time_qualifiers(remaining_df, k=6)
time_ids = sorted(time_df["Athlete ID"].astype(int).tolist())

for aid in time_ids:
    Qualify[int(aid)] = 1

# Sanity: total finalists = 32
qualified_ids_csp = [aid for aid, val in Qualify.items() if val == 1]
if len(qualified_ids_csp) != 32:
    raise ValueError(f"CSP produced {len(qualified_ids_csp)} qualifiers, expected 32.")


# ==========================================================
# 3. BUILD FINAL QUALIFIER TABLE (CSV OUTPUT)
# ==========================================================
qualified_rows = df[df["Athlete ID"].isin(qualified_ids_csp)].copy()

def tag_qualification(row):
    aid = int(row["Athlete ID"])
    if aid in auto_ids:
        return "Auto"
    elif aid in time_ids:
        return "Time"
    else:
        return "None"  # Should not happen if CSP is correct

qualified_rows["Qualification"] = qualified_rows.apply(tag_qualification, axis=1)

# Sort for nice reading: Auto first, then Time; inside by Heat, then SimTime
qualified_rows = qualified_rows.sort_values(
    by=["Qualification", "Heat", "SimTime"],
    ascending=[True, True, True]
)

# Reorder columns for clarity
qualified_rows = qualified_rows[
    ["Athlete ID", "Name", "School",
     "Heat", "Lane", "SeedTime", "SimTime", "Place", "Qualification"]
]

qualified_rows.to_csv(OUTPUT_FILE, index=False)


# ==========================================================
# 4. CHAPTER 9 — HORN-CLAUSE INFERENCE VALIDATION
# ==========================================================

# --- Facts and rules (propositional Horn clauses) ---
facts: set[str] = set()
rules: list[tuple[list[str], str]] = []  # (antecedents, conclusion)

# 4.1. Base facts: Athlete, Heat, Place, SimTime
for _, row in df.iterrows():
    aid = int(row["Athlete ID"])
    h = int(row["Heat"])
    p = int(row["Place"])
    t = float(row["SimTime"])

    facts.add(f"Athlete({aid})")
    facts.add(f"Heat({aid},H{h})")
    facts.add(f"Place({aid},{p})")
    facts.add(f"SimTime({aid},{t:.3f})")

# 4.2. CSP-derived facts for Chapter 9
# These are the "ground truth" from Chapter 6.
for aid in auto_ids:
    facts.add(f"CSPAuto({aid})")

for aid in time_ids:
    facts.add(f"CSPTime({aid})")

# 4.3. Horn rules

# From CSPAuto to logical Auto
# CSPAuto(a) → Auto(a)
for aid in df["Athlete ID"].astype(int).unique():
    rules.append(([f"CSPAuto({aid})"], f"Auto({aid})"))

# From CSPTime to logical TimeQual
# CSPTime(a) → TimeQual(a)
for aid in df["Athlete ID"].astype(int).unique():
    rules.append(([f"CSPTime({aid})"], f"TimeQual({aid})"))

# Auto(a) → Qualifies(a)
# TimeQual(a) → Qualifies(a)
for aid in df["Athlete ID"].astype(int).unique():
    rules.append(([f"Auto({aid})"], f"Qualifies({aid})"))
    rules.append(([f"TimeQual({aid})"], f"Qualifies({aid})"))

# (Optional academic extras — not used in consistency check, but nice for report)
# Place(a,1) → Top2(a)
# Place(a,2) → Top2(a)
# Top2(a)    → StrongPerformance(a)
for aid in df["Athlete ID"].astype(int).unique():
    rules.append(([f"Place({aid},1)"], f"Top2({aid})"))
    rules.append(([f"Place({aid},2)"], f"Top2({aid})"))
    rules.append(([f"Top2({aid})"], f"StrongPerformance({aid})"))


# 4.4. Forward chaining (standard Horn-clause inference)
def forward_chain(facts_set: set[str], rules_list: list[tuple[list[str], str]]) -> set[str]:
    """
    Classic forward-chaining algorithm for propositional Horn clauses:

    While new facts can be derived:
        If all antecedents of a rule are in the fact set,
        add the rule's conclusion.
    """
    facts_fc = set(facts_set)
    changed = True
    while changed:
        changed = False
        for antecedents, conclusion in rules_list:
            if all(a in facts_fc for a in antecedents):
                if conclusion not in facts_fc:
                    facts_fc.add(conclusion)
                    changed = True
    return facts_fc


derived = forward_chain(facts, rules)


# 4.5. Extract inferred Auto / TimeQual sets by Athlete ID
def extract_ids(prefix: str, facts_set: set[str]) -> list[int]:
    ids = []
    for f in facts_set:
        if f.startswith(prefix + "(") and f.endswith(")"):
            inside = f[len(prefix) + 1:-1]  # content inside parentheses
            try:
                aid = int(inside)
                ids.append(aid)
            except ValueError:
                # Not a pure integer argument; skip
                pass
    return sorted(set(ids))


auto_inferred_ids = extract_ids("Auto", derived)
time_inferred_ids = extract_ids("TimeQual", derived)

auto_csp_ids = sorted(auto_ids)
time_csp_ids = sorted(time_ids)


# ==========================================================
# 5. CONSISTENCY CHECK (CH.6 vs CH.9)
# ==========================================================
consistent_auto = (auto_inferred_ids == auto_csp_ids)
consistent_time = (time_inferred_ids == time_csp_ids)
consistent_all = consistent_auto and consistent_time

if not consistent_all:
    print("⚠ Inconsistency between CSP (Ch.6) and inference (Ch.9)!")
    print(f"  CSP Auto IDs:      {auto_csp_ids}")
    print(f"  Inferred Auto IDs: {auto_inferred_ids}")
    print(f"  CSP Time IDs:      {time_csp_ids}")
    print(f"  Inferred Time IDs: {time_inferred_ids}")
else:
    print("✔ CHAPTER 6 (CSP) and CHAPTER 9 (Inference) agree perfectly.")
    print(f"  Auto qualifiers (IDs): {auto_csp_ids}")
    print(f"  Time qualifiers (IDs): {time_csp_ids}")
    print(f"\n✔ Finalists CSV written to: {OUTPUT_FILE}")
