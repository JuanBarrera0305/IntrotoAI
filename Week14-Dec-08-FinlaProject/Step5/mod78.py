"""
CHAPTER 7 + CHAPTER 8 — Logical Agents & Inference on Semifinalists
-------------------------------------------------------------------

Input:
    Semifinal_Heats_By_Prelims.csv

Columns:
    Athlete ID, Name, School, Heat, Lane, SeedTime, SimTime, Place, Qualification

Goal:
    Use *logical agents* (Ch.7) + *first-order inference* (Ch.8) to reason about
    semifinalists in a knowledge-based way.

We:

 1) Build a FIRST-ORDER style knowledge base (KB) of tuple facts, e.g.:

        ('SeedTime', aid, t)
        ('LaneOf', aid, lane)
        ('AutoQualCSV', aid)
        ('TimeQualCSV', aid)

 2) Apply domain axioms (Horn-like rules) with a forward-chaining engine
    similar in spirit to the homework for Chapter 8.

    Examples of axioms:

        If qualification from CSV = "Auto"  → AutoQual(a) ∧ Qualifies(a)
        If qualification from CSV = "Time"  → TimeQual(a) ∧ Qualifies(a)
        If SeedTime(a) ≤ 10.50              → FastSeed(a)
        If Lane ∈ {3,4,5,6}                 → LaneAdv(a)
        If AutoQual(a) ∧ FastSeed(a)        → StrongFinalist(a)
        If TimeQual(a) ∧ FastSeed(a) ∧ LaneAdv(a) → DarkHorse(a)

 3) Compare:

        - Auto / Time qualifiers in CSV
        - Auto / Time qualifiers entailed by the KB

 4) Print logical groups:

        - StrongFinalist (Auto ∧ FastSeed)
        - DarkHorse     (Time ∧ FastSeed ∧ LaneAdv)

This file is explicitly “Chapter 7 + Chapter 8”:
   • Ch.7: logical agent view of semifinalists + properties/goals.
   • Ch.8: implementation of forward chaining over Horn-like axioms.
"""

import pandas as pd

INPUT_FILE = "Semifinal_Heats_By_Prelims.csv"

# -------------------------------------------------------------------
# 1. LOAD SEMIFINALISTS
# -------------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()

required_cols = [
    "Athlete ID", "Name", "School",
    "Heat", "Lane", "Prelims", "Place", "Qualification"
]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}' in {INPUT_FILE}")


# Make sure types are clean
df["Athlete ID"] = df["Athlete ID"].astype(int)
df["Heat"] = df["Heat"].astype(int)
df["Lane"] = df["Lane"].astype(int)
df["Place"] = df["Place"].astype(int)
df["Qualification"] = df["Qualification"].astype(str).str.strip()


# -------------------------------------------------------------------
# 2. BUILD INITIAL FACT SET (KB) — CHAPTER 7 “LOGICAL AGENT STATE”
# -------------------------------------------------------------------
# Facts are tuples: ('Predicate', arg1, arg2, ...)
facts: set[tuple] = set()

for _, row in df.iterrows():
    aid = int(row["Athlete ID"])
    name = str(row["Name"]).strip()
    school = str(row["School"]).strip()
    heat = int(row["Heat"])
    lane = int(row["Lane"])
    seed = float(row["Prelims"])
    place = int(row["Place"])
    qual = row["Qualification"]

    # Base object & attributes
    facts.add(("Athlete", aid))
    facts.add(("NameOf", aid, name))
    facts.add(("SchoolOf", aid, school))
    facts.add(("HeatOf", aid, heat))
    facts.add(("LaneOf", aid, lane))
    facts.add(("SeedTime", aid, seed))
    facts.add(("Place", aid, place))

    # Qualification facts from CSV (these are like given evidence)
    if qual == "Auto":
        facts.add(("AutoQualCSV", aid))
    elif qual == "Time":
        facts.add(("TimeQualCSV", aid))
    else:
        # Should not happen in this project, but we keep it safe
        facts.add(("OtherQualCSV", aid))


# -------------------------------------------------------------------
# 3. AXIOMS / RULES — CHAPTER 8 STYLE (FORWARD CHAINING)
# -------------------------------------------------------------------

FAST_SEED_THRESHOLD = 10.50
ADV_LANES = {3, 4, 5, 6}
DISADV_LANES = {1, 8}


def apply_axioms(facts_in: set[tuple]) -> set[tuple]:
    """
    Forward-chaining style deduction over tuple facts, in the spirit
    of the Chapter 8 homework (course/professor example).

    We repeatedly scan the fact base and add any facts implied by our axioms
    until saturation (no more new facts).
    """
    new_facts = set(facts_in)
    changed = True

    while changed:
        changed = False
        current = set(new_facts)

        # ----------------------------
        # Unary rules (depend on one fact)
        # ----------------------------
        for f in current:
            pred, *args = f

            # Axiom 1 & 2:
            #   AutoQualCSV(a) → AutoQual(a) ∧ Qualifies(a)
            #   TimeQualCSV(a) → TimeQual(a) ∧ Qualifies(a)
            if pred == "AutoQualCSV":
                a = args[0]
                if ("AutoQual", a) not in new_facts:
                    new_facts.add(("AutoQual", a))
                    changed = True
                if ("Qualifies", a) not in new_facts:
                    new_facts.add(("Qualifies", a))
                    changed = True

            if pred == "TimeQualCSV":
                a = args[0]
                if ("TimeQual", a) not in new_facts:
                    new_facts.add(("TimeQual", a))
                    changed = True
                if ("Qualifies", a) not in new_facts:
                    new_facts.add(("Qualifies", a))
                    changed = True

            # Axiom 3:
            #   SeedTime(a,t) ∧ (t <= FAST_SEED_THRESHOLD) → FastSeed(a)
            if pred == "SeedTime":
                a, t = args
                if t <= FAST_SEED_THRESHOLD and ("FastSeed", a) not in new_facts:
                    new_facts.add(("FastSeed", a))
                    changed = True

            # Axiom 4:
            #   LaneOf(a,l) ∧ l ∈ {3,4,5,6} → LaneAdv(a)
            #   LaneOf(a,l) ∧ l ∈ {1,8}     → LaneDisadv(a)
            if pred == "LaneOf":
                a, l = args
                if l in ADV_LANES and ("LaneAdv", a) not in new_facts:
                    new_facts.add(("LaneAdv", a))
                    changed = True
                if l in DISADV_LANES and ("LaneDisadv", a) not in new_facts:
                    new_facts.add(("LaneDisadv", a))
                    changed = True

        # ----------------------------
        # Binary / multi-fact rules
        # ----------------------------

        # Precompute some index lists for the current iteration
        auto_q = [f for f in new_facts if f[0] == "AutoQual"]
        time_q = [f for f in new_facts if f[0] == "TimeQual"]
        fast_s = [f for f in new_facts if f[0] == "FastSeed"]
        lane_a = [f for f in new_facts if f[0] == "LaneAdv"]

        # Axiom 5:
        #   AutoQual(a) ∧ FastSeed(a) → StrongFinalist(a)
        for (_, a1) in auto_q:
            if ("FastSeed", a1) in new_facts and ("StrongFinalist", a1) not in new_facts:
                new_facts.add(("StrongFinalist", a1))
                changed = True

        # New Axiom 6:
        #   TimeQual(a) ∧ SeedTime(a,t) ∧ (t <= 10.55) → DarkHorse(a)
        for (_, a1, t) in [f for f in new_facts if f[0] == "SeedTime"]:
            if t <= 10.55 and ("TimeQual", a1) in new_facts:
                if ("DarkHorse", a1) not in new_facts:
                    new_facts.add(("DarkHorse", a1))
                    changed = True

        # Axiom 7 (optional academic flavor):
        #   Place(a,1) → HeatWinner(a)
        #   Place(a,2) → HeatPodium(a)
        #   HeatWinner(a) → StrongPerformance(a)
        #   HeatPodium(a) → StrongPerformance(a)
        for f in [f for f in new_facts if f[0] == "Place"]:
            _, a, p = f
            if p == 1 and ("HeatWinner", a) not in new_facts:
                new_facts.add(("HeatWinner", a))
                changed = True
            if p == 2 and ("HeatPodium", a) not in new_facts:
                new_facts.add(("HeatPodium", a))
                changed = True

        for tag in ["HeatWinner", "HeatPodium"]:
            for f in [f for f in new_facts if f[0] == tag]:
                _, a = f
                if ("StrongPerformance", a) not in new_facts:
                    new_facts.add(("StrongPerformance", a))
                    changed = True

    return new_facts


deduced_facts = apply_axioms(facts)


# -------------------------------------------------------------------
# 4. HELPERS TO EXTRACT IDS FROM FACT SET
# -------------------------------------------------------------------
def ids_with(pred: str) -> set[int]:
    return {f[1] for f in deduced_facts if f[0] == pred}


auto_ids_from_csv = set(df[df["Qualification"] == "Auto"]["Athlete ID"].tolist())
time_ids_from_csv = set(df[df["Qualification"] == "Time"]["Athlete ID"].tolist())

auto_ids_kb = ids_with("AutoQual")
time_ids_kb = ids_with("TimeQual")

# Limit StrongFinalists to top-15 fastest Auto qualifiers
auto_sorted = df[df["Athlete ID"].isin(auto_ids_kb)].sort_values(by="Prelims")
top15_auto = set(auto_sorted.head(15)["Athlete ID"].tolist())

# Override StrongFinalist facts to only include top15_auto
# Remove StrongFinalist tags for Auto qualifiers not in top 15
for fact in list(deduced_facts):
    if fact[0] == "StrongFinalist":
        a = fact[1]
        if a not in top15_auto:
            deduced_facts.remove(fact)

# Ensure top 15 are all marked StrongFinalist
for a in top15_auto:
    deduced_facts.add(("StrongFinalist", a))

# -------------------------------------------------------------------
# 5. CONSISTENCY CHECK: CH.7/8 LOGIC vs CSV
# -------------------------------------------------------------------
print("=== CH.7/8 LOGIC vs CSV CONSISTENCY CHECK ===")

consistent_auto = (auto_ids_from_csv == auto_ids_kb)
consistent_time = (time_ids_from_csv == time_ids_kb)

if consistent_auto and consistent_time:
    print("✔ All CSV qualifiers are logically entailed by the KB.\n")
else:
    print("⚠ Inconsistency detected between CSV and KB!")
    print(f"  CSV Auto IDs: {sorted(auto_ids_from_csv)}")
    print(f"   KB Auto IDs: {sorted(auto_ids_kb)}")
    print(f"  CSV Time IDs: {sorted(time_ids_from_csv)}")
    print(f"   KB Time IDs: {sorted(time_ids_kb)}")
    print()


# -------------------------------------------------------------------
# 6. REPORT LOGICAL CATEGORIES (STRONGFINALIST, DARKHORSE, etc.)
# -------------------------------------------------------------------
strong_ids = sorted(ids_with("StrongFinalist"))
darkhorse_ids = sorted(ids_with("DarkHorse"))

def formatted_row(aid: int) -> str:
    row = df[df["Athlete ID"] == aid].iloc[0]
    return (
        f"{aid:4d}  "
        f"{row['Name']:<18} "
        f"Seed={row['Prelims']:.2f}  "
        f"Lane={int(row['Lane'])}  "
        f"Qual={row['Qualification']}"
    )

print(f"=== StrongFinalist (Auto ∧ FastSeed) (|{len(strong_ids)}|) ===")
for aid in strong_ids:
    print(" ", formatted_row(aid))

print()
print(f"=== DarkHorse (TimeQual ∧ Prelims ≤ 10.55) (|{len(darkhorse_ids)}|) ===")
for aid in darkhorse_ids:
    print(" ", formatted_row(aid))
print()

# ---------------------------------------------------------
# SAVE CH.7/8 CLASSIFICATION FOR CHAPTER 11
# ---------------------------------------------------------

classified = df.copy()

classified["StrongFinalist"] = classified["Athlete ID"].isin(strong_ids).astype(int)
classified["DarkHorse"]      = classified["Athlete ID"].isin(darkhorse_ids).astype(int)

classified.to_csv("Finalists_Classified_CH7_8.csv", index=False)

print("\nSaved CH7/8 classifications → Finalists_Classified_CH7_8.csv")