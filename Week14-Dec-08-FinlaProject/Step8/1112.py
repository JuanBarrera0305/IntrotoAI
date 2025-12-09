"""
CH.11 + CH.12 + CH.13 + CH.14 — Final Race Intelligence Pipeline
-----------------------------------------------------------------

Input:
    Finalists_CH13_CH14.csv
        Athlete ID, Name, School, SeedTime, SimTime, DisplayedSemiTime

Steps:
  • Chapter 12: Knowledge Representation on finalists
      - SeedClass (Elite / Competitive / Borderline)
      - StrongFinalistFinal, DarkHorseFinal using time-based logic

  • Chapter 11: Planning Final Lanes
      - STRIPS-style AssignLane(a, F1, lane)
      - Center lanes first for strong finalists, then dark horses, then others

  • Chapters 13 + 14: Monte Carlo final race simulation
      - Uses DisplayedSemiTime as posterior mean (PostMean)
      - Posterior predictive sampling for final times
      - ProbGold, ProbMedal, PredictedFinalTime

Output:
    FinalRace_CH11_12_13_14.csv
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import heapq

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 0. CONFIG & RANDOM SEEDS
# ---------------------------------------------------------
INPUT_FILE = "Finalists_CH13_CH14.csv"
OUTPUT_FILE = "FinalRace_CH11_12_13_14.csv"
PLAN_LOG = "Final_Plan_CH11.log"

# Final round performance noise (slightly lower variability)
SIGMA_FINAL = 0.045

# Monte Carlo simulation count
N_SIMS = 20000

random.seed(2025)
np.random.seed(2025)


# =========================================================
# CHAPTER 12 — Knowledge Representation on Finalists
# =========================================================

def seed_class_from_time(t: float) -> str:
    """Simple ontology for performance categories."""
    if t <= 10.37:
        return "Elite"
    elif t <= 10.40:
        return "Competitive"
    else:
        return "Borderline"


def build_finalists_kb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Ch.12-style knowledge:
      - SeedClass based on DisplayedSemiTime
      - StrongFinalistFinal: top 3 by DisplayedSemiTime
      - DarkHorseFinal: athletes who improved vs SeedTime
    """
    df = df.copy()
    df["DisplayedSemiTime"] = df["DisplayedSemiTime"].astype(float)
    df["SeedTime"] = df["SeedTime"].astype(float)

    # SeedClass from DisplayedSemiTime (posterior performance estimate)
    df["SeedClass"] = df["DisplayedSemiTime"].apply(seed_class_from_time)

    # Rank by semifinal performance (lower is better)
    df = df.sort_values(by="DisplayedSemiTime").reset_index(drop=True)

    # Strong finalists in final: top 3 by semifinal performance
    df["StrongFinalistFinal"] = 0
    df.loc[df.index[:3], "StrongFinalistFinal"] = 1

    # Dark horses in final: those who improved vs seed time
    df["DarkHorseFinal"] = (df["DisplayedSemiTime"] < df["SeedTime"]).astype(int)

    return df


# =========================================================
# CHAPTER 11 — STRIPS Planner for FINAL LANE ASSIGNMENT
# =========================================================

@dataclass(frozen=True)
class Action:
    name: str
    precond_pos: frozenset
    precond_neg: frozenset
    add: frozenset
    delete: frozenset


def make_fluent(pred: str, *args) -> str:
    return f"{pred}({', '.join(str(a) for a in args)})"


def applicable(state: frozenset, action: Action) -> bool:
    return action.precond_pos.issubset(state) and state.isdisjoint(action.precond_neg)


def apply(state: frozenset, action: Action) -> frozenset:
    return frozenset((state - action.delete) | action.add)


def goal_test(state: frozenset, goal: frozenset) -> bool:
    return goal.issubset(state)


def heuristic_unsatisfied_goals(state: frozenset, finalists: List[int]) -> int:
    assigned = {f for f in state if f.startswith("Assigned(")}
    return len(finalists) - len(assigned)


def astar(initial: frozenset,
          goal: frozenset,
          actions: List[Action],
          finalists: List[int]) -> Optional[List[Action]]:
    """
    A* search over STRIPS states for final lane assignment.
    Cost of every action = 1.
    """
    frontier: List[Tuple[int, int, frozenset, List[Action]]] = []
    g0 = 0
    h0 = heuristic_unsatisfied_goals(initial, finalists)
    heapq.heappush(frontier, (g0 + h0, g0, initial, []))
    best_g: Dict[frozenset, int] = {initial: 0}

    while frontier:
        f, g, state, plan = heapq.heappop(frontier)

        if goal_test(state, goal):
            return plan

        for a in actions:
            if not applicable(state, a):
                continue
            next_state = apply(state, a)
            g2 = g + 1
            if next_state not in best_g or g2 < best_g[next_state]:
                best_g[next_state] = g2
                h2 = heuristic_unsatisfied_goals(next_state, finalists)
                heapq.heappush(frontier, (g2 + h2, g2, next_state, plan + [a]))

    return None


def plan_final_lanes(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Action], frozenset]:
    """
    Use a Ch.11 STRIPS planner to assign final lanes for the 8 finalists.
    - One final heat: F1
    - Lanes 1..8
    - Priority: StrongFinalistFinal → DarkHorseFinal → others
    - Lane priority: [4, 5, 3, 6, 2, 7, 1, 8]
    """
    df = df.copy()
    finalists: List[int] = df["Athlete ID"].astype(int).tolist()

    if len(finalists) != 8:
        raise ValueError(
            f"Expected exactly 8 finalists for the final, found {len(finalists)}."
        )

    # Domain objects
    FINAL_HEAT = "F1"
    LANES = list(range(1, 9))
    LANE_PRIORITY = [4, 5, 3, 6, 2, 7, 1, 8]

    # Priority ordering based on Ch.12 knowledge
    strong_ids = set(df[df["StrongFinalistFinal"] == 1]["Athlete ID"])
    dark_ids = set(df[(df["DarkHorseFinal"] == 1) & (df["StrongFinalistFinal"] == 0)]["Athlete ID"])

    ordered_finalists: List[int] = (
        [a for a in finalists if a in strong_ids] +
        [a for a in finalists if a in dark_ids and a not in strong_ids] +
        [a for a in finalists if a not in strong_ids and a not in dark_ids]
    )

    # Initial state: all unassigned, all lanes free
    init_fluents = set()
    for a in finalists:
        init_fluents.add(make_fluent("Unassigned", a))
    for lane in LANES:
        init_fluents.add(make_fluent("LaneFree", FINAL_HEAT, lane))
    init_state = frozenset(init_fluents)

    # Goal: Assigned(a) for all finalists
    goal = frozenset({make_fluent("Assigned", a) for a in finalists})

    # Action schema: AssignLane(a, F1, lane)
    def make_assign_action(a: int, lane: int) -> Action:
        return Action(
            name=f"AssignLane(a={a}, heat={FINAL_HEAT}, lane={lane})",
            precond_pos=frozenset({
                make_fluent("Unassigned", a),
                make_fluent("LaneFree", FINAL_HEAT, lane),
            }),
            precond_neg=frozenset(),
            add=frozenset({
                make_fluent("Assigned", a),
                make_fluent("InHeat", a, FINAL_HEAT),
                make_fluent("InLane", a, lane),
            }),
            delete=frozenset({
                make_fluent("Unassigned", a),
                make_fluent("LaneFree", FINAL_HEAT, lane),
            }),
        )

    # Ground actions with good ordering: strong finalists in best lanes first
    ground_actions: List[Action] = []
    for a in ordered_finalists:
        for lane in LANE_PRIORITY:
            ground_actions.append(make_assign_action(a, lane))

    # Run A* planner
    plan = astar(init_state, goal, ground_actions, finalists)
    if plan is None:
        raise RuntimeError("No plan found for final lanes (this should not happen).")

    # Simulate plan to get final state
    state = init_state
    for act in plan:
        if not applicable(state, act):
            raise RuntimeError(f"Action not applicable during simulation: {act.name}")
        state = apply(state, act)
    final_state = state

    # Extract lane assignments from final_state
    assignments: Dict[int, int] = {}  # Athlete ID -> FinalLane

    for f in final_state:
        if f.startswith("InLane("):
            inside = f[len("InLane("):-1]
            a_str, lane_str = [x.strip() for x in inside.split(",")]
            aid = int(a_str)
            lane = int(lane_str)
            assignments[aid] = lane

    if len(assignments) != len(finalists):
        raise RuntimeError(
            f"Planner produced {len(assignments)} lane assignments for {len(finalists)} finalists."
        )

    # Attach FinalLane to df
    df["FinalLane"] = df["Athlete ID"].astype(int).map(assignments)

    return df, plan, final_state


# =========================================================
# CH.13 + CH.14 — Final Race Simulation (Your Logic)
# =========================================================

def sample_final_time(mu: float, sigma: float) -> float:
    # sigma here is the posterior std; total variance adds SIGMA_FINAL^2
    return random.gauss(mu, math.sqrt(sigma ** 2 + SIGMA_FINAL ** 2))


def run_final_sim(df: pd.DataFrame, n_sims: int):
    win_counts = defaultdict(int)
    medal_counts = defaultdict(int)
    final_time_samples = defaultdict(list)

    athletes = df.to_dict("records")

    for _ in range(n_sims):
        results = []

        for a in athletes:
            t = sample_final_time(a["PostMean"], a["PostStd"])
            final_time_samples[a["Athlete ID"]].append(t)
            results.append((a["Athlete ID"], t))

        # Sort by time (lower is better)
        results.sort(key=lambda x: x[1])

        # Winner
        win_counts[results[0][0]] += 1

        # Medalists (top 3)
        for aid, _ in results[:3]:
            medal_counts[aid] += 1

    # Probabilities & predicted time
    win_prob = {aid: win_counts[aid] / n_sims for aid in df["Athlete ID"]}
    medal_prob = {aid: medal_counts[aid] / n_sims for aid in df["Athlete ID"]}
    pred_time = {aid: float(np.mean(final_time_samples[aid])) for aid in df["Athlete ID"]}

    return win_prob, medal_prob, pred_time


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():
    # 1) Load finalists
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    needed_cols = ["Athlete ID", "Name", "School", "SeedTime", "SimTime", "DisplayedSemiTime"]
    for c in needed_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {INPUT_FILE}")

    df["Athlete ID"] = df["Athlete ID"].astype(int)

    print("\n=== Loaded finalists for CH.11 + CH.12 + CH.13 + CH.14 ===")
    print(df[["Athlete ID", "Name", "School", "SeedTime", "SimTime", "DisplayedSemiTime"]])

    # 2) Chapter 12: build knowledge on finalists
    df_kb = build_finalists_kb(df)
    print("\n=== Chapter 12 — Knowledge Representation on Finalists ===")
    print(df_kb[["Athlete ID", "Name", "SeedClass", "StrongFinalistFinal", "DarkHorseFinal"]])

    # 3) Chapter 11: plan final lanes using that knowledge
    df_planned, plan, final_state = plan_final_lanes(df_kb)

    print("\n=== Chapter 11 — Planned Final Lanes ===")
    print(df_planned[["Athlete ID", "Name", "FinalLane", "SeedClass",
                      "StrongFinalistFinal", "DarkHorseFinal"]].sort_values("FinalLane"))

    # Write a human-readable plan log
    with open(PLAN_LOG, "w") as f:
        f.write("CHAPTER 11 PLAN — Final Lane Assignment\n")
        f.write("=======================================\n\n")
        for i, act in enumerate(plan, 1):
            f.write(f"{i:2d}. {act.name}\n")

        f.write("\nFinal state sample fluents:\n")
        for fluent in sorted(list(final_state))[:30]:
            f.write(f"  {fluent}\n")

    print(f"\n📝 Final lane plan written to {PLAN_LOG}")

    # 4) Chapters 13 + 14: posterior predictive simulation for final times

    # Posterior mean: DisplayedSemiTime (updated from semifinal)
    df_planned["PostMean"] = df_planned["DisplayedSemiTime"].astype(float)

    # Posterior std (confidence) — constant for now, could be refined per athlete
    df_planned["PostStd"] = 0.05

    win_prob, medal_prob, pred_time = run_final_sim(df_planned, N_SIMS)

    df_planned["ProbGold"] = df_planned["Athlete ID"].map(win_prob)
    df_planned["ProbMedal"] = df_planned["Athlete ID"].map(medal_prob)
    df_planned["PredictedFinalTime"] = df_planned["Athlete ID"].map(pred_time)

    # Sort by winning probability for projected ranking
    df_out = df_planned.sort_values(by="ProbGold", ascending=False).reset_index(drop=True)

    # Save final race output
    df_out[[
        "Athlete ID", "Name", "School", "FinalLane",
        "ProbGold", "ProbMedal", "PredictedFinalTime"
    ]].to_csv(OUTPUT_FILE, index=False)

    print("\n=== FINAL RACE SIMULATION WRITTEN TO", OUTPUT_FILE, "===\n")
    print(df_out[[
        "Athlete ID", "Name", "School", "FinalLane",
        "ProbGold", "ProbMedal", "PredictedFinalTime"
    ]])


if __name__ == "__main__":
    main()
