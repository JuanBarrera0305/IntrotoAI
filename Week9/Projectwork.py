import random
import math
from typing import List, Dict, Tuple

random.seed(42)

LANES = 8
LANE_PATTERN = [4, 5, 3, 6, 2, 7, 1, 8]  # assignment order for ranks 1..8

# --- Data structures ---
class Athlete:
    def __init__(self, athlete_id: int, name: str, seed_time: float):
        self.id = athlete_id
        self.name = name
        self.seed_time = seed_time

class Assignment:
    def __init__(self, round_name: str, heat_index: int, lane: int, athlete: Athlete):
        self.round = round_name
        self.heat_index = heat_index
        self.lane = lane
        self.athlete = athlete

class Result:
    def __init__(self, assignment: Assignment, time: float):
        self.assignment = assignment
        self.time = time

# --- Helpers ---
def generate_athletes(n: int) -> List[Athlete]:
    # Create athletes with seed times centered around 10.80 ± 0.60s
    athletes = []
    for i in range(n):
        base = 10.8 + random.gauss(0, 0.6)
        seed = max(9.8, min(12.5, base))  # clamp to reasonable sprint times
        athletes.append(Athlete(i+1, f"Athlete_{i+1}", round(seed, 3)))
    # Sort by seed time ascending (fastest first)
    athletes.sort(key=lambda a: a.seed_time)
    return athletes

def serpentine_groups(athletes: List[Athlete], lanes: int) -> List[List[Athlete]]:
    # Group athletes into heats of 'lanes' using serpentine distribution
    n = len(athletes)
    heats = math.ceil(n / lanes)
    groups = [[] for _ in range(heats)]
    idx = 0
    direction = 1  # 1 forward, -1 backward
    while idx < n:
        # fill a "row" across heats
        if direction == 1:
            for h in range(heats):
                if idx < n:
                    groups[h].append(athletes[idx])
                    idx += 1
            direction = -1
        else:
            for h in reversed(range(heats)):
                if idx < n:
                    groups[h].append(athletes[idx])
                    idx += 1
            direction = 1
    # Ensure each group length <= lanes
    for g in groups:
        if len(g) > lanes:
            raise ValueError("Group overflow; check serpentine logic.")
    return groups

def assign_lanes(group: List[Athlete], round_name: str, heat_index: int) -> List[Assignment]:
    # Assign lanes according to LANE_PATTERN for ranks within the heat
    # Rank group by seed_time ascending (fastest first)
    ranked = sorted(group, key=lambda a: a.seed_time)
    assignments = []
    for i, athlete in enumerate(ranked):
        lane = LANE_PATTERN[i] if i < len(LANE_PATTERN) else None
        assignments.append(Assignment(round_name, heat_index, lane, athlete))
    return assignments

def simulate_heat_results(assignments: List[Assignment]) -> List[Result]:
    # Simulate performance around seed_time with Gaussian noise
    results = []
    for a in assignments:
        noise = random.gauss(0, 0.08)  # 80ms std dev
        time = max(9.6, a.athlete.seed_time + noise)
        results.append(Result(a, round(time, 3)))
    # Sort by time ascending within heat
    results.sort(key=lambda r: r.time)
    return results

def progress_next_round(all_heat_results: List[List[Result]], auto_slots: int, target_size: int) -> List[Athlete]:
    # Auto-qualify winners of each heat
    winners = [heat_res[0].assignment.athlete for heat_res in all_heat_results]
    # Collect remaining athletes + times across all heats
    remaining = []
    for heat_res in all_heat_results:
        for r in heat_res[1:]:
            remaining.append((r.assignment.athlete, r.time))
    # Sort remaining by time ascending
    remaining.sort(key=lambda x: x[1])
    # Fill remaining slots to reach target_size
    need = target_size - len(winners)
    qualifiers = winners + [ath for ath, _ in remaining[:need]]
    # De-duplicate and ensure size
    unique = []
    seen = set()
    for a in qualifiers:
        if a.id not in seen:
            unique.append(a)
            seen.add(a.id)
    if len(unique) != target_size:
        raise ValueError("Progression size mismatch; check inputs and rule.")
    return unique

def round_assignments(athletes: List[Athlete], round_name: str) -> Tuple[List[List[Assignment]], List[List[Result]]]:
    groups = serpentine_groups(athletes, LANES)
    all_assignments = []
    all_results = []
    for h_idx, group in enumerate(groups, start=1):
        assigns = assign_lanes(group, round_name, h_idx)
        results = simulate_heat_results(assigns)
        all_assignments.append(assigns)
        all_results.append(results)
    return all_assignments, all_results

def print_round(all_assignments: List[List[Assignment]], all_results: List[List[Result]]):
    for h_idx, (assigns, results) in enumerate(zip(all_assignments, all_results), start=1):
        print(f"Round {assigns[0].round} — Heat {h_idx}")
        print("  Assignments (seed_time):")
        for a in sorted(assigns, key=lambda x: x.lane):
            print(f"    Lane {a.lane}: {a.athlete.name} ({a.athlete.seed_time}s)")
        print("  Results (simulated times):")
        for place, r in enumerate(results, start=1):
            print(f"    {place}. {r.assignment.athlete.name} — {r.time}s (Lane {r.assignment.lane})")
        print()

# --- Meet pipeline for 80 athletes on 8 lanes ---
def run_meet():
    athletes = generate_athletes(80)

    # Round 1: Heats (10 heats of 8 lanes)
    R1_assigns, R1_results = round_assignments(athletes, "Heats")
    print_round(R1_assigns, R1_results)

    # Progress to Semifinals: target 24 athletes
    semifinalists = progress_next_round(R1_results, auto_slots=len(R1_results), target_size=24)

    # Round 2: Semifinals (3 heats of 8 lanes)
    R2_assigns, R2_results = round_assignments(semifinalists, "Semifinals")
    print_round(R2_assigns, R2_results)

    # Progress to Final: target 8 athletes
    finalists = progress_next_round(R2_results, auto_slots=len(R2_results), target_size=8)

    # Round 3: Final (1 heat of 8 lanes)
    R3_assigns, R3_results = round_assignments(finalists, "Final")
    print_round(R3_assigns, R3_results)

if __name__ == "__main__":
    run_meet()