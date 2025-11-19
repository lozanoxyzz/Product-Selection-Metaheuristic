from typing import List, Dict, Tuple
import random
import pandas as pd

from constraints import is_feasible, selection_metrics
from greedy import top_k_per_category


def build_initial_solution(df: pd.DataFrame,
                           buckets: Dict[str, List[int]],
                           max_items: int = 6,
                           max_tries: int = 1000) -> List[int]:
    """
    Builds an initial feasible solution:
    - one product per category
    - taken from the top-k list
    """

    categories = list(buckets.keys())

    # Deterministic attempt: best item of each category
    initial = []
    for cat in categories:
        if buckets[cat]:
            initial.append(buckets[cat][0])

    if len(initial) == 0 or not is_feasible(initial, df, max_items=max_items):
        # Random attempts until a feasible combination is found
        for _ in range(max_tries):
            candidate = [random.choice(buckets[cat]) for cat in categories if len(buckets[cat]) > 0]
            if candidate and is_feasible(candidate, df, max_items=max_items):
                return candidate
        raise RuntimeError("Failed to build a feasible initial solution.")

    return initial


def tabu_search(df: pd.DataFrame,
                k_per_cat: int = 3,
                max_items: int = 6,
                max_iter: int = 100,
                tabu_tenure: int = 7) -> Tuple[List[int], float, float, float]:
    """
    Tabu Search:
    - solution: 6 products, one per category
    - neighborhood: swap item within the same category
    - objective: maximize profit
    - candidate pool: top-k per category
    """

    # Build category → top-k candidates
    buckets = top_k_per_category(df, k=k_per_cat)
    categories = list(buckets.keys())

    # Initial solution
    current = build_initial_solution(df, buckets, max_items=max_items)
    best = current.copy()
    best_profit, best_cogs, best_rating = selection_metrics(current, df)

    # Tabu list (category, product_id) → remaining iterations
    tabu: Dict[Tuple[str, int], int] = {}

    print("Initial solution:", current)
    print(f"Initial profit = {best_profit:.3f}, COGS = {best_cogs:.3f}, rating = {best_rating:.3f}")

    # Build product → category mapping
    id_to_cat = {row["product id"]: row["category"] for _, row in df.iterrows()}

    for it in range(1, max_iter + 1):
        best_neighbor = None
        best_neighbor_profit = float("-inf")
        best_neighbor_cogs = 0.0
        best_neighbor_rating = 0.0
        best_move = None  # (category, old_id, new_id)

        # Find current category assignment
        cat_to_current = {}
        for pid in current:
            cat_to_current[id_to_cat[pid]] = pid

        # Explore neighborhood
        for cat in categories:
            if cat not in cat_to_current:
                continue

            current_pid = cat_to_current[cat]

            for cand_pid in buckets[cat]:
                if cand_pid == current_pid:
                    continue

                new_sol = current.copy()
                idx = new_sol.index(current_pid)
                new_sol[idx] = cand_pid

                if not is_feasible(new_sol, df, max_items=max_items):
                    continue

                profit, cogs, rating = selection_metrics(new_sol, df)

                # Tabu check
                is_tabu = (cat, cand_pid) in tabu
                if is_tabu and profit <= best_profit:
                    continue  # tabu and not improving global best

                if profit > best_neighbor_profit:
                    best_neighbor = new_sol
                    best_neighbor_profit = profit
                    best_neighbor_cogs = cogs
                    best_neighbor_rating = rating
                    best_move = (cat, current_pid, cand_pid)

        if best_neighbor is None:
            print(f"Iteration {it}: no admissible neighbors, stopping.")
            break

        # Move to best neighbor
        current = best_neighbor
        moved_cat, old_id, new_id = best_move
        tabu[(moved_cat, old_id)] = tabu_tenure

        # Update tabu tenures
        to_remove = []
        for key in tabu:
            tabu[key] -= 1
            if tabu[key] <= 0:
                to_remove.append(key)
        for key in to_remove:
            del tabu[key]

        # Update global best
        if best_neighbor_profit > best_profit:
            best = current.copy()
            best_profit = best_neighbor_profit
            best_cogs = best_neighbor_cogs
            best_rating = best_neighbor_rating

        print(f"Iter {it:3d} | Current Profit = {best_neighbor_profit:.3f} | "
              f"Best = {best_profit:.3f} | Solution = {current}")

    print("\nBest solution found:", best)
    print(f"Profit = {best_profit:.3f}, COGS = {best_cogs:.3f}, Rating = {best_rating:.3f}")

    return best, best_profit, best_cogs, best_rating

