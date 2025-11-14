from typing import List, Dict, Tuple
import random
import pandas as pd

from constraints import is_feasible, selection_metrics
from greedy import top_k_per_category


def build_initial_solution(df: pd.DataFrame,
                           buckets: Dict[str, List[int]],
                           max_items: int = 6,
                           max_tries: int = 1000) -> List[int]:
    
    categories = list(buckets.keys())

    
    initial = []
    for cat in categories:
        if not buckets[cat]:
            continue
        initial.append(buckets[cat][0])

    if len(initial) == 0 or not is_feasible(initial, df, max_items=max_items):
        
        best = None
        for _ in range(max_tries):
            candidate = []
            for cat in categories:
                if not buckets[cat]:
                    continue
                candidate.append(random.choice(buckets[cat]))
            if len(candidate) == 0:
                continue
            if is_feasible(candidate, df, max_items=max_items):
                best = candidate
                break
        if best is None:
            raise RuntimeError("No se encontró solución inicial factible.")
        return best

    return initial


def tabu_search(df: pd.DataFrame,
                k_per_cat: int = 3,
                max_items: int = 6,
                max_iter: int = 100,
                tabu_tenure: int = 7) -> Tuple[List[int], float, float, float]:

    buckets = top_k_per_category(df, k=k_per_cat)

    categories = list(buckets.keys())

    current = build_initial_solution(df, buckets, max_items=max_items)
    best = current.copy()
    best_profit, best_cogs, best_rating = selection_metrics(current, df)

    
    tabu: Dict[Tuple[str, int], int] = {}

    print("Solución inicial:", current)
    print(f"Profit inicial = {best_profit:.3f}, COGS = {best_cogs:.3f}, rating = {best_rating:.3f}")

    for it in range(1, max_iter + 1):
        best_neighbor = None
        best_neighbor_profit = float("-inf")
        best_neighbor_cogs = 0.0
        best_neighbor_rating = 0.0
        best_move = None  # (category, old_id, new_id)

        
        cat_to_current: Dict[str, int] = {}
        
        id_to_cat = {}
        for _, row in df.iterrows():
            id_to_cat[row["product id"]] = row["category"]

        for pid in current:
            cat = id_to_cat[pid]
            cat_to_current[cat] = pid

        
        for cat in categories:
            if cat not in cat_to_current:
                continue
            current_pid = cat_to_current[cat]

            for candidate_pid in buckets[cat]:
                if candidate_pid == current_pid:
                    continue

                new_solution = current.copy()
                
                idx = new_solution.index(current_pid)
                new_solution[idx] = candidate_pid

                if not is_feasible(new_solution, df, max_items=max_items):
                    continue

                profit, cogs, rating = selection_metrics(new_solution, df)

                move_is_tabu = (cat, candidate_pid) in tabu

            
                if move_is_tabu and profit <= best_profit:
                    continue

               
                if profit > best_neighbor_profit:
                    best_neighbor_profit = profit
                    best_neighbor_cogs = cogs
                    best_neighbor_rating = rating
                    best_neighbor = new_solution
                    best_move = (cat, current_pid, candidate_pid)

        if best_neighbor is None:
            print(f"Iteración {it}: sin vecinos admisibles, se detiene.")
            break

        
        current = best_neighbor

    
        moved_cat, old_id, new_id = best_move
        tabu[(moved_cat, old_id)] = tabu_tenure

        to_delete = []
        for key in tabu:
            tabu[key] -= 1
            if tabu[key] <= 0:
                to_delete.append(key)
        for key in to_delete:
            del tabu[key]

        if best_neighbor_profit > best_profit:
            best_profit = best_neighbor_profit
            best_cogs = best_neighbor_cogs
            best_rating = best_neighbor_rating
            best = current.copy()

        print(f"Iter {it:3d} | Profit actual = {best_neighbor_profit:.3f} | "
              f"Mejor global = {best_profit:.3f} | Solución = {current}")

    print("\nMejor solución encontrada:", best)
    print(f"Profit = {best_profit:.3f}, COGS = {best_cogs:.3f}, rating = {best_rating:.3f}")

    return best, best_profit, best_cogs, best_rating


if __name__ == "__main__":
    from data_loading import load_datasets

    df = load_datasets()
    if df is not None:
        best_sel, best_p, best_c, best_r = tabu_search(df)
