import pandas as pd
from tabu import tabu_search

def load_df(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def wait():
    input("\nPress ENTER to continue...")

def run_for_dataset(name, path):
    print(f"\n===== DATASET {name} =====")
    df = load_df(path)

    best_sel, best_profit, best_cogs, best_rating = tabu_search(
        df,
        k_per_cat=3,
        max_items=6,
        max_iter=100,
        tabu_tenure=7
    )

    return {
        "dataset": name,
        "n_items": len(df),
        "profit": best_profit,
        "cogs": best_cogs,
        "rating": best_rating,
        "eff": best_profit / best_cogs if best_cogs > 0 else 0,
        "selection": best_sel,
    }

def analyze(results):
    print("\n===== SIZE-BASED BEHAVIOR ANALYSIS =====")

    results = sorted(results, key=lambda r: r["n_items"])

    for i in range(1, len(results)):
        small = results[i-1]
        large = results[i]

        print(f"\nFrom {small['n_items']} → {large['n_items']} items:")

        dp = large["profit"] - small["profit"]
        de = large["eff"] - small["eff"]
        dr = large["rating"] - small["rating"]

        print(f"- Profit change: {dp:+.3f}")
        print(f"- Efficiency change (profit/COGS): {de:+.3f}")
        print(f"- Rating change: {dr:+.3f}")

        print("Interpretation:")
        if abs(dp) < 1 and abs(de) < 0.002:
            print("  → Tabu remains stable as dataset size increases.")
        elif dp > 0:
            print("  → With more data, Tabu finds slightly better combinations.")
        else:
            print("  → Tabu performance declines with larger datasets.")

    print("\nGeneral conclusion:")
    print("This analysis shows how the method scales with dataset size,")
    print("and whether complexity affects Tabu's ability to find strong solutions.\n")

def main():
    results = []

    results.append(run_for_dataset("50", "dataset_PIA_50.csv"))
    wait()

    results.append(run_for_dataset("100", "dataset_PIA_100.csv"))
    wait()

    results.append(run_for_dataset("150", "dataset_PIA_150.csv"))
    wait()

    print("\n===== FINAL RESULTS =====")
    for r in results:
        print(f"\nDataset {r['dataset']} ({r['n_items']} items):")
        print(f"  Profit : {r['profit']:.3f}")
        print(f"  COGS   : {r['cogs']:.3f}")
        print(f"  Rating : {r['rating']:.3f}")
        print(f"  Eff    : {r['eff']:.4f}")
        print(f"  Sel    : {r['selection']}")

    analyze(results)

if __name__ == "__main__":
    main()
