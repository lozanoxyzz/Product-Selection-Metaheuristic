import pandas as pd
from constraints import compute_budget_B,compute_rmin, is_feasible
from greedy import greedy_secure_filter, top_k_per_category, compute_greedy_score
from data_loading import load_datasets
# === Mini dataset ===
data = {
    "category": ["Drinks", "Snacks", "Fruits", "Drinks", "Snacks"],
    "cogs": [5.0, 4.0, 3.8, 4.5, 3.0],
    "profit": [8.0, 6.5, 5.2, 7.0, 5.8],
    "rating": [4.7, 4.4, 4.6, 4.1, 4.3],
    "product id": [1, 2, 3, 4, 5]
}

def debug_top_k(df, k=3, col_cat="category", col_profit="profit",
                col_cogs="cogs", col_rating="rating", col_id="product id"):

    df = df.copy()
    df["_score"] = compute_greedy_score(df, col_profit, col_cogs, col_rating)

    categories = df[col_cat].unique()

    for cat in categories:
        print("\n==============================")
        print(f"Category: {cat}")
        print("==============================")

        group = df[df[col_cat] == cat].sort_values("_score", ascending=False)

        print("\nALL PRODUCTS (sorted by score):")
        print(group[[col_id, col_profit, col_cogs, col_rating, "_score"]])

        print(f"\nTOP {k} FOR THIS CATEGORY:")
        print(group[[col_id, "_score"]].head(k))



### TESTS
if __name__ == "__main__":
    # B = compute_budget_B(df)
    # Rmin = compute_rmin(df)

    # print(f"Budget B = {B:.2f}")
    # print(f"Minimum rating Rmin = {Rmin:.2f}\n")

    # tests = {
    # "A": [1, 2, 3],
    # "B": [1, 2],
    # "C": [1, 3],
    # "D": [1, 4],
    # "E": [1, 2, 3, 4, 5, 1]
    # }

    # for name, sel in tests.items():
    #     result = is_feasible(sel, df, B, Rmin)
    #     print(f"Selection {name} {sel} -> {'Feasible' if result else 'Not feasible'}")
    # print("")

    df = load_datasets()

    B = compute_budget_B(df)
    Rmin = compute_rmin(df)

    df_filtered = greedy_secure_filter(df, quantile=0.5)
    print("")
    print(df_filtered)

    buckets = top_k_per_category(df_filtered, k=3)
    debug_top_k(df_filtered, k=3)
    print("")
    print(buckets)