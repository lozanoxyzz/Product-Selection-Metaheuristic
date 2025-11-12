import pandas as pd
from constraints import compute_budget_B,compute_rmin, is_feasible
# === Mini dataset ===
data = {
    "category": ["Drinks", "Snacks", "Fruits", "Drinks", "Snacks"],
    "cogs": [5.0, 4.0, 3.8, 4.5, 3.0],
    "profit": [8.0, 6.5, 5.2, 7.0, 5.8],
    "rating": [4.7, 4.4, 4.6, 4.1, 4.3],
    "product id": [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)


### TESTS
if __name__ == "__main__":
    B = compute_budget_B(df)
    Rmin = compute_rmin(df)

    print(f"Budget B = {B:.2f}")
    print(f"Minimum rating Rmin = {Rmin:.2f}\n")

    tests = {
    "A": [1, 2, 3],
    "B": [1, 2],
    "C": [1, 3],
    "D": [1, 4],
    "E": [1, 2, 3, 4, 5, 1]
    }

    for name, sel in tests.items():
        result = is_feasible(sel, df, B, Rmin)
        print(f"Selection {name} {sel} -> {'Feasible' if result else 'Not feasible'}")
    print("")