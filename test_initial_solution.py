from data_loading import load_datasets
from constraints import compute_budget_B, compute_rmin, is_feasible, selection_metrics
from greedy import greedy_secure_filter, top_k_per_category
from tabu import generate_initial_solution, generate_neighbors

print("\n=== STEP 1: Load dataset ===")
df = load_datasets()

print(f"Dataset size")

# B and Rmin
B = compute_budget_B(df)
Rmin = compute_rmin(df)

print(f"Budget B = {B:.2f}")
print(f"Rmin    = {Rmin:.2f}")

# Greedy filter
df_filtered = greedy_secure_filter(df, quantile=1)

# Buckets
buckets = top_k_per_category(
    df_filtered,
    k=3,
    col_cat="category",
    col_profit="profit",
    col_cogs="cogs",
    col_rating="rating",
    col_id="product id"
)

print("\n=== Buckets ===")
for cat, ids in buckets.items():
    print(f"{cat}: {ids}")

# Generate initial solution
print("\n=== Generating initial solution ===")
initial = generate_initial_solution(buckets, df, B, Rmin)

print("Initial solution:", initial)

# Check feasibility
if initial is None:
    print("No feasible solution found!")
else:
    feasible = is_feasible(initial, df, B, Rmin)
    print("Feasible?:", feasible)

    profit, cogs, rating = selection_metrics(initial, df)
    print(f"Profit = {profit:.2f}")
    print(f"COGS   = {cogs:.2f}   (<= B={B:.2f})")
    print(f"Rating = {rating:.3f} (>= Rmin={Rmin:.3f})")

neighbors = generate_neighbors(initial, buckets)
for n in neighbors:
    print(n)