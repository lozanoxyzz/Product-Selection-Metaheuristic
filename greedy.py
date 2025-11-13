import pandas as pd
from typing import Dict, List

def compute_greedy_score(df: pd.DataFrame,
                         col_profit="profit",
                         col_cogs="cogs",
                         col_rating="rating") -> pd.Series:
    
    ### score = (profit / cogs) * rating.

    safe_cogs = df[col_cogs].replace(0, 1e-9)
    return (df[col_profit] / safe_cogs) * df[col_rating]


def greedy_secure_filter(df: pd.DataFrame,
                         quantile: float = 0.5,
                         col_cat="category",
                         col_profit="profit",
                         col_cogs="cogs",
                         col_rating="rating") -> pd.DataFrame:
   
    df = df.copy()
    df["_score"] = compute_greedy_score(df, col_profit, col_cogs, col_rating)

    # Global threshold (percentile)
    threshold = df["_score"].quantile(quantile)
    filtered = df[df["_score"] >= threshold]

    # Ensure each category survives
    categories = df[col_cat].unique()
    for cat in categories:
        if cat not in filtered[col_cat].values:
            # Rescue best product of this category
            best_cat_item = df[df[col_cat] == cat].sort_values("_score", ascending=False).head(1)
            filtered = pd.concat([filtered, best_cat_item], ignore_index=True)

    # Remove helper column
    return filtered.drop(columns=["_score"])


def top_k_per_category(df: pd.DataFrame,
                       k: int = 3,
                       col_cat="category",
                       col_profit="profit",
                       col_cogs="cogs",
                       col_rating="rating",
                       col_id="product id") -> Dict[str, List]:
    
    df = df.copy()
    df["_score"] = compute_greedy_score(df, col_profit, col_cogs, col_rating)

    df_sorted = df.sort_values("_score", ascending=False)

    buckets = {}
    for cat, group in df_sorted.groupby(col_cat):
        top_k = group.head(k)[col_id].tolist()
        buckets[str(cat)] = top_k

    return buckets
