from typing import List, Tuple, Optional
import pandas as pd

def compute_budget_B(df: pd.DataFrame, 
                    alpha: float = 0.7, 
                    col_cat: str = "category",
                    col_cogs: str = "cogs"):
    
    # Group by category and take the maximum COGS in each category
    max_per_cat = df.groupby(col_cat, as_index=False)[col_cogs].max()

    # Sum all and multiply by alpha
    return float(max_per_cat[col_cogs].sum() * alpha)

def compute_rmin(df: pd.DataFrame, buffer: float= 0.1, col_rating: str= "rating") -> float :
    #Rmin = dataset average rating + buffer
    return float(df[col_rating].mean() + buffer)

def selection_metrics(selection: List, df: pd.DataFrame, col_id = "product id", col_profit="profit", col_cogs="cogs", col_rating="rating") -> Tuple[float,float,float]:
    #If no items selected, return zeros
    if not selection:
        return (0.0,0.0,0.0)
    
    #Filter dataset to include only the selected items
    sub = df[df[col_id].isin(selection)]

    #Compute the three key metrics
    return (float(sub[col_profit].sum()),
            float(sub[col_cogs].sum()),
            float(sub[col_rating].mean()))

def is_feasible(selection: List, df: pd.DataFrame,
                B: Optional[float] = None, Rmin: Optional[float] = None,
                max_items: int = 6, one_per_category: bool = True,
                col_id="product id", col_cat="category", col_cogs="cogs", col_rating="rating",
                alpha_budget: float = 0.7, rating_buffer: float = 0.1,
                eps: float = 1e-9) -> bool:
    #1. Selection must not be empty
    if not selection:
        return False
    
    #2. Limit number of selected products
    if len(selection) > max_items:
        return False
    
    sub = df[df[col_id].isin(selection)]

    #3. One product per category (no duplicates)
    if one_per_category and sub[col_cat].duplicated().any():
        return False
    
    #4. Budget constraint
    B = compute_budget_B(df, alpha_budget, col_cat, col_cogs) if B is None else B
    if float(sub[col_cogs].sum()) > B + eps:
        return False
    
    #5. Minimum average rating
    Rmin = compute_rmin(df, rating_buffer, col_rating) if Rmin is None else Rmin
    if float(sub[col_rating].mean()) + eps < Rmin:
        return False
    
    # If all constraints satisfied
    return True



