# Association Rules modules (Apriori)

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def run_association_rules(df: pd.DataFrame, min_support=0.05, min_confidence=0.5):
    # Excepting 0/1 numeric basket columns
    basket = df.copy()

    # Ensure only 0/1 int
    basket = basket.apply(lambda col: col.fillna(0).astype(int))

    freq_items = apriori(basket, min_support=min_support, use_colnames=True)

    rules = association_rules(
        freq_items, metric="confidence", min_threshold=min_confidence
    )
    rules = rules.sort_values(["confidence", "lift"], ascending=False)

    return {"frequent_itemsets_df": freq_items, "rules_df": rules}
