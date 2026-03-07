# Detector module -> decides supervised type

import pandas as pd

def detect_supervised_type(y: pd.Series) -> str:
    # If categorical/object/bool -> Classification
    if y.dtype == 'object' or str(y.dtype).startswith('category') or y.dtype == 'bool':
        return 'classification'
    # If numeric :
    y_non_null = y.dropna()
    n_unique = y_non_null.nunique()
    unique_ratio = n_unique / max(len(y_non_null), 1)

    # If few uniques -> Classification
    if n_unique <= 10 or unique_ratio <= 0.05:
        return 'classification'
    
    # Otherwise -> Regression
    return 'regression'

# Unsupervised Detection Section

import numpy as np

def is_transaction_like(df: pd.DataFrame) -> bool:
    # mostly numeric  (0/1) item cols are present
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return False
    
    bin_cols = 0
    for c in num.columns:
        vals = set(num[c].dropna().unique())
        if vals.issubset({0, 1}) and len(vals) > 0:
            bin_cols += 1

    return bin_cols >= max(10, int(0.6 * num.shape[1]))

def detect_unsupervised_type(df: pd.DataFrame) -> str:
    if is_transaction_like(df):
        return 'association_rules'
    
    if df.shape[1] >= 30:
        return 'dim_reduction_plus_clustering'
    
    return 'clustering'  # default to clustering for now


# Deep Learing Suggestion Logic

def suggest_deep_learning(df: pd.DataFrame) -> dict:
    """ Heuristic suggestions for deep learning usage"""

    colnames = ' ' .join(df.columns).lower()

    text_hint = any(k in colnames for k in ['text', 'review', 'comment', 'message', 'sentence'])
    image_hint = any(k in colnames for k in ['image', 'img', 'filepath', 'path', 'file'])

    big_data = (df.shape[0] > 50000) or (df.shape[1] >= 2000)

    if image_hint:
        return {'recommend': True, 'type': 'image', 'reason': 'Columns suggest image/file paths'}
    
    if text_hint:
        return {'recommend': True, 'type': 'text', 'reason': 'Columns suggest text data'}
    
    if big_data:
        return {'recommend': True, 'type': 'tabular_deep', 'reason': 'Very large dataset/features'}
    
    return {'recommend': False, 'type': None, 'reason': 'Machine Learning is sufficient'}