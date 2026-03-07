import os
import joblib
import pandas as pd
import re
import pandas as pd
import numpy as np
from difflib import get_close_matches


def save_artifact(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_artifact(path: str):
    return joblib.load(path)

def save_model(model, path):
    save_artifact(model, path)

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+','', str(s).lower())

def resolve_target_column(df: pd.DataFrame, target_name: str):
    '''Case-insensitive + underscore/space ignore + fuzzy match'''
    cols = list(df.columns)
    norm_map = {_norm(c): c for c in cols}
    t =  _norm(target_name)

    # Exact normalized
    if t in norm_map:
        return norm_map[t], 'normalized_exact'
    
    # Fuzzy
    close = get_close_matches(t, list(norm_map.keys()), n=1, cutoff=0.82)
    if close:
        return norm_map[close[0]], 'fuzzy_match'
    
    return None, 'not_found'

def validate_dataframe(df: pd.DataFrame):
    if df is None or df.empty:
        return False, 'CSV is empty or could not be read'
    if df.shape[1] < 2:
        return False, 'CSV must have at least 2 columns.'
    if df.shape[0] < 20:
        return False, 'Too few rows (<20). Model training may be unreliable.'
    return True, 'ok'

def validate_target(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        return False, f'Target column "{target_col}" not found.'
    if df[target_col].dropna().empty:
        return False, f'Target column "{target_col}" has no values.'
    return True, 'ok'

def detect_target_column(df: pd.DataFrame):
    """
    Heuristics:
    1) If common target-like names exist -> pick them
    2) Else try last column (very common)
    3) Else score columns: categorical low-unique => classification,
       numeric high-variance => regression
    """
    cols = list(df.columns)
    if len(cols) < 2:
        return None, "too_few_columns"

    # 1) Name-based priority
    priority_keywords = [
    # Most common ML targets first
    "survived", "churn", "default", "fraud", "target", "label", "outcome", "status",
    # Regression type
    "price", "salary", "charges", "sales", "profit",
    # Keep generic words last
    "class", "output","y"
]
    norm_cols = { _norm(c): c for c in cols }

    for kw in priority_keywords:
        for nc, original in norm_cols.items():
            if kw in nc:
                return original, f"keyword_match:{kw}"

    # helper to classify column type
    def col_kind(s: pd.Series):
        s2 = s.dropna()
        if s2.empty:
            return "empty"
        # treat numeric
        if pd.api.types.is_numeric_dtype(s2):
            uniq = s2.nunique()
            if uniq <= 20:
                return "discrete_numeric"
            return "continuous_numeric"
        # categorical/text
        uniq = s2.nunique()
        if uniq <= 2:
            return "binary"
        if uniq <= 20:
            return "multiclass"
        return "text_like"

    # 2) Try last column heuristic
    last = cols[-1]
    k = col_kind(df[last])
    if k in ["binary", "multiclass"]:
        return last, "last_column_heuristic"

    # 3) Score all columns
    best = None
    best_score = -1
    best_reason = None

    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue

        kind = col_kind(df[c])
        uniq = s.nunique()

        score = 0
        reason = []

        # classification-friendly
        if kind in ["binary"]:
            score += 90; reason.append("binary_target_like")
        elif kind in ["multiclass"]:
            score += 75; reason.append("multiclass_target_like")
        elif kind in ["discrete_numeric"] and uniq <= 20:
            score += 60; reason.append("discrete_numeric_target_like")

        # regression-friendly
        if kind == "continuous_numeric":
            score += 70; reason.append("continuous_numeric_target_like")

        # penalize text-like huge unique
        if kind == "text_like":
            score -= 50; reason.append("text_like_penalty")

        # avoid obvious ID columns
        nc = _norm(c)
        if "id" in nc or "uuid" in nc:
            score -= 60; reason.append("id_penalty")

        if score > best_score:
            best_score = score
            best = c
            best_reason = ",".join(reason) if reason else "scored_best"

    if best_score <= 0:
        return None, "no_confident_target"
    return best, best_reason