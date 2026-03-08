import pandas as pd
from scipy.stats import ks_2samp


def detect_drift(train_df, new_df):

    drift_report = {}

    for col in train_df.columns:
        if train_df[col].dtype != "object":

            stat, p_value = ks_2samp(train_df[col], new_df[col])

            drift_report[col] = {"p_value": p_value, "drift_detected": p_value < 0.05}

    return drift_report
