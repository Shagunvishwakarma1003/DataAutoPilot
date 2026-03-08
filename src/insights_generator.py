import pandas as pd


def generate_insights(df, target):

    rows, cols = df.shape

    insights = []

    insights.append(f"Dataset contains {rows} rows and {cols} columns.")
    insights.append(f"Target variable detected: {target}")

    if df[target].nunique() <= 10:
        insights.append("Problem type: Classification")
    else:
        insights.append("Problem type: Regression")

    insights.append(f"Average value of target: {df[target].mean():.2f}")

    return insights
