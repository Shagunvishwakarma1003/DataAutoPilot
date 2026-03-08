import pandas as pd


def create_leaderboard(results):

    df = pd.DataFrame(results)

    # choose metric automatically
    if "accuracy" in df.columns:
        metric = "accuracy"
    elif "f1 Score" in df.columns:
        metric = "f1 Score"
    elif "r2" in df.columns:
        metric = "r2"
    else:
        metric = df.columns[1]

    df = df.sort_values(by=metric, ascending=False)

    df["rank"] = range(1, len(df) + 1)

    # select only available columns
    cols = ["rank"] + [
        c for c in ["model", "accuracy", "f1 Score", "r2"] if c in df.columns
    ]

    return df[cols]
