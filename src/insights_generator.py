import pandas as pd

def generate_insights(df, target):
    insights = []

    # Target column insights
    if target in df.columns:

        # Numeric target → calculate average
        if pd.api.types.is_numeric_dtype(df[target]):
            insights.append(
                f"Average value of target: {df[target].mean():.2f}"
            )

        # Categorical target → show class distribution
        else:
            value_counts = df[target].value_counts()

            insights.append(
                f"Target contains {df[target].nunique()} unique categories."
            )

            insights.append(
                f"Most common target category: {value_counts.index[0]}"
            )

            insights.append(
                f"Most common category count: {value_counts.iloc[0]}"
            )

    # Dataset information
    insights.append(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    # Missing values
    missing = df.isnull().sum().sum()

    if missing > 0:
        insights.append(
            f"Dataset contains {missing} missing values."
        )
    else:
        insights.append(
            "Dataset has no missing values."
        )

    return insights