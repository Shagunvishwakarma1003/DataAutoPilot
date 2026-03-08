import matplotlib.pyplot as plt
import seaborn as sns


def create_dashboard(df, target):

    # Target distribution
    if target in df.columns:
        plt.figure()
        df[target].value_counts().plot(kind="bar")
        plt.title("Target Distribution")
        plt.savefig("output/target_distribution.png")
        plt.close()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("output/dashboard_correlation.png")
    plt.close()
