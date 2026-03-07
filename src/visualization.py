# Visualization utilities for Unsupervised results

import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_k_silhouette(scores, save_path= 'output/k_vs_silhouette.png'):
    # scores: list of dicts like [{'k': 2, 'silhouette': 0.21}, ...]

    if not scores:
        print('No silhouette scores found to plot.')
        return
    
    df = pd.DataFrame(scores).sort_values('k')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure()
    plt.plot(df['k'], df['silhouette'], marker='o')
    plt.title('K vs Silhouette Score')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_pca_clusters(embeddings, labels, save_path='output/pca_clusters.png'):
    # embeddings: array-like of shape (n_samples, 2)
    # labels: array-like of shape (n_samples,)

    if embeddings is None or labels is None:
        print(' PCA embeddings / labels missing, cannot plot.')
        return
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure()
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, s=20)
    plt.title('PCA (2D) Cluster Visualization')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

# -------------------------------

# Supervised Visualization

import numpy as np
import pandas as pd


def save_missing_values_plot(df: pd.DataFrame, out_path="output/eda/missing_values.png", top_n=30):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0].head(top_n)

    if miss.empty:
        return None

    plt.figure()
    plt.bar(miss.index.astype(str), miss.values)
    plt.xticks(rotation=90)
    plt.title("Missing Values (Top)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path

def save_model_leaderboard(results, metric="RMSE", out_path="output/model_leaderboard.png"):

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    models = [r["model"] for r in results]
    scores = [r[metric] for r in results]

    pairs = sorted(zip(scores, models))
    scores_sorted, models_sorted = zip(*pairs)

    plt.figure(figsize=(8,5))
    plt.barh(models_sorted, scores_sorted)

    plt.xlabel(metric)
    plt.title("Model Leaderboard")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def save_correlation_heatmap(df: pd.DataFrame, out_path="output/eda/correlation_heatmap.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None

    corr = num_df.corr(numeric_only=True).values
    cols = num_df.columns.astype(str).tolist()

    plt.figure()
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlation Heatmap (Numeric)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def save_feature_importance(best_model, x: pd.DataFrame, out_path="output/feature_importance.png", top_n=15):

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    model = best_model.named_steps.get("model")
    pre = best_model.named_steps.get("preprocessor")

    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)

    if importances is None:
        return None

    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = np.array([f"f{i}" for i in range(len(importances))])

    idx = np.argsort(importances)[::-1][:top_n]
    top_names = np.array(feat_names)[idx].astype(str)
    top_vals = importances[idx]

    plt.figure()
    plt.barh(top_names[::-1], top_vals[::-1])
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path