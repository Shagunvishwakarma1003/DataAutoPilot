import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def save_permutation_importance(best_model, X_test, y_test, out_path="output/permutation_importance.png", top_n=15):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    r = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="f1"
    )

    importances = r.importances_mean
    try:
        feat_names = best_model.named_steps["preprocessor"].get_feature_names_out()
        feat_names = np.array([f.split("__")[-1] for f in feat_names])
    except Exception:
        feat_names = np.array([f"f{i}" for i in range(len(importances))])

    idx = np.argsort(importances)[::-1][:top_n]
    names = feat_names[idx].astype(str)
    vals = importances[idx]

    plt.figure()
    plt.barh(names[::-1], vals[::-1])
    plt.title("Permutation Importance (Top)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path