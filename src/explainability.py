import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def explain_with_shap(
    best_model, x: "pd.DataFrame", out_dir="output/shap", max_display=10
):
    """
    Works with:
    - Tree models: RandomForest, DecisionTree, XGBoost (TreeExplainer)
    - Linear models: LogisticRegression, LinearSVC (LinearExplainer)
    Returns:
    {
        "summary_png": "...",
        "waterfall_png": "...",
        "positive_reasons": [(feat, val), ...],
        "negative_reasons": [(feat, val), ...],
    }
    """

    os.makedirs(out_dir, exist_ok=True)

    # large dataset
    if len(x) > 5000:
        x = x.sample(2000, random_state=42)

    pre = best_model.named_steps.get("preprocessor")
    model = best_model.named_steps.get("model")

    # --- transform x to model input space ---
    x_t = pre.transform(x) if pre is not None else x

    # Make dense if sparse
    try:
        if hasattr(x_t, "toarray"):
            x_dense = x_t.toarray()
        else:
            x_dense = np.array(x_t)
    except Exception:
        x_dense = np.array(x_t)

    # feature names after preprocessing
    try:
        feat_names = pre.get_feature_names_out()
        feat_names = np.array([str(f).split("__")[-1] for f in feat_names])
    except Exception:
        n_feats = x_dense.shape[1]
        feat_names = np.array([f"f{i}" for i in range(n_feats)])

    model_name = model.__class__.__name__.lower()

    # Decide explainer type
    is_tree = (
        hasattr(model, "feature_importances_")
        or ("xgb" in model_name)
        or ("randomforest" in model_name)
        or ("decisiontree" in model_name)
    )
    is_linear = (
        hasattr(model, "coef_")
        or ("logistic" in model_name)
        or ("linearsvc" in model_name)
        or ("linear" in model_name)
        or ("ridge" in model_name)
        or ("lasso" in model_name)
    )
    # Build explainer
    if is_tree:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(x_dense)
    elif is_linear:
        # use a small background for speed
        bg = x_dense[: min(100, len(x_dense))]
        explainer = shap.LinearExplainer(model, bg)
        sv = explainer.shap_values(x_dense)
    else:
        raise ValueError(
            f"Model not supported by SHAP in this function: {model._class.name_}"
        )

    # Handle binary classification shapes:
    # sv can be list [class0, class1] OR array (n, f)
    if isinstance(sv, list):
        # prefer class 1 explanations if available
        sv_arr = sv[1] if len(sv) > 1 else sv[0]
        base_val = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            and len(explainer.expected_value) > 1
            else explainer.expected_value
        )
    else:
        sv_arr = sv
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[0]

    # ✅ IMPORTANT: always use POSITION index (0..n-1), NOT dataset index like 718
    idx = 0
    if sv_arr.shape[0] == 0:
        raise ValueError("Empty SHAP values.")

    if idx >= sv_arr.shape[0]:
        idx = 0

    # ---------- Summary plot ----------
    summary_png = os.path.join(out_dir, "shap_summary.png")
    plt.figure()
    shap.summary_plot(
        sv_arr,
        x_dense,
        feature_names=list(feat_names),
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    plt.savefig(summary_png, dpi=200, bbox_inches="tight")
    plt.close()

    # ---------- Waterfall plot ----------
    waterfall_png = os.path.join(out_dir, "shap_waterfall.png")

    idx = 0  # position index
    # sv_arr can be list (class-wise), 3D (n,f,k), or 2D (n,f)
    if isinstance(sv_arr, list):
        sv_use = sv_arr[1] if len(sv_arr) > 1 else sv_arr[0]
    elif getattr(sv_arr, "ndim", 0) == 3:
        sv_use = sv_arr[:, :, 1]
    else:
        sv_use = sv_arr

    values_1 = sv_use[idx, :]  # ✅ feature-wise vector
    data_1 = x_dense[idx]

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1] if len(base_val) > 1 else base_val[0]

    exp = shap.Explanation(
        values=values_1,
        base_values=base_val,
        data=data_1,
        feature_names=list(feat_names),
    )

    plt.figure()
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(waterfall_png, dpi=200, bbox_inches="tight")
    plt.close()

    # ---------- Top reasons ----------
    pairs = list(zip(feat_names, values_1))
    pairs_sorted = sorted(pairs, key=lambda z: abs(z[1]), reverse=True)[:max_display]

    positive_reasons = [(str(n), float(v)) for n, v in pairs_sorted if v > 0][
        :max_display
    ]
    negative_reasons = [(str(n), float(v)) for n, v in pairs_sorted if v < 0][
        :max_display
    ]

    return {
        "summary_png": summary_png,
        "waterfall_png": waterfall_png,
        "positive_reasons": positive_reasons,
        "negative_reasons": negative_reasons,
    }
