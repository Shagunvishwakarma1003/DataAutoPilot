# Supervised training module (Classification/Regression)

import pandas as pd
import numpy as np
import shap
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def should_use_class_weight(y, threshold=0.20):
    """
    If minority class ratio < threshold => imbalance => use class_weight
    """
    classes, counts = np.unique(y, return_counts=True)
    ratios = counts / counts.sum()
    minority_ratio = ratios.min()
    return (minority_ratio < threshold), float(minority_ratio)

from tuner import tune_pipeline_top2
from report_generator import generate_html_report
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from visualization import save_feature_importance, save_model_leaderboard
from explainability import explain_with_shap
from importance import save_permutation_importance

def build_preprocessor(X: pd.DataFrame):
    # numeric + categorical preprocessing
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    pre = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols)
        ],
        remainder='drop'
    )
    return pre

def train_regression(df: pd.DataFrame, target: str, random_state=42, cv=None):
    # Split, train multiple regression models, pick best (lowest RMSE

    x = df.drop(columns=[target])
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
    pre = build_preprocessor(x)
    models = [
        ('LinearRegression', LinearRegression()),
        ('Ridge', Ridge(alpha=0.1, max_iter=5000)),
        ('Lasso', Lasso(alpha=0.001)),
        ('RandomForestRegressor', RandomForestRegressor(n_estimators=300, random_state=random_state)),
        ('XGBRegressor', XGBRegressor()),
        ('SVR', SVR()),
        ('KNNRegressor', KNeighborsRegressor(n_neighbors=7)),
    ]

    results = []
    best_model = None
    best_rmse = float('inf')

    for name, model in models:
        pipe = Pipeline(steps=[('preprocessor', pre), ('model', model)])
        pipe.fit(x_train, y_train)
        preds = pipe.predict(x_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        # Cross Validation;
        if cv and cv >=3:
            kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            cv_rmse = -cross_val_score(pipe, x, y, cv=kf,
                                       scoring='neg_root_mean_squared_error').mean()
        else:
            cv_rmse = None

        results.append({
            'model': name,
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'CV_RMSE': float(cv_rmse) if cv_rmse else None
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipe

    # Leaderboard Graph
    print("Model Comparison (Regression):")

    for r in results:
        print(r)

    leaderboard_path = save_model_leaderboard(results, metric="RMSE")
    print("📊 Leaderboard graph saved:", leaderboard_path)
    
    preds =  best_model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Report PDF/HTML
    from report_generator import generate_html_report
    best_model_name = best_model.named_steps["model"].__class__.__name__
    best_metric = {
        "rmse": float(rmse),
        "mae":float(mae),
        "r2": float(r2)
    }

    artifacts = {
        "eda_report_txt": "output/eda/eda_report.txt",
        "corr_heatmap_png": "output/eda/correlation_heatmap.png",
        "feature_importance_png": "output/feature_importance.png",
        "permutation_importance_png": "output/permutation_importance.png",
        "shap_summary_png": "output/shap/shap_summary.png",
        "shap_waterfall_png": "output/shap/shap_waterfall.png"
    }

    report_path = generate_html_report(
        output_dir="output",
        task_type="regression",
        dataset_path="",
        target=str(target),
        data_shape=df.shape,
        best_model_name=best_model_name,
        best_metric=best_metric,
        all_results=results,
        artifacts=artifacts
    )

    print("📄 Report saved:", report_path)
    
    # SHAP Explainability (Regression)
    try:
        x_small = x_test.head(50)
        shap_out = explain_with_shap(best_model, x_small, out_dir="output/shap", max_display=10)
        print("✅ SHAP saved:")
        print(" -", shap_out["summary_png"])
        print(" -", shap_out["waterfall_png"])
        
        print("\n🔎 Why predicted value (Top reasons):")
        for n, v in shap_out["positive_reasons"]:
            print(f" + {n} ({v:.4f})")
        for n, v in shap_out["negative_reasons"]:
            print(f" - {n} ({v:.4f})")
            
    except Exception as e:
        print("ℹ️ SHAP not available:", str(e))

    return best_model, results

def train_classification(df: pd.DataFrame, target: str, random_state=42, cv=None):
    # Split, train multiple classification models, pick best (highest F1)

    x = df.drop(columns=[target])
    y = df[target].astype('category').cat.codes  # Convert to numeric codes for classification

    use_cw, min_ratio = should_use_class_weight(y)
    
    if use_cw:
        print(f"⚠️ Imbalance detected (minority={min_ratio:.2f}) → enabling class_weight='balanced'")
    else:
        print(f"✅ Balanced data (minority={min_ratio:.2f})")
    
    # ✅ Class Imbalance Detector
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    ratios = counts / total
    
    minority_idx = np.argmin(counts)
    minority_class = classes[minority_idx]
    minority_ratio = ratios[minority_idx]

    # simple rule: minority < 20% OR max/min ratio >= 3
    imbalance_ratio = counts.max() / counts.min()
    
    print("\n📊 Class Distribution:")
    
    for c, cnt, r in zip(classes, counts, ratios):
        print(f"  Class {c}: {cnt} samples ({r*100:.2f}%)")

    if (minority_ratio < 0.20) or (imbalance_ratio >= 3):
        print("\n⚠️ Class Imbalance detected!")
        print(f"  Minority class: {minority_class} ({minority_ratio*100:.2f}%)")
        print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}")
        print("✅ Suggestions:")
        print("  - Use class_weight='balanced' (LogReg / SVM / RF / DT)")
        print("  - Prefer F1 / PR-AUC over Accuracy")
        print("  - Use StratifiedKFold (already good)")
    else:
        print("\n✅ Classes look balanced enough.")

    # XGBoost imbalance handling
    if use_cw and len(np.unique(y)) == 2:
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        spw = neg / max(pos, 1)
    else:
        spw = None

    if use_cw and len(np.unique(y)) == 2:
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        spw = neg / max(pos, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state, stratify=y)
    pre = build_preprocessor(x)
    
    #XGBosst parameters;
    xgb_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        eval_metric='logloss',
        verbosity=0
    )
    if spw is not None:
        xgb_params["scale_pos_weight"] = spw
    
    models = [
        ('LogisticRegression', LogisticRegression(max_iter=2000,
                                                  class_weight="balanced" if use_cw else None)),
        ('RandomForestClassifier', RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced" if use_cw else None)),
        ('DecisionTree', DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced" if use_cw else None)),
        ('SVM', SVC(probability=True,
                    class_weight="balanced" if use_cw else None)),
        ('XGBClassifier', XGBClassifier(**xgb_params)),
        ('KNNClassifier', KNeighborsClassifier(n_neighbors=7)),
    ]

    results = []
    best_model = None
    best_f1 = -1
    
    n_classes = len(np.unique(y))
    avg_type ='binary' if n_classes == 2 else 'weighted'
    print(f'\nDetected {n_classes} classes -> using {avg_type} average')

    for name, model in models:
        pipe = Pipeline(steps=[('preprocessor', pre), ('model', model)])
        pipe.fit(x_train, y_train)
        preds = pipe.predict(x_test)
        acc = accuracy_score(y_test, preds)        
        prec = precision_score(y_test, preds, average=avg_type, zero_division=0)
        rec = recall_score(y_test, preds, average=avg_type, zero_division=0)
        f1 = f1_score(y_test, preds, average=avg_type, zero_division=0)
        cm = confusion_matrix(y_test, preds).tolist()

        if cv and cv >= 3:
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            cv_f1 = cross_val_score(pipe, x, y, cv=skf, scoring='f1_weighted').mean()
        else:
            cv_f1 = None

        results.append({
            'model': name,
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1 Score': float(f1),
            'cv_f1': float(cv_f1) if cv_f1 else None,
            'confusion matrix': cm
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = pipe

    # Leaderboard graph (Classification)
    print("\n Model Comparison (Classification):")
    for r in results:
        print(r)

    leaderboard_path = save_model_leaderboard(results, metric="f1 Score")
    print("=📊 Leaderboard graph saved:", leaderboard_path)
    
    # --- AFTER models loop ends (best_model ready) ---
    def get_metric(r, *keys, default=None):
        """Hinglish: results dict me multiple possible keys check karta hai."""

        for k in keys:
            if k in r and r[k] is not None:
                return r[k]
        return default
    # ✅ Leaderboard sort by CV_F1 (fallback to F1)
    results_sorted = sorted(
        results,
        key=lambda r: (
            get_metric(r, "cv_f1", "cv_f1_score", default=-1)
            if get_metric(r, "cv_f1", "cv_f1_score", default=None) is not None
            else get_metric(r, "f1_score", "f1 Score", "f1", default=-1)
        ),
        reverse=True
    )
    
    print("\n🏆 Leaderboard (sorted by CV F1 → fallback F1):")
    for i, r in enumerate(results_sorted, 1):
        acc = get_metric(r, "accuracy", "acc", default=None)
        f1  = get_metric(r, "f1_score", "f1 Score", "f1", default=None)
        cvf = get_metric(r, "cv_f1", "cv_f1_score", default=None)
        print(f"{i:02d}. {r.get('model',''): <20} | CV_F1={cvf} | F1={f1} | Acc={acc}")

    # ✅ Tune TOP-2 models (optional but AutoML vibe)
    from tuner import tune_pipeline_top2
    
    top2 = [results_sorted[0]["model"]] if len(results_sorted) == 1 else [results_sorted[0]["model"], results_sorted[1]["model"]]
    
    name_to_pipe = {}
    for name, model in models:
        name_to_pipe[name] = Pipeline(steps=[("preprocessor", pre), ("model", model)])
    
    tune_logs = []
    best_after_tune = best_model
    best_after_tune_cv = get_metric(results_sorted[0], "cv_f1", "cv_f1_score", default=None)
    
    for nm in top2:
        try:
            tuned_pipe, info = tune_pipeline_top2(
                task_type="classification",
                model_name=nm,
                pipe=name_to_pipe[nm],
                x_train=x_train,
                y_train=y_train,
                cv=(cv if cv else 5),
                random_state=random_state,
                n_iter=25
            )
            
            # tuned evaluate on test (quick)
            
            tuned_pipe.fit(x_train, y_train)
            preds_t = tuned_pipe.predict(x_test)
            
            # compute F1 (binary/weighted already handled in your earlier logic ideally)
            n_classes = len(np.unique(y_train))
            avg_type = "binary" if n_classes == 2 else "weighted"
            
            f1_t = float(f1_score(y_test, preds_t, average=avg_type, zero_division=0))
            acc_t = float(accuracy_score(y_test, preds_t))

            info["test_f1"] = f1_t
            info["test_acc"] = acc_t
            tune_logs.append(info)
            
            # Compare using test_f1 (or you can compare by CV score if you want)
            # Here: prefer higher test_f1
            current_best_test_f1 = float(get_metric({"x": None}, "x", default=-1))  # dummy
            if best_after_tune is best_model:
                # set baseline from best_model on test
                preds_base = best_model.predict(x_test)
                base_f1 = float(f1_score(y_test, preds_base, average=avg_type, zero_division=0))
                current_best_test_f1 = base_f1
            else:
                current_best_test_f1 = max([t.get("test_f1", -1) for t in tune_logs if t.get("tuned")], default=-1)

            if f1_t > current_best_test_f1:
                best_after_tune = tuned_pipe
                
        except Exception as e:
            tune_logs.append({"model": nm, "tuned": False, "error": str(e)})

    print("\n🛠 Tuning Summary (Top-2):")
    for t in tune_logs:
        print(t)

    # ✅ Replace best_model if tuned won (best_after_tune changed)
    best_model = best_after_tune

    # ✅ SHAP ONLY ONCE (best model only)
    try:
        x_small = x_test.head(50).reset_index(drop=True)
        shap_out = explain_with_shap(best_model, x_small, out_dir="output/shap", max_display=10)

        print("✅ SHAP saved:")
        print(" -", shap_out["summary_png"])
        print(" -", shap_out["waterfall_png"])

        print("\n🔎 Why predicted (Top reasons):")
        for n, v in shap_out["positive_reasons"][:5]:
            print(f" + {n} ({v:.4f})")
        for n, v in shap_out["negative_reasons"][:5]:
            print(f" - {n} ({v:.4f})")
            
    except Exception as e:
        print("ℹ️ SHAP not available:", str(e))

    # PDF/HTML Report
    from report_generator import generate_html_report
    best_model_name = best_model.named_steps["model"].__class__.__name__

    best_metric = {
        "cv_f1": get_metric(results_sorted[0], "cv_f1", "cv_f1_score", default=None),
        "f1": get_metric(results_sorted[0], "f1_score", "f1 Score", "f1", default=None),
        "accuracy": get_metric(results_sorted[0], "accuracy", "acc", default=None),
        "tuning_top2": tune_logs
    }

    artifacts = {
        "eda_report_txt": "output/eda/eda_report.txt",
        "corr_heatmap_png": "output/eda/correlation_heatmap.png",
        "feature_importance_png": "output/feature_importance.png",
        "permutation_importance_png": "output/permutation_importance.png",
        "shap_summary_png": "output/shap/shap_summary.png",
        "shap_waterfall_png": "output/shap/shap_waterfall.png"
    }

    report_path = generate_html_report(
         output_dir="output",
        task_type="classification",
        dataset_path="",
        target=str(target),
        data_shape=df.shape,
        best_model_name=best_model_name,
        best_metric=best_metric,
        all_results=results_sorted,
        artifacts=artifacts
    )

    print("📄 Report saved:", report_path)

    # Best metric for report: take top of leaderboard (after sorting)
    best_metric = {
        "cv_f1": get_metric(results_sorted[0], "cv_f1", "cv_f1_score", default=None),
        "f1": get_metric(results_sorted[0], "f1_score", "f1 Score", "f1", default=None),
        "accuracy": get_metric(results_sorted[0], "accuracy", "acc", default=None),
        "tuning_top2": tune_logs
    }

    artifacts = {
        "eda_report_txt": "output/eda/eda_report.txt",
        "corr_heatmap_png": "output/eda/correlation_heatmap.png",
        "feature_importance_png": "output/feature_importance.png",
        "permutation_importance_png": "output/permutation_importance.png",
        "shap_summary_png": "output/shap/shap_summary.png",
        "shap_waterfall_png": "output/shap/shap_waterfall.png",
    }

    report_path = generate_html_report(
        output_dir="output",
        task_type="classification",
        dataset_path="",
        target=str(target),
        data_shape=df.shape if "df" in locals() else (None, None),
        best_model_name=best_model_name,
        best_metric=best_metric,
        all_results=results_sorted,
        artifacts=artifacts,
        notes={"note": "Leaderboard + top2 tuning + SHAP (best model only)"},
    )

    print("📄 Report saved:", report_path)

    return best_model, results_sorted