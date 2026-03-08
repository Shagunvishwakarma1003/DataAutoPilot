from typing import Any, Dict, Tuple
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score


def tune_pipeline_top2(
    task_type: str,
    model_name: str,
    pipe,
    x_train,
    y_train,
    cv: int = 5,
    random_state: int = 42,
    n_iter: int = 25,
    n_jobs: int = -1,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Tune ONE sklearn Pipeline (preprocessor + model).
    You call it only for TOP-2 models.
    Returns: best_pipe, info_dict
    """
    task_type = task_type.lower().strip()
    model = pipe.named_steps["model"]
    m = model.__class__.__name__.lower()

    # --- Param grids (light, strong) ---
    param_dist = None

    if "randomforest" in m:
        param_dist = {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 4, 6, 10, 14],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        }

    elif "decisiontree" in m:
        param_dist = {
            "model__max_depth": [None, 3, 5, 8, 12],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
        }

    elif "xgb" in m or "xgboost" in m:
        param_dist = {
            "model__n_estimators": [300, 600, 900],
            "model__max_depth": [2, 3, 4, 5, 6],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
        }

    elif "ridge" in m or "lasso" in m:
        param_dist = {"model__alpha": np.logspace(-4, 2, 30).tolist()}

    elif "svr" in m:
        param_dist = {
            "model__C": [0.5, 1, 5, 10, 20],
            "model__epsilon": [0.01, 0.05, 0.1, 0.2],
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["rbf", "poly"],
        }

    elif "kneighbors" in m:
        param_dist = {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }

    else:
        return pipe, {
            "model": model_name,
            "tuned": False,
            "reason": f"No tuning grid for {m}",
        }

    # --- scoring ---
    if task_type == "classification":
        scoring = make_scorer(f1_score)  # binary F1
    else:
        scoring = "neg_root_mean_squared_error"

    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )
    rs.fit(x_train, y_train)

    info = {
        "model": model_name,
        "tuned": True,
        "best_score": float(rs.best_score_),
        "best_params": rs.best_params_,
    }
    return rs.best_estimator_, info
