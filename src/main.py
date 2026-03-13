import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import os
import argparse
from drift_detector import detect_drift
from detector import (
    detect_supervised_type,
    detect_unsupervised_type,
    suggest_deep_learning,
)
from unsupervised import (
    run_clustering,
    run_pca,
    generate_cluster_report,
    interpret_clusters,
    interpret_clusters_smart,
    auto_name_cluster,
    generate_business_recommendations,
)
from association import run_association_rules
from supervised import train_regression, train_classification
from visualization import (
    plot_k_silhouette,
    plot_pca_clusters,
    save_missing_values_plot,
    save_correlation_heatmap,
)
from utils import (
    validate_dataframe,
    validate_target,
    resolve_target_column,
    save_artifact,
    save_model,
    load_artifact,
    detect_target_column,
)
from eda import make_eda_report
from leaderboard import create_leaderboard
from predictor import predict_to_csv
from insights_generator import generate_insights
from dashboard import create_dashboard

# create folder automatically
os.makedirs("output, exist_ok=True")
os.makedirs("output/eda", exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to CSV file")
    parser.add_argument("--target", required=False, help="Target column (optional)")
    parser.add_argument(
        "--use_deep", action="store_true", help="Force deep learning mode (future)"
    )
    parser.add_argument(
        "--predict", type=str, default=None, help="CSV path for prediction"
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Load saved .pkl model (used with --predict only)",
    )
    parser.add_argument(
        "--predict-on-train", action="store_true", help="Predict on training dataset"
    )
    parser.add_argument(
        "--cv", type=int, default=0, help="Number of CV folds (e.g., 5). 0 = no CV"
    )

    args = parser.parse_args()
    print("\nDataAutoPilot Started...")
    print("Running automated data science pipeline...")

    # Load data
    df = pd.read_csv(args.data)
    if len(df) > 10000:
        df = df.sample(10000, random_state=42)
    ok, msg = validate_dataframe(df)
    if not ok:
        print(f"❎ {msg}")
        print("=👉 Tip: please provide a valid CSV.")
        raise SystemExit(1)
    print("\nDataset Loaded Successfully!")
    print("Shape:", df.shape)

    # Auto Target Detection;
    if args.target is None:
        auto_target, why = detect_target_column(df)
        if auto_target is None:
            print("❎ Could not auto-detect target column.")
            print("👉 Please pass --target <column_name>")
            print("✅ Available columns:", list(df.columns))
            raise SystemExit(1)
        args.target = auto_target
        print(f' Auto-detected target:" {args.target}" ({why})')

    # If user provided target, solve it safely;

    if args.target:
        resolved, reason = resolve_target_column(df, args.target)

        if resolved is None:
            print(f' ❎ Target "{args.target}" not found')
            print("✅Available columns:", list(df.columns))
            raise SystemExit(1)

        if resolved != args.target:
            print(f'ℹ Using target "{resolved}" ({reason})')

        # Replace with actual column name;
        args.target = resolved

        ok, msg = validate_target(df, args.target)
        if not ok:
            print(f"❎ {msg}")
            raise SystemExit(1)

        # AUTO PROBLEM DETECTION
        y = df[args.target]
        task = detect_supervised_type(y)
        print(f"\nDetected Problem Type: {task.upper()} ")

        # AUTO EDA + VISUALIZATION:
        # ✅ AUTO EDA (always)
        report_path = make_eda_report(df, target=args.target, out_dir="output/eda")
        print(f"✅ EDA report saved: {report_path}")

        # DATASET INSIGHTS
        insights = generate_insights(df, args.target)

        print("\n DATASET INSIGHTS")
        for i in insights:
            print("-", i)

        with open("output/insights.txt", "w") as f:
            for i in insights:
                f.write(i + "\n")

        # AUTO DASHBOARD
        create_dashboard(df, args.target)
        print("Dashboard charts saved in output")

        # ✅ AUTO VISUALS (EDA visuals)
        mv = save_missing_values_plot(df, out_path="output/eda/missing_values.png")
        ch = save_correlation_heatmap(df, out_path="output/eda/correlation_heatmap.png")

        if mv:
            print(f"✅ Missing plot saved: {mv}")
        if ch:
            print(f"✅ Correlation heatmap saved: {ch}")

        # Case-1: Only predict (no training);
        if args.load_model and args.predict:
            artifact = load_artifact(args.load_model)
            model = artifact["model"] if isinstance(artifact, dict) else artifact

            pred_df = pd.read_csv(args.predict)
            predict_to_csv(
                model, pred_df, target=None, save_path="output/predictions.csv"
            )
            print("✅ Predictions saved: output/predictions.csv")
            raise SystemExit(0)

    # Deep Learning
    dl_info = suggest_deep_learning(df)
    print("\nDeep Learning Suggestions:", dl_info)

    if args.use_deep:
        print("\nDeep Learning mode selected (phase-2 implementation required).")

    # If target provided -> Supervised
    if args.target:
        task = detect_supervised_type(df[args.target])
        if task is None:
            print("task None,check detector.py ")
            return
        print(f"\nDetected: Supervised -> {task.upper()}")

        if task == "regression":
            best_model, results = train_regression(df, args.target, cv=5)

            print("\n Model Comparison (Regression):")
            for r in results:
                print(r)

            # Model Leaderboard
            leaderboard = create_leaderboard(results)

            print("\n🏆 MODEL LEADERBOARD")
            print(leaderboard)

            best_model_name = leaderboard.iloc[0]["model"]

            print("\n🏆 BEST MODEL RECOMMENDATION")
            print(f"Best Model: {best_model_name}")
            print("Reason: Highest performance on validation data")

            # save model with metadata;
            artifact = {"model": best_model, "target": args.target, "task": task}
            save_artifact(artifact, f"output/best_{task}_model.pkl")
            print(f" ✅Saved: output/best_{task}_model.pkl")

        elif task == "classification":
            best_model, results = train_classification(df, target=args.target, cv=5)

            print("\n Model Comparison (Classification):")
            for r in results:
                print(r)

            # Model Leaderboard
            leaderboard = create_leaderboard(results)

            print("\n🏆 MODEL LEADERBOARD")
            print(leaderboard)

            best_model_name = leaderboard.iloc[0]["model"]

            print("\n🏆 BEST MODEL RECOMMENDATION")
            print(f"Best Model: {best_model_name}")
            print("Reason: Highest performance on validation data")

            # save model with metadata;
            artifact = {"model": best_model, "target": args.target, "task": task}
            save_artifact(artifact, f"output/best_{task}_model.pkl")
            print(f"✅ Saved: output/best_{task}_model.pkl")

            # after best_model ready (in regression/classification);

            # Predict on training data (optional);
            if args.predict_on_train:
                predict_to_csv(
                    best_model,
                    df,
                    target=args.target,
                    save_path="output/predictions_train.csv",
                )
                print("✅ Train predictions saved: output/predictions_train.csv")

            # Predict on new CSV (optional);
            if args.predict:
                new_df = pd.read_csv(args.predict)
                predict_to_csv(
                    best_model,
                    new_df,
                    target=None,
                    save_path="output/predictions_new.csv",
                )
                print(" ✅New CSV predictions saved: output/predictions_new.csv")

    else:

        # Unsupervised route
        unsup_task = detect_unsupervised_type(df)
        print(f"\nDetected: Unsupervised -> {unsup_task.upper()}")

        if unsup_task == "association_rules":
            ar = run_association_rules(df)
            ar["rules_df"].to_csv("output/association_rules.csv", index=False)
            print("\n Association rules saved: output/association_rules.csv")

        elif unsup_task == "dim_reduction_plus_clustering":
            pca_out = run_pca(df, n_components=2)
            cl_out = run_clustering(df)

            emb_df = pd.DataFrame(pca_out["embeddings"], columns=["PC1", "PC2"])
            emb_df["cluster"] = cl_out["labels"]
            emb_df.to_csv("output/embeddings_clusters.csv", index=False)

            save_model(pca_out["pca_model"], "output/pca_model.pkl")
            save_model(cl_out["best_cluster_model"], "output/cluster_model.pkl")

            print("\n PCA + Clustering saved in output/")

        else:
            cl_out = run_clustering(df)

            print(f'\n Best k selected: {cl_out["best_k"]}')
            print(f' Best silhouette score: {cl_out["best_score"]: .4f}')

            # Save score summary
            pd.DataFrame(cl_out["scores"]).to_csv(
                "output/k_silhouette_scores.csv", index=False
            )

            # Save labels
            pd.DataFrame({"cluster_label": cl_out["labels"]}).to_csv(
                "output/clusters.csv", index=False
            )

            # Save model
            save_model(cl_out["best_cluster_model"], "output/cluster_model.pkl")

            # Cluster report
            report_df = generate_cluster_report(df, cl_out["labels"])

            interpretations = interpret_clusters(report_df)

            print("\n Cluster Interpretations:")
            for text in interpretations:
                print("-", text)

            print("\n Cluster Report Generated:")
            print(report_df)
            print("Saved: output/cluster_report.csv")

            # Auto Cluster Naming ---
            cluster_names = auto_name_cluster(report_df)
            print("\n Auto Cluster Names:")
            for cid, name in cluster_names.items():
                print(f" - Cluster {cid}: {name}")

            # Business Recommendations ---

            reco_lines = generate_business_recommendations(cluster_names, report_df)
            print(
                "\n Business Recommendations Saved: output/business_recommendations.txt"
            )

            # Business-smart interpretation
            smart_lines = interpret_clusters_smart(
                df, report_df, save_path="output/cluster_interpretation.txt"
            )
            print("\n Smart Interpretation Saved: output/cluster_interpretation.txt")

            print("\n Saved:")
            print(" - output/clusters.csv")
            print(" - output/k_silhouette_scores.csv")
            print(" - output/best_cluster_model.pkl")

            # Visualization
            plot_k_silhouette(cl_out["scores"], "output/k_vs_silhouette.png")
            pca_out = run_pca(df, n_components=2)
            plot_pca_clusters(
                pca_out["components"], cl_out["labels"], "output/pca_clusters.png"
            )
            print("\n Plots saved:")
            print(" - output/k_vs_silhouette.png")
            print(" - output/pca_clusters.png")


if __name__ == "__main__":
    main()
