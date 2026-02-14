import os
import json
import pandas as pd
import joblib
from datasets import load_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from huggingface_hub import HfApi, create_repo

DATASET_REPO = "mukherjee78/predictive-maintenance-engine-data"
MODEL_REPO = "mukherjee78/predictive-maintenance-random-forest"
MODEL_PATH = "models/best_model.pkl"
METRICS_PATH = "models/metrics.json"
REPORTS_FIGURES_DIR = "reports/figures"
TARGET = "engine_condition"


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs(REPORTS_FIGURES_DIR, exist_ok=True)

    print("Loading train/test data from Hugging Face...")
    dataset = load_dataset(
        DATASET_REPO,
        data_files={"train": "train.csv", "test": "test.csv"},
    )
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # ========== Phase 1: Train all models (no evaluation yet) ==========
    print("\n" + "=" * 60)
    print("PHASE 1 — Train all models")
    print("=" * 60)

    # --- Decision Tree (Baseline) ---
    print("\n--- Decision Tree (Baseline) ---")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    print("Fitted.")

    # --- Random Forest (GridSearchCV) ---
    print("\n--- Random Forest (GridSearchCV) ---")
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2],
    }
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="recall",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")

    # --- Logistic Regression (scaled features) ---
    print("\n--- Logistic Regression ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    print("Fitted.")

    # --- Gradient Boosting (GridSearchCV) ---
    print("\n--- Gradient Boosting (GridSearchCV) ---")
    gb = GradientBoostingClassifier(random_state=42)
    gb_param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_samples_split": [2, 5],
    }
    gb_search = GridSearchCV(
        estimator=gb,
        param_grid=gb_param_grid,
        scoring="recall",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    gb_search.fit(X_train, y_train)
    best_gb = gb_search.best_estimator_
    print(f"Best params: {gb_search.best_params_}")

    # --- XGBoost (GridSearchCV) ---
    print("\n--- XGBoost (GridSearchCV) ---")
    xgb = XGBClassifier(random_state=42, eval_metric="logloss")
    xgb_param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    xgb_search = GridSearchCV(
        estimator=xgb,
        param_grid=xgb_param_grid,
        scoring="recall",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    print(f"Best params: {xgb_search.best_params_}")

    # ========== Phase 2: Log all tuned parameters ==========
    print("\n" + "=" * 60)
    print("PHASE 2 — Log tuned parameters (RF, GB, XGBoost)")
    print("=" * 60)

    for name, search in [
        ("Random Forest", grid_search),
        ("Gradient Boosting", gb_search),
        ("XGBoost", xgb_search),
    ]:
        log_df = pd.DataFrame(search.cv_results_)[
            ["params", "mean_test_score", "rank_test_score"]
        ].sort_values("rank_test_score")
        print(f"\n--- {name} (top 5) ---")
        print(log_df.head().to_string())

    # ========== Phase 3: Evaluate all models on test set ==========
    print("\n" + "=" * 60)
    print("PHASE 3 — Evaluate all models on test data")
    print("=" * 60)

    dt_preds = dt_model.predict(X_test)
    dt_probs = dt_model.predict_proba(X_test)[:, 1]
    rf_preds = best_rf.predict(X_test)
    rf_probs = best_rf.predict_proba(X_test)[:, 1]
    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    gb_preds = best_gb.predict(X_test)
    gb_probs = best_gb.predict_proba(X_test)[:, 1]
    xgb_preds = best_xgb.predict(X_test)
    xgb_probs = best_xgb.predict_proba(X_test)[:, 1]

    for name, preds, probs in [
        ("Decision Tree", dt_preds, dt_probs),
        ("Random Forest", rf_preds, rf_probs),
        ("Logistic Regression", lr_preds, lr_probs),
        ("Gradient Boosting", gb_preds, gb_probs),
        ("XGBoost", xgb_preds, xgb_probs),
    ]:
        print(f"\n--- {name} ---")
        print(classification_report(y_test, preds))
        print("ROC-AUC:", roc_auc_score(y_test, probs))

    # --- Model comparison (all 5) ---
    print("\n" + "=" * 60)
    print("Model comparison (summary)")
    comparison_df = pd.DataFrame({
        "Model": [
            "Decision Tree",
            "Random Forest",
            "Logistic Regression",
            "Gradient Boosting",
            "XGBoost",
        ],
        "Precision": [
            precision_score(y_test, dt_preds, zero_division=0),
            precision_score(y_test, rf_preds, zero_division=0),
            precision_score(y_test, lr_preds, zero_division=0),
            precision_score(y_test, gb_preds, zero_division=0),
            precision_score(y_test, xgb_preds, zero_division=0),
        ],
        "Recall": [
            recall_score(y_test, dt_preds),
            recall_score(y_test, rf_preds),
            recall_score(y_test, lr_preds),
            recall_score(y_test, gb_preds),
            recall_score(y_test, xgb_preds),
        ],
        "F1 Score": [
            f1_score(y_test, dt_preds),
            f1_score(y_test, rf_preds),
            f1_score(y_test, lr_preds),
            f1_score(y_test, gb_preds),
            f1_score(y_test, xgb_preds),
        ],
        "ROC-AUC": [
            roc_auc_score(y_test, dt_probs),
            roc_auc_score(y_test, rf_probs),
            roc_auc_score(y_test, lr_probs),
            roc_auc_score(y_test, gb_probs),
            roc_auc_score(y_test, xgb_probs),
        ],
    })
    print(comparison_df.to_string(index=False))

    # --- Best model by recall (for deployment) ---
    models_for_best = [
        ("Decision Tree", dt_model, recall_score(y_test, dt_preds)),
        ("Random Forest", best_rf, recall_score(y_test, rf_preds)),
        ("Logistic Regression", lr_model, recall_score(y_test, lr_preds)),
        ("Gradient Boosting", best_gb, recall_score(y_test, gb_preds)),
        ("XGBoost", best_xgb, recall_score(y_test, xgb_preds)),
    ]
    best_name, best_estimator, best_recall = max(models_for_best, key=lambda x: x[2])
    print(f"\nBest model by recall: {best_name} (recall={best_recall:.4f})")

    # --- Confusion matrix for best model ---
    try:
        import matplotlib.pyplot as plt
        X_plot = X_test_scaled if best_name == "Logistic Regression" else X_test
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(best_estimator, X_plot, y_test, ax=ax)
        cm_path = os.path.join(REPORTS_FIGURES_DIR, "confusion_matrix_best_model.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"\nConfusion matrix saved to {cm_path}")
    except Exception as e:
        print(f"Skipping confusion matrix figure: {e}")

    # --- Save best model locally (include scaler when best is Logistic Regression) ---
    to_save = {
        "model": best_estimator,
        "scaler": scaler if best_name == "Logistic Regression" else None,
        "best_model_name": best_name,
    }
    joblib.dump(to_save, MODEL_PATH)
    print(f"\nBest model ({best_name}) saved to {MODEL_PATH}")

    # --- Metrics (all 5 models) ---
    metrics = {
        "best_model_name": best_name,
        "best_params_rf": grid_search.best_params_,
        "best_params_gb": gb_search.best_params_,
        "best_params_xgb": xgb_search.best_params_,
        "decision_tree": {
            "precision": float(precision_score(y_test, dt_preds, zero_division=0)),
            "recall": float(recall_score(y_test, dt_preds)),
            "f1": float(f1_score(y_test, dt_preds)),
            "roc_auc": float(roc_auc_score(y_test, dt_probs)),
        },
        "random_forest": {
            "precision": float(precision_score(y_test, rf_preds, zero_division=0)),
            "recall": float(recall_score(y_test, rf_preds)),
            "f1": float(f1_score(y_test, rf_preds)),
            "roc_auc": float(roc_auc_score(y_test, rf_probs)),
        },
        "logistic_regression": {
            "precision": float(precision_score(y_test, lr_preds, zero_division=0)),
            "recall": float(recall_score(y_test, lr_preds)),
            "f1": float(f1_score(y_test, lr_preds)),
            "roc_auc": float(roc_auc_score(y_test, lr_probs)),
        },
        "gradient_boosting": {
            "precision": float(precision_score(y_test, gb_preds, zero_division=0)),
            "recall": float(recall_score(y_test, gb_preds)),
            "f1": float(f1_score(y_test, gb_preds)),
            "roc_auc": float(roc_auc_score(y_test, gb_probs)),
        },
        "xgboost": {
            "precision": float(precision_score(y_test, xgb_preds, zero_division=0)),
            "recall": float(recall_score(y_test, xgb_preds)),
            "f1": float(f1_score(y_test, xgb_preds)),
            "roc_auc": float(roc_auc_score(y_test, xgb_probs)),
        },
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {METRICS_PATH}")

    # --- Hugging Face Model Hub ---
    print("\nRegistering model on Hugging Face Model Hub...")
    api = HfApi()
    create_repo(
        repo_id=MODEL_REPO,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo="best_model.pkl",
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=METRICS_PATH,
        path_in_repo="metrics.json",
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print(f"Model registered at: https://huggingface.co/{MODEL_REPO}")
    print("\nModel training and registration complete.")


if __name__ == "__main__":
    main()
