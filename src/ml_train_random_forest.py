from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# CODEGUARD AI - RANDOM FOREST TEMPORAL BASELINE
# ============================================================

BASE_DIR = Path("data/processed/github")

TRAIN_FILE = BASE_DIR / "requests_temporal_train.csv"
VALIDATION_FILE = BASE_DIR / "requests_temporal_validation.csv"
TEST_FILE = BASE_DIR / "requests_temporal_test.csv"

MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "random_forest_temporal_baseline.joblib"

RESULT_FILE = BASE_DIR / "requests_random_forest_predictions.csv"


# ------------------------------------------------------------
# Temporal feature columns
# ------------------------------------------------------------

FEATURE_COLUMNS = [
    "prior_commit_count",
    "prior_additions",
    "prior_deletions",
    "prior_changes",
    "mean_prior_changes",
    "median_prior_changes",
    "max_prior_changes",
    "prior_churn_per_commit",
    "file_age_days",
    "days_since_last_change",
    "prior_contributor_count",
    "prior_top_contributor_share",
    "prior_contribution_entropy",
    "prior_bus_factor",
    "prior_defect_count",
    "prior_defect_rate",
    "commits_last_30d",
    "changes_last_30d",
    "additions_last_30d",
    "deletions_last_30d",
    "commits_last_90d",
    "changes_last_90d",
    "additions_last_90d",
    "deletions_last_90d",
    "commits_last_180d",
    "changes_last_180d",
    "additions_last_180d",
    "deletions_last_180d",
]


# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------

RANDOM_STATE = 42

MODEL = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_dataset(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    missing_columns = [
        col
        for col in FEATURE_COLUMNS + ["defect_target"]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path} is missing columns: "
            f"{missing_columns}"
        )

    return df


def evaluate_predictions(
    name,
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        predictions
    )

    print_section(f"{name} EVALUATION")

    print(f"Threshold : {threshold:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "non_defect",
                "defect",
            ],
            zero_division=0
        )
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "confusion_matrix": cm,
    }


def choose_validation_threshold(
    y_true,
    probabilities,
):
    """
    Select threshold using ONLY the validation set.

    The test set is not used for threshold selection.
    """

    thresholds = np.arange(
        0.05,
        0.96,
        0.01
    )

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    return (
        float(best_row["threshold"]),
        results_df
    )


def main():

    print("=" * 70)
    print("CODEGUARD AI - RANDOM FOREST TEMPORAL BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print_section("LOADING DATASETS")

    train = load_dataset(TRAIN_FILE)
    validation = load_dataset(VALIDATION_FILE)
    test = load_dataset(TEST_FILE)

    print(f"Train rows      : {len(train)}")
    print(f"Validation rows : {len(validation)}")
    print(f"Test rows       : {len(test)}")

    # --------------------------------------------------------
    # Prepare X / y
    # --------------------------------------------------------

    X_train = train[FEATURE_COLUMNS]
    y_train = train["defect_target"].astype(int)

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation["defect_target"].astype(int)

    X_test = test[FEATURE_COLUMNS]
    y_test = test["defect_target"].astype(int)

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    print_section("CLASS DISTRIBUTION")

    print(
        f"Train positives      : {y_train.sum()}"
    )

    print(
        f"Train negatives      : {(y_train == 0).sum()}"
    )

    print(
        f"Validation positives : {y_validation.sum()}"
    )

    print(
        f"Validation negatives : {(y_validation == 0).sum()}"
    )

    print(
        f"Test positives       : {y_test.sum()}"
    )

    print(
        f"Test negatives       : {(y_test == 0).sum()}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print_section("TRAINING RANDOM FOREST")

    print(
        f"Features : {len(FEATURE_COLUMNS)}"
    )

    print(
        "Class weighting : balanced"
    )

    print(
        "Estimators      : 500"
    )

    MODEL.fit(
        X_train,
        y_train
    )

    print(
        "\nRandom Forest training complete."
    )

    # --------------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------------

    validation_probabilities = (
        MODEL.predict_proba(
            X_validation
        )[:, 1]
    )

    validation_roc_auc = roc_auc_score(
        y_validation,
        validation_probabilities
    )

    validation_pr_auc = average_precision_score(
        y_validation,
        validation_probabilities
    )

    print_section("VALIDATION PROBABILITY PERFORMANCE")

    print(
        f"Validation ROC-AUC : "
        f"{validation_roc_auc:.4f}"
    )

    print(
        f"Validation PR-AUC  : "
        f"{validation_pr_auc:.4f}"
    )

    # --------------------------------------------------------
    # Threshold selection
    # --------------------------------------------------------

    print_section("VALIDATION THRESHOLD SELECTION")

    threshold, threshold_results = (
        choose_validation_threshold(
            y_validation,
            validation_probabilities
        )
    )

    best_threshold_row = threshold_results.loc[
        threshold_results["f1"].idxmax()
    ]

    print(
        f"Selected threshold : "
        f"{threshold:.4f}"
    )

    print(
        f"Validation precision: "
        f"{best_threshold_row['precision']:.4f}"
    )

    print(
        f"Validation recall   : "
        f"{best_threshold_row['recall']:.4f}"
    )

    print(
        f"Validation F1       : "
        f"{best_threshold_row['f1']:.4f}"
    )

    print("\nTop threshold candidates:")

    print(
        threshold_results
        .sort_values(
            "f1",
            ascending=False
        )
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------

    validation_metrics = evaluate_predictions(
        "VALIDATION",
        y_validation,
        validation_probabilities,
        threshold,
    )

    # --------------------------------------------------------
    # Final TEST evaluation
    #
    # The threshold was selected using validation only.
    # The test set is now used once for final evaluation.
    # --------------------------------------------------------

    test_probabilities = (
        MODEL.predict_proba(
            X_test
        )[:, 1]
    )

    test_metrics = evaluate_predictions(
        "FINAL TEST",
        y_test,
        test_probabilities,
        threshold,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print_section("RANDOM FOREST FEATURE IMPORTANCE")

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": MODEL.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    print(
        importance_df
        .round(6)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": MODEL,
            "features": FEATURE_COLUMNS,
            "threshold": threshold,
            "random_state": RANDOM_STATE,
        },
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    predictions = test[
        [
            "commit_sha",
            "commit_date",
            "file_path",
            "defect_target",
            "clean_defect_category",
            "commit_seq",
        ]
    ].copy()

    predictions["predicted_probability"] = (
        test_probabilities
    )

    predictions["predicted_target"] = (
        test_probabilities >= threshold
    ).astype(int)

    predictions["threshold"] = threshold

    predictions = predictions.sort_values(
        "predicted_probability",
        ascending=False
    )

    predictions.to_csv(
        RESULT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_section("FINAL TEST SUMMARY")

    print(
        f"Test ROC-AUC : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"Test PR-AUC  : "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"Test Precision : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1 : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"\nModel saved : {MODEL_FILE}"
    )

    print(
        f"Predictions saved : {RESULT_FILE}"
    )

    print(
        "\nRandom Forest temporal baseline complete."
    )


if __name__ == "__main__":
    main()