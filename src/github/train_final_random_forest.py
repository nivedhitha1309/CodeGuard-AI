from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_final_temporal_train.csv"
)

VAL_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_final_temporal_val.csv"
)

TEST_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_final_temporal_test.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "final_random_forest_temporal.joblib"
PREDICTION_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_final_random_forest_predictions.csv"
)


TARGET_COLUMN = "ml_defect_target"

# These are the temporal features created by
# build_final_temporal_ml_dataset.py.
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
    "rolling_30d_commits",
    "rolling_30d_changes",
    "rolling_30d_additions",
    "rolling_30d_deletions",
    "rolling_90d_commits",
    "rolling_90d_changes",
    "rolling_90d_additions",
    "rolling_90d_deletions",
    "rolling_180d_commits",
    "rolling_180d_changes",
    "rolling_180d_additions",
    "rolling_180d_deletions",
]


def load_split(path):
    df = pd.read_csv(path)

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns in {path.name}: "
            f"{missing_features}"
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column '{TARGET_COLUMN}' in {path.name}"
        )

    return df


def validate_split(name, df):
    print(f"{name}:")
    print(f"  Rows      : {len(df)}")
    print(f"  Positives : {int(df[TARGET_COLUMN].sum())}")
    print(f"  Negatives : {int((df[TARGET_COLUMN] == 0).sum())}")

    if len(df) > 0:
        print(
            f"  Rate      : "
            f"{df[TARGET_COLUMN].mean():.4f}"
        )

    missing = df[FEATURE_COLUMNS].isna().sum().sum()

    if missing != 0:
        raise ValueError(
            f"{name} contains {missing} missing feature values."
        )

    print(f"  Missing features: {missing}")
    print()


def find_best_threshold(y_true, probabilities):
    """
    Select threshold using validation F1 only.

    The test set is never used to select this threshold.
    """

    thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


def evaluate_model(name, y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Threshold : {threshold:.2f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")

    print()
    print("Confusion matrix:")
    print(matrix)

    print()

    return {
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def main():

    print("=" * 70)
    print("CODEGUARD AI - FINAL RANDOM FOREST")
    print("=" * 70)
    print()

    train_df = load_split(TRAIN_FILE)
    val_df = load_split(VAL_FILE)
    test_df = load_split(TEST_FILE)

    print("DATASET SUMMARY")
    print("-" * 70)

    validate_split("TRAIN", train_df)
    validate_split("VALIDATION", val_df)
    validate_split("TEST", test_df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].astype(int)

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN].astype(int)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].astype(int)

    print("FEATURE COUNT")
    print("-" * 70)
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print()

    print("TRAINING RANDOM FOREST")
    print("-" * 70)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")
    print()

    # ---------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------

    val_probabilities = model.predict_proba(
        X_val
    )[:, 1]

    validation_threshold, validation_f1 = (
        find_best_threshold(
            y_val,
            val_probabilities,
        )
    )

    print("VALIDATION THRESHOLD SELECTION")
    print("-" * 70)
    print(
        f"Selected threshold : "
        f"{validation_threshold:.2f}"
    )
    print(
        f"Validation F1      : "
        f"{validation_f1:.4f}"
    )
    print()

    validation_metrics = evaluate_model(
        "VALIDATION RESULTS",
        y_val,
        val_probabilities,
        validation_threshold,
    )

    # ---------------------------------------------------------------
    # TEST
    # ---------------------------------------------------------------

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_metrics = evaluate_model(
        "FINAL TEST RESULTS",
        y_test,
        test_probabilities,
        validation_threshold,
    )

    # ---------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print("=" * 70)
    print("TOP FEATURE IMPORTANCES")
    print("=" * 70)

    print(
        importance_df.head(15).to_string(
            index=False
        )
    )

    print()

    # ---------------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------------

    model_package = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "validation_threshold": validation_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    joblib.dump(
        model_package,
        MODEL_FILE,
    )

    # ---------------------------------------------------------------
    # SAVE PREDICTIONS
    # ---------------------------------------------------------------

    prediction_df = test_df.copy()

    prediction_df["predicted_probability"] = (
        test_probabilities
    )

    prediction_df["predicted_label"] = (
        test_probabilities >= validation_threshold
    ).astype(int)

    prediction_df["model"] = (
        "final_random_forest_temporal"
    )

    prediction_df.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Model      : {MODEL_FILE}"
    )

    print(
        f"Predictions: {PREDICTION_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()