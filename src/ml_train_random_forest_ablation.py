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
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE_DIR / "data" / "processed" / "github" / "requests_temporal_train.csv"
VALIDATION_PATH = BASE_DIR / "data" / "processed" / "github" / "requests_temporal_validation.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "github" / "requests_temporal_test.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "random_forest_temporal_ablation.joblib"

PREDICTION_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_random_forest_ablation_predictions.csv"
)


# ============================================================
# FEATURES
# ============================================================

ALL_FEATURES = [
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


# Ablation:
# Remove features that directly summarize historical defect labels.
ABLATED_FEATURES = [
    feature
    for feature in ALL_FEATURES
    if feature not in {
        "prior_defect_count",
        "prior_defect_rate",
    }
]


# ============================================================
# HELPERS
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }


def select_threshold(y_true, probabilities):
    thresholds = np.round(
        np.arange(0.05, 0.951, 0.01),
        2,
    )

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

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

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    results_df = pd.DataFrame(results)

    best = results_df.sort_values(
        ["f1", "recall", "precision"],
        ascending=[False, False, False],
    ).iloc[0]

    return float(best["threshold"]), results_df


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CODEGUARD AI - RANDOM FOREST TEMPORAL ABLATION")
print("=" * 70)

print()
print("=" * 70)
print("ABLATION")
print("=" * 70)

print("Removed features:")
print("  - prior_defect_count")
print("  - prior_defect_rate")

print()
print("Remaining features:", len(ABLATED_FEATURES))

for feature in ABLATED_FEATURES:
    print("  -", feature)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train rows      : {len(train_df)}")
print(f"Validation rows : {len(validation_df)}")
print(f"Test rows       : {len(test_df)}")


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = set(ABLATED_FEATURES + ["defect_target"])

for name, df in [
    ("train", train_df),
    ("validation", validation_df),
    ("test", test_df),
]:
    missing = sorted(required_columns - set(df.columns))

    if missing:
        raise ValueError(
            f"{name} dataset is missing required columns: {missing}"
        )


# ============================================================
# TARGET
# ============================================================

y_train = train_df["defect_target"].astype(int)
y_validation = validation_df["defect_target"].astype(int)
y_test = test_df["defect_target"].astype(int)


print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(f"Train positives      : {int(y_train.sum())}")
print(f"Train negatives      : {int((y_train == 0).sum())}")

print(f"Validation positives : {int(y_validation.sum())}")
print(f"Validation negatives : {int((y_validation == 0).sum())}")

print(f"Test positives       : {int(y_test.sum())}")
print(f"Test negatives       : {int((y_test == 0).sum())}")


# ============================================================
# FEATURE MATRICES
# ============================================================

X_train = train_df[ABLATED_FEATURES].copy()
X_validation = validation_df[ABLATED_FEATURES].copy()
X_test = test_df[ABLATED_FEATURES].copy()

for df in [X_train, X_validation, X_test]:
    for column in ABLATED_FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

if X_train.isna().any().any():
    raise ValueError("Missing values detected in training features.")

if X_validation.isna().any().any():
    raise ValueError("Missing values detected in validation features.")

if X_test.isna().any().any():
    raise ValueError("Missing values detected in test features.")


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print()
print("=" * 70)
print("TRAINING RANDOM FOREST ABLATION")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

print(f"Features        : {len(ABLATED_FEATURES)}")
print("Class weighting : balanced")
print("Estimators      : 500")

model.fit(X_train, y_train)

print()
print("Random Forest ablation training complete.")


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]

validation_roc_auc = roc_auc_score(
    y_validation,
    validation_probabilities,
)

validation_pr_auc = average_precision_score(
    y_validation,
    validation_probabilities,
)

print()
print("=" * 70)
print("VALIDATION PROBABILITY PERFORMANCE")
print("=" * 70)

print(f"Validation ROC-AUC : {validation_roc_auc:.4f}")
print(f"Validation PR-AUC  : {validation_pr_auc:.4f}")


# ============================================================
# THRESHOLD SELECTION
# ============================================================

selected_threshold, threshold_results = select_threshold(
    y_validation,
    validation_probabilities,
)

best_row = threshold_results[
    threshold_results["threshold"] == selected_threshold
].iloc[0]

print()
print("=" * 70)
print("VALIDATION THRESHOLD SELECTION")
print("=" * 70)

print(f"Selected threshold : {selected_threshold:.4f}")
print(f"Validation precision: {best_row['precision']:.4f}")
print(f"Validation recall   : {best_row['recall']:.4f}")
print(f"Validation F1       : {best_row['f1']:.4f}")

print()
print("Top threshold candidates:")

print(
    threshold_results
    .sort_values(
        ["f1", "recall", "precision"],
        ascending=[False, False, False],
    )
    .head(10)
    .to_string(index=False)
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

validation_metrics = calculate_metrics(
    y_validation,
    validation_probabilities,
    selected_threshold,
)

validation_predictions = (
    validation_probabilities >= selected_threshold
).astype(int)

print()
print("=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)

print(f"Threshold : {selected_threshold:.4f}")
print(f"ROC-AUC   : {validation_metrics['roc_auc']:.4f}")
print(f"PR-AUC    : {validation_metrics['pr_auc']:.4f}")
print(f"Precision : {validation_metrics['precision']:.4f}")
print(f"Recall    : {validation_metrics['recall']:.4f}")
print(f"F1        : {validation_metrics['f1']:.4f}")

print()
print("Confusion matrix:")
print(
    confusion_matrix(
        y_validation,
        validation_predictions,
    )
)

print()
print("Classification report:")
print(
    classification_report(
        y_validation,
        validation_predictions,
        target_names=["non_defect", "defect"],
        zero_division=0,
    )
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probabilities >= selected_threshold
).astype(int)

test_metrics = calculate_metrics(
    y_test,
    test_probabilities,
    selected_threshold,
)

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

print(f"Threshold : {selected_threshold:.4f}")
print(f"ROC-AUC   : {test_metrics['roc_auc']:.4f}")
print(f"PR-AUC    : {test_metrics['pr_auc']:.4f}")
print(f"Precision : {test_metrics['precision']:.4f}")
print(f"Recall    : {test_metrics['recall']:.4f}")
print(f"F1        : {test_metrics['f1']:.4f}")

print()
print("Confusion matrix:")
print(
    confusion_matrix(
        y_test,
        test_predictions,
    )
)

print()
print("Classification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["non_defect", "defect"],
        zero_division=0,
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(
    {
        "feature": ABLATED_FEATURES,
        "importance": model.feature_importances_,
    }
).sort_values(
    "importance",
    ascending=False,
)

print()
print("=" * 70)
print("ABLATION FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance_df.to_string(
        index=False,
        formatters={
            "importance": "{:.6f}".format,
        },
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_package = {
    "model": model,
    "features": ABLATED_FEATURES,
    "threshold": selected_threshold,
    "removed_features": [
        "prior_defect_count",
        "prior_defect_rate",
    ],
    "random_state": 42,
    "model_type": "RandomForestClassifier",
    "experiment": "temporal_ablation_without_prior_defect_features",
}

joblib.dump(
    model_package,
    MODEL_PATH,
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_output = test_df[
    [
        "commit_sha",
        "commit_date",
        "file_path",
        "defect_target",
    ]
].copy()

prediction_output["predicted_probability"] = test_probabilities
prediction_output["predicted_class"] = test_predictions
prediction_output["threshold"] = selected_threshold

prediction_output.to_csv(
    PREDICTION_PATH,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL ABLATION SUMMARY")
print("=" * 70)

print(f"Test ROC-AUC   : {test_metrics['roc_auc']:.4f}")
print(f"Test PR-AUC    : {test_metrics['pr_auc']:.4f}")
print(f"Test Precision : {test_metrics['precision']:.4f}")
print(f"Test Recall    : {test_metrics['recall']:.4f}")
print(f"Test F1        : {test_metrics['f1']:.4f}")

print()
print(f"Model saved : {MODEL_PATH}")
print(f"Predictions saved : {PREDICTION_PATH}")

print()
print("Random Forest temporal ablation complete.")