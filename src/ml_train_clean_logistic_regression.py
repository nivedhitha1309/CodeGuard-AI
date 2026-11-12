from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

TRAIN_PATH = DATA_DIR / "requests_clean_temporal_train.csv"
VALIDATION_PATH = DATA_DIR / "requests_clean_temporal_validation.csv"
TEST_PATH = DATA_DIR / "requests_clean_temporal_test.csv"

MODEL_PATH = (
    MODEL_DIR
    / "clean_logistic_regression_temporal.joblib"
)

PREDICTION_PATH = (
    DATA_DIR
    / "requests_clean_logistic_regression_predictions.csv"
)


# ============================================================
# FEATURES
# ============================================================

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


# ============================================================
# EVALUATION FUNCTION
# ============================================================

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

    matrix = confusion_matrix(
        y_true,
        predictions
    )

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")
    print(f"Threshold : {threshold:.2f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")

    print("\nConfusion matrix:")
    print(matrix)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix,
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("CODEGUARD AI - CLEAN TEMPORAL LOGISTIC REGRESSION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Train rows      : {len(train)}")
print(f"Validation rows : {len(validation)}")
print(f"Test rows       : {len(test)}")


# ============================================================
# VALIDATE FEATURES
# ============================================================

for name, dataframe in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing features: {missing}"
        )


# ============================================================
# BUILD X / Y
# ============================================================

X_train = train[FEATURE_COLUMNS].copy()
y_train = train["target"].astype(int)

X_validation = validation[FEATURE_COLUMNS].copy()
y_validation = validation["target"].astype(int)

X_test = test[FEATURE_COLUMNS].copy()
y_test = test["target"].astype(int)


# ============================================================
# PREVALENCE
# ============================================================

print("\n" + "=" * 70)
print("TARGET PREVALENCE")
print("=" * 70)

train_prevalence = y_train.mean()
validation_prevalence = y_validation.mean()
test_prevalence = y_test.mean()

print(
    f"Train prevalence      : "
    f"{train_prevalence:.4f}"
)

print(
    f"Validation prevalence : "
    f"{validation_prevalence:.4f}"
)

print(
    f"Test prevalence       : "
    f"{test_prevalence:.4f}"
)


# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                C=1.0,
                max_iter=5000,
                random_state=42,
            ),
        ),
    ]
)

model.fit(
    X_train,
    y_train
)

print(
    "Logistic Regression training complete."
)


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]


# ============================================================
# THRESHOLD SELECTION
# ============================================================

print("\n" + "=" * 70)
print("SELECTING VALIDATION THRESHOLD")
print("=" * 70)

threshold_results = []

for threshold in np.arange(
    0.05,
    0.96,
    0.01
):

    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )

    threshold_results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

threshold_table = pd.DataFrame(
    threshold_results
)

best_row = threshold_table.loc[
    threshold_table["f1"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)

print(
    f"Selected threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Validation precision : "
    f"{best_row['precision']:.4f}"
)

print(
    f"Validation recall : "
    f"{best_row['recall']:.4f}"
)

print(
    f"Validation F1 : "
    f"{best_row['f1']:.4f}"
)


# ============================================================
# VALIDATION RESULTS
# ============================================================

validation_results = evaluate_predictions(
    "VALIDATION RESULTS",
    y_validation,
    validation_probabilities,
    best_threshold,
)


# ============================================================
# TEST RESULTS
# ============================================================

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

test_results = evaluate_predictions(
    "TEST RESULTS",
    y_test,
    test_probabilities,
    best_threshold,
)


# ============================================================
# COEFFICIENTS
# ============================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 70)

classifier = model.named_steps[
    "classifier"
]

coefficients = classifier.coef_[0]

coefficient_table = pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS,
        "coefficient": coefficients,
        "absolute_coefficient": np.abs(
            coefficients
        ),
    }
).sort_values(
    "absolute_coefficient",
    ascending=False
)

print(
    coefficient_table.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    {
        "model": model,
        "features": FEATURE_COLUMNS,
        "threshold": best_threshold,
    },
    MODEL_PATH
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_output = test[
    [
        "commit_sha",
        "commit_date",
        "commit_seq",
        "file_path",
        "commit_subject",
        "clean_defect_category",
        "target",
    ]
].copy()

prediction_output[
    "predicted_probability"
] = test_probabilities

prediction_output[
    "predicted_target"
] = (
    test_probabilities >= best_threshold
).astype(int)

prediction_output.to_csv(
    PREDICTION_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLEAN LOGISTIC REGRESSION COMPLETE")
print("=" * 70)

print(
    f"Validation ROC-AUC : "
    f"{validation_results['roc_auc']:.4f}"
)

print(
    f"Validation PR-AUC  : "
    f"{validation_results['pr_auc']:.4f}"
)

print(
    f"Test ROC-AUC       : "
    f"{test_results['roc_auc']:.4f}"
)

print(
    f"Test PR-AUC        : "
    f"{test_results['pr_auc']:.4f}"
)

print(
    f"Test F1            : "
    f"{test_results['f1']:.4f}"
)

print(
    f"\nModel saved to : "
    f"{MODEL_PATH}"
)

print(
    f"Predictions saved to : "
    f"{PREDICTION_PATH}"
)

print("=" * 70)