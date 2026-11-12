from pathlib import Path
import math
import re

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

HISTORY_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "github"
    / "requests_commit_file_history.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "final_random_forest_temporal.joblib"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_current_ml_defect_risk.csv"
)


# ============================================================
# FILE FILTERING
# ============================================================

PRODUCTION_SOURCE_DIRS = (
    "src/",
    "requests/",
)

EXCLUDED_PATH_PARTS = (
    "/tests/",
    "/test/",
    "/docs/",
    "/examples/",
    "/scripts/",
    "/tools/",
    "/vendor/",
    "/.github/",
)

EXCLUDED_FILENAME_PATTERNS = (
    r"(^|/)test_.*\.py$",
    r"(^|/).*_test\.py$",
)


# ============================================================
# MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
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
# HELPERS
# ============================================================

def is_bot_author(author):
    """Return True for GitHub bot-style authors."""

    if pd.isna(author):
        return False

    author = str(author).strip().lower()

    return author.endswith("[bot]")


def is_production_python_file(file_path):
    """
    Keep production Python source files and exclude tests,
    documentation, examples, tools, CI, etc.
    """

    if pd.isna(file_path):
        return False

    path = str(file_path).replace("\\", "/").strip()

    if not path.endswith(".py"):
        return False

    if not path.startswith(PRODUCTION_SOURCE_DIRS):
        return False

    lowered = path.lower()

    for excluded in EXCLUDED_PATH_PARTS:
        if excluded in lowered:
            return False

    for pattern in EXCLUDED_FILENAME_PATTERNS:
        if re.search(pattern, lowered):
            return False

    return True


def normalized_entropy(counts):
    """
    Shannon entropy normalized to [0, 1].
    """

    counts = np.asarray(
        counts,
        dtype=float,
    )

    counts = counts[counts > 0]

    if len(counts) <= 1:
        return 0.0

    total = counts.sum()

    if total <= 0:
        return 0.0

    probabilities = counts / total

    entropy = -np.sum(
        probabilities * np.log(probabilities)
    )

    max_entropy = math.log(
        len(probabilities)
    )

    if max_entropy <= 0:
        return 0.0

    return float(
        entropy / max_entropy
    )


def calculate_bus_factor(contribution_counts):
    """
    Minimum number of contributors responsible
    for at least 50% of prior human commits.
    """

    counts = pd.Series(
        contribution_counts,
        dtype=float,
    )

    counts = counts[
        counts > 0
    ]

    if counts.empty:
        return 0

    counts = counts.sort_values(
        ascending=False
    )

    total = counts.sum()

    if total <= 0:
        return 0

    cumulative_share = (
        counts.cumsum()
        / total
    ).to_numpy()

    position = np.argmax(
        cumulative_share >= 0.50
    )

    return int(
        position + 1
    )


def calculate_features(
    file_history,
    as_of_date,
):
    """
    Calculate the 26 temporal features expected by
    the final Random Forest.

    The current/latest event is excluded.

    Therefore every feature represents information
    available strictly before the prediction event.
    """

    history = file_history.copy()

    history["commit_date"] = pd.to_datetime(
        history["commit_date"],
        utc=True,
        errors="coerce",
    )

    history = history[
        history["commit_date"] < as_of_date
    ].copy()

    if history.empty:
        return None

    history = history.sort_values(
        "commit_date"
    )

    # --------------------------------------------------------
    # Change metrics
    # --------------------------------------------------------

    additions = pd.to_numeric(
        history["additions"],
        errors="coerce",
    ).fillna(0)

    deletions = pd.to_numeric(
        history["deletions"],
        errors="coerce",
    ).fillna(0)

    changes = pd.to_numeric(
        history["changes"],
        errors="coerce",
    )

    changes = changes.fillna(
        additions + deletions
    )

    prior_commit_count = len(
        history
    )

    prior_additions = float(
        additions.sum()
    )

    prior_deletions = float(
        deletions.sum()
    )

    prior_changes = float(
        changes.sum()
    )

    mean_prior_changes = float(
        changes.mean()
    )

    median_prior_changes = float(
        changes.median()
    )

    max_prior_changes = float(
        changes.max()
    )

    prior_churn_per_commit = (
        prior_changes
        / prior_commit_count
        if prior_commit_count > 0
        else 0.0
    )

    # --------------------------------------------------------
    # File age
    # --------------------------------------------------------

    first_date = history[
        "commit_date"
    ].min()

    last_date = history[
        "commit_date"
    ].max()

    file_age_days = (
        as_of_date - first_date
    ).total_seconds() / 86400.0

    days_since_last_change = (
        as_of_date - last_date
    ).total_seconds() / 86400.0

    file_age_days = max(
        file_age_days,
        0.0,
    )

    days_since_last_change = max(
        days_since_last_change,
        0.0,
    )

    # --------------------------------------------------------
    # Human contributor concentration
    # --------------------------------------------------------

    human_history = history[
        ~history["author"].apply(
            is_bot_author
        )
    ].copy()

    if human_history.empty:

        prior_contributor_count = 0

        prior_top_contributor_share = 0.0

        prior_contribution_entropy = 0.0

        prior_bus_factor = 0

    else:

        contributor_counts = (
            human_history
            .groupby("author")
            .size()
            .sort_values(
                ascending=False
            )
        )

        total_human_commits = (
            contributor_counts.sum()
        )

        prior_contributor_count = int(
            len(contributor_counts)
        )

        if total_human_commits > 0:

            prior_top_contributor_share = float(
                contributor_counts.iloc[0]
                / total_human_commits
            )

            prior_contribution_entropy = (
                normalized_entropy(
                    contributor_counts.values
                )
            )

            prior_bus_factor = (
                calculate_bus_factor(
                    contributor_counts.values
                )
            )

        else:

            prior_top_contributor_share = 0.0

            prior_contribution_entropy = 0.0

            prior_bus_factor = 0

    # --------------------------------------------------------
    # Rolling windows
    # --------------------------------------------------------

    rolling_features = {}

    for label, days in [
        ("30d", 30),
        ("90d", 90),
        ("180d", 180),
    ]:

        cutoff = (
            as_of_date
            - pd.Timedelta(
                days=days
            )
        )

        recent = history[
            history["commit_date"] >= cutoff
        ]

        rolling_features[
            f"rolling_{label}_commits"
        ] = float(
            len(recent)
        )

        rolling_features[
            f"rolling_{label}_changes"
        ] = float(
            pd.to_numeric(
                recent["changes"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        rolling_features[
            f"rolling_{label}_additions"
        ] = float(
            pd.to_numeric(
                recent["additions"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        rolling_features[
            f"rolling_{label}_deletions"
        ] = float(
            pd.to_numeric(
                recent["deletions"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

    # --------------------------------------------------------
    # Final 26 features
    # --------------------------------------------------------

    features = {

        "prior_commit_count":
            float(prior_commit_count),

        "prior_additions":
            prior_additions,

        "prior_deletions":
            prior_deletions,

        "prior_changes":
            prior_changes,

        "mean_prior_changes":
            mean_prior_changes,

        "median_prior_changes":
            median_prior_changes,

        "max_prior_changes":
            max_prior_changes,

        "prior_churn_per_commit":
            float(prior_churn_per_commit),

        "file_age_days":
            float(file_age_days),

        "days_since_last_change":
            float(days_since_last_change),

        "prior_contributor_count":
            float(prior_contributor_count),

        "prior_top_contributor_share":
            float(prior_top_contributor_share),

        "prior_contribution_entropy":
            float(prior_contribution_entropy),

        "prior_bus_factor":
            float(prior_bus_factor),

        "rolling_30d_commits":
            rolling_features[
                "rolling_30d_commits"
            ],

        "rolling_30d_changes":
            rolling_features[
                "rolling_30d_changes"
            ],

        "rolling_30d_additions":
            rolling_features[
                "rolling_30d_additions"
            ],

        "rolling_30d_deletions":
            rolling_features[
                "rolling_30d_deletions"
            ],

        "rolling_90d_commits":
            rolling_features[
                "rolling_90d_commits"
            ],

        "rolling_90d_changes":
            rolling_features[
                "rolling_90d_changes"
            ],

        "rolling_90d_additions":
            rolling_features[
                "rolling_90d_additions"
            ],

        "rolling_90d_deletions":
            rolling_features[
                "rolling_90d_deletions"
            ],

        "rolling_180d_commits":
            rolling_features[
                "rolling_180d_commits"
            ],

        "rolling_180d_changes":
            rolling_features[
                "rolling_180d_changes"
            ],

        "rolling_180d_additions":
            rolling_features[
                "rolling_180d_additions"
            ],

        "rolling_180d_deletions":
            rolling_features[
                "rolling_180d_deletions"
            ],
    }

    return features


def risk_level(score):
    """
    Presentation bands for the continuous ML risk score.

    These are NOT calibrated probabilities.
    """

    if score >= 0.50:
        return "Critical"

    if score >= 0.25:
        return "High"

    if score >= 0.10:
        return "Medium"

    return "Low"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CODEGUARD AI - CURRENT ML DEFECT RISK")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model_package = joblib.load(
        MODEL_PATH
    )

    if isinstance(
        model_package,
        dict,
    ):

        model = model_package.get(
            "model",
            model_package.get(
                "classifier",
                model_package,
            ),
        )

        expected_features = (
            model_package.get(
                "feature_columns"
            )
        )

    else:

        model = model_package

        expected_features = None

    if expected_features is None:
        expected_features = MODEL_FEATURES

    expected_features = list(
        expected_features
    )

    print(
        f"Model loaded           : "
        f"{MODEL_PATH}"
    )

    print(
        f"Expected features      : "
        f"{len(expected_features)}"
    )

    # --------------------------------------------------------
    # Load history
    # --------------------------------------------------------

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"History file not found: "
            f"{HISTORY_PATH}"
        )

    history = pd.read_csv(
        HISTORY_PATH
    )

    print(
        f"Historical rows loaded : "
        f"{len(history)}"
    )

    # --------------------------------------------------------
    # Validate actual dataset schema
    # --------------------------------------------------------

    required_columns = {
        "commit_sha",
        "author",
        "commit_date",
        "file_path",
        "additions",
        "deletions",
        "changes",
    }

    missing_columns = (
        required_columns
        - set(history.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    # --------------------------------------------------------
    # Normalize actual history schema
    # --------------------------------------------------------

    history["commit_date"] = pd.to_datetime(
        history["commit_date"],
        utc=True,
        errors="coerce",
    )

    history = history.dropna(
        subset=["commit_date"]
    ).copy()

    history["file_path"] = (
        history["file_path"]
        .astype(str)
        .str.replace(
            "\\",
            "/",
            regex=False,
        )
    )

    # --------------------------------------------------------
    # Production Python files
    # --------------------------------------------------------

    history = history[
        history["file_path"].apply(
            is_production_python_file
        )
    ].copy()

    print(
        f"Production Python rows : "
        f"{len(history)}"
    )

    print(
        f"Production Python files: "
        f"{history['file_path'].nunique()}"
    )

    # --------------------------------------------------------
    # Chronological ordering
    # --------------------------------------------------------

    history = history.sort_values(
        [
            "file_path",
            "commit_date",
            "commit_sha",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Generate current file-level risk
    # --------------------------------------------------------

    total_files = (
        history["file_path"]
        .nunique()
    )

    print()

    print(
        f"Generating current risk for "
        f"{total_files} production Python files..."
    )

    results = []

    for file_number, (
        file_path,
        file_history,
    ) in enumerate(
        history.groupby(
            "file_path",
            sort=True,
        ),
        start=1,
    ):

        file_history = (
            file_history
            .sort_values(
                [
                    "commit_date",
                    "commit_sha",
                ]
            )
            .copy()
        )

        # Latest event is the current/as-of point.
        latest_event = (
            file_history.iloc[-1]
        )

        as_of_date = (
            latest_event["commit_date"]
        )

        features = calculate_features(
            file_history,
            as_of_date,
        )

        if features is None:
            continue

        feature_row = pd.DataFrame(
            [features]
        )

        # Exact feature order expected by model.
        feature_row = feature_row[
            expected_features
        ]

        feature_row = feature_row.apply(
            pd.to_numeric,
            errors="coerce",
        ).fillna(0.0)

        # ----------------------------------------------------
        # Predict continuous ML risk
        # ----------------------------------------------------

        probability = float(
            model.predict_proba(
                feature_row
            )[0, 1]
        )

        results.append(
            {
                "file_path":
                    file_path,

                "as_of_date":
                    as_of_date,

                "ml_defect_risk":
                    probability,

                "ml_risk_level":
                    risk_level(
                        probability
                    ),

                "prior_commit_count":
                    features[
                        "prior_commit_count"
                    ],

                "prior_contributor_count":
                    features[
                        "prior_contributor_count"
                    ],

                "prior_bus_factor":
                    features[
                        "prior_bus_factor"
                    ],

                "prior_contribution_entropy":
                    features[
                        "prior_contribution_entropy"
                    ],

                "days_since_last_change":
                    features[
                        "days_since_last_change"
                    ],
            }
        )

        if (
            file_number % 25 == 0
            or file_number == total_files
        ):

            print(
                f"  Processed "
                f"{file_number}/"
                f"{total_files}"
            )

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    if result_df.empty:
        raise RuntimeError(
            "No current ML risk predictions "
            "were generated."
        )

    result_df["as_of_date"] = (
        pd.to_datetime(
            result_df["as_of_date"],
            utc=True,
        )
        .dt.strftime(
            "%Y-%m-%d %H:%M:%S%z"
        )
    )

    result_df = result_df.sort_values(
        "ml_defect_risk",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()

    print("=" * 70)
    print("CURRENT ML RISK SUMMARY")
    print("=" * 70)

    print(
        f"Files scored            : "
        f"{len(result_df)}"
    )

    print(
        f"Average ML risk         : "
        f"{result_df['ml_defect_risk'].mean():.4f}"
    )

    print(
        f"Minimum ML risk         : "
        f"{result_df['ml_defect_risk'].min():.4f}"
    )

    print(
        f"Maximum ML risk         : "
        f"{result_df['ml_defect_risk'].max():.4f}"
    )

    print()

    print("Risk levels:")

    print(
        result_df[
            "ml_risk_level"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print(
        "Top 15 current ML-risk files:"
    )

    display_columns = [
        "file_path",
        "ml_defect_risk",
        "ml_risk_level",
        "prior_commit_count",
        "prior_contributor_count",
        "prior_bus_factor",
        "days_since_last_change",
    ]

    print(
        result_df[
            display_columns
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print()

    print("=" * 70)
    print(
        "CURRENT ML DEFECT RISK COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()