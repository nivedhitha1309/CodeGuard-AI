from pathlib import Path
import re

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

TARGET_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_clean_defect_targets.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_clean_temporal_ml_dataset.csv"
)


# ============================================================
# TARGET DEFINITIONS
# ============================================================

POSITIVE_CATEGORIES = {
    "likely_defect_candidate",
    "regression_defect_candidate",
    "security_defect_candidate",
}

EXCLUDED_CATEGORIES = {
    "excluded_merge_commit",
    "excluded_documentation",
    "excluded_non_source",
    "excluded_ci_build",
    "excluded_test_only",
    "excluded_style_or_lint",
    "excluded_documentation_language",
    "excluded_dependency_maintenance",
    "excluded_typo_or_style",
    "excluded_refactor_or_cleanup",
    "excluded_typing_maintenance",
    "excluded_ci_language",
    "excluded_version_or_dependency_bump",
    "uncertain_fix",
    "test_only_defect_evidence",
    "test_only_regression_evidence",
    "test_only_security_evidence",
}


# ============================================================
# FEATURE DEFINITIONS
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
# HELPER FUNCTIONS
# ============================================================

def normalize_path(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().replace("\\", "/")

    while "//" in value:
        value = value.replace("//", "/")

    if value.startswith("./"):
        value = value[2:]

    return value


def is_production_python_file(path):
    path = normalize_path(path).lower()

    if not path.endswith(".py"):
        return False

    excluded_prefixes = (
        "tests/",
        "test/",
        ".github/",
        ".git/",
        "docs/",
        "doc/",
    )

    if path.startswith(excluded_prefixes):
        return False

    excluded_names = {
        "conftest.py",
        "setup.py",
    }

    if Path(path).name in excluded_names:
        return False

    return True


def is_merge_commit(subject):
    if pd.isna(subject):
        return False

    subject = str(subject).strip().lower()

    return (
        subject.startswith("merge ")
        or subject.startswith("merge:")
        or subject.startswith("merge/")
    )


def derive_commit_subject(message):
    if pd.isna(message):
        return ""

    message = str(message)

    return message.splitlines()[0].strip()


def normalized_entropy(counts):
    counts = np.asarray(counts, dtype=float)

    total = counts.sum()

    if total <= 0:
        return 0.0

    probabilities = counts[counts > 0] / total

    if len(probabilities) <= 1:
        return 0.0

    entropy = -np.sum(probabilities * np.log(probabilities))

    max_entropy = np.log(len(probabilities))

    if max_entropy <= 0:
        return 0.0

    return float(entropy / max_entropy)


def calculate_prior_features(history, target_seq, file_path):
    """
    Calculate all features using only history with:

        commit_seq < target_seq

    This prevents future information from entering the feature vector.
    """

    prior = history[
        (history["file_path"] == file_path)
        & (history["commit_seq"] < target_seq)
    ].copy()

    if prior.empty:
        return {
            "prior_commit_count": 0,
            "prior_additions": 0.0,
            "prior_deletions": 0.0,
            "prior_changes": 0.0,
            "mean_prior_changes": 0.0,
            "median_prior_changes": 0.0,
            "max_prior_changes": 0.0,
            "prior_churn_per_commit": 0.0,
            "file_age_days": 0.0,
            "days_since_last_change": 0.0,
            "prior_contributor_count": 0,
            "prior_top_contributor_share": 0.0,
            "prior_contribution_entropy": 0.0,
            "prior_bus_factor": 0.0,
            "rolling_30d_commits": 0,
            "rolling_30d_changes": 0.0,
            "rolling_30d_additions": 0.0,
            "rolling_30d_deletions": 0.0,
            "rolling_90d_commits": 0,
            "rolling_90d_changes": 0.0,
            "rolling_90d_additions": 0.0,
            "rolling_90d_deletions": 0.0,
            "rolling_180d_commits": 0,
            "rolling_180d_changes": 0.0,
            "rolling_180d_additions": 0.0,
            "rolling_180d_deletions": 0.0,
        }

    prior = prior.sort_values(["commit_date", "commit_sha"])

    target_date = history.loc[
        history["commit_seq"] == target_seq,
        "commit_date"
    ].iloc[0]

    # --------------------------------------------------------
    # Basic prior history
    # --------------------------------------------------------

    prior_commit_count = prior["commit_sha"].nunique()

    prior_additions = prior["additions"].sum()
    prior_deletions = prior["deletions"].sum()
    prior_changes = prior["changes"].sum()

    mean_prior_changes = prior["changes"].mean()
    median_prior_changes = prior["changes"].median()
    max_prior_changes = prior["changes"].max()

    prior_churn_per_commit = (
        prior_changes / prior_commit_count
        if prior_commit_count > 0
        else 0.0
    )

    # --------------------------------------------------------
    # File age / recency
    # --------------------------------------------------------

    first_change = prior["commit_date"].min()
    last_change = prior["commit_date"].max()

    file_age_days = max(
        0.0,
        (target_date - first_change).total_seconds() / 86400.0
    )

    days_since_last_change = max(
        0.0,
        (target_date - last_change).total_seconds() / 86400.0
    )

    # --------------------------------------------------------
    # Contributor concentration
    # --------------------------------------------------------

    contributor_counts = (
        prior.groupby("author")["commit_sha"]
        .nunique()
        .sort_values(ascending=False)
    )

    prior_contributor_count = len(contributor_counts)

    if prior_contributor_count > 0:
        top_share = (
            contributor_counts.iloc[0]
            / contributor_counts.sum()
        )

        entropy = normalized_entropy(contributor_counts.values)

        cumulative = contributor_counts.cumsum()

        bus_factor = int(
            np.searchsorted(
                cumulative.values,
                contributor_counts.sum() * 0.5
            ) + 1
        )

    else:
        top_share = 0.0
        entropy = 0.0
        bus_factor = 0

    # --------------------------------------------------------
    # Rolling activity
    # --------------------------------------------------------

    def rolling_features(days):
        cutoff = target_date - pd.Timedelta(days=days)

        window = prior[
            prior["commit_date"] >= cutoff
        ]

        return {
            "commits": window["commit_sha"].nunique(),
            "changes": window["changes"].sum(),
            "additions": window["additions"].sum(),
            "deletions": window["deletions"].sum(),
        }

    r30 = rolling_features(30)
    r90 = rolling_features(90)
    r180 = rolling_features(180)

    return {
        "prior_commit_count": int(prior_commit_count),
        "prior_additions": float(prior_additions),
        "prior_deletions": float(prior_deletions),
        "prior_changes": float(prior_changes),
        "mean_prior_changes": float(mean_prior_changes),
        "median_prior_changes": float(median_prior_changes),
        "max_prior_changes": float(max_prior_changes),
        "prior_churn_per_commit": float(prior_churn_per_commit),
        "file_age_days": float(file_age_days),
        "days_since_last_change": float(days_since_last_change),
        "prior_contributor_count": int(prior_contributor_count),
        "prior_top_contributor_share": float(top_share),
        "prior_contribution_entropy": float(entropy),
        "prior_bus_factor": int(bus_factor),
        "rolling_30d_commits": int(r30["commits"]),
        "rolling_30d_changes": float(r30["changes"]),
        "rolling_30d_additions": float(r30["additions"]),
        "rolling_30d_deletions": float(r30["deletions"]),
        "rolling_90d_commits": int(r90["commits"]),
        "rolling_90d_changes": float(r90["changes"]),
        "rolling_90d_additions": float(r90["additions"]),
        "rolling_90d_deletions": float(r90["deletions"]),
        "rolling_180d_commits": int(r180["commits"]),
        "rolling_180d_changes": float(r180["changes"]),
        "rolling_180d_additions": float(r180["additions"]),
        "rolling_180d_deletions": float(r180["deletions"]),
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("CODEGUARD AI - CLEAN SOURCE-CHANGE TEMPORAL ML DATASET")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING INPUT DATA")
print("=" * 70)

history = pd.read_csv(HISTORY_PATH)
targets = pd.read_csv(TARGET_PATH)

print(f"History rows : {len(history)}")
print(f"Target rows  : {len(targets)}")


# ============================================================
# VALIDATE HISTORY SCHEMA
# ============================================================

required_history_columns = [
    "commit_sha",
    "author",
    "commit_date",
    "file_path",
    "additions",
    "deletions",
    "changes",
    "commit_message",
]

missing_history = [
    col
    for col in required_history_columns
    if col not in history.columns
]

if missing_history:
    raise ValueError(
        f"History dataset is missing columns: {missing_history}"
    )


# ============================================================
# VALIDATE TARGET SCHEMA
# ============================================================

required_target_columns = [
    "commit_sha",
    "file_path",
    "clean_defect_category",
    "defect_target",
]

missing_target = [
    col
    for col in required_target_columns
    if col not in targets.columns
]

if missing_target:
    raise ValueError(
        f"Target dataset is missing columns: {missing_target}"
    )


# ============================================================
# PREPARE HISTORY
# ============================================================

print("\n" + "=" * 70)
print("PREPARING COMMIT HISTORY")
print("=" * 70)

history["commit_sha"] = history["commit_sha"].astype(str).str.strip()
history["file_path"] = history["file_path"].map(normalize_path)
history["author"] = history["author"].fillna("UNKNOWN").astype(str)

history["commit_date"] = pd.to_datetime(
    history["commit_date"],
    errors="coerce",
    utc=True,
)

for column in ["additions", "deletions", "changes"]:
    history[column] = pd.to_numeric(
        history[column],
        errors="coerce"
    ).fillna(0)

history["commit_message"] = (
    history["commit_message"]
    .fillna("")
    .astype(str)
)

history["commit_subject"] = history["commit_message"].apply(
    derive_commit_subject
)

history = history.dropna(
    subset=["commit_date"]
).copy()

before_duplicates = len(history)

history = history.drop_duplicates(
    subset=["commit_sha", "file_path"]
).copy()

duplicates_removed = before_duplicates - len(history)

print(f"Duplicate history rows removed : {duplicates_removed}")


# ============================================================
# CHRONOLOGICAL COMMIT ORDER
# ============================================================

print("\n" + "=" * 70)
print("CREATING CHRONOLOGICAL COMMIT ORDER")
print("=" * 70)

commit_order = (
    history[
        ["commit_sha", "commit_date"]
    ]
    .drop_duplicates()
    .sort_values(
        ["commit_date", "commit_sha"]
    )
    .reset_index(drop=True)
)

commit_order["commit_seq"] = np.arange(
    len(commit_order)
)

history = history.merge(
    commit_order,
    on=["commit_sha", "commit_date"],
    how="left",
    validate="many_to_one",
)

print(f"Unique commits : {len(commit_order)}")
print(
    "History date range : "
    f"{history['commit_date'].min()} to "
    f"{history['commit_date'].max()}"
)


# ============================================================
# CREATE COMMIT SUBJECT LOOKUP
# ============================================================

commit_subject_lookup = (
    history[
        ["commit_sha", "commit_subject"]
    ]
    .drop_duplicates("commit_sha")
    .set_index("commit_sha")["commit_subject"]
)


# ============================================================
# PREPARE TARGET DATA
# ============================================================

print("\n" + "=" * 70)
print("PREPARING CLEAN TARGET POPULATION")
print("=" * 70)

targets["commit_sha"] = (
    targets["commit_sha"]
    .astype(str)
    .str.strip()
)

targets["file_path"] = (
    targets["file_path"]
    .map(normalize_path)
)

targets["clean_defect_category"] = (
    targets["clean_defect_category"]
    .fillna("")
    .astype(str)
    .str.strip()
)

targets["defect_target"] = pd.to_numeric(
    targets["defect_target"],
    errors="coerce"
).fillna(0).astype(int)

# ------------------------------------------------------------
# IMPORTANT FIX:
# The target CSV does not contain commit_subject.
# Derive it from the original history using commit_sha.
# ------------------------------------------------------------

targets["commit_subject"] = (
    targets["commit_sha"]
    .map(commit_subject_lookup)
    .fillna("")
)

targets["commit_seq"] = (
    targets["commit_sha"]
    .map(
        commit_order.set_index("commit_sha")["commit_seq"]
    )
)

targets["commit_date"] = (
    targets["commit_sha"]
    .map(
        commit_order.set_index("commit_sha")["commit_date"]
    )
)

print(
    "Targets with matching history commit : "
    f"{targets['commit_seq'].notna().sum()} / {len(targets)}"
)

targets = targets[
    targets["commit_seq"].notna()
].copy()

targets["is_merge_commit"] = targets[
    "commit_subject"
].apply(is_merge_commit)


# ============================================================
# SELECT POSITIVE AND NEGATIVE POPULATIONS
# ============================================================

positive_targets = targets[
    targets["clean_defect_category"].isin(
        POSITIVE_CATEGORIES
    )
    & (targets["defect_target"] == 1)
].copy()

negative_targets = targets[
    (~targets["clean_defect_category"].isin(
        POSITIVE_CATEGORIES
    ))
    & (~targets["clean_defect_category"].isin(
        EXCLUDED_CATEGORIES
    ))
    & (targets["defect_target"] == 0)
].copy()

print("\nTarget population before source filtering:")
print(f"Positive target rows : {len(positive_targets)}")
print(f"Negative target rows : {len(negative_targets)}")


# ============================================================
# REMOVE MERGE COMMITS
# ============================================================

positive_targets = positive_targets[
    ~positive_targets["is_merge_commit"]
].copy()

negative_targets = negative_targets[
    ~negative_targets["is_merge_commit"]
].copy()


# ============================================================
# PRODUCTION PYTHON FILTER
# ============================================================

positive_targets = positive_targets[
    positive_targets["file_path"].apply(
        is_production_python_file
    )
].copy()

negative_targets = negative_targets[
    negative_targets["file_path"].apply(
        is_production_python_file
    )
].copy()


print("\nAfter merge/source filtering:")
print(f"Positive rows : {len(positive_targets)}")
print(f"Negative rows : {len(negative_targets)}")


# ============================================================
# COMBINE TARGET EVENTS
# ============================================================

positive_targets["target"] = 1
negative_targets["target"] = 0

events = pd.concat(
    [
        positive_targets,
        negative_targets,
    ],
    ignore_index=True,
)

events = events[
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

events = events.drop_duplicates(
    subset=["commit_sha", "file_path"]
).copy()

events = events.sort_values(
    ["commit_seq", "file_path"]
).reset_index(drop=True)


# ============================================================
# DATASET OVERVIEW
# ============================================================

print("\n" + "=" * 70)
print("CLEAN DATASET OVERVIEW")
print("=" * 70)

print(f"Rows              : {len(events)}")
print(
    f"Unique commits     : {events['commit_sha'].nunique()}"
)
print(
    f"Unique files       : {events['file_path'].nunique()}"
)
print(
    f"Positive rows      : {(events['target'] == 1).sum()}"
)
print(
    f"Negative rows      : {(events['target'] == 0).sum()}"
)

print("\nTarget categories:")

category_counts = (
    events["clean_defect_category"]
    .value_counts()
)

for category, count in category_counts.items():
    print(f"  {category:<40} {count}")


# ============================================================
# TEMPORAL FEATURE GENERATION
# ============================================================

print("\n" + "=" * 70)
print("GENERATING LEAKAGE-SAFE TEMPORAL FEATURES")
print("=" * 70)

feature_rows = []

for index, row in events.iterrows():

    if index % 100 == 0:
        print(
            f"Processing event {index + 1} / {len(events)}"
        )

    features = calculate_prior_features(
        history=history,
        target_seq=int(row["commit_seq"]),
        file_path=row["file_path"],
    )

    output_row = {
        "commit_sha": row["commit_sha"],
        "commit_date": row["commit_date"],
        "commit_seq": row["commit_seq"],
        "file_path": row["file_path"],
        "commit_subject": row["commit_subject"],
        "clean_defect_category": row[
            "clean_defect_category"
        ],
        "target": row["target"],
    }

    output_row.update(features)

    feature_rows.append(output_row)


dataset = pd.DataFrame(feature_rows)


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

for column in FEATURE_COLUMNS:
    dataset[column] = pd.to_numeric(
        dataset[column],
        errors="coerce"
    )

dataset[FEATURE_COLUMNS] = (
    dataset[FEATURE_COLUMNS]
    .replace([np.inf, -np.inf], np.nan)
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING DATASET")
print("=" * 70)

missing_values = dataset[
    FEATURE_COLUMNS
].isna().sum().sum()

print(f"Missing feature values : {missing_values}")

duplicate_events = dataset.duplicated(
    subset=["commit_sha", "file_path"]
).sum()

print(
    f"Duplicate commit/file events : "
    f"{duplicate_events}"
)


# ------------------------------------------------------------
# Chronological validation
# ------------------------------------------------------------

chronological = dataset["commit_seq"].is_monotonic_increasing

print(
    f"Chronological order PASS : "
    f"{chronological}"
)


# ------------------------------------------------------------
# Leakage validation
# ------------------------------------------------------------

leakage_failures = 0

history_commit_lookup = (
    history.groupby("file_path")["commit_seq"]
    .apply(set)
    .to_dict()
)

for _, row in dataset.iterrows():

    file_path = row["file_path"]
    target_seq = int(row["commit_seq"])

    prior_commits = {
        seq
        for seq in history_commit_lookup.get(
            file_path,
            set()
        )
        if seq < target_seq
    }

    expected_prior_count = len(prior_commits)

    if int(row["prior_commit_count"]) != expected_prior_count:
        leakage_failures += 1


print(
    f"Leakage validation failures : "
    f"{leakage_failures}"
)


# ============================================================
# TARGET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TARGET SUMMARY")
print("=" * 70)

target_counts = (
    dataset["target"]
    .value_counts()
    .sort_index()
)

for target_value, count in target_counts.items():
    print(
        f"Target {target_value} : "
        f"{count}"
    )

positive_count = int(
    (dataset["target"] == 1).sum()
)

negative_count = int(
    (dataset["target"] == 0).sum()
)

if len(dataset) > 0:
    positive_rate = positive_count / len(dataset)
else:
    positive_rate = 0.0

print(
    f"Positive rate : "
    f"{positive_rate:.4f}"
)


# ============================================================
# FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE MEANS BY TARGET")
print("=" * 70)

feature_means = (
    dataset
    .groupby("target")[FEATURE_COLUMNS]
    .mean()
    .T
)

print(feature_means.to_string())


# ============================================================
# DATE RANGE
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET RANGE")
print("=" * 70)

print(
    f"Start : {dataset['commit_date'].min()}"
)

print(
    f"End   : {dataset['commit_date'].max()}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

if missing_values != 0:
    raise ValueError(
        "Dataset contains missing feature values."
    )

if duplicate_events != 0:
    raise ValueError(
        "Dataset contains duplicate commit/file events."
    )

if not chronological:
    raise ValueError(
        "Dataset is not chronologically ordered."
    )

if leakage_failures != 0:
    raise ValueError(
        "Leakage validation failed."
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

dataset.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print(
    f"Output : {OUTPUT_PATH}"
)

print(
    f"Rows   : {len(dataset)}"
)

print(
    f"Columns: {len(dataset.columns)}"
)

print("=" * 70)