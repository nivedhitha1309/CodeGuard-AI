import os
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

HISTORY_PATH = Path(
    "data/raw/github/requests_commit_file_history.csv"
)

TARGET_PATH = Path(
    "data/processed/github/requests_clean_defect_targets.csv"
)

OUTPUT_PATH = Path(
    "data/processed/github/requests_temporal_defect_dataset.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Only production Python source files are used for the
# temporal ML dataset.
#
# Tests, documentation, CI/configuration and packaging files
# are excluded.

EXCLUDED_PREFIXES = (
    "tests/",
    "docs/",
    ".github/",
)

EXCLUDED_FILES = {
    "setup.py",
    "setup.cfg",
    "tox.ini",
    "conftest.py",
}

# Recent-history windows.
WINDOWS_DAYS = [30, 90, 180]


# ============================================================
# HELPERS
# ============================================================

def normalize_path(path):
    """Normalize repository paths."""
    if pd.isna(path):
        return ""

    path = str(path).replace("\\", "/").strip()

    while path.startswith("./"):
        path = path[2:]

    return path


def is_production_python_file(path):
    """
    Return True for production Python source files.

    Excludes:
    - tests
    - documentation
    - GitHub/CI files
    - packaging/configuration files
    """
    path = normalize_path(path)

    if not path.lower().endswith(".py"):
        return False

    lower = path.lower()

    if any(
        lower.startswith(prefix)
        for prefix in EXCLUDED_PREFIXES
    ):
        return False

    if lower in {
        filename.lower()
        for filename in EXCLUDED_FILES
    }:
        return False

    return True


def safe_numeric(series):
    """Convert a pandas Series to numeric safely."""
    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)


def concentration_metrics(history):
    """
    Calculate contributor concentration using only
    historical human contributions.

    Returns:
        contributor_count
        top_contributor_share
        normalized_entropy
        bus_factor
    """

    if history.empty:
        return 0, 0.0, 0.0, 0

    counts = (
        history
        .groupby("author_key")
        .size()
    )

    counts = counts[counts > 0]

    if counts.empty:
        return 0, 0.0, 0.0, 0

    total = counts.sum()

    shares = counts / total

    contributor_count = len(counts)

    top_share = float(
        shares.max()
    )

    entropy = float(
        -(shares * np.log(shares)).sum()
    )

    if contributor_count <= 1:
        normalized_entropy = 0.0
    else:
        normalized_entropy = (
            entropy / np.log(contributor_count)
        )

    # Approximate bus factor:
    # minimum contributors required to
    # reach 50% of historical contributions.
    sorted_shares = shares.sort_values(
        ascending=False
    )

    cumulative = sorted_shares.cumsum()

    bus_factor = int(
        (cumulative < 0.5).sum() + 1
    )

    return (
        contributor_count,
        top_share,
        normalized_entropy,
        bus_factor,
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CODEGUARD AI - TEMPORAL DEFECT DATASET")
print("=" * 70)

print("\nLoading history...")
history = pd.read_csv(
    HISTORY_PATH
)

print("Loading cleaned targets...")
targets = pd.read_csv(
    TARGET_PATH
)

print(
    f"History rows : {len(history)}"
)

print(
    f"Target rows  : {len(targets)}"
)


# ============================================================
# VALIDATION
# ============================================================

required_history_columns = {
    "commit_sha",
    "author",
    "author_email",
    "commit_date",
    "file_path",
    "additions",
    "deletions",
    "changes",
}

required_target_columns = {
    "commit_sha",
    "commit_date",
    "file_path",
    "defect_target",
    "clean_defect_category",
}

missing_history = (
    required_history_columns
    - set(history.columns)
)

missing_targets = (
    required_target_columns
    - set(targets.columns)
)

if missing_history:
    raise ValueError(
        f"Missing history columns: "
        f"{sorted(missing_history)}"
    )

if missing_targets:
    raise ValueError(
        f"Missing target columns: "
        f"{sorted(missing_targets)}"
    )


# ============================================================
# NORMALIZE PATHS AND DATES
# ============================================================

history["file_path"] = (
    history["file_path"]
    .map(normalize_path)
)

targets["file_path"] = (
    targets["file_path"]
    .map(normalize_path)
)

history["commit_date"] = pd.to_datetime(
    history["commit_date"],
    errors="coerce",
    utc=True,
)

targets["commit_date"] = pd.to_datetime(
    targets["commit_date"],
    errors="coerce",
    utc=True,
)

history = history.dropna(
    subset=[
        "commit_sha",
        "commit_date",
        "file_path",
    ]
).copy()

targets = targets.dropna(
    subset=[
        "commit_sha",
        "commit_date",
        "file_path",
    ]
).copy()


# ============================================================
# AUTHOR NORMALIZATION
# ============================================================

history["author_key"] = (
    history["author"]
    .fillna("unknown")
    .astype(str)
    .str.strip()
    .str.lower()
)

history["is_bot"] = (
    history["author_key"]
    .str.endswith("[bot]")
)


# ============================================================
# NUMERIC COLUMNS
# ============================================================

for column in [
    "additions",
    "deletions",
    "changes",
]:
    history[column] = safe_numeric(
        history[column]
    )


# ============================================================
# PRODUCTION PYTHON HISTORY
# ============================================================

history = history[
    history["file_path"].map(
        is_production_python_file
    )
].copy()

targets = targets[
    targets["file_path"].map(
        is_production_python_file
    )
].copy()

print(
    f"\nProduction Python history rows : "
    f"{len(history)}"
)

print(
    f"Production Python target rows  : "
    f"{len(targets)}"
)


# ============================================================
# REMOVE DUPLICATE FILE-HISTORY ROWS
# ============================================================

history = history.drop_duplicates(
    subset=[
        "commit_sha",
        "file_path",
    ]
).copy()

targets = targets.drop_duplicates(
    subset=[
        "commit_sha",
        "file_path",
    ]
).copy()


# ============================================================
# CREATE DETERMINISTIC COMMIT ORDER
# ============================================================

print(
    "\nCreating chronological commit order..."
)

commit_table = (
    history[
        [
            "commit_sha",
            "commit_date",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "commit_date",
            "commit_sha",
        ]
    )
    .reset_index(drop=True)
)

commit_table["commit_seq"] = np.arange(
    len(commit_table)
)

commit_seq_map = dict(
    zip(
        commit_table["commit_sha"],
        commit_table["commit_seq"],
    )
)

history["commit_seq"] = (
    history["commit_sha"]
    .map(commit_seq_map)
)

targets["commit_seq"] = (
    targets["commit_sha"]
    .map(commit_seq_map)
)


# ============================================================
# KEEP TARGET EVENTS THAT EXIST IN HISTORY
# ============================================================

before_target_count = len(targets)

targets = targets[
    targets["commit_seq"].notna()
].copy()

print(
    f"Targets with valid historical sequence : "
    f"{len(targets)}"
)

print(
    f"Targets removed due to missing sequence : "
    f"{before_target_count - len(targets)}"
)


# ============================================================
# HUMAN HISTORY
# ============================================================

# IMPORTANT:
# This is created AFTER commit_seq has been added.
#
# This fixes the KeyError from the previous version.

history_human = history[
    ~history["is_bot"]
].copy()


# ============================================================
# CACHE FILE HISTORY
# ============================================================

print(
    "\nPreparing file-history caches..."
)

file_history_groups = {
    path: group.sort_values(
        "commit_seq"
    ).copy()
    for path, group in history.groupby(
        "file_path",
        sort=False,
    )
}

human_history_groups = {
    path: group.sort_values(
        "commit_seq"
    ).copy()
    for path, group in history_human.groupby(
        "file_path",
        sort=False,
    )
}


# ============================================================
# BUILD TEMPORAL FEATURES
# ============================================================

print(
    "\nBuilding leakage-safe temporal features..."
)

records = []

sorted_targets = targets.sort_values(
    [
        "commit_seq",
        "file_path",
    ]
)


for _, target in sorted_targets.iterrows():

    file_path = target["file_path"]

    target_seq = int(
        target["commit_seq"]
    )

    target_date = target["commit_date"]


    # --------------------------------------------------------
    # FILE HISTORY
    # --------------------------------------------------------

    file_hist = file_history_groups.get(
        file_path,
        pd.DataFrame(),
    )

    if file_hist.empty:
        continue


    # --------------------------------------------------------
    # STRICT TEMPORAL FILTER
    # --------------------------------------------------------
    #
    # ONLY commits strictly before the target commit
    # are allowed to contribute features.
    #
    # The target commit itself is NEVER included.
    # --------------------------------------------------------

    prior_file = file_hist[
        file_hist["commit_seq"] < target_seq
    ].copy()


    # --------------------------------------------------------
    # HUMAN CONTRIBUTOR HISTORY
    # --------------------------------------------------------

    human_file = human_history_groups.get(
        file_path,
        pd.DataFrame(),
    )

    if not human_file.empty:

        human_prior = human_file[
            human_file["commit_seq"] < target_seq
        ].copy()

    else:

        human_prior = pd.DataFrame()


    # --------------------------------------------------------
    # BASIC PRIOR HISTORY
    # --------------------------------------------------------

    prior_commit_count = len(
        prior_file
    )

    if prior_commit_count:

        prior_additions = float(
            prior_file["additions"].sum()
        )

        prior_deletions = float(
            prior_file["deletions"].sum()
        )

        prior_changes = float(
            prior_file["changes"].sum()
        )

        mean_prior_changes = float(
            prior_file["changes"].mean()
        )

        median_prior_changes = float(
            prior_file["changes"].median()
        )

        max_prior_changes = float(
            prior_file["changes"].max()
        )

    else:

        prior_additions = 0.0
        prior_deletions = 0.0
        prior_changes = 0.0
        mean_prior_changes = 0.0
        median_prior_changes = 0.0
        max_prior_changes = 0.0


    # --------------------------------------------------------
    # CHANGE INTENSITY
    # --------------------------------------------------------

    if prior_commit_count:

        prior_churn_per_commit = (
            prior_changes
            / prior_commit_count
        )

    else:

        prior_churn_per_commit = 0.0


    # --------------------------------------------------------
    # FILE AGE
    # --------------------------------------------------------

    if prior_commit_count:

        first_seen = prior_file[
            "commit_date"
        ].min()

        last_changed = prior_file[
            "commit_date"
        ].max()

        file_age_days = max(
            0.0,
            (
                target_date - first_seen
            ).total_seconds()
            / 86400.0,
        )

        days_since_last_change = max(
            0.0,
            (
                target_date - last_changed
            ).total_seconds()
            / 86400.0,
        )

    else:

        file_age_days = 0.0
        days_since_last_change = 0.0


    # --------------------------------------------------------
    # KNOWLEDGE CONCENTRATION
    # --------------------------------------------------------

    (
        prior_contributor_count,
        prior_top_contributor_share,
        prior_contribution_entropy,
        prior_bus_factor,
    ) = concentration_metrics(
        human_prior
    )


    # --------------------------------------------------------
    # RECENT ACTIVITY WINDOWS
    # --------------------------------------------------------

    window_features = {}

    for days in WINDOWS_DAYS:

        cutoff = (
            target_date
            - pd.Timedelta(
                days=days
            )
        )

        recent = prior_file[
            prior_file["commit_date"]
            >= cutoff
        ]

        window_features[
            f"commits_last_{days}d"
        ] = int(
            len(recent)
        )

        window_features[
            f"changes_last_{days}d"
        ] = float(
            recent["changes"].sum()
        ) if not recent.empty else 0.0

        window_features[
            f"additions_last_{days}d"
        ] = float(
            recent["additions"].sum()
        ) if not recent.empty else 0.0

        window_features[
            f"deletions_last_{days}d"
        ] = float(
            recent["deletions"].sum()
        ) if not recent.empty else 0.0


    # --------------------------------------------------------
    # PRIOR DEFECT HISTORY
    # --------------------------------------------------------
    #
    # Only previous defect targets for this file are counted.
    #
    # The current target is excluded.
    # --------------------------------------------------------

    prior_target_rows = targets[
        (
            targets["file_path"]
            == file_path
        )
        &
        (
            targets["commit_seq"]
            < target_seq
        )
        &
        (
            targets["defect_target"]
            == 1
        )
    ]

    prior_defect_count = len(
        prior_target_rows
    )

    if prior_commit_count:

        prior_defect_rate = (
            prior_defect_count
            / prior_commit_count
        )

    else:

        prior_defect_rate = 0.0


    # --------------------------------------------------------
    # CREATE RECORD
    # --------------------------------------------------------

    record = {

        # Event identity
        "commit_sha": target[
            "commit_sha"
        ],

        "commit_date": target[
            "commit_date"
        ],

        "file_path": file_path,

        # Target
        "defect_target": int(
            target["defect_target"]
        ),

        "clean_defect_category": target[
            "clean_defect_category"
        ],

        # Temporal sequence
        "commit_seq": target_seq,

        # Prior history
        "prior_commit_count":
            prior_commit_count,

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
            prior_churn_per_commit,

        # File age/activity
        "file_age_days":
            file_age_days,

        "days_since_last_change":
            days_since_last_change,

        # Knowledge concentration
        "prior_contributor_count":
            prior_contributor_count,

        "prior_top_contributor_share":
            prior_top_contributor_share,

        "prior_contribution_entropy":
            prior_contribution_entropy,

        "prior_bus_factor":
            prior_bus_factor,

        # Previous defects
        "prior_defect_count":
            prior_defect_count,

        "prior_defect_rate":
            prior_defect_rate,
    }

    record.update(
        window_features
    )

    records.append(
        record
    )


# ============================================================
# CREATE DATASET
# ============================================================

dataset = pd.DataFrame(
    records
)

if dataset.empty:
    raise RuntimeError(
        "No temporal records were created."
    )


# ============================================================
# NUMERIC CLEANUP
# ============================================================

numeric_columns = [
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
]

for days in WINDOWS_DAYS:

    numeric_columns.extend(
        [
            f"commits_last_{days}d",
            f"changes_last_{days}d",
            f"additions_last_{days}d",
            f"deletions_last_{days}d",
        ]
    )


for column in numeric_columns:

    if column in dataset.columns:

        dataset[column] = pd.to_numeric(
            dataset[column],
            errors="coerce",
        ).fillna(0)


# ============================================================
# CLIP NORMALIZED FEATURES
# ============================================================

dataset[
    "prior_top_contributor_share"
] = dataset[
    "prior_top_contributor_share"
].clip(
    lower=0,
    upper=1,
)

dataset[
    "prior_contribution_entropy"
] = dataset[
    "prior_contribution_entropy"
].clip(
    lower=0,
    upper=1,
)

dataset[
    "prior_defect_rate"
] = dataset[
    "prior_defect_rate"
].clip(
    lower=0,
    upper=1,
)


# ============================================================
# SORT
# ============================================================

dataset = dataset.sort_values(
    [
        "commit_seq",
        "file_path",
    ]
).reset_index(
    drop=True
)


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TEMPORAL DATASET VALIDATION"
)

print(
    "=" * 70
)

print(
    f"Rows                    : "
    f"{len(dataset)}"
)

print(
    f"Unique commits          : "
    f"{dataset['commit_sha'].nunique()}"
)

print(
    f"Unique files            : "
    f"{dataset['file_path'].nunique()}"
)

print(
    f"Defect targets          : "
    f"{int(dataset['defect_target'].sum())}"
)

print(
    f"Non-defect events       : "
    f"{int((dataset['defect_target'] == 0).sum())}"
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print(
    "\nTARGET DISTRIBUTION"
)

print(
    dataset[
        "clean_defect_category"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# MISSING VALUE CHECK
# ============================================================

print(
    "\nFEATURE MISSING VALUES"
)

missing = dataset.isna().sum()

missing = missing[
    missing > 0
]

if missing.empty:

    print(
        "No missing values."
    )

else:

    print(
        missing.to_string()
    )


# ============================================================
# LEAKAGE CHECK
# ============================================================

print(
    "\nLEAKAGE CHECK"
)

violations = []


for _, row in dataset.iterrows():

    target_seq = int(
        row["commit_seq"]
    )

    file_path = row[
        "file_path"
    ]

    stored_prior_count = int(
        row["prior_commit_count"]
    )

    file_hist = file_history_groups.get(
        file_path,
        pd.DataFrame(),
    )

    actual_prior_count = len(
        file_hist[
            file_hist["commit_seq"]
            < target_seq
        ]
    )

    if (
        stored_prior_count
        != actual_prior_count
    ):

        violations.append(
            {
                "commit_sha":
                    row["commit_sha"],

                "file_path":
                    file_path,

                "stored_prior_count":
                    stored_prior_count,

                "actual_prior_count":
                    actual_prior_count,
            }
        )


if violations:

    print(
        f"WARNING: {len(violations)} "
        f"temporal violations detected."
    )

    print(
        pd.DataFrame(
            violations
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Temporal leakage validation failed."
    )

else:

    print(
        "PASS - all temporal features use "
        "strictly prior file history."
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

dataset.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "OUTPUT"
)

print(
    "=" * 70
)

print(
    f"Saved file: {OUTPUT_PATH}"
)

print(
    "\nTemporal ML dataset successfully created."
)

print(
    "All historical features are calculated "
    "strictly before each target event."
)