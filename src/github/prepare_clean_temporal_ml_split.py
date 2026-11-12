from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_clean_temporal_ml_dataset.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("CODEGUARD AI - CLEAN TEMPORAL ML SPLIT")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING CLEAN TEMPORAL DATASET")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

df["commit_date"] = pd.to_datetime(
    df["commit_date"],
    errors="coerce",
    utc=True,
)

df = df.sort_values(
    ["commit_seq", "commit_date", "commit_sha", "file_path"]
).reset_index(drop=True)

print(f"Total rows : {len(df)}")
print(f"Unique commits : {df['commit_sha'].nunique()}")
print(f"Unique files : {df['file_path'].nunique()}")
print(f"Positive rows : {(df['target'] == 1).sum()}")
print(f"Negative rows : {(df['target'] == 0).sum()}")


# ============================================================
# VALIDATION
# ============================================================

if df["commit_date"].isna().any():
    raise ValueError("Dataset contains invalid commit dates.")

if df["commit_seq"].isna().any():
    raise ValueError("Dataset contains missing commit sequence values.")

if df.duplicated(
    subset=["commit_sha", "file_path"]
).any():
    raise ValueError(
        "Dataset contains duplicate commit/file events."
    )


# ============================================================
# SPLIT BY UNIQUE COMMITS
# ============================================================

print("\n" + "=" * 70)
print("CREATING CHRONOLOGICAL COMMIT SPLIT")
print("=" * 70)

commit_table = (
    df[
        [
            "commit_sha",
            "commit_seq",
            "commit_date",
        ]
    ]
    .drop_duplicates("commit_sha")
    .sort_values(
        ["commit_seq", "commit_date", "commit_sha"]
    )
    .reset_index(drop=True)
)

n_commits = len(commit_table)

train_commit_count = int(
    n_commits * TRAIN_RATIO
)

validation_commit_count = int(
    n_commits * VALIDATION_RATIO
)

test_commit_count = (
    n_commits
    - train_commit_count
    - validation_commit_count
)

print(f"Total commits      : {n_commits}")
print(f"Train commits      : {train_commit_count}")
print(f"Validation commits : {validation_commit_count}")
print(f"Test commits       : {test_commit_count}")


# ============================================================
# COMMIT BOUNDARIES
# ============================================================

train_commits = set(
    commit_table.iloc[
        :train_commit_count
    ]["commit_sha"]
)

validation_commits = set(
    commit_table.iloc[
        train_commit_count:
        train_commit_count + validation_commit_count
    ]["commit_sha"]
)

test_commits = set(
    commit_table.iloc[
        train_commit_count + validation_commit_count:
    ]["commit_sha"]
)


# ============================================================
# CREATE SPLITS
# ============================================================

train = df[
    df["commit_sha"].isin(train_commits)
].copy()

validation = df[
    df["commit_sha"].isin(validation_commits)
].copy()

test = df[
    df["commit_sha"].isin(test_commits)
].copy()


# ============================================================
# SORT
# ============================================================

train = train.sort_values(
    ["commit_seq", "commit_date", "commit_sha", "file_path"]
).reset_index(drop=True)

validation = validation.sort_values(
    ["commit_seq", "commit_date", "commit_sha", "file_path"]
).reset_index(drop=True)

test = test.sort_values(
    ["commit_seq", "commit_date", "commit_sha", "file_path"]
).reset_index(drop=True)


# ============================================================
# SPLIT SUMMARY FUNCTION
# ============================================================

def print_split_summary(name, data):

    print("\n" + "-" * 70)
    print(name)

    print(f"Rows           : {len(data)}")
    print(
        f"Unique commits : "
        f"{data['commit_sha'].nunique()}"
    )
    print(
        f"Unique files   : "
        f"{data['file_path'].nunique()}"
    )

    positive = int(
        (data["target"] == 1).sum()
    )

    negative = int(
        (data["target"] == 0).sum()
    )

    print(f"Positive       : {positive}")
    print(f"Negative       : {negative}")

    if len(data) > 0:
        print(
            f"Positive rate  : "
            f"{positive / len(data):.4f}"
        )

        print(
            f"Date range     : "
            f"{data['commit_date'].min()} "
            f"to "
            f"{data['commit_date'].max()}"
        )

        print(
            f"Commit seq     : "
            f"{data['commit_seq'].min()} "
            f"to "
            f"{data['commit_seq'].max()}"
        )


# ============================================================
# PRINT SUMMARIES
# ============================================================

print_split_summary("TRAIN", train)
print_split_summary("VALIDATION", validation)
print_split_summary("TEST", test)


# ============================================================
# TEMPORAL BOUNDARY VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING TEMPORAL BOUNDARIES")
print("=" * 70)

train_max_seq = train["commit_seq"].max()
validation_min_seq = validation["commit_seq"].min()
validation_max_seq = validation["commit_seq"].max()
test_min_seq = test["commit_seq"].min()

print(
    f"Train max seq      : {train_max_seq}"
)

print(
    f"Validation min seq : {validation_min_seq}"
)

print(
    f"Validation max seq : {validation_max_seq}"
)

print(
    f"Test min seq       : {test_min_seq}"
)

if not (
    train_max_seq < validation_min_seq
    and validation_max_seq < test_min_seq
):
    raise ValueError(
        "Temporal boundary validation failed."
    )

print("Temporal boundaries PASS")


# ============================================================
# COMMIT OVERLAP VALIDATION
# ============================================================

train_validation_overlap = (
    set(train["commit_sha"])
    & set(validation["commit_sha"])
)

train_test_overlap = (
    set(train["commit_sha"])
    & set(test["commit_sha"])
)

validation_test_overlap = (
    set(validation["commit_sha"])
    & set(test["commit_sha"])
)

print(
    f"Train/Validation commit overlap : "
    f"{len(train_validation_overlap)}"
)

print(
    f"Train/Test commit overlap       : "
    f"{len(train_test_overlap)}"
)

print(
    f"Validation/Test commit overlap  : "
    f"{len(validation_test_overlap)}"
)

if (
    train_validation_overlap
    or train_test_overlap
    or validation_test_overlap
):
    raise ValueError(
        "Commit overlap detected between temporal splits."
    )

print("Commit separation PASS")


# ============================================================
# FILE OVERLAP
# ============================================================

train_files = set(train["file_path"])
validation_files = set(validation["file_path"])
test_files = set(test["file_path"])

print("\nFile overlap:")
print(
    f"Train/Validation shared files : "
    f"{len(train_files & validation_files)}"
)

print(
    f"Train/Test shared files       : "
    f"{len(train_files & test_files)}"
)

print(
    f"Validation/Test shared files  : "
    f"{len(validation_files & test_files)}"
)

print(
    "\nNote: file overlap across temporal splits "
    "is allowed because the task predicts future "
    "changes to files using their prior history."
)


# ============================================================
# SAVE
# ============================================================

train_path = (
    OUTPUT_DIR
    / "requests_clean_temporal_train.csv"
)

validation_path = (
    OUTPUT_DIR
    / "requests_clean_temporal_validation.csv"
)

test_path = (
    OUTPUT_DIR
    / "requests_clean_temporal_test.csv"
)

train.to_csv(
    train_path,
    index=False
)

validation.to_csv(
    validation_path,
    index=False
)

test.to_csv(
    test_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("CLEAN TEMPORAL SPLIT CREATED SUCCESSFULLY")
print("=" * 70)

print(f"Train      : {train_path}")
print(f"Validation : {validation_path}")
print(f"Test       : {test_path}")

print("=" * 70)