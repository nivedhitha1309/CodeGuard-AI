from pathlib import Path

import pandas as pd


# ============================================================
# CODEGUARD AI - TEMPORAL ML SPLIT PREPARATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/github/requests_temporal_defect_dataset.csv"
)

OUTPUT_DIR = Path(
    "data/processed/github"
)

TRAIN_FILE = OUTPUT_DIR / "requests_temporal_train.csv"
VALIDATION_FILE = OUTPUT_DIR / "requests_temporal_validation.csv"
TEST_FILE = OUTPUT_DIR / "requests_temporal_test.csv"


# Chronological proportions.
#
# 60% -> training
# 20% -> validation
# 20% -> test
#
# The split is performed using commit sequence rather than
# random sampling.
TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def describe_split(name, df):

    print(f"\n{name}")

    print("-" * 70)

    print(f"Rows             : {len(df)}")
    print(f"Unique commits   : {df['commit_sha'].nunique()}")
    print(f"Unique files     : {df['file_path'].nunique()}")

    print(
        f"Commit sequence  : "
        f"{df['commit_seq'].min()} -> {df['commit_seq'].max()}"
    )

    print(
        f"Date range       : "
        f"{df['commit_date'].min()} -> {df['commit_date'].max()}"
    )

    positive = int(df["defect_target"].sum())
    negative = int((df["defect_target"] == 0).sum())

    print(f"Defect events    : {positive}")
    print(f"Non-defect       : {negative}")

    if len(df) > 0:
        print(
            f"Defect rate      : "
            f"{positive / len(df):.4f}"
        )

    print("\nTarget distribution:")

    print(
        df["defect_target"]
        .value_counts()
        .sort_index()
        .to_string()
    )


def main():

    print("=" * 70)
    print("CODEGUARD AI - TEMPORAL ML SPLIT PREPARATION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print_section("INPUT DATASET")

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------------
    # Date conversion
    # --------------------------------------------------------

    df["commit_date"] = pd.to_datetime(
        df["commit_date"],
        utc=True,
        errors="coerce"
    )

    if df["commit_date"].isna().any():
        raise ValueError(
            "Invalid commit_date values detected."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        ["commit_seq", "commit_date", "commit_sha", "file_path"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Validate commit sequence
    # --------------------------------------------------------

    if not df["commit_seq"].is_monotonic_increasing:
        raise ValueError(
            "commit_seq is not monotonically increasing."
        )

    # --------------------------------------------------------
    # Calculate row boundaries
    # --------------------------------------------------------

    n = len(df)

    train_end = int(n * TRAIN_RATIO)

    validation_end = int(
        n * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    if train_end <= 0:
        raise ValueError("Training split is empty.")

    if validation_end <= train_end:
        raise ValueError("Validation split is empty.")

    if validation_end >= n:
        raise ValueError("Test split is empty.")

    # --------------------------------------------------------
    # Create chronological splits
    # --------------------------------------------------------

    train = df.iloc[:train_end].copy()

    validation = df.iloc[
        train_end:validation_end
    ].copy()

    test = df.iloc[
        validation_end:
    ].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        TRAIN_FILE,
        index=False
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False
    )

    test.to_csv(
        TEST_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Display split information
    # --------------------------------------------------------

    print_section("CHRONOLOGICAL SPLITS")

    describe_split(
        "TRAIN",
        train
    )

    describe_split(
        "VALIDATION",
        validation
    )

    describe_split(
        "TEST",
        test
    )

    # --------------------------------------------------------
    # Boundary information
    # --------------------------------------------------------

    print_section("SPLIT BOUNDARIES")

    print(
        "Train -> Validation boundary:"
    )

    print(
        f"Last train commit : "
        f"{train['commit_seq'].max()}"
    )

    print(
        f"First validation : "
        f"{validation['commit_seq'].min()}"
    )

    print(
        f"Train last date  : "
        f"{train['commit_date'].max()}"
    )

    print(
        f"Validation first : "
        f"{validation['commit_date'].min()}"
    )

    print()

    print(
        "Validation -> Test boundary:"
    )

    print(
        f"Last validation : "
        f"{validation['commit_seq'].max()}"
    )

    print(
        f"First test      : "
        f"{test['commit_seq'].min()}"
    )

    print(
        f"Validation last : "
        f"{validation['commit_date'].max()}"
    )

    print(
        f"Test first      : "
        f"{test['commit_date'].min()}"
    )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    print_section("TEMPORAL LEAKAGE CHECK")

    train_max = train["commit_seq"].max()
    validation_min = validation["commit_seq"].min()

    validation_max = validation["commit_seq"].max()
    test_min = test["commit_seq"].min()

    train_validation_ok = (
        train_max < validation_min
    )

    validation_test_ok = (
        validation_max < test_min
    )

    print(
        f"Train before validation : "
        f"{'PASS' if train_validation_ok else 'FAIL'}"
    )

    print(
        f"Validation before test  : "
        f"{'PASS' if validation_test_ok else 'FAIL'}"
    )

    # --------------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------------

    print_section("DUPLICATE CHECK")

    train_duplicates = train.duplicated(
        ["commit_sha", "file_path"]
    ).sum()

    validation_duplicates = validation.duplicated(
        ["commit_sha", "file_path"]
    ).sum()

    test_duplicates = test.duplicated(
        ["commit_sha", "file_path"]
    ).sum()

    print(
        f"Train duplicates      : {train_duplicates}"
    )

    print(
        f"Validation duplicates : {validation_duplicates}"
    )

    print(
        f"Test duplicates       : {test_duplicates}"
    )

    # --------------------------------------------------------
    # File overlap information
    #
    # Same files appearing in different time periods is
    # expected and useful for this temporal prediction task.
    # We report it rather than treating it as leakage.
    # --------------------------------------------------------

    print_section("FILE OVERLAP ACROSS TIME")

    train_files = set(train["file_path"])
    validation_files = set(validation["file_path"])
    test_files = set(test["file_path"])

    print(
        f"Train files      : {len(train_files)}"
    )

    print(
        f"Validation files : {len(validation_files)}"
    )

    print(
        f"Test files       : {len(test_files)}"
    )

    print(
        f"Train ∩ Validation : "
        f"{len(train_files & validation_files)}"
    )

    print(
        f"Validation ∩ Test  : "
        f"{len(validation_files & test_files)}"
    )

    print(
        f"Train ∩ Test       : "
        f"{len(train_files & test_files)}"
    )

    # --------------------------------------------------------
    # Positive event check
    # --------------------------------------------------------

    print_section("POSITIVE EVENT CHECK")

    for name, split in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:

        positives = split[
            split["defect_target"] == 1
        ]

        print(
            f"{name:12s}: "
            f"{len(positives)} positive events"
        )

    # --------------------------------------------------------
    # Category distribution
    # --------------------------------------------------------

    print_section("POSITIVE CATEGORY DISTRIBUTION")

    for name, split in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:

        positives = split[
            split["defect_target"] == 1
        ]

        print(f"\n{name}")

        if positives.empty:
            print("No positive events.")
        else:
            print(
                positives[
                    "clean_defect_category"
                ]
                .value_counts()
                .to_string()
            )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print_section("FINAL SPLIT VALIDATION")

    total_rows_match = (
        len(train)
        + len(validation)
        + len(test)
        == len(df)
    )

    no_overlap = (
        train.index.intersection(
            validation.index
        ).empty
        and train.index.intersection(
            test.index
        ).empty
        and validation.index.intersection(
            test.index
        ).empty
    )

    all_checks = [
        train_validation_ok,
        validation_test_ok,
        train_duplicates == 0,
        validation_duplicates == 0,
        test_duplicates == 0,
        total_rows_match,
        no_overlap,
    ]

    print(
        f"All checks passed : "
        f"{'YES' if all(all_checks) else 'NO'}"
    )

    print_section("OUTPUT")

    print(f"Train file      : {TRAIN_FILE}")
    print(f"Validation file : {VALIDATION_FILE}")
    print(f"Test file       : {TEST_FILE}")

    print(
        "\nTemporal ML split preparation complete."
    )


if __name__ == "__main__":
    main()