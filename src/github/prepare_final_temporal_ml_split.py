import os

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "data/processed/github/"
    "requests_final_temporal_ml_dataset.csv"
)

OUTPUT_DIR = (
    "data/processed/github/"
)

TRAIN_FILE = (
    OUTPUT_DIR
    + "requests_final_temporal_train.csv"
)

VAL_FILE = (
    OUTPUT_DIR
    + "requests_final_temporal_val.csv"
)

TEST_FILE = (
    OUTPUT_DIR
    + "requests_final_temporal_test.csv"
)


# ============================================================
# SPLIT FUNCTION
# ============================================================

def assign_split(sequence, train_end, val_end):

    if sequence <= train_end:
        return "train"

    if sequence <= val_end:
        return "validation"

    return "test"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CODEGUARD AI - FINAL TEMPORAL ML SPLIT"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    df["commit_date"] = pd.to_datetime(
        df["commit_date"],
        utc=True,
        errors="coerce",
    )

    df = (
        df
        .sort_values(
            [
                "temporal_commit_sequence",
                "commit_date",
                "commit_sha",
                "file_path",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"\nRows: {len(df):,}"
    )

    print(
        f"Commits: "
        f"{df['commit_sha'].nunique():,}"
    )

    print(
        f"Files: "
        f"{df['file_path'].nunique():,}"
    )

    print(
        f"Positive rows: "
        f"{int(df['ml_defect_target'].sum()):,}"
    )

    print(
        f"Negative rows: "
        f"{int((df['ml_defect_target'] == 0).sum()):,}"
    )

    # --------------------------------------------------------
    # UNIQUE COMMIT ORDER
    # --------------------------------------------------------

    commits = (
        df[
            [
                "commit_sha",
                "commit_date",
                "temporal_commit_sequence",
            ]
        ]
        .drop_duplicates(
            subset=[
                "commit_sha",
            ]
        )
        .sort_values(
            [
                "temporal_commit_sequence",
                "commit_date",
                "commit_sha",
            ]
        )
        .reset_index(drop=True)
    )

    n_commits = len(commits)

    print(
        f"\nUnique chronological commits: "
        f"{n_commits:,}"
    )

    # --------------------------------------------------------
    # 70 / 15 / 15 CHRONOLOGICAL SPLIT
    #
    # Split boundaries are determined using commit count,
    # not individual rows.
    # --------------------------------------------------------

    train_end_index = (
        int(n_commits * 0.70) - 1
    )

    val_end_index = (
        int(n_commits * 0.85) - 1
    )

    train_end_sequence = int(
        commits.iloc[
            train_end_index
        ][
            "temporal_commit_sequence"
        ]
    )

    val_end_sequence = int(
        commits.iloc[
            val_end_index
        ][
            "temporal_commit_sequence"
        ]
    )

    print(
        "\nChronological boundaries:"
    )

    print(
        f"Train ends at commit sequence: "
        f"{train_end_sequence}"
    )

    print(
        f"Validation ends at commit sequence: "
        f"{val_end_sequence}"
    )

    print(
        f"Test begins after sequence: "
        f"{val_end_sequence}"
    )

    # --------------------------------------------------------
    # ASSIGN SPLITS
    # --------------------------------------------------------

    df["split"] = (
        df[
            "temporal_commit_sequence"
        ]
        .apply(
            lambda x:
                assign_split(
                    x,
                    train_end_sequence,
                    val_end_sequence,
                )
        )
    )

    # --------------------------------------------------------
    # CREATE DATASETS
    # --------------------------------------------------------

    train = (
        df[
            df["split"] == "train"
        ]
        .copy()
    )

    validation = (
        df[
            df["split"] == "validation"
        ]
        .copy()
    )

    test = (
        df[
            df["split"] == "test"
        ]
        .copy()
    )

    # --------------------------------------------------------
    # SUMMARY FUNCTION
    # --------------------------------------------------------

    def print_summary(name, data):

        positives = int(
            data[
                "ml_defect_target"
            ].sum()
        )

        negatives = int(
            (
                data[
                    "ml_defect_target"
                ] == 0
            ).sum()
        )

        positive_rate = (
            positives / len(data)
            if len(data) > 0
            else 0.0
        )

        print(
            f"\n{name}"
        )

        print(
            f"Rows: {len(data):,}"
        )

        print(
            f"Commits: "
            f"{data['commit_sha'].nunique():,}"
        )

        print(
            f"Files: "
            f"{data['file_path'].nunique():,}"
        )

        print(
            f"Positive: {positives:,}"
        )

        print(
            f"Negative: {negatives:,}"
        )

        print(
            f"Positive rate: "
            f"{positive_rate:.4f}"
        )

        if len(data) > 0:

            print(
                f"Date range: "
                f"{data['commit_date'].min()} "
                f"to "
                f"{data['commit_date'].max()}"
            )

            print(
                f"Sequence range: "
                f"{data['temporal_commit_sequence'].min()} "
                f"to "
                f"{data['temporal_commit_sequence'].max()}"
            )

    # --------------------------------------------------------
    # PRINT SPLIT SUMMARIES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    print_summary(
        "TRAIN",
        train,
    )

    print_summary(
        "VALIDATION",
        validation,
    )

    print_summary(
        "TEST",
        test,
    )

    # --------------------------------------------------------
    # TEMPORAL BOUNDARY CHECK
    # --------------------------------------------------------

    temporal_pass = True

    if len(train) > 0 and len(validation) > 0:

        if (
            train["commit_date"].max()
            >= validation["commit_date"].min()
        ):
            temporal_pass = False

    if len(validation) > 0 and len(test) > 0:

        if (
            validation["commit_date"].max()
            >= test["commit_date"].min()
        ):
            temporal_pass = False

    print(
        "\nTemporal boundaries: "
        f"{'PASS' if temporal_pass else 'FAIL'}"
    )

    # --------------------------------------------------------
    # COMMIT OVERLAP CHECK
    # --------------------------------------------------------

    train_commits = set(
        train["commit_sha"]
    )

    val_commits = set(
        validation["commit_sha"]
    )

    test_commits = set(
        test["commit_sha"]
    )

    train_val_overlap = (
        train_commits
        & val_commits
    )

    train_test_overlap = (
        train_commits
        & test_commits
    )

    val_test_overlap = (
        val_commits
        & test_commits
    )

    print(
        f"Train/validation commit overlap: "
        f"{len(train_val_overlap)}"
    )

    print(
        f"Train/test commit overlap: "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Validation/test commit overlap: "
        f"{len(val_test_overlap)}"
    )

    commit_overlap_pass = (
        len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
    )

    print(
        "Commit overlap check: "
        f"{'PASS' if commit_overlap_pass else 'FAIL'}"
    )

    # --------------------------------------------------------
    # POSITIVE DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nPositive distribution:"
    )

    for name, data in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:

        counts = (
            data[
                data[
                    "ml_defect_target"
                ] == 1
            ][
                "clean_refined_label"
            ]
            .value_counts()
        )

        print(
            f"\n{name}:"
        )

        if len(counts) == 0:
            print("No positive events")
        else:
            print(counts)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    train.to_csv(
        TRAIN_FILE,
        index=False,
    )

    validation.to_csv(
        VAL_FILE,
        index=False,
    )

    test.to_csv(
        TEST_FILE,
        index=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FILES SAVED"
    )

    print(
        f"Train: {os.path.abspath(TRAIN_FILE)}"
    )

    print(
        f"Validation: "
        f"{os.path.abspath(VAL_FILE)}"
    )

    print(
        f"Test: {os.path.abspath(TEST_FILE)}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()