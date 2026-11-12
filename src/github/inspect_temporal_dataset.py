from pathlib import Path

import pandas as pd


# ============================================================
# CODEGUARD AI - TEMPORAL DATASET INSPECTION
# ============================================================

INPUT_FILE = Path(
    "data/processed/github/requests_temporal_defect_dataset.csv"
)


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():

    print("=" * 70)
    print("CODEGUARD AI - TEMPORAL DATASET INSPECTION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Temporal dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print_section("BASIC DATASET INFORMATION")

    print(f"Rows              : {len(df)}")
    print(f"Columns           : {len(df.columns)}")
    print(f"Unique commits    : {df['commit_sha'].nunique()}")
    print(f"Unique files      : {df['file_path'].nunique()}")

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    df["commit_date"] = pd.to_datetime(
        df["commit_date"],
        utc=True,
        errors="coerce"
    )

    print_section("TEMPORAL RANGE")

    print(f"First commit date : {df['commit_date'].min()}")
    print(f"Last commit date  : {df['commit_date'].max()}")
    print(
        f"Total time span   : "
        f"{(df['commit_date'].max() - df['commit_date'].min()).days} days"
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print_section("TARGET DISTRIBUTION")

    target_counts = df["defect_target"].value_counts().sort_index()

    print(target_counts.to_string())

    target_percent = (
        df["defect_target"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nTarget percentages:")
    print(target_percent.to_string())

    # --------------------------------------------------------
    # Category distribution
    # --------------------------------------------------------

    print_section("CLEAN DEFECT CATEGORY DISTRIBUTION")

    print(
        df["clean_defect_category"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Positive target categories
    # --------------------------------------------------------

    print_section("POSITIVE TARGET CATEGORIES")

    positive = df[df["defect_target"] == 1]

    if positive.empty:
        print("No positive target rows found.")
    else:
        print(
            positive["clean_defect_category"]
            .value_counts()
            .to_string()
        )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    metadata_columns = [
        "commit_sha",
        "commit_date",
        "file_path",
        "defect_target",
        "clean_defect_category",
        "commit_seq",
    ]

    feature_columns = [
        col
        for col in df.columns
        if col not in metadata_columns
    ]

    print_section("FEATURE COLUMNS")

    print(f"Feature count : {len(feature_columns)}")

    for i, col in enumerate(feature_columns, start=1):
        print(f"{i:2d}. {col}")

    # --------------------------------------------------------
    # Feature statistics
    # --------------------------------------------------------

    print_section("FEATURE STATISTICS")

    stats = df[feature_columns].describe().T

    stats["missing"] = df[feature_columns].isna().sum()

    print(
        stats[
            [
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
                "missing",
            ]
        ].round(4).to_string()
    )

    # --------------------------------------------------------
    # Positive vs negative feature means
    # --------------------------------------------------------

    print_section("FEATURE MEANS: NON-DEFECT VS DEFECT")

    grouped_means = (
        df.groupby("defect_target")[feature_columns]
        .mean()
        .T
    )

    grouped_means.columns = [
        "non_defect_mean" if col == 0 else "defect_mean"
        for col in grouped_means.columns
    ]

    if "non_defect_mean" in grouped_means.columns:
        grouped_means["difference"] = (
            grouped_means["defect_mean"]
            - grouped_means["non_defect_mean"]
        )

    print(
        grouped_means
        .sort_values(
            "difference",
            ascending=False
        )
        .round(4)
        .to_string()
    )

    # --------------------------------------------------------
    # Correlation with target
    # --------------------------------------------------------

    print_section("FEATURE / TARGET CORRELATION")

    correlations = (
        df[feature_columns + ["defect_target"]]
        .corr(numeric_only=True)["defect_target"]
        .drop("defect_target")
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False
        )
    )

    print(
        correlations
        .round(4)
        .to_string()
    )

    # --------------------------------------------------------
    # Temporal target distribution
    # --------------------------------------------------------

    print_section("DEFECT TARGETS BY TIME PERIOD")

    df["year"] = df["commit_date"].dt.year

    yearly = (
        df.groupby("year")
        .agg(
            total_events=("defect_target", "size"),
            defect_events=("defect_target", "sum"),
        )
    )

    yearly["defect_rate"] = (
        yearly["defect_events"]
        / yearly["total_events"]
    )

    print(
        yearly
        .round(4)
        .to_string()
    )

    # --------------------------------------------------------
    # Target distribution by commit sequence
    # --------------------------------------------------------

    print_section("TARGET DISTRIBUTION BY COMMIT SEQUENCE")

    seq_bins = pd.qcut(
        df["commit_seq"],
        q=5,
        duplicates="drop"
    )

    sequence_summary = (
        df.groupby(
            seq_bins,
            observed=True
        )
        .agg(
            events=("defect_target", "size"),
            defect_events=("defect_target", "sum"),
            min_commit_seq=("commit_seq", "min"),
            max_commit_seq=("commit_seq", "max"),
        )
    )

    sequence_summary["defect_rate"] = (
        sequence_summary["defect_events"]
        / sequence_summary["events"]
    )

    print(
        sequence_summary
        .round(4)
        .to_string()
    )

    # --------------------------------------------------------
    # Per-file event counts
    # --------------------------------------------------------

    print_section("FILE ACTIVITY DISTRIBUTION")

    file_counts = (
        df.groupby("file_path")
        .size()
        .sort_values(ascending=False)
    )

    print(f"Files with 1 event : {(file_counts == 1).sum()}")
    print(f"Files with 2+ events: {(file_counts >= 2).sum()}")
    print(f"Maximum events/file: {file_counts.max()}")

    print("\nTop 20 files by number of events:")
    print(file_counts.head(20).to_string())

    # --------------------------------------------------------
    # Defect events per file
    # --------------------------------------------------------

    print_section("DEFECT EVENTS BY FILE")

    defect_by_file = (
        positive.groupby("file_path")
        .size()
        .sort_values(ascending=False)
    )

    print(
        defect_by_file.head(20).to_string()
    )

    # --------------------------------------------------------
    # Prior defect feature inspection
    # --------------------------------------------------------

    print_section("PRIOR DEFECT FEATURE CHECK")

    print(
        "Rows where prior_defect_count > 0 :",
        (df["prior_defect_count"] > 0).sum()
    )

    print(
        "Maximum prior_defect_count        :",
        df["prior_defect_count"].max()
    )

    print(
        "Rows where prior_defect_rate > 0 :",
        (df["prior_defect_rate"] > 0).sum()
    )

    print(
        "Maximum prior_defect_rate         :",
        df["prior_defect_rate"].max()
    )

    # --------------------------------------------------------
    # Leakage sanity checks
    # --------------------------------------------------------

    print_section("LEAKAGE SANITY CHECKS")

    checks = {}

    checks["commit_seq_monotonic"] = (
        df["commit_seq"].is_monotonic_increasing
    )

    checks["no_missing_features"] = (
        df[feature_columns].isna().sum().sum() == 0
    )

    checks["no_duplicate_commit_file"] = (
        df.duplicated(
            ["commit_sha", "file_path"]
        ).sum() == 0
    )

    checks["non_negative_prior_commit_count"] = (
        (df["prior_commit_count"] >= 0).all()
    )

    checks["non_negative_prior_defect_count"] = (
        (df["prior_defect_count"] >= 0).all()
    )

    checks["defect_rate_valid"] = (
        (
            (df["prior_defect_rate"] >= 0)
            &
            (df["prior_defect_rate"] <= 1)
        ).all()
    )

    checks["top_contributor_share_valid"] = (
        (
            (df["prior_top_contributor_share"] >= 0)
            &
            (df["prior_top_contributor_share"] <= 1)
        ).all()
    )

    for name, result in checks.items():
        print(
            f"{name:40s}: "
            f"{'PASS' if result else 'FAIL'}"
        )

    all_pass = all(checks.values())

    print(
        "\nOverall sanity check:",
        "PASS" if all_pass else "FAIL"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print_section("INSPECTION COMPLETE")

    print(
        "The temporal dataset is ready for the next "
        "dataset-design stage."
    )


if __name__ == "__main__":
    main()