import os

import pandas as pd


# ===============================================================
# Configuration
# ===============================================================

BUS_FACTOR_FILE = (
    "data/processed/bus_factor/"
    "requests_bus_factor_refined.csv"
)

SATD_FILE = (
    "data/processed/satd/"
    "requests_satd_aggregated.csv"
)

OUTPUT_FILE = (
    "data/processed/fusion/"
    "requests_bus_satd_merged.csv"
)


# ===============================================================
# Helper
# ===============================================================

def normalize_file_path(series):
    """
    Normalize repository file paths so that the two datasets
    can be merged reliably.
    """

    return (
        series
        .astype(str)
        .str.replace(
            "\\",
            "/",
            regex=False,
        )
        .str.strip()
    )


# ===============================================================
# Main
# ===============================================================

def main():

    print("=" * 70)
    print("CodeGuard AI - Bus Factor + SATD Fusion")
    print("=" * 70)

    # -----------------------------------------------------------
    # Validate input files
    # -----------------------------------------------------------

    if not os.path.exists(
        BUS_FACTOR_FILE
    ):

        raise FileNotFoundError(
            f"Bus-factor file not found: "
            f"{BUS_FACTOR_FILE}"
        )

    if not os.path.exists(
        SATD_FILE
    ):

        raise FileNotFoundError(
            f"SATD file not found: "
            f"{SATD_FILE}"
        )

    # -----------------------------------------------------------
    # Load datasets
    # -----------------------------------------------------------

    bus_df = pd.read_csv(
        BUS_FACTOR_FILE
    )

    satd_df = pd.read_csv(
        SATD_FILE
    )

    if bus_df.empty:

        raise ValueError(
            "Bus-factor dataset is empty."
        )

    if satd_df.empty:

        print(
            "WARNING: SATD dataset is empty. "
            "All files will receive SATD = 0."
        )

    # -----------------------------------------------------------
    # Validate required columns
    # -----------------------------------------------------------

    bus_required = {
        "file_path",
        "total_commits",
        "contributor_count",
        "top_contributor_share",
        "contribution_entropy",
        "bus_factor",
        "knowledge_concentration",
        "activity_score",
        "refined_bus_risk",
    }

    missing_bus = (
        bus_required
        - set(bus_df.columns)
    )

    if missing_bus:

        raise ValueError(
            "Missing bus-factor columns: "
            + ", ".join(
                sorted(missing_bus)
            )
        )

    satd_required = {
        "file_path",
        "satd_count",
        "satd_present",
    }

    missing_satd = (
        satd_required
        - set(satd_df.columns)
    )

    if missing_satd:

        raise ValueError(
            "Missing SATD columns: "
            + ", ".join(
                sorted(missing_satd)
            )
        )

    # -----------------------------------------------------------
    # Normalize paths
    # -----------------------------------------------------------

    bus_df["file_path"] = normalize_file_path(
        bus_df["file_path"]
    )

    satd_df["file_path"] = normalize_file_path(
        satd_df["file_path"]
    )

    # -----------------------------------------------------------
    # Defensive duplicate check
    # -----------------------------------------------------------

    if bus_df["file_path"].duplicated().any():

        duplicate_count = (
            bus_df["file_path"]
            .duplicated()
            .sum()
        )

        raise ValueError(
            "Bus-factor dataset contains "
            f"{duplicate_count} duplicate file paths."
        )

    if satd_df["file_path"].duplicated().any():

        duplicate_count = (
            satd_df["file_path"]
            .duplicated()
            .sum()
        )

        raise ValueError(
            "SATD dataset contains "
            f"{duplicate_count} duplicate file paths."
        )

    # -----------------------------------------------------------
    # Select SATD columns
    # -----------------------------------------------------------

    satd_columns = [
        "file_path",
        "satd_count",
        "satd_present",
        "todo_count",
        "fixme_count",
        "hack_count",
        "xxx_count",
        "workaround_count",
        "technical_debt_count",
        "techdebt_count",
        "debt_count",
        "refactor_count",
        "revisit_count",
        "cleanup_count",
        "first_satd_line",
        "last_satd_line",
    ]

    # Some columns may not exist if an older aggregation file
    # was used. Create them as zero/NA so the merge remains
    # robust.

    for column in satd_columns:

        if column == "file_path":
            continue

        if column not in satd_df.columns:

            if column in {
                "first_satd_line",
                "last_satd_line",
            }:

                satd_df[column] = pd.NA

            elif column == "satd_present":

                satd_df[column] = False

            else:

                satd_df[column] = 0

    satd_df = satd_df[
        satd_columns
    ].copy()

    # -----------------------------------------------------------
    # Merge
    # -----------------------------------------------------------

    merged = bus_df.merge(
        satd_df,
        on="file_path",
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------------
    # Fill SATD values for files without SATD
    # -----------------------------------------------------------

    count_columns = [
        "satd_count",
        "todo_count",
        "fixme_count",
        "hack_count",
        "xxx_count",
        "workaround_count",
        "technical_debt_count",
        "techdebt_count",
        "debt_count",
        "refactor_count",
        "revisit_count",
        "cleanup_count",
    ]

    for column in count_columns:

        merged[column] = (
            pd.to_numeric(
                merged[column],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

    merged["satd_present"] = (
        merged["satd_present"]
        .fillna(False)
        .astype(bool)
    )

    # -----------------------------------------------------------
    # SATD coverage
    # -----------------------------------------------------------

    merged["satd_coverage"] = (
        merged["satd_present"]
        .astype(int)
    )

    # -----------------------------------------------------------
    # Create a simple normalized SATD score
    # -----------------------------------------------------------

    # We do NOT yet call this the final CodeGuard SATD risk.
    #
    # The score is simply a normalized representation of the
    # observed SATD count within this dataset.
    #
    # This will be revisited when LOC/file-size information is
    # incorporated.

    max_satd = (
        merged["satd_count"].max()
    )

    if max_satd > 0:

        merged["satd_score"] = (
            merged["satd_count"]
            / max_satd
        )

    else:

        merged["satd_score"] = 0.0

    merged["satd_score"] = (
        merged["satd_score"]
        .clip(0, 1)
        .round(4)
    )

    # -----------------------------------------------------------
    # Check merge coverage
    # -----------------------------------------------------------

    files_with_satd = int(
        merged["satd_present"].sum()
    )

    files_without_satd = (
        len(merged)
        - files_with_satd
    )

    # -----------------------------------------------------------
    # Output ordering
    # -----------------------------------------------------------

    preferred_columns = [
        "file_path",

        # Bus-factor / knowledge metrics
        "total_commits",
        "contributor_count",
        "total_contributions",
        "top_contributor_share",
        "contribution_entropy",
        "bus_factor",
        "knowledge_concentration",
        "activity_score",
        "refined_bus_risk",
        "bus_risk_level",
        "file_category",
        "knowledge_relevant",

        # SATD metrics
        "satd_count",
        "satd_present",
        "satd_coverage",
        "satd_score",
        "todo_count",
        "fixme_count",
        "hack_count",
        "xxx_count",
        "workaround_count",
        "technical_debt_count",
        "techdebt_count",
        "debt_count",
        "refactor_count",
        "revisit_count",
        "cleanup_count",
        "first_satd_line",
        "last_satd_line",
    ]

    # Keep only columns that actually exist.
    preferred_columns = [
        column
        for column in preferred_columns
        if column in merged.columns
    ]

    remaining_columns = [
        column
        for column in merged.columns
        if column not in preferred_columns
    ]

    merged = merged[
        preferred_columns
        + remaining_columns
    ]

    # -----------------------------------------------------------
    # Save
    # -----------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    # -----------------------------------------------------------
    # Summary
    # -----------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUS FACTOR + SATD MERGE SUMMARY")
    print("=" * 70)

    print(
        f"Bus-factor files       : "
        f"{len(bus_df)}"
    )

    print(
        f"SATD-affected files    : "
        f"{len(satd_df)}"
    )

    print(
        f"Merged files           : "
        f"{len(merged)}"
    )

    print(
        f"Files with SATD        : "
        f"{files_with_satd}"
    )

    print(
        f"Files without SATD     : "
        f"{files_without_satd}"
    )

    print(
        f"Total SATD occurrences : "
        f"{merged['satd_count'].sum()}"
    )

    print(
        f"Average SATD count     : "
        f"{merged['satd_count'].mean():.4f}"
    )

    print(
        f"Maximum SATD count     : "
        f"{merged['satd_count'].max()}"
    )

    print(
        f"Average SATD score     : "
        f"{merged['satd_score'].mean():.4f}"
    )

    print("\nFILES WITH SATD")
    print("-" * 70)

    satd_files = (
        merged[
            merged["satd_present"]
        ][
            [
                "file_path",
                "satd_count",
                "satd_score",
                "refined_bus_risk",
                "knowledge_concentration",
            ]
        ]
        .sort_values(
            by=[
                "satd_count",
                "refined_bus_risk",
            ],
            ascending=False,
        )
    )

    if not satd_files.empty:

        print(
            satd_files.to_string(
                index=False
            )
        )

    else:

        print(
            "No files with SATD."
        )

    print("\n" + "=" * 70)

    print(
        f"Output saved to: {OUTPUT_FILE}"
    )

    print("=" * 70)


# ===============================================================
# Entry point
# ===============================================================

if __name__ == "__main__":
    main()