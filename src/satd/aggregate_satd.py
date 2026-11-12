import os

import pandas as pd


# ===============================================================
# Configuration
# ===============================================================

INPUT_FILE = (
    "data/raw/satd/"
    "requests_satd_occurrences.csv"
)

OUTPUT_FILE = (
    "data/processed/satd/"
    "requests_satd_aggregated.csv"
)


# ===============================================================
# Main
# ===============================================================

def main():

    print("=" * 70)
    print("CodeGuard AI - SATD Aggregation")
    print("=" * 70)

    # -----------------------------------------------------------
    # Validate input
    # -----------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    if df.empty:

        raise ValueError(
            "SATD occurrence CSV is empty."
        )

    required_columns = {
        "file_path",
        "line_number",
        "satd_type",
        "matched_text",
        "comment_text",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    # -----------------------------------------------------------
    # Clean data
    # -----------------------------------------------------------

    df["file_path"] = (
        df["file_path"]
        .astype(str)
        .str.replace(
            "\\",
            "/",
            regex=False,
        )
        .str.strip()
    )

    df["satd_type"] = (
        df["satd_type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Defensive duplicate removal.
    df = (
        df.drop_duplicates(
            subset=[
                "file_path",
                "line_number",
                "satd_type",
                "matched_text",
                "comment_text",
            ]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------
    # Aggregate SATD by file
    # -----------------------------------------------------------

    grouped = (
        df.groupby("file_path")
        .agg(
            satd_count=(
                "satd_type",
                "count",
            ),
            todo_count=(
                "satd_type",
                lambda x: (
                    x == "TODO"
                ).sum(),
            ),
            fixme_count=(
                "satd_type",
                lambda x: (
                    x == "FIXME"
                ).sum(),
            ),
            hack_count=(
                "satd_type",
                lambda x: (
                    x == "HACK"
                ).sum(),
            ),
            xxx_count=(
                "satd_type",
                lambda x: (
                    x == "XXX"
                ).sum(),
            ),
            workaround_count=(
                "satd_type",
                lambda x: (
                    x == "WORKAROUND"
                ).sum(),
            ),
            technical_debt_count=(
                "satd_type",
                lambda x: (
                    x == "TECHNICAL_DEBT"
                ).sum(),
            ),
            techdebt_count=(
                "satd_type",
                lambda x: (
                    x == "TECHDEBT"
                ).sum(),
            ),
            debt_count=(
                "satd_type",
                lambda x: (
                    x == "DEBT"
                ).sum(),
            ),
            refactor_count=(
                "satd_type",
                lambda x: (
                    x == "REFACTOR"
                ).sum(),
            ),
            revisit_count=(
                "satd_type",
                lambda x: (
                    x == "REVISIT"
                ).sum(),
            ),
            cleanup_count=(
                "satd_type",
                lambda x: (
                    x == "CLEANUP"
                ).sum(),
            ),
            first_satd_line=(
                "line_number",
                "min",
            ),
            last_satd_line=(
                "line_number",
                "max",
            ),
        )
        .reset_index()
    )

    # -----------------------------------------------------------
    # SATD presence
    # -----------------------------------------------------------

    grouped["satd_present"] = (
        grouped["satd_count"] > 0
    )

    # -----------------------------------------------------------
    # SATD density
    # -----------------------------------------------------------

    # Since we do not yet have total source-code line counts,
    # this is intentionally NOT called "SATD density".
    #
    # satd_count is kept as the primary raw SATD measure.
    #
    # A normalized SATD score can be created later when we have
    # file size / LOC information.

    # -----------------------------------------------------------
    # Reorder columns
    # -----------------------------------------------------------

    output_columns = [
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

    result = grouped[
        output_columns
    ].copy()

    # -----------------------------------------------------------
    # Sort
    # -----------------------------------------------------------

    result = (
        result
        .sort_values(
            by=[
                "satd_count",
                "file_path",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------
    # Save
    # -----------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    # -----------------------------------------------------------
    # Summary
    # -----------------------------------------------------------

    print("\n" + "=" * 70)
    print("SATD AGGREGATION SUMMARY")
    print("=" * 70)

    print(
        f"Input SATD occurrences : "
        f"{len(df)}"
    )

    print(
        f"Files with SATD        : "
        f"{len(result)}"
    )

    print(
        f"Total SATD count       : "
        f"{result['satd_count'].sum()}"
    )

    print("\nSATD TYPE TOTALS")
    print("-" * 70)

    type_columns = [
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

    type_totals = (
        result[type_columns]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        type_totals.to_string()
    )

    print("\nFILES BY SATD COUNT")
    print("-" * 70)

    print(
        result[
            [
                "file_path",
                "satd_count",
                "todo_count",
                "xxx_count",
                "satd_present",
            ]
        ]
        .to_string(index=False)
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