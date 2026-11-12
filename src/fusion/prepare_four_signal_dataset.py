from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CODEGUARD AI - FOUR-SIGNAL FUSION DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

THREE_SIGNAL_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "fusion"
    / "requests_three_signal.csv"
)

ML_RISK_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_current_ml_defect_risk.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "fusion"
    / "requests_four_signal_dataset.csv"
)


def normalize_file_path(series):
    return (
        series.astype(str)
        .str.replace("\\", "/", regex=False)
        .str.strip()
    )


def main():

    print("=" * 70)
    print("CODEGUARD AI - FOUR-SIGNAL FUSION DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load three-signal dataset
    # --------------------------------------------------------

    if not THREE_SIGNAL_PATH.exists():
        raise FileNotFoundError(
            f"Three-signal dataset not found:\n{THREE_SIGNAL_PATH}"
        )

    three = pd.read_csv(THREE_SIGNAL_PATH)

    print(f"\nThree-signal rows loaded : {len(three)}")

    three["file_path"] = normalize_file_path(
        three["file_path"]
    )

    # --------------------------------------------------------
    # 2. Display actual columns
    # --------------------------------------------------------

    print("\nThree-signal columns:")

    for column in three.columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # 3. Validate actual signal columns
    #
    # Actual dataset columns:
    #
    #   refined_bus_risk
    #   satd_score
    #   code_risk_for_fusion
    #
    # These are the three signals we want.
    # --------------------------------------------------------

    required_columns = [
        "file_path",
        "refined_bus_risk",
        "satd_score",
        "code_risk_for_fusion",
    ]

    missing = [
        column
        for column in required_columns
        if column not in three.columns
    ]

    if missing:
        raise ValueError(
            "Three-signal dataset is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    # --------------------------------------------------------
    # 4. Validate file uniqueness
    # --------------------------------------------------------

    duplicate_files = (
        three["file_path"]
        .duplicated()
        .sum()
    )

    if duplicate_files > 0:
        raise ValueError(
            f"Three-signal dataset contains "
            f"{duplicate_files} duplicate file_path rows."
        )

    print(
        f"Duplicate three-signal files: {duplicate_files}"
    )

    # --------------------------------------------------------
    # 5. Load current ML risk
    # --------------------------------------------------------

    if not ML_RISK_PATH.exists():
        raise FileNotFoundError(
            f"Current ML risk file not found:\n{ML_RISK_PATH}"
        )

    ml = pd.read_csv(ML_RISK_PATH)

    print(
        f"Current ML-risk rows loaded : {len(ml)}"
    )

    if "file_path" not in ml.columns:
        raise ValueError(
            "Current ML-risk dataset is missing 'file_path'."
        )

    if "ml_defect_risk" not in ml.columns:
        raise ValueError(
            "Current ML-risk dataset is missing "
            "'ml_defect_risk'."
        )

    ml["file_path"] = normalize_file_path(
        ml["file_path"]
    )

    # --------------------------------------------------------
    # 6. Validate ML uniqueness
    # --------------------------------------------------------

    ml_duplicate_files = (
        ml["file_path"]
        .duplicated()
        .sum()
    )

    if ml_duplicate_files > 0:
        raise ValueError(
            f"Current ML-risk dataset contains "
            f"{ml_duplicate_files} duplicate file_path rows."
        )

    print(
        f"Duplicate ML-risk files       : "
        f"{ml_duplicate_files}"
    )

    # --------------------------------------------------------
    # 7. Keep required ML columns
    # --------------------------------------------------------

    ml_merge = ml[
        [
            "file_path",
            "ml_defect_risk",
            "ml_risk_level",
        ]
    ].copy()

    ml_merge = ml_merge.rename(
        columns={
            "ml_defect_risk":
                "ml_defect_risk_score"
        }
    )

    # --------------------------------------------------------
    # 8. Merge ML signal
    #
    # Keep all 284 files.
    #
    # Files without ML risk remain NaN.
    # They are NOT assigned zero.
    # --------------------------------------------------------

    four = three.merge(
        ml_merge,
        on="file_path",
        how="left",
        validate="one_to_one",
    )

    print(
        f"\nFour-signal rows after merge : "
        f"{len(four)}"
    )

    # --------------------------------------------------------
    # 9. Create explicit signal availability flags
    # --------------------------------------------------------

    four["bus_signal_available"] = (
        four["refined_bus_risk"].notna()
    )

    four["satd_signal_available"] = (
        four["satd_score"].notna()
    )

    four["code_signal_available"] = (
        four["code_risk_for_fusion"].notna()
    )

    four["ml_signal_available"] = (
        four["ml_defect_risk_score"].notna()
    )

    # --------------------------------------------------------
    # 10. Four-signal completeness
    # --------------------------------------------------------

    availability_columns = [
        "bus_signal_available",
        "satd_signal_available",
        "code_signal_available",
        "ml_signal_available",
    ]

    four["four_signal_available_count"] = (
        four[availability_columns]
        .sum(axis=1)
        .astype(int)
    )

    four["four_signal_completeness"] = (
        four["four_signal_available_count"] / 4.0
    )

    four["four_signal_complete"] = (
        four["four_signal_available_count"] == 4
    )

    # --------------------------------------------------------
    # 11. Signal coverage
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SIGNAL COVERAGE")
    print("=" * 70)

    print(
        f"Bus signal available : "
        f"{four['bus_signal_available'].sum()} / "
        f"{len(four)}"
    )

    print(
        f"SATD signal available: "
        f"{four['satd_signal_available'].sum()} / "
        f"{len(four)}"
    )

    print(
        f"Code signal available: "
        f"{four['code_signal_available'].sum()} / "
        f"{len(four)}"
    )

    print(
        f"ML signal available  : "
        f"{four['ml_signal_available'].sum()} / "
        f"{len(four)}"
    )

    print(
        f"Four-signal complete : "
        f"{four['four_signal_complete'].sum()} / "
        f"{len(four)}"
    )

    # --------------------------------------------------------
    # 12. Completeness distribution
    # --------------------------------------------------------

    print("\nFour-signal completeness distribution:")

    completeness_counts = (
        four["four_signal_completeness"]
        .value_counts()
        .sort_index()
    )

    for completeness, count in completeness_counts.items():

        print(
            f"  {completeness:.2f} : "
            f"{count} files"
        )

    # --------------------------------------------------------
    # 13. Current ML signal summary
    # --------------------------------------------------------

    available_ml = four.loc[
        four["ml_signal_available"],
        "ml_defect_risk_score",
    ]

    if len(available_ml) > 0:

        print("\n" + "=" * 70)
        print("CURRENT ML SIGNAL SUMMARY")
        print("=" * 70)

        print(
            f"Files with ML risk : "
            f"{len(available_ml)}"
        )

        print(
            f"Average ML risk    : "
            f"{available_ml.mean():.4f}"
        )

        print(
            f"Minimum ML risk    : "
            f"{available_ml.min():.4f}"
        )

        print(
            f"Maximum ML risk    : "
            f"{available_ml.max():.4f}"
        )

        print("\nML risk levels:")

        level_counts = (
            four.loc[
                four["ml_signal_available"],
                "ml_risk_level",
            ]
            .value_counts()
        )

        for level, count in level_counts.items():

            print(
                f"  {level:<8} {count}"
            )

    # --------------------------------------------------------
    # 14. Complete four-signal subset
    # --------------------------------------------------------

    complete = four[
        four["four_signal_complete"]
    ].copy()

    print("\n" + "=" * 70)
    print("FOUR-SIGNAL COMPLETE SUBSET")
    print("=" * 70)

    print(
        f"Complete files: {len(complete)}"
    )

    signal_columns = [
        "refined_bus_risk",
        "satd_score",
        "code_risk_for_fusion",
        "ml_defect_risk_score",
    ]

    if len(complete) > 0:

        print("\nSignal statistics:")

        for column in signal_columns:

            values = complete[column]

            print(
                f"{column:<26} "
                f"mean={values.mean():.4f} "
                f"min={values.min():.4f} "
                f"max={values.max():.4f}"
            )

        print("\nSignal correlations:")

        correlation = complete[
            signal_columns
        ].corr()

        print(
            correlation.to_string(
                float_format=lambda x: f"{x:.4f}"
            )
        )

    else:

        print(
            "\nNo files have all four signals available."
        )

    # --------------------------------------------------------
    # 15. Top current ML-risk files
    # --------------------------------------------------------

    if len(available_ml) > 0:

        print("\n" + "=" * 70)
        print("TOP CURRENT ML-RISK FILES")
        print("=" * 70)

        top_ml = (
            four[
                four["ml_signal_available"]
            ]
            .sort_values(
                "ml_defect_risk_score",
                ascending=False,
            )
            [
                [
                    "file_path",
                    "ml_defect_risk_score",
                    "ml_risk_level",
                    "refined_bus_risk",
                    "satd_score",
                    "code_risk_for_fusion",
                    "four_signal_complete",
                ]
            ]
            .head(15)
        )

        print(
            top_ml.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

    # --------------------------------------------------------
    # 16. Methodological checks
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("METHODOLOGICAL CHECKS")
    print("=" * 70)

    print(
        "ML predictions used here : "
        "current file-level deployment risk"
    )

    print(
        "Test-event predictions   : "
        "NOT merged into this dataset"
    )

    print(
        "Unavailable signals      : "
        "kept as NaN, not replaced with zero"
    )

    print(
        "Fusion score calculation : "
        "NOT performed yet"
    )

    print(
        "Fusion weights           : "
        "NOT learned yet"
    )

    # --------------------------------------------------------
    # 17. Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    four.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"Saved four-signal dataset:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"Final shape: {four.shape}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()