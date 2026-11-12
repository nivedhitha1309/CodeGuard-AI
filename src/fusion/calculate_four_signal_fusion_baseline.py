from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CODEGUARD AI - FOUR-SIGNAL FUSION BASELINE
# ============================================================
#
# Purpose:
#   Combine the four available risk signals into one unified
#   file-level risk score.
#
# Signals:
#   1. refined_bus_risk
#   2. satd_score
#   3. code_risk_for_fusion
#   4. ml_defect_risk_score
#
# Method:
#   Equal-weight baseline with availability-aware
#   renormalization.
#
# Important:
#   Missing signals are NOT treated as zero.
#   We average only over signals that are actually available.
#
# This is a transparent baseline, NOT a learned fusion model.
# ============================================================


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "fusion"
    / "requests_four_signal_dataset.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "fusion"
    / "requests_four_signal_fusion_baseline.csv"
)


# ------------------------------------------------------------
# Equal baseline weights
# ------------------------------------------------------------

WEIGHTS = {
    "refined_bus_risk": 0.25,
    "satd_score": 0.25,
    "code_risk_for_fusion": 0.25,
    "ml_defect_risk_score": 0.25,
}


def calculate_available_weighted_score(row):
    """
    Calculate the equal-weight fusion score using only signals
    that are actually available for this file.

    Missing signals do not contribute zero risk.
    Instead, their weights are removed and the remaining weights
    are renormalized.
    """

    available_signals = []

    for column, weight in WEIGHTS.items():

        value = row[column]

        if pd.notna(value):
            available_signals.append(
                (float(value), weight)
            )

    if not available_signals:
        return np.nan

    total_weight = sum(
        weight
        for _, weight in available_signals
    )

    weighted_sum = sum(
        value * weight
        for value, weight in available_signals
    )

    return weighted_sum / total_weight


def assign_risk_level(score):
    """
    Presentation bands for the unified risk score.

    These are baseline reporting bands, not calibrated
    probabilities and not empirically optimized thresholds.
    """

    if pd.isna(score):
        return "Unavailable"

    if score >= 0.75:
        return "Critical"

    if score >= 0.50:
        return "High"

    if score >= 0.25:
        return "Medium"

    return "Low"


def main():

    print("=" * 70)
    print("CODEGUARD AI - FOUR-SIGNAL FUSION BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Four-signal dataset not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(
        f"\nInput rows: {len(df)}"
    )

    # --------------------------------------------------------
    # 2. Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "file_path",
        "refined_bus_risk",
        "satd_score",
        "code_risk_for_fusion",
        "ml_defect_risk_score",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Input dataset is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    # --------------------------------------------------------
    # 3. Validate uniqueness
    # --------------------------------------------------------

    duplicate_files = (
        df["file_path"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate files: {duplicate_files}"
    )

    if duplicate_files > 0:
        raise ValueError(
            "Input dataset contains duplicate file_path values."
        )

    # --------------------------------------------------------
    # 4. Calculate signal availability
    # --------------------------------------------------------

    signal_columns = list(
        WEIGHTS.keys()
    )

    df["fusion_signal_count"] = (
        df[signal_columns]
        .notna()
        .sum(axis=1)
        .astype(int)
    )

    df["fusion_signal_completeness"] = (
        df["fusion_signal_count"] / 4.0
    )

    # --------------------------------------------------------
    # 5. Calculate unified risk
    # --------------------------------------------------------

    print(
        "\nCalculating availability-aware fusion..."
    )

    df["unified_risk"] = df.apply(
        calculate_available_weighted_score,
        axis=1,
    )

    # --------------------------------------------------------
    # 6. Assign risk levels
    # --------------------------------------------------------

    df["unified_risk_level"] = (
        df["unified_risk"]
        .apply(assign_risk_level)
    )

    # --------------------------------------------------------
    # 7. Record which signals contributed
    # --------------------------------------------------------

    def contributing_signals(row):

        names = []

        if pd.notna(row["refined_bus_risk"]):
            names.append("bus")

        if pd.notna(row["satd_score"]):
            names.append("satd")

        if pd.notna(row["code_risk_for_fusion"]):
            names.append("code")

        if pd.notna(row["ml_defect_risk_score"]):
            names.append("ml")

        return "+".join(names)

    df["signals_used"] = df.apply(
        contributing_signals,
        axis=1,
    )

    # --------------------------------------------------------
    # 8. Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FUSION SUMMARY")
    print("=" * 70)

    print(
        f"Files scored: "
        f"{df['unified_risk'].notna().sum()} / {len(df)}"
    )

    print(
        f"Average unified risk: "
        f"{df['unified_risk'].mean():.4f}"
    )

    print(
        f"Minimum unified risk: "
        f"{df['unified_risk'].min():.4f}"
    )

    print(
        f"Maximum unified risk: "
        f"{df['unified_risk'].max():.4f}"
    )

    # --------------------------------------------------------
    # 9. Signal-count distribution
    # --------------------------------------------------------

    print("\nSignal availability:")

    count_distribution = (
        df["fusion_signal_count"]
        .value_counts()
        .sort_index()
    )

    for count, number in count_distribution.items():

        print(
            f"  {count}/4 signals : "
            f"{number} files"
        )

    # --------------------------------------------------------
    # 10. Risk-level distribution
    # --------------------------------------------------------

    print("\nUnified risk levels:")

    risk_distribution = (
        df["unified_risk_level"]
        .value_counts()
        .reindex(
            [
                "Critical",
                "High",
                "Medium",
                "Low",
                "Unavailable",
            ],
            fill_value=0,
        )
    )

    for level, number in risk_distribution.items():

        print(
            f"  {level:<11} {number}"
        )

    # --------------------------------------------------------
    # 11. Compare complete vs incomplete files
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPLETE VS INCOMPLETE")
    print("=" * 70)

    complete = df[
        df["fusion_signal_count"] == 4
    ]

    incomplete = df[
        df["fusion_signal_count"] < 4
    ]

    print(
        f"Complete four-signal files   : "
        f"{len(complete)}"
    )

    print(
        f"Incomplete four-signal files : "
        f"{len(incomplete)}"
    )

    if len(complete) > 0:

        print(
            f"Complete subset mean risk    : "
            f"{complete['unified_risk'].mean():.4f}"
        )

        print(
            f"Complete subset max risk     : "
            f"{complete['unified_risk'].max():.4f}"
        )

    if len(incomplete) > 0:

        print(
            f"Incomplete subset mean risk  : "
            f"{incomplete['unified_risk'].mean():.4f}"
        )

        print(
            f"Incomplete subset max risk   : "
            f"{incomplete['unified_risk'].max():.4f}"
        )

    # --------------------------------------------------------
    # 12. Top unified-risk files
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP UNIFIED-RISK FILES")
    print("=" * 70)

    display_columns = [
        "file_path",
        "unified_risk",
        "unified_risk_level",
        "refined_bus_risk",
        "satd_score",
        "code_risk_for_fusion",
        "ml_defect_risk_score",
        "fusion_signal_count",
        "signals_used",
    ]

    top_files = (
        df
        .sort_values(
            "unified_risk",
            ascending=False,
        )
        [display_columns]
        .head(20)
    )

    print(
        top_files.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # 13. Complete four-signal ranking
    # --------------------------------------------------------

    if len(complete) > 0:

        print("\n" + "=" * 70)
        print("TOP COMPLETE FOUR-SIGNAL FILES")
        print("=" * 70)

        complete_top = (
            complete
            .sort_values(
                "unified_risk",
                ascending=False,
            )
            [display_columns]
            .head(20)
        )

        print(
            complete_top.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    # --------------------------------------------------------
    # 14. Methodological notes
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("METHODOLOGICAL NOTES")
    print("=" * 70)

    print(
        "Fusion method          : equal-weight availability-aware"
    )

    print(
        "Bus weight             : 0.25"
    )

    print(
        "SATD weight            : 0.25"
    )

    print(
        "Code-risk weight       : 0.25"
    )

    print(
        "ML-risk weight         : 0.25"
    )

    print(
        "Missing signal handling: weight renormalization"
    )

    print(
        "Fusion weights learned : No"
    )

    print(
        "Test-event predictions : Not used"
    )

    print(
        "Risk bands calibrated  : No"
    )

    print(
        "Ground-truth fusion    : Not available"
    )

    # --------------------------------------------------------
    # 15. Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"Saved:\n{OUTPUT_PATH}"
    )

    print(
        f"Final shape: {df.shape}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()