from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "data/processed/fusion/requests_refactoring_priorities.csv"
)


# ============================================================
# HELPERS
# ============================================================

def check_equal(df, left, right):
    """
    Compare two columns while handling NaN values safely.
    """
    left_values = df[left]
    right_values = df[right]

    both_missing = left_values.isna() & right_values.isna()

    equal_values = (
        left_values.astype(str).str.strip()
        == right_values.astype(str).str.strip()
    )

    return both_missing | equal_values


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CODEGUARD AI - REFACTORING PRIORITY VALIDATION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput rows    : {len(df)}")
    print(f"Input columns : {len(df.columns)}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "file_path",
        "applicability_category",
        "risk_applicable",
        "prioritization_eligible",
        "unified_risk",
        "unified_risk_level",
        "refined_bus_risk",
        "satd_score",
        "satd_present",
        "code_risk_for_fusion",
        "ml_defect_risk_score",
        "review_priority",
        "primary_recommendation",
        "secondary_recommendations",
        "recommendation_reason",
        "dominant_signal",
        "dominant_signal_value",
        "recommendation_rule",
        "final_action_status",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("\nFAIL")
        print("Missing required columns:")
        for column in missing_columns:
            print(f"  - {column}")
        raise ValueError(
            "Required columns are missing."
        )

    print("\nRequired columns: PASS")

    # --------------------------------------------------------
    # Duplicate file check
    # --------------------------------------------------------

    duplicate_files = int(
        df["file_path"].duplicated().sum()
    )

    print(
        f"Duplicate file_path values: {duplicate_files}"
    )

    if duplicate_files != 0:
        raise ValueError(
            "Duplicate file_path values detected."
        )

    print("Duplicate check: PASS")

    # --------------------------------------------------------
    # Missing recommendation fields
    # --------------------------------------------------------

    recommendation_columns = [
        "review_priority",
        "primary_recommendation",
        "recommendation_reason",
        "dominant_signal",
        "recommendation_rule",
        "final_action_status",
    ]

    print("\nRecommendation completeness:")

    recommendation_failures = 0

    for column in recommendation_columns:

        missing_count = int(
            df[column].isna().sum()
            +
            (
                df[column]
                .astype(str)
                .str.strip()
                .eq("")
                .sum()
            )
        )

        print(
            f"{column:30s}: {missing_count}"
        )

        if missing_count > 0:
            recommendation_failures += 1

    if recommendation_failures:
        raise ValueError(
            "One or more recommendation columns contain missing values."
        )

    print("Recommendation completeness: PASS")

    # --------------------------------------------------------
    # Risk range
    # --------------------------------------------------------

    numeric_risk = pd.to_numeric(
        df["unified_risk"],
        errors="coerce"
    )

    risk_missing = int(
        numeric_risk.isna().sum()
    )

    risk_outside = int(
        (
            (numeric_risk < 0)
            | (numeric_risk > 1)
        )
        .fillna(False)
        .sum()
    )

    print("\nUnified risk validation:")
    print(f"Missing unified risk : {risk_missing}")
    print(f"Outside [0,1]        : {risk_outside}")

    if risk_missing != 0 or risk_outside != 0:
        raise ValueError(
            "Unified risk contains invalid values."
        )

    print("Unified risk range: PASS")

    # --------------------------------------------------------
    # Risk-level consistency
    # --------------------------------------------------------

    def expected_risk_level(value):

        if value >= 0.75:
            return "Critical"
        elif value >= 0.50:
            return "High"
        elif value >= 0.25:
            return "Medium"
        else:
            return "Low"

    expected_levels = numeric_risk.apply(
        expected_risk_level
    )

    risk_level_mismatches = int(
        (
            expected_levels.astype(str)
            != df["unified_risk_level"].astype(str)
        ).sum()
    )

    print("\nRisk-level consistency:")
    print(
        f"Mismatches: {risk_level_mismatches}"
    )

    if risk_level_mismatches != 0:
        raise ValueError(
            "Unified risk levels do not match unified risk."
        )

    print("Risk-level consistency: PASS")

    # --------------------------------------------------------
    # Eligibility consistency
    # --------------------------------------------------------

    eligible = (
        df["prioritization_eligible"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    risk_applicable = (
        df["risk_applicable"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    # Every prioritization-eligible file should also be
    # risk-applicable.
    invalid_eligible = int(
        (eligible & ~risk_applicable).sum()
    )

    print("\nEligibility consistency:")
    print(
        f"Eligible but not risk-applicable: "
        f"{invalid_eligible}"
    )

    if invalid_eligible != 0:
        raise ValueError(
            "Some files are eligible despite not being risk-applicable."
        )

    print("Eligibility consistency: PASS")

    # --------------------------------------------------------
    # Final action status consistency
    # --------------------------------------------------------

    expected_status = []

    for _, row in df.iterrows():

        is_eligible = bool(
            str(row["prioritization_eligible"])
            .strip()
            .lower()
            in ["true", "1", "yes"]
        )

        level = str(
            row["unified_risk_level"]
        ).strip()

        if not is_eligible:
            status = "Traceability only"

        elif level == "Critical":
            status = "Immediate refactoring/review"

        elif level == "High":
            status = "High-priority refactoring/review"

        elif level == "Medium":
            status = "Refactoring/review candidate"

        else:
            status = "Monitor"

        expected_status.append(status)

    status_mismatches = int(
        (
            pd.Series(expected_status)
            .astype(str)
            != df["final_action_status"]
            .astype(str)
        ).sum()
    )

    print("\nFinal action-status consistency:")
    print(
        f"Mismatches: {status_mismatches}"
    )

    if status_mismatches != 0:
        raise ValueError(
            "Final action status is inconsistent with eligibility/risk level."
        )

    print("Final action-status consistency: PASS")

    # --------------------------------------------------------
    # Review-priority consistency
    # --------------------------------------------------------

    expected_priority = []

    for _, row in df.iterrows():

        level = str(
            row["unified_risk_level"]
        ).strip()

        priority_map = {
            "Critical": "Immediate review",
            "High": "High-priority review",
            "Medium": "Refactoring/review candidate",
            "Low": "Monitor",
        }

        expected_priority.append(
            priority_map.get(
                level,
                "Review"
            )
        )

    priority_mismatches = int(
        (
            pd.Series(expected_priority)
            .astype(str)
            != df["review_priority"]
            .astype(str)
        ).sum()
    )

    print("\nReview-priority consistency:")
    print(
        f"Mismatches: {priority_mismatches}"
    )

    if priority_mismatches != 0:
        raise ValueError(
            "Review priority is inconsistent with risk level."
        )

    print("Review-priority consistency: PASS")

    # --------------------------------------------------------
    # Dominant signal validation
    # --------------------------------------------------------

    signal_columns = {
        "bus_factor_risk": "refined_bus_risk",
        "satd": "satd_score",
        "code_risk": "code_risk_for_fusion",
        "ml_defect_risk": "ml_defect_risk_score",
    }

    dominant_failures = 0
    dominant_value_failures = 0

    for index, row in df.iterrows():

        available = {}

        for signal_name, column_name in signal_columns.items():

            value = pd.to_numeric(
                row[column_name],
                errors="coerce"
            )

            if not pd.isna(value):
                available[signal_name] = float(value)

        actual_signal = str(
            row["dominant_signal"]
        ).strip()

        actual_value = pd.to_numeric(
            row["dominant_signal_value"],
            errors="coerce"
        )

        if not available:

            if actual_signal != "none":
                dominant_failures += 1

            if not pd.isna(actual_value):
                dominant_value_failures += 1

        else:

            expected_signal = max(
                available,
                key=available.get
            )

            expected_value = available[
                expected_signal
            ]

            if actual_signal != expected_signal:
                dominant_failures += 1

            if pd.isna(actual_value):
                dominant_value_failures += 1
            elif not np.isclose(
                float(actual_value),
                expected_value,
                atol=1e-9
            ):
                dominant_value_failures += 1

    print("\nDominant-signal validation:")
    print(
        f"Signal mismatches: {dominant_failures}"
    )
    print(
        f"Value mismatches : {dominant_value_failures}"
    )

    if (
        dominant_failures != 0
        or dominant_value_failures != 0
    ):
        raise ValueError(
            "Dominant signal information is inconsistent."
        )

    print("Dominant-signal validation: PASS")

    # --------------------------------------------------------
    # Recommendation-rule consistency
    # --------------------------------------------------------

    rule_values = (
        df["recommendation_rule"]
        .astype(str)
        .str.strip()
    )

    invalid_rules = int(
        (
            rule_values
            != "rule_based_refactoring_v1"
        ).sum()
    )

    print("\nRecommendation rule:")
    print(
        f"Invalid rule values: {invalid_rules}"
    )

    if invalid_rules != 0:
        raise ValueError(
            "Unexpected recommendation rule detected."
        )

    print("Recommendation rule: PASS")

    # --------------------------------------------------------
    # Coverage accounting
    # --------------------------------------------------------

    total_files = len(df)

    eligible_count = int(
        eligible.sum()
    )

    traceability_count = int(
        (~eligible).sum()
    )

    print("\nCoverage:")
    print(f"Total files              : {total_files}")
    print(f"Eligible                 : {eligible_count}")
    print(f"Traceability only        : {traceability_count}")
    print(
        f"Accounting check         : "
        f"{eligible_count + traceability_count}"
    )

    if (
        eligible_count
        + traceability_count
        != total_files
    ):
        raise ValueError(
            "Eligibility accounting does not equal total files."
        )

    print("Coverage accounting: PASS")

    # --------------------------------------------------------
    # Recommendation distribution
    # --------------------------------------------------------

    print("\nFinal action status distribution:")
    print(
        df["final_action_status"]
        .value_counts()
        .to_string()
    )

    print("\nRisk-level distribution:")
    print(
        df["unified_risk_level"]
        .value_counts()
        .to_string()
    )

    print("\nApplicability distribution:")
    print(
        df["applicability_category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Eligible-file summary
    # --------------------------------------------------------

    eligible_df = df.loc[
        eligible
    ].copy()

    print("\nEligible files by category:")
    print(
        eligible_df["applicability_category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nEligible files by review priority:")
    print(
        eligible_df["review_priority"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Final PASS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION: PASS")
    print("=" * 70)

    print(
        "\nThe refactoring-priority dataset passed all "
        "structural and consistency checks."
    )

    print(
        "\nValidated:"
    )
    print("- Required columns")
    print("- Unique file coverage")
    print("- Recommendation completeness")
    print("- Unified risk range")
    print("- Risk-level consistency")
    print("- Eligibility consistency")
    print("- Final action status")
    print("- Review priority")
    print("- Dominant signal")
    print("- Recommendation rule")
    print("- File accounting")

    print("\nDone.")


if __name__ == "__main__":
    main()