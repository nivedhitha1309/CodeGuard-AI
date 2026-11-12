from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/processed/fusion/requests_four_signal_fusion_applicability.csv"
)

OUTPUT_FILE = Path(
    "data/processed/fusion/requests_final_risk_prioritization.csv"
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# These are the four signals used by the current CodeGuard fusion model.
SIGNAL_COLUMNS = [
    "refined_bus_risk",
    "satd_score",
    "code_risk_for_fusion",
    "ml_defect_risk_score",
]

# Risk levels are inherited from the validated fusion baseline.
RISK_LEVELS = [
    "Low",
    "Medium",
    "High",
    "Critical",
]


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def calculate_available_signal_count(row):
    """Count the number of actually available risk signals."""
    return int(
        row[SIGNAL_COLUMNS]
        .notna()
        .sum()
    )


def calculate_signal_completeness(row):
    """Calculate fraction of the four signals that are available."""
    return (
        calculate_available_signal_count(row)
        / len(SIGNAL_COLUMNS)
    )


def calculate_available_signal_mean(row):
    """
    Availability-aware equal-weight fusion.

    Missing signals are excluded from the denominator.
    Missing is therefore NOT interpreted as zero.
    """
    values = row[SIGNAL_COLUMNS].dropna()

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def determine_risk_level(score):
    """
    Apply the existing CodeGuard risk bands.

    These thresholds are the same thresholds used by the
    validated four-signal fusion baseline.
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


def risk_priority_value(level):
    """
    Numeric ordering used only to sort the final prioritisation table.

    This does not create a new risk score.
    """
    priority_map = {
        "Critical": 4,
        "High": 3,
        "Medium": 2,
        "Low": 1,
        "Unavailable": 0,
    }

    return priority_map.get(level, 0)


def applicability_priority(category):
    """
    Human-readable prioritisation group.

    Production source and test code remain distinct from
    configuration because they represent different artifact types.
    """
    mapping = {
        "production_source": "Production Source",
        "test_code": "Test Code",
        "configuration": "Configuration",
        "documentation": "Documentation",
        "binary_asset": "Binary Asset",
        "other": "Other",
    }

    return mapping.get(category, "Other")


def describe_signal_profile(row):
    """Create a compact explanation of which signals contributed."""
    used = []

    if pd.notna(row["refined_bus_risk"]):
        used.append("Bus")

    if pd.notna(row["satd_score"]):
        used.append("SATD")

    if pd.notna(row["code_risk_for_fusion"]):
        used.append("Code")

    if pd.notna(row["ml_defect_risk_score"]):
        used.append("ML")

    if not used:
        return "None"

    return "+".join(used)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    print("=" * 70)
    print("CODEGUARD AI - FINAL RISK PRIORITIZATION DATASET")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load applicability dataset
    # ------------------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput file: {INPUT_FILE}")
    print(f"Input rows: {len(df)}")
    print(f"Input columns: {len(df.columns)}")

    # ------------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------------

    required_columns = {
        "file_path",
        "applicability_category",
        "risk_applicable",
        "prioritization_eligible",
        "unified_risk",
        "unified_risk_level",
        "fusion_signal_count",
        "fusion_signal_completeness",
        "signals_used",
        *SIGNAL_COLUMNS,
    }

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    # ------------------------------------------------------------------
    # Basic integrity checks
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("INPUT INTEGRITY")
    print("=" * 70)

    duplicate_files = int(
        df["file_path"].duplicated().sum()
    )

    print(f"Duplicate file paths: {duplicate_files}")

    if duplicate_files != 0:
        raise ValueError(
            "Duplicate file paths detected."
        )

    if df["file_path"].isna().any():
        raise ValueError(
            "Missing file paths detected."
        )

    print("PASS")

    # ------------------------------------------------------------------
    # Recalculate signal availability
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SIGNAL CONSISTENCY")
    print("=" * 70)

    calculated_signal_count = (
        df[SIGNAL_COLUMNS]
        .notna()
        .sum(axis=1)
    )

    count_mismatch = (
        calculated_signal_count.to_numpy()
        != df["fusion_signal_count"].to_numpy()
    )

    print(
        "Signal-count mismatches:",
        int(count_mismatch.sum())
    )

    if count_mismatch.any():
        raise ValueError(
            "Signal-count values are inconsistent."
        )

    calculated_completeness = (
        calculated_signal_count
        / len(SIGNAL_COLUMNS)
    )

    completeness_mismatch = ~np.isclose(
        calculated_completeness,
        df["fusion_signal_completeness"],
        atol=1e-10,
    )

    print(
        "Signal-completeness mismatches:",
        int(completeness_mismatch.sum())
    )

    if completeness_mismatch.any():
        raise ValueError(
            "Signal-completeness values are inconsistent."
        )

    print("PASS")

    # ------------------------------------------------------------------
    # Recalculate fusion score
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FUSION SCORE CONSISTENCY")
    print("=" * 70)

    recalculated_fusion = df.apply(
        calculate_available_signal_mean,
        axis=1,
    )

    fusion_mismatch = ~np.isclose(
        recalculated_fusion,
        df["unified_risk"],
        atol=1e-10,
        equal_nan=True,
    )

    print(
        "Fusion-score mismatches:",
        int(fusion_mismatch.sum())
    )

    if fusion_mismatch.any():
        raise ValueError(
            "Unified-risk values do not match "
            "availability-aware equal-weight fusion."
        )

    print("PASS")

    # ------------------------------------------------------------------
    # Recalculate risk levels
    # ------------------------------------------------------------------

    calculated_levels = (
        df["unified_risk"]
        .apply(determine_risk_level)
    )

    level_mismatch = (
        calculated_levels.to_numpy()
        != df["unified_risk_level"].to_numpy()
    )

    print(
        "Risk-level mismatches:",
        int(level_mismatch.sum())
    )

    if level_mismatch.any():
        raise ValueError(
            "Risk-level values are inconsistent."
        )

    print("PASS")

    # ------------------------------------------------------------------
    # Add final prioritisation fields
    # ------------------------------------------------------------------

    df["prioritization_group"] = (
        df["applicability_category"]
        .apply(applicability_priority)
    )

    df["signal_profile"] = (
        df.apply(
            describe_signal_profile,
            axis=1,
        )
    )

    df["available_signal_count"] = (
        df.apply(
            calculate_available_signal_count,
            axis=1,
        )
    )

    df["signal_completeness"] = (
        df.apply(
            calculate_signal_completeness,
            axis=1,
        )
    )

    df["risk_priority_order"] = (
        df["unified_risk_level"]
        .apply(risk_priority_value)
    )

    # ------------------------------------------------------------------
    # Final prioritisation eligibility
    # ------------------------------------------------------------------

    # Only files marked as risk-applicable are eligible for the
    # primary software-risk prioritisation view.
    #
    # The complete repository remains in the output for traceability.

    df["final_prioritization_status"] = np.where(
        df["prioritization_eligible"],
        "Eligible",
        "Traceability Only",
    )

    # ------------------------------------------------------------------
    # Sort for final prioritisation
    # ------------------------------------------------------------------

    df = df.sort_values(
        by=[
            "prioritization_eligible",
            "risk_priority_order",
            "unified_risk",
            "signal_completeness",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL PRIORITIZATION SUMMARY")
    print("=" * 70)

    total_files = len(df)

    eligible_files = int(
        df["prioritization_eligible"].sum()
    )

    traceability_files = (
        total_files - eligible_files
    )

    print(f"Total repository files : {total_files}")
    print(f"Prioritization eligible: {eligible_files}")
    print(f"Traceability only      : {traceability_files}")

    # ------------------------------------------------------------------
    # Eligible-file risk distribution
    # ------------------------------------------------------------------

    eligible_df = df[
        df["prioritization_eligible"]
    ].copy()

    print("\nRisk distribution - eligible files:")

    eligible_distribution = (
        eligible_df["unified_risk_level"]
        .value_counts()
        .reindex(
            RISK_LEVELS,
            fill_value=0,
        )
    )

    print(
        eligible_distribution.to_string()
    )

    print(
        f"\nAverage eligible-file risk: "
        f"{eligible_df['unified_risk'].mean():.4f}"
    )

    print(
        f"Maximum eligible-file risk: "
        f"{eligible_df['unified_risk'].max():.4f}"
    )

    # ------------------------------------------------------------------
    # Applicability category distribution
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ELIGIBLE FILES BY CATEGORY")
    print("=" * 70)

    category_distribution = (
        eligible_df["applicability_category"]
        .value_counts()
        .sort_index()
    )

    print(
        category_distribution.to_string()
    )

    # ------------------------------------------------------------------
    # Top production source files
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP PRODUCTION-SOURCE FILES")
    print("=" * 70)

    production_df = eligible_df[
        eligible_df["applicability_category"]
        == "production_source"
    ].copy()

    production_columns = [
        "file_path",
        "unified_risk",
        "unified_risk_level",
        "refined_bus_risk",
        "satd_score",
        "code_risk_for_fusion",
        "ml_defect_risk_score",
        "available_signal_count",
        "signal_profile",
    ]

    top_production = (
        production_df[
            production_columns
        ]
        .sort_values(
            "unified_risk",
            ascending=False,
        )
        .head(20)
    )

    if len(top_production) > 0:
        print(
            top_production.to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # Top test-code files
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP TEST-CODE FILES")
    print("=" * 70)

    test_df = eligible_df[
        eligible_df["applicability_category"]
        == "test_code"
    ].copy()

    top_test = (
        test_df[
            production_columns
        ]
        .sort_values(
            "unified_risk",
            ascending=False,
        )
        .head(10)
    )

    if len(top_test) > 0:
        print(
            top_test.to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # Top configuration files
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP CONFIGURATION FILES")
    print("=" * 70)

    config_df = eligible_df[
        eligible_df["applicability_category"]
        == "configuration"
    ].copy()

    top_config = (
        config_df[
            production_columns
        ]
        .sort_values(
            "unified_risk",
            ascending=False,
        )
        .head(10)
    )

    if len(top_config) > 0:
        print(
            top_config.to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # Signal completeness
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SIGNAL COMPLETENESS - ELIGIBLE FILES")
    print("=" * 70)

    completeness_distribution = (
        eligible_df["available_signal_count"]
        .value_counts()
        .sort_index()
    )

    print(
        completeness_distribution.to_string()
    )

    # ------------------------------------------------------------------
    # Highest-risk files with incomplete signals
    # ------------------------------------------------------------------

    incomplete_df = eligible_df[
        eligible_df["available_signal_count"]
        < len(SIGNAL_COLUMNS)
    ].copy()

    print("\n" + "=" * 70)
    print("HIGH-RISK ELIGIBLE FILES WITH INCOMPLETE SIGNALS")
    print("=" * 70)

    incomplete_columns = [
        "file_path",
        "applicability_category",
        "unified_risk",
        "unified_risk_level",
        "available_signal_count",
        "signal_completeness",
        "signal_profile",
    ]

    top_incomplete = (
        incomplete_df[
            incomplete_columns
        ]
        .sort_values(
            "unified_risk",
            ascending=False,
        )
        .head(15)
    )

    if len(top_incomplete) > 0:
        print(
            top_incomplete.to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # Output column ordering
    # ------------------------------------------------------------------

    preferred_columns = [
        "file_path",
        "applicability_category",
        "risk_applicable",
        "prioritization_eligible",
        "final_prioritization_status",
        "prioritization_group",
        "unified_risk",
        "unified_risk_level",
        "risk_priority_order",
        "refined_bus_risk",
        "satd_score",
        "code_risk_for_fusion",
        "ml_defect_risk_score",
        "available_signal_count",
        "signal_completeness",
        "signal_profile",
        "fusion_signal_count",
        "fusion_signal_completeness",
        "signals_used",
        "applicability_reason",
    ]

    remaining_columns = [
        column
        for column in df.columns
        if column not in preferred_columns
    ]

    df = df[
        preferred_columns
        + remaining_columns
    ]

    # ------------------------------------------------------------------
    # Save final dataset
    # ------------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL OUTPUT")
    print("=" * 70)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Final shape: {df.shape}")

    print("\nThe final dataset contains:")
    print("  - all repository files")
    print("  - four-signal unified risk")
    print("  - applicability category")
    print("  - prioritization eligibility")
    print("  - signal completeness")
    print("  - contributing signal profile")
    print("  - risk-level and priority ordering")

    print("\nImportant methodological properties:")
    print("  - Missing signals are not treated as zero.")
    print("  - Fusion weights remain equal at 0.25.")
    print("  - Missing signals are handled by weight renormalization.")
    print("  - No test-event predictions are used.")
    print("  - Non-applicable repository files remain for traceability.")
    print("  - No learned fusion weights are claimed.")
    print("  - Risk thresholds remain the existing baseline thresholds.")

    print("\nFinal risk prioritization dataset created successfully.")


if __name__ == "__main__":
    main()