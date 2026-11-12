from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "data/processed/fusion/requests_refactoring_priorities.csv"
)

OUTPUT_FILE = Path(
    "data/processed/fusion/requests_codeguard_report.csv"
)

SUMMARY_FILE = Path(
    "data/processed/fusion/requests_codeguard_summary.txt"
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def format_number(value, digits=4):
    value = safe_float(value)

    if np.isnan(value):
        return "N/A"

    return f"{value:.{digits}f}"


def format_percentage(value):
    value = safe_float(value)

    if np.isnan(value):
        return "N/A"

    return f"{value * 100:.2f}%"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("CODEGUARD AI - FINAL IMPLEMENTATION REPORT")
    print("=" * 75)

    # --------------------------------------------------------
    # Load final validated dataset
    # --------------------------------------------------------

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
        "signal_completeness",
        "available_signal_count",
        "review_priority",
        "primary_recommendation",
        "recommendation_reason",
        "dominant_signal",
        "final_action_status",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_columns
            )
        )

    print("Required columns: PASS")

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicate_count = int(
        df["file_path"].duplicated().sum()
    )

    print(f"Duplicate files: {duplicate_count}")

    if duplicate_count != 0:
        raise ValueError(
            "Duplicate file_path values detected."
        )

    # --------------------------------------------------------
    # Convert boolean-like fields
    # --------------------------------------------------------

    eligible = (
        df["prioritization_eligible"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    applicable = (
        df["risk_applicable"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    # --------------------------------------------------------
    # Signal availability
    # --------------------------------------------------------

    bus_available = (
        pd.to_numeric(
            df["refined_bus_risk"],
            errors="coerce"
        ).notna()
    )

    satd_available = (
        pd.to_numeric(
            df["satd_score"],
            errors="coerce"
        ).notna()
    )

    code_available = (
        pd.to_numeric(
            df["code_risk_for_fusion"],
            errors="coerce"
        ).notna()
    )

    ml_available = (
        pd.to_numeric(
            df["ml_defect_risk_score"],
            errors="coerce"
        ).notna()
    )

    # --------------------------------------------------------
    # Basic repository statistics
    # --------------------------------------------------------

    total_files = len(df)

    eligible_count = int(
        eligible.sum()
    )

    traceability_count = (
        total_files - eligible_count
    )

    applicable_count = int(
        applicable.sum()
    )

    not_applicable_count = (
        total_files - applicable_count
    )

    # --------------------------------------------------------
    # Signal counts
    # --------------------------------------------------------

    bus_count = int(
        bus_available.sum()
    )

    satd_count = int(
        satd_available.sum()
    )

    code_count = int(
        code_available.sum()
    )

    ml_count = int(
        ml_available.sum()
    )

    signal_count_distribution = (
        df["available_signal_count"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Risk statistics
    # --------------------------------------------------------

    risk = pd.to_numeric(
        df["unified_risk"],
        errors="coerce"
    )

    average_risk = risk.mean()
    minimum_risk = risk.min()
    maximum_risk = risk.max()

    risk_distribution = (
        df["unified_risk_level"]
        .value_counts()
        .reindex(
            ["Critical", "High", "Medium", "Low"],
            fill_value=0
        )
    )

    eligible_df = df.loc[
        eligible
    ].copy()

    eligible_risk = pd.to_numeric(
        eligible_df["unified_risk"],
        errors="coerce"
    )

    eligible_average_risk = (
        eligible_risk.mean()
        if len(eligible_df) > 0
        else np.nan
    )

    eligible_maximum_risk = (
        eligible_risk.max()
        if len(eligible_df) > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Applicability distribution
    # --------------------------------------------------------

    applicability_distribution = (
        df["applicability_category"]
        .value_counts()
        .sort_index()
    )

    eligible_category_distribution = (
        eligible_df["applicability_category"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Recommendation distribution
    # --------------------------------------------------------

    recommendation_distribution = (
        eligible_df["primary_recommendation"]
        .value_counts()
    )

    dominant_signal_distribution = (
        eligible_df["dominant_signal"]
        .value_counts()
    )

    final_action_distribution = (
        df["final_action_status"]
        .value_counts()
    )

    review_priority_distribution = (
        df["review_priority"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Top eligible files
    # --------------------------------------------------------

    top_eligible = (
        eligible_df
        .sort_values(
            by=[
                "unified_risk",
                "signal_completeness",
            ],
            ascending=[
                False,
                False,
            ]
        )
        [
            [
                "file_path",
                "applicability_category",
                "unified_risk",
                "unified_risk_level",
                "signal_completeness",
                "available_signal_count",
                "dominant_signal",
                "primary_recommendation",
                "final_action_status",
            ]
        ]
        .head(20)
        .copy()
    )

    # --------------------------------------------------------
    # Top production source files
    # --------------------------------------------------------

    production_df = eligible_df.loc[
        eligible_df["applicability_category"]
        == "production_source"
    ].copy()

    top_production = (
        production_df
        .sort_values(
            by="unified_risk",
            ascending=False
        )
        [
            [
                "file_path",
                "unified_risk",
                "unified_risk_level",
                "refined_bus_risk",
                "satd_score",
                "code_risk_for_fusion",
                "ml_defect_risk_score",
                "signal_completeness",
                "primary_recommendation",
            ]
        ]
        .head(20)
        .copy()
    )

    # --------------------------------------------------------
    # Top test files
    # --------------------------------------------------------

    test_df = eligible_df.loc[
        eligible_df["applicability_category"]
        == "test_code"
    ].copy()

    top_tests = (
        test_df
        .sort_values(
            by="unified_risk",
            ascending=False
        )
        [
            [
                "file_path",
                "unified_risk",
                "unified_risk_level",
                "refined_bus_risk",
                "satd_score",
                "code_risk_for_fusion",
                "ml_defect_risk_score",
                "signal_completeness",
                "primary_recommendation",
            ]
        ]
        .head(10)
        .copy()
    )

    # --------------------------------------------------------
    # Top configuration files
    # --------------------------------------------------------

    config_df = eligible_df.loc[
        eligible_df["applicability_category"]
        == "configuration"
    ].copy()

    top_configuration = (
        config_df
        .sort_values(
            by="unified_risk",
            ascending=False
        )
        [
            [
                "file_path",
                "unified_risk",
                "unified_risk_level",
                "refined_bus_risk",
                "satd_score",
                "code_risk_for_fusion",
                "ml_defect_risk_score",
                "signal_completeness",
                "primary_recommendation",
            ]
        ]
        .head(10)
        .copy()
    )

    # --------------------------------------------------------
    # Signal averages
    # --------------------------------------------------------

    signal_averages = {
        "Bus-factor risk": pd.to_numeric(
            df["refined_bus_risk"],
            errors="coerce"
        ).mean(),

        "SATD score": pd.to_numeric(
            df["satd_score"],
            errors="coerce"
        ).mean(),

        "Code risk": pd.to_numeric(
            df["code_risk_for_fusion"],
            errors="coerce"
        ).mean(),

        "ML defect risk": pd.to_numeric(
            df["ml_defect_risk_score"],
            errors="coerce"
        ).mean(),
    }

    # --------------------------------------------------------
    # Signal correlations
    # --------------------------------------------------------

    correlation_columns = [
        "refined_bus_risk",
        "satd_score",
        "code_risk_for_fusion",
        "ml_defect_risk_score",
    ]

    correlations = (
        df[correlation_columns]
        .corr()
    )

    # --------------------------------------------------------
    # Create final report dataset
    #
    # This is a compact file-level report containing the
    # information needed to inspect CodeGuard's output.
    # --------------------------------------------------------

    report_columns = [
        "file_path",
        "applicability_category",
        "risk_applicable",
        "prioritization_eligible",
        "unified_risk",
        "unified_risk_level",
        "available_signal_count",
        "signal_completeness",
        "refined_bus_risk",
        "satd_score",
        "satd_present",
        "code_risk_for_fusion",
        "ml_defect_risk_score",
        "dominant_signal",
        "dominant_signal_value",
        "review_priority",
        "primary_recommendation",
        "secondary_recommendations",
        "recommendation_reason",
        "final_action_status",
    ]

    report_df = (
        df[report_columns]
        .copy()
    )

    report_df = report_df.sort_values(
        by=[
            "prioritization_eligible",
            "unified_risk",
            "signal_completeness",
        ],
        ascending=[
            False,
            False,
            False,
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save report dataset
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Build text summary
    # --------------------------------------------------------

    summary_lines = []

    summary_lines.append(
        "CODEGUARD AI - FINAL IMPLEMENTATION SUMMARY"
    )
    summary_lines.append(
        "=" * 75
    )

    summary_lines.append("")
    summary_lines.append(
        "1. REPOSITORY COVERAGE"
    )
    summary_lines.append(
        f"Total files analysed: {total_files}"
    )
    summary_lines.append(
        f"Risk-applicable files: {applicable_count}"
    )
    summary_lines.append(
        f"Traceability/non-applicable files: "
        f"{not_applicable_count}"
    )
    summary_lines.append(
        f"Prioritization-eligible files: {eligible_count}"
    )
    summary_lines.append(
        f"Traceability-only files: {traceability_count}"
    )

    summary_lines.append("")
    summary_lines.append(
        "2. SIGNAL COVERAGE"
    )
    summary_lines.append(
        f"Bus-factor signal: {bus_count}/{total_files}"
    )
    summary_lines.append(
        f"SATD signal: {satd_count}/{total_files}"
    )
    summary_lines.append(
        f"Code-risk signal: {code_count}/{total_files}"
    )
    summary_lines.append(
        f"ML defect-risk signal: {ml_count}/{total_files}"
    )

    summary_lines.append("")
    summary_lines.append(
        "Signal-count distribution:"
    )

    for signal_count, count in (
        signal_count_distribution.items()
    ):
        summary_lines.append(
            f"  {signal_count}/4 signals: {count}"
        )

    summary_lines.append("")
    summary_lines.append(
        "3. UNIFIED RISK"
    )
    summary_lines.append(
        f"Average unified risk: "
        f"{format_number(average_risk)}"
    )
    summary_lines.append(
        f"Minimum unified risk: "
        f"{format_number(minimum_risk)}"
    )
    summary_lines.append(
        f"Maximum unified risk: "
        f"{format_number(maximum_risk)}"
    )

    for level in [
        "Critical",
        "High",
        "Medium",
        "Low",
    ]:
        summary_lines.append(
            f"{level}: "
            f"{int(risk_distribution[level])}"
        )

    summary_lines.append("")
    summary_lines.append(
        f"Eligible-file average risk: "
        f"{format_number(eligible_average_risk)}"
    )
    summary_lines.append(
        f"Eligible-file maximum risk: "
        f"{format_number(eligible_maximum_risk)}"
    )

    summary_lines.append("")
    summary_lines.append(
        "4. APPLICABILITY CATEGORIES"
    )

    for category, count in (
        applicability_distribution.items()
    ):
        summary_lines.append(
            f"{category}: {count}"
        )

    summary_lines.append("")
    summary_lines.append(
        "Eligible categories:"
    )

    for category, count in (
        eligible_category_distribution.items()
    ):
        summary_lines.append(
            f"{category}: {count}"
        )

    summary_lines.append("")
    summary_lines.append(
        "5. RECOMMENDATIONS"
    )

    for recommendation, count in (
        recommendation_distribution.items()
    ):
        summary_lines.append(
            f"{count}: {recommendation}"
        )

    summary_lines.append("")
    summary_lines.append(
        "Dominant observed signals:"
    )

    for signal, count in (
        dominant_signal_distribution.items()
    ):
        summary_lines.append(
            f"{signal}: {count}"
        )

    summary_lines.append("")
    summary_lines.append(
        "6. SIGNAL AVERAGES"
    )

    for signal_name, value in (
        signal_averages.items()
    ):
        summary_lines.append(
            f"{signal_name}: "
            f"{format_number(value)}"
        )

    summary_lines.append("")
    summary_lines.append(
        "7. SIGNAL CORRELATIONS"
    )

    summary_lines.append(
        correlations.to_string()
    )

    summary_lines.append("")
    summary_lines.append(
        "8. FINAL ACTION STATUS"
    )

    for status, count in (
        final_action_distribution.items()
    ):
        summary_lines.append(
            f"{status}: {count}"
        )

    summary_lines.append("")
    summary_lines.append(
        "9. REVIEW PRIORITY"
    )

    for priority, count in (
        review_priority_distribution.items()
    ):
        summary_lines.append(
            f"{priority}: {count}"
        )

    summary_lines.append("")
    summary_lines.append(
        "10. TOP ELIGIBLE FILES"
    )

    for rank, (_, row) in enumerate(
        top_eligible.iterrows(),
        start=1
    ):
        summary_lines.append(
            f"{rank}. "
            f"{row['file_path']} | "
            f"risk={row['unified_risk']:.4f} | "
            f"level={row['unified_risk_level']} | "
            f"signals={int(row['available_signal_count'])}/4 | "
            f"dominant={row['dominant_signal']} | "
            f"action={row['primary_recommendation']}"
        )

    summary_lines.append("")
    summary_lines.append(
        "11. TOP PRODUCTION SOURCE FILES"
    )

    for rank, (_, row) in enumerate(
        top_production.iterrows(),
        start=1
    ):
        summary_lines.append(
            f"{rank}. "
            f"{row['file_path']} | "
            f"risk={row['unified_risk']:.4f} | "
            f"bus={row['refined_bus_risk']:.4f} | "
            f"SATD={row['satd_score']:.4f} | "
            f"code={format_number(row['code_risk_for_fusion'])} | "
            f"ML={format_number(row['ml_defect_risk_score'])}"
        )

    summary_lines.append("")
    summary_lines.append(
        "12. TOP TEST FILES"
    )

    for rank, (_, row) in enumerate(
        top_tests.iterrows(),
        start=1
    ):
        summary_lines.append(
            f"{rank}. "
            f"{row['file_path']} | "
            f"risk={row['unified_risk']:.4f} | "
            f"action={row['primary_recommendation']}"
        )

    summary_lines.append("")
    summary_lines.append(
        "13. TOP CONFIGURATION FILES"
    )

    for rank, (_, row) in enumerate(
        top_configuration.iterrows(),
        start=1
    ):
        summary_lines.append(
            f"{rank}. "
            f"{row['file_path']} | "
            f"risk={row['unified_risk']:.4f} | "
            f"action={row['primary_recommendation']}"
        )

    summary_lines.append("")
    summary_lines.append(
        "14. METHODOLOGICAL NOTES"
    )
    summary_lines.append(
        "- Bus-factor, SATD, code-risk and ML-risk are retained "
        "as separate signals."
    )
    summary_lines.append(
        "- Missing signals are not interpreted as zero."
    )
    summary_lines.append(
        "- Four-signal fusion uses availability-aware "
        "equal-weight baseline fusion."
    )
    summary_lines.append(
        "- Current ML risk is generated from the deployment-style "
        "historical model."
    )
    summary_lines.append(
        "- Test-event ML predictions are not used in the final "
        "repository prioritization."
    )
    summary_lines.append(
        "- Refactoring recommendations are rule-based and "
        "explainable."
    )
    summary_lines.append(
        "- Unified risk is a prioritization signal, not a calibrated "
        "probability of failure."
    )
    summary_lines.append(
        "- Historical defect labels are heuristic labels, not "
        "ground-truth defect annotations."
    )
    summary_lines.append(
        "- Dominant signal identifies the largest available "
        "normalized component and is not a causal attribution."
    )

    summary_lines.append("")
    summary_lines.append(
        "15. OUTPUT FILES"
    )
    summary_lines.append(
        f"File-level report: {OUTPUT_FILE}"
    )
    summary_lines.append(
        f"Text summary: {SUMMARY_FILE}"
    )

    summary_lines.append("")
    summary_lines.append(
        "END OF CODEGUARD AI IMPLEMENTATION SUMMARY"
    )

    summary_text = "\n".join(
        summary_lines
    )

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    SUMMARY_FILE.write_text(
        summary_text,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("CODEGUARD AI IMPLEMENTATION SUMMARY")
    print("=" * 75)

    print("\nREPOSITORY")
    print(f"Total files              : {total_files}")
    print(f"Risk-applicable          : {applicable_count}")
    print(f"Prioritization eligible  : {eligible_count}")
    print(f"Traceability only        : {traceability_count}")

    print("\nSIGNAL COVERAGE")
    print(
        f"Bus factor               : "
        f"{bus_count}/{total_files}"
    )
    print(
        f"SATD                     : "
        f"{satd_count}/{total_files}"
    )
    print(
        f"Code risk                : "
        f"{code_count}/{total_files}"
    )
    print(
        f"ML defect risk           : "
        f"{ml_count}/{total_files}"
    )

    print("\nUNIFIED RISK")
    print(
        f"Average                 : "
        f"{format_number(average_risk)}"
    )
    print(
        f"Minimum                 : "
        f"{format_number(minimum_risk)}"
    )
    print(
        f"Maximum                 : "
        f"{format_number(maximum_risk)}"
    )

    print("\nRisk levels:")
    for level in [
        "Critical",
        "High",
        "Medium",
        "Low",
    ]:
        print(
            f"  {level:<10}: "
            f"{int(risk_distribution[level])}"
        )

    print("\nAPPLICABILITY")
    for category, count in (
        applicability_distribution.items()
    ):
        print(
            f"  {category:<20}: {count}"
        )

    print("\nELIGIBLE CATEGORIES")
    for category, count in (
        eligible_category_distribution.items()
    ):
        print(
            f"  {category:<20}: {count}"
        )

    print("\nTOP ELIGIBLE FILES")

    print(
        top_eligible.to_string(
            index=False
        )
    )

    print("\n" + "=" * 75)
    print("OUTPUT FILES")
    print("=" * 75)

    print(
        f"File-level report : {OUTPUT_FILE}"
    )

    print(
        f"Text summary      : {SUMMARY_FILE}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()