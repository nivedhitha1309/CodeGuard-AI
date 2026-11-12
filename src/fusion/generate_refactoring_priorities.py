from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "data/processed/fusion/requests_final_risk_prioritization.csv"
)

OUTPUT_FILE = Path(
    "data/processed/fusion/requests_refactoring_priorities.csv"
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    """Convert a value to float, returning NaN when unavailable."""
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def is_positive_signal(value, threshold):
    """Return True when a signal exists and meets the threshold."""
    value = safe_float(value)
    return not np.isnan(value) and value >= threshold


def build_recommendations(row):
    """
    Generate explainable refactoring recommendations from the
    existing CodeGuard signals.

    This is a rule-based recommendation layer. It does not retrain
    the ML model and does not invent a new risk score.
    """

    category = str(row.get("applicability_category", "")).strip().lower()
    risk_level = str(row.get("unified_risk_level", "")).strip()

    bus = safe_float(row.get("refined_bus_risk"))
    satd = safe_float(row.get("satd_score"))
    code = safe_float(row.get("code_risk_for_fusion"))
    ml = safe_float(row.get("ml_defect_risk_score"))

    satd_present = bool(row.get("satd_present", False))

    recommendations = []
    reasons = []
    signal_values = {}

    # --------------------------------------------------------
    # Production source
    # --------------------------------------------------------

    if category == "production_source":

        if is_positive_signal(code, 0.50):
            recommendations.append(
                "Refactor high-complexity or high-code-risk logic"
            )
            reasons.append(
                f"code risk is {code:.3f}"
            )
            signal_values["code_risk"] = code

        elif is_positive_signal(code, 0.35):
            recommendations.append(
                "Review code complexity and consider targeted refactoring"
            )
            reasons.append(
                f"code risk is {code:.3f}"
            )
            signal_values["code_risk"] = code

        if satd_present or is_positive_signal(satd, 0.30):
            recommendations.append(
                "Resolve or review self-admitted technical debt"
            )
            reasons.append(
                f"SATD score is {satd:.3f}"
            )
            signal_values["satd"] = satd

        if is_positive_signal(bus, 0.50):
            recommendations.append(
                "Reduce knowledge concentration by increasing contributor ownership"
            )
            reasons.append(
                f"bus-factor risk is {bus:.3f}"
            )
            signal_values["bus_risk"] = bus

        elif is_positive_signal(bus, 0.35):
            recommendations.append(
                "Review maintainer/contributor concentration"
            )
            reasons.append(
                f"bus-factor risk is {bus:.3f}"
            )
            signal_values["bus_risk"] = bus

        if is_positive_signal(ml, 0.25):
            recommendations.append(
                "Review defect-prone behavior and strengthen regression coverage"
            )
            reasons.append(
                f"ML defect risk is {ml:.3f}"
            )
            signal_values["ml_risk"] = ml

        elif is_positive_signal(ml, 0.15):
            recommendations.append(
                "Review historical defect-risk indicators and regression coverage"
            )
            reasons.append(
                f"ML defect risk is {ml:.3f}"
            )
            signal_values["ml_risk"] = ml

    # --------------------------------------------------------
    # Test code
    # --------------------------------------------------------

    elif category == "test_code":

        recommendations.append(
            "Review and strengthen test coverage and maintainability"
        )

        if is_positive_signal(code, 0.35):
            recommendations.append(
                "Refactor complex test logic"
            )
            reasons.append(
                f"code risk is {code:.3f}"
            )
            signal_values["code_risk"] = code

        if satd_present or is_positive_signal(satd, 0.30):
            recommendations.append(
                "Resolve technical-debt markers in test code"
            )
            reasons.append(
                f"SATD score is {satd:.3f}"
            )
            signal_values["satd"] = satd

        if is_positive_signal(bus, 0.50):
            recommendations.append(
                "Increase test-code ownership across contributors"
            )
            reasons.append(
                f"bus-factor risk is {bus:.3f}"
            )
            signal_values["bus_risk"] = bus

        if is_positive_signal(ml, 0.25):
            recommendations.append(
                "Add or strengthen regression tests around defect-prone behavior"
            )
            reasons.append(
                f"ML defect risk is {ml:.3f}"
            )
            signal_values["ml_risk"] = ml

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    elif category == "configuration":

        recommendations.append(
            "Review configuration and build/automation dependencies"
        )

        if is_positive_signal(bus, 0.50):
            recommendations.append(
                "Document configuration ownership and reduce maintainer concentration"
            )
            reasons.append(
                f"bus-factor risk is {bus:.3f}"
            )
            signal_values["bus_risk"] = bus

        if satd_present or is_positive_signal(satd, 0.30):
            recommendations.append(
                "Review technical-debt markers in configuration"
            )
            reasons.append(
                f"SATD score is {satd:.3f}"
            )
            signal_values["satd"] = satd

    # --------------------------------------------------------
    # Other categories
    # --------------------------------------------------------

    else:

        recommendations.append(
            "Review file manually before applying software refactoring"
        )

        if satd_present or is_positive_signal(satd, 0.30):
            recommendations.append(
                "Review technical-debt markers"
            )
            reasons.append(
                f"SATD score is {satd:.3f}"
            )
            signal_values["satd"] = satd

        if is_positive_signal(bus, 0.50):
            recommendations.append(
                "Review contributor concentration and ownership"
            )
            reasons.append(
                f"bus-factor risk is {bus:.3f}"
            )
            signal_values["bus_risk"] = bus

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not recommendations:
        recommendations.append(
            "Monitor and review during future maintenance"
        )

    # --------------------------------------------------------
    # Determine dominant observed signal.
    #
    # This is descriptive only: it identifies the largest
    # available normalized signal, not causal importance.
    # --------------------------------------------------------

    available_signals = {
        "bus_factor_risk": bus,
        "satd": satd,
        "code_risk": code,
        "ml_defect_risk": ml,
    }

    available_signals = {
        name: value
        for name, value in available_signals.items()
        if not np.isnan(value)
    }

    if available_signals:
        dominant_signal = max(
            available_signals,
            key=available_signals.get
        )
        dominant_value = available_signals[dominant_signal]
    else:
        dominant_signal = "none"
        dominant_value = np.nan

    # --------------------------------------------------------
    # Priority status comes from the existing unified risk level.
    # No new numerical risk score is created here.
    # --------------------------------------------------------

    priority_map = {
        "Critical": "Immediate review",
        "High": "High-priority review",
        "Medium": "Refactoring/review candidate",
        "Low": "Monitor",
    }

    review_priority = priority_map.get(
        risk_level,
        "Review"
    )

    # --------------------------------------------------------
    # Explainability text
    # --------------------------------------------------------

    if reasons:
        reason_text = "; ".join(reasons)
    else:
        reason_text = (
            "No individual signal crossed the recommendation "
            "thresholds used by this rule-based layer."
        )

    return pd.Series(
        {
            "review_priority": review_priority,
            "primary_recommendation": recommendations[0],
            "secondary_recommendations": " | ".join(
                recommendations[1:]
            ),
            "recommendation_reason": reason_text,
            "dominant_signal": dominant_signal,
            "dominant_signal_value": dominant_value,
            "recommendation_rule": "rule_based_refactoring_v1",
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CODEGUARD AI - REFACTORING PRIORITY GENERATION")
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
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicate_files = df["file_path"].duplicated().sum()

    print(f"Duplicate files: {duplicate_files}")

    if duplicate_files > 0:
        raise ValueError(
            "Duplicate file_path values detected."
        )

    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    print("\nGenerating rule-based refactoring recommendations...")

    recommendations = df.apply(
        build_recommendations,
        axis=1
    )

    df = pd.concat(
        [df, recommendations],
        axis=1
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    def build_status(row):

        eligible = bool(row["prioritization_eligible"])
        risk_level = str(row["unified_risk_level"])

        if not eligible:
            return "Traceability only"

        if risk_level == "Critical":
            return "Immediate refactoring/review"
        elif risk_level == "High":
            return "High-priority refactoring/review"
        elif risk_level == "Medium":
            return "Refactoring/review candidate"
        else:
            return "Monitor"

    df["final_action_status"] = df.apply(
        build_status,
        axis=1
    )

    # --------------------------------------------------------
    # Keep eligible files first.
    #
    # Existing unified risk determines ordering.
    # No new score is introduced.
    # --------------------------------------------------------

    df["eligibility_sort"] = (
        df["prioritization_eligible"]
        .astype(bool)
        .astype(int)
    )

    risk_order = {
        "Critical": 4,
        "High": 3,
        "Medium": 2,
        "Low": 1,
        "Unavailable": 0,
    }

    df["risk_sort"] = (
        df["unified_risk_level"]
        .map(risk_order)
        .fillna(0)
    )

    df = df.sort_values(
        by=[
            "eligibility_sort",
            "risk_sort",
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

    df.drop(
        columns=[
            "eligibility_sort",
            "risk_sort",
        ],
        inplace=True,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("REFACTORING PRIORITY SUMMARY")
    print("=" * 70)

    print(f"Total files                 : {len(df)}")

    eligible_count = int(
        df["prioritization_eligible"].sum()
    )

    print(f"Prioritization eligible     : {eligible_count}")
    print(
        f"Traceability-only files    : "
        f"{len(df) - eligible_count}"
    )

    print("\nFinal action status:")
    print(
        df["final_action_status"]
        .value_counts()
        .to_string()
    )

    print("\nReview priority:")
    print(
        df["review_priority"]
        .value_counts()
        .to_string()
    )

    print("\nPrimary recommendation:")
    print(
        df.loc[
            df["prioritization_eligible"],
            "primary_recommendation"
        ]
        .value_counts()
        .head(10)
        .to_string()
    )

    print("\nDominant observed signal:")
    print(
        df.loc[
            df["prioritization_eligible"],
            "dominant_signal"
        ]
        .value_counts()
        .to_string()
    )

    print("\nTop eligible files:")

    display_columns = [
        "file_path",
        "applicability_category",
        "unified_risk",
        "unified_risk_level",
        "signal_completeness",
        "dominant_signal",
        "primary_recommendation",
        "final_action_status",
    ]

    print(
        df.loc[
            df["prioritization_eligible"],
            display_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(f"Saved: {OUTPUT_FILE}")

    print("\nMethodological notes:")
    print("- Rule-based recommendation layer")
    print("- Existing unified risk is preserved")
    print("- No new risk score is created")
    print("- Missing signals are not treated as zero")
    print("- Recommendations are explainable and traceable")
    print("- Dominant signal is descriptive, not causal")
    print("- Test-event ML predictions are not used here")

    print("\nDone.")


if __name__ == "__main__":
    main()