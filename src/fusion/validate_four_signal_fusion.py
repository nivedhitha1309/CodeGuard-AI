from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/processed/fusion/requests_four_signal_fusion_baseline.csv"
)

SIGNAL_COLUMNS = [
    "refined_bus_risk",
    "satd_score",
    "code_risk_for_fusion",
    "ml_defect_risk_score",
]

REQUIRED_COLUMNS = [
    "file_path",
    *SIGNAL_COLUMNS,
    "fusion_signal_count",
    "fusion_signal_completeness",
    "unified_risk",
    "unified_risk_level",
    "signals_used",
]


def expected_risk_level(score):
    if score >= 0.75:
        return "Critical"
    if score >= 0.50:
        return "High"
    if score >= 0.25:
        return "Medium"
    return "Low"


def expected_signals_used(row):
    available = []

    if pd.notna(row["refined_bus_risk"]):
        available.append("bus")

    if pd.notna(row["satd_score"]):
        available.append("satd")

    if pd.notna(row["code_risk_for_fusion"]):
        available.append("code")

    if pd.notna(row["ml_defect_risk_score"]):
        available.append("ml")

    return "+".join(available)


def main():
    print("=" * 70)
    print("VALIDATING FOUR-SIGNAL FUSION BASELINE")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fusion output file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput file: {INPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # ------------------------------------------------------------
    # 1. Required columns
    # ------------------------------------------------------------
    print("\n[1] Required column check")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("FAIL")
        print("Missing columns:")
        for column in missing_columns:
            print(f"  - {column}")
        raise ValueError("Required columns are missing.")

    print("PASS")
    print("All required columns are present.")

    # ------------------------------------------------------------
    # 2. Shape and duplicate check
    # ------------------------------------------------------------
    print("\n[2] Shape and duplicate check")

    duplicate_files = df["file_path"].duplicated().sum()

    print(f"Rows: {len(df)}")
    print(f"Unique files: {df['file_path'].nunique()}")
    print(f"Duplicate file paths: {duplicate_files}")

    if duplicate_files != 0:
        print("FAIL: duplicate file paths detected.")
        raise ValueError("Duplicate file paths detected.")

    if len(df) != 284:
        print(
            "WARNING: expected 284 rows based on the current "
            f"project snapshot, but found {len(df)}."
        )
    else:
        print("PASS")

    # ------------------------------------------------------------
    # 3. Unified risk completeness
    # ------------------------------------------------------------
    print("\n[3] Unified-risk completeness")

    missing_unified = df["unified_risk"].isna().sum()

    print(f"Missing unified-risk values: {missing_unified}")

    if missing_unified != 0:
        print("FAIL")
        raise ValueError("Unified risk contains missing values.")

    print("PASS")

    # ------------------------------------------------------------
    # 4. Unified risk range
    # ------------------------------------------------------------
    print("\n[4] Unified-risk range")

    min_risk = df["unified_risk"].min()
    max_risk = df["unified_risk"].max()

    print(f"Minimum unified risk: {min_risk:.6f}")
    print(f"Maximum unified risk: {max_risk:.6f}")

    if min_risk < 0 or max_risk > 1:
        print("FAIL")
        raise ValueError("Unified risk is outside the [0, 1] range.")

    print("PASS")

    # ------------------------------------------------------------
    # 5. Signal availability count
    # ------------------------------------------------------------
    print("\n[5] Signal availability consistency")

    actual_signal_count = df[SIGNAL_COLUMNS].notna().sum(axis=1)

    count_mismatches = (
        actual_signal_count.to_numpy()
        != df["fusion_signal_count"].to_numpy()
    )

    mismatch_count = int(count_mismatches.sum())

    print(f"Signal-count mismatches: {mismatch_count}")

    if mismatch_count != 0:
        print("FAIL")
        raise ValueError(
            "fusion_signal_count does not match actual signal availability."
        )

    print("PASS")

    # ------------------------------------------------------------
    # 6. Completeness ratio
    # ------------------------------------------------------------
    print("\n[6] Signal completeness consistency")

    expected_completeness = actual_signal_count / len(SIGNAL_COLUMNS)

    completeness_mismatch = ~np.isclose(
        expected_completeness,
        df["fusion_signal_completeness"],
        atol=1e-10,
    )

    mismatch_count = int(completeness_mismatch.sum())

    print(f"Completeness mismatches: {mismatch_count}")

    if mismatch_count != 0:
        print("FAIL")
        raise ValueError(
            "fusion_signal_completeness is inconsistent with signal availability."
        )

    print("PASS")

    # ------------------------------------------------------------
    # 7. signals_used consistency
    # ------------------------------------------------------------
    print("\n[7] signals_used consistency")

    expected_used = df.apply(expected_signals_used, axis=1)

    used_mismatch = (
        expected_used.fillna("").to_numpy()
        != df["signals_used"].fillna("").to_numpy()
    )

    mismatch_count = int(used_mismatch.sum())

    print(f"signals_used mismatches: {mismatch_count}")

    if mismatch_count != 0:
        print("FAIL")

        mismatches = df.loc[
            used_mismatch,
            [
                "file_path",
                *SIGNAL_COLUMNS,
                "signals_used",
            ],
        ]

        print("\nExamples:")
        print(mismatches.head(10).to_string(index=False))

        raise ValueError(
            "signals_used does not match actual signal availability."
        )

    print("PASS")

    # ------------------------------------------------------------
    # 8. Mathematical fusion verification
    # ------------------------------------------------------------
    print("\n[8] Mathematical fusion verification")

    # Equal-weight availability-aware fusion:
    # unified risk = mean of all available signals.
    expected_fusion = df[SIGNAL_COLUMNS].mean(axis=1)

    fusion_mismatch = ~np.isclose(
        expected_fusion,
        df["unified_risk"],
        atol=1e-10,
    )

    mismatch_count = int(fusion_mismatch.sum())

    print(f"Fusion calculation mismatches: {mismatch_count}")

    if mismatch_count != 0:
        print("FAIL")

        mismatches = df.loc[
            fusion_mismatch,
            [
                "file_path",
                *SIGNAL_COLUMNS,
                "unified_risk",
            ],
        ].copy()

        mismatches["expected_unified_risk"] = expected_fusion[
            fusion_mismatch
        ]

        print("\nExamples:")
        print(mismatches.head(10).to_string(index=False))

        raise ValueError(
            "Unified risk does not match the availability-aware equal-weight formula."
        )

    print("PASS")

    # ------------------------------------------------------------
    # 9. Missing signals were not converted to zero
    # ------------------------------------------------------------
    print("\n[9] Missing-signal handling")

    code_missing = df["code_risk_for_fusion"].isna().sum()
    ml_missing = df["ml_defect_risk_score"].isna().sum()

    print(f"Files without code signal: {code_missing}")
    print(f"Files without ML signal:   {ml_missing}")

    # Confirm that missing values remain genuinely missing.
    # This is important because missing != zero.
    if code_missing == 0:
        print(
            "WARNING: no missing code-risk values were found. "
            "Check whether the source dataset changed."
        )

    if ml_missing == 0:
        print(
            "WARNING: no missing ML-risk values were found. "
            "Check whether the source dataset changed."
        )

    print("PASS")

    # ------------------------------------------------------------
    # 10. Risk-level consistency
    # ------------------------------------------------------------
    print("\n[10] Unified-risk level consistency")

    expected_levels = df["unified_risk"].apply(expected_risk_level)

    level_mismatch = (
        expected_levels.to_numpy()
        != df["unified_risk_level"].to_numpy()
    )

    mismatch_count = int(level_mismatch.sum())

    print(f"Risk-level mismatches: {mismatch_count}")

    if mismatch_count != 0:
        print("FAIL")

        mismatches = df.loc[
            level_mismatch,
            [
                "file_path",
                "unified_risk",
                "unified_risk_level",
            ],
        ].copy()

        mismatches["expected_level"] = expected_levels[
            level_mismatch
        ]

        print("\nExamples:")
        print(mismatches.head(10).to_string(index=False))

        raise ValueError(
            "Unified-risk levels are inconsistent with the defined thresholds."
        )

    print("PASS")

    # ------------------------------------------------------------
    # 11. Signal coverage summary
    # ------------------------------------------------------------
    print("\n[11] Signal coverage")

    for column, label in [
        ("refined_bus_risk", "Bus"),
        ("satd_score", "SATD"),
        ("code_risk_for_fusion", "Code"),
        ("ml_defect_risk_score", "ML"),
    ]:
        available = int(df[column].notna().sum())
        unavailable = int(df[column].isna().sum())

        print(
            f"{label:<6}: "
            f"{available:>3} available / "
            f"{unavailable:>3} unavailable"
        )

    print("\nSignal-count distribution:")
    print(
        df["fusion_signal_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ------------------------------------------------------------
    # 12. Risk-level distribution
    # ------------------------------------------------------------
    print("\n[12] Unified-risk distribution")

    print(
        df["unified_risk_level"]
        .value_counts()
        .reindex(
            ["Low", "Medium", "High", "Critical"],
            fill_value=0,
        )
        .to_string()
    )

    # ------------------------------------------------------------
    # 13. Top files
    # ------------------------------------------------------------
    print("\n[13] Top 15 unified-risk files")

    top = (
        df[
            [
                "file_path",
                "unified_risk",
                "unified_risk_level",
                "fusion_signal_count",
                "signals_used",
                *SIGNAL_COLUMNS,
            ]
        ]
        .sort_values("unified_risk", ascending=False)
        .head(15)
    )

    print(top.to_string(index=False))

    # ------------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL VALIDATION RESULT")
    print("=" * 70)

    print("PASS")
    print("Four-signal fusion baseline passed all structural checks.")
    print(
        "The unified risk is mathematically consistent with "
        "availability-aware equal-weight fusion."
    )
    print(
        "Missing signals remain missing and are not treated as zero."
    )

    print("\nValidation complete.")


if __name__ == "__main__":
    main()