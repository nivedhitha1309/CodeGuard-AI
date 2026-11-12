from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

FINAL_RISK = (
    ROOT / "data" / "processed" / "fusion"
    / "requests_final_risk_prioritization.csv"
)

REFACTORING = (
    ROOT / "data" / "processed" / "fusion"
    / "requests_refactoring_priorities.csv"
)

REPORT = (
    ROOT / "data" / "processed" / "fusion"
    / "requests_codeguard_report.csv"
)

FOUR_SIGNAL = (
    ROOT / "data" / "processed" / "fusion"
    / "requests_four_signal_fusion_baseline.csv"
)

APPLICABILITY = (
    ROOT / "data" / "processed" / "fusion"
    / "requests_four_signal_fusion_applicability.csv"
)

CURRENT_ML = (
    ROOT / "data" / "processed" / "github"
    / "requests_current_ml_defect_risk.csv"
)

TEMPORAL_DATASET = (
    ROOT / "data" / "processed" / "github"
    / "requests_final_temporal_ml_dataset.csv"
)

TRAIN = (
    ROOT / "data" / "processed" / "github"
    / "requests_final_temporal_train.csv"
)

VAL = (
    ROOT / "data" / "processed" / "github"
    / "requests_final_temporal_val.csv"
)

TEST = (
    ROOT / "data" / "processed" / "github"
    / "requests_final_temporal_test.csv"
)

RF_PRED = (
    ROOT / "data" / "processed" / "github"
    / "requests_final_random_forest_predictions.csv"
)

LOG_PRED = (
    ROOT / "data" / "processed" / "github"
    / "requests_final_logistic_regression_predictions.csv"
)


def check(condition, message, failures):
    if condition:
        print(f"[PASS] {message}")
    else:
        print(f"[FAIL] {message}")
        failures.append(message)


def load_csv(path, name, failures):
    if not path.exists():
        print(f"[FAIL] Missing file: {name}")
        failures.append(f"Missing file: {name}")
        return None

    try:
        df = pd.read_csv(path)
        print(f"[OK]   {name}: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as exc:
        print(f"[FAIL] Cannot read {name}: {exc}")
        failures.append(f"Cannot read {name}")
        return None


def compare_series(a, b, tolerance=1e-9):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")

    if len(a) != len(b):
        return False

    return np.allclose(
        a.to_numpy(),
        b.to_numpy(),
        equal_nan=True,
        atol=tolerance,
        rtol=tolerance,
    )


def main():

    failures = []

    print("=" * 80)
    print("CODEGUARD AI — FINAL CONSISTENCY & REPRODUCIBILITY AUDIT")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("LOADING FINAL ARTIFACTS")
    print("-" * 80)

    final_risk = load_csv(FINAL_RISK, "Final risk prioritization", failures)
    refactoring = load_csv(REFACTORING, "Refactoring priorities", failures)
    report = load_csv(REPORT, "CodeGuard report", failures)
    four_signal = load_csv(FOUR_SIGNAL, "Four-signal fusion", failures)
    applicability = load_csv(APPLICABILITY, "Fusion applicability", failures)
    current_ml = load_csv(CURRENT_ML, "Current ML risk", failures)

    temporal = load_csv(TEMPORAL_DATASET, "Final temporal dataset", failures)
    train = load_csv(TRAIN, "Temporal train", failures)
    val = load_csv(VAL, "Temporal validation", failures)
    test = load_csv(TEST, "Temporal test", failures)

    rf = load_csv(RF_PRED, "Random Forest predictions", failures)
    logistic = load_csv(LOG_PRED, "Logistic predictions", failures)

    if failures:
        print("\nCannot continue because required files could not be loaded.")
        print(f"Failures: {len(failures)}")
        return

    # ------------------------------------------------------------------
    # 1. FINAL REPOSITORY COVERAGE
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("1. FINAL REPOSITORY COVERAGE")
    print("-" * 80)

    check(
        len(final_risk) == 284,
        "Final prioritization contains exactly 284 files",
        failures,
    )

    check(
        len(refactoring) == 284,
        "Refactoring output contains exactly 284 files",
        failures,
    )

    check(
        len(report) == 284,
        "CodeGuard report contains exactly 284 files",
        failures,
    )

    check(
        final_risk["file_path"].nunique() == len(final_risk),
        "Final prioritization has no duplicate file paths",
        failures,
    )

    check(
        refactoring["file_path"].nunique() == len(refactoring),
        "Refactoring output has no duplicate file paths",
        failures,
    )

    check(
        report["file_path"].nunique() == len(report),
        "Report has no duplicate file paths",
        failures,
    )

    # ------------------------------------------------------------------
    # 2. FILE COVERAGE BETWEEN FINAL OUTPUTS
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("2. FINAL OUTPUT FILE COVERAGE")
    print("-" * 80)

    final_files = set(final_risk["file_path"])
    refactoring_files = set(refactoring["file_path"])
    report_files = set(report["file_path"])

    check(
        final_files == refactoring_files,
        "Final prioritization and refactoring outputs contain identical files",
        failures,
    )

    check(
        final_files == report_files,
        "Final prioritization and report contain identical files",
        failures,
    )

    # ------------------------------------------------------------------
    # 3. APPLICABILITY CONSISTENCY
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("3. APPLICABILITY CONSISTENCY")
    print("-" * 80)

    applicability_files = set(applicability["file_path"])

    check(
        final_files == applicability_files,
        "Final prioritization covers exactly the applicability dataset files",
        failures,
    )

    eligible_count = int(final_risk["prioritization_eligible"].sum())

    check(
        eligible_count == 192,
        "Final prioritization has exactly 192 eligible files",
        failures,
    )

    traceability_count = int((~final_risk["prioritization_eligible"]).sum())

    check(
        traceability_count == 92,
        "Final prioritization has exactly 92 traceability-only files",
        failures,
    )

    check(
        eligible_count + traceability_count == len(final_risk),
        "Eligible + traceability-only files account for all 284 files",
        failures,
    )

    # ------------------------------------------------------------------
    # 4. SIGNAL COVERAGE
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("4. SIGNAL COVERAGE")
    print("-" * 80)

    bus_available = int(final_risk["bus_signal_available"].sum())
    satd_available = int(final_risk["satd_signal_available"].sum())
    code_available = int(final_risk["code_data_available"].sum())
    ml_available = int(final_risk["ml_signal_available"].sum())

    print(f"Bus signal available : {bus_available}")
    print(f"SATD signal available: {satd_available}")
    print(f"Code signal available: {code_available}")
    print(f"ML signal available  : {ml_available}")

    check(
        bus_available == 284,
        "Bus-factor signal available for all 284 files",
        failures,
    )

    check(
        satd_available == 284,
        "SATD signal available for all 284 files",
        failures,
    )

    check(
        code_available == 35,
        "Code-risk signal available for exactly 35 files",
        failures,
    )

    check(
        ml_available == 124,
        "ML signal available for exactly 124 files",
        failures,
    )

    # ------------------------------------------------------------------
    # 5. SIGNAL COUNT / COMPLETENESS
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("5. SIGNAL COUNT AND COMPLETENESS")
    print("-" * 80)

    expected_count = (
        final_risk["bus_signal_available"].astype(int)
        + final_risk["satd_signal_available"].astype(int)
        + final_risk["code_data_available"].astype(int)
        + final_risk["ml_signal_available"].astype(int)
    )

    check(
        (expected_count == final_risk["available_signal_count"]).all(),
        "Available signal counts are mathematically correct",
        failures,
    )

    expected_completeness = expected_count / 4.0

    check(
        compare_series(
            expected_completeness,
            final_risk["signal_completeness"],
        ),
        "Signal completeness values are mathematically correct",
        failures,
    )

    counts = final_risk["available_signal_count"].value_counts().sort_index()

    print("\nSignal-count distribution:")
    for count, number in counts.items():
        print(f"  {int(count)}/4 signals : {number} files")

    # ------------------------------------------------------------------
    # 6. FUSION SCORE REPRODUCTION
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("6. FUSION SCORE REPRODUCTION")
    print("-" * 80)

    signal_columns = [
        "refined_bus_risk",
        "satd_score",
        "code_risk_for_fusion",
        "ml_defect_risk_score",
    ]

    reproduced_scores = []

    for _, row in final_risk.iterrows():

        values = []

        for column in signal_columns:
            value = pd.to_numeric(
                pd.Series([row[column]]),
                errors="coerce",
            ).iloc[0]

            if pd.notna(value):
                values.append(float(value))

        if values:
            reproduced_scores.append(
                float(np.mean(values))
            )
        else:
            reproduced_scores.append(np.nan)

    reproduced_scores = pd.Series(
        reproduced_scores,
        index=final_risk.index,
    )

    check(
        compare_series(
            reproduced_scores,
            final_risk["unified_risk"],
        ),
        "Unified risk exactly matches availability-aware equal-weight fusion",
        failures,
    )

    print(
        f"Reproduced average unified risk: "
        f"{reproduced_scores.mean():.6f}"
    )

    print(
        f"Stored average unified risk:    "
        f"{pd.to_numeric(final_risk['unified_risk']).mean():.6f}"
    )

    # ------------------------------------------------------------------
    # 7. RISK LEVEL CONSISTENCY
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("7. RISK LEVEL CONSISTENCY")
    print("-" * 80)

    def expected_level(score):
        if pd.isna(score):
            return "Unavailable"

        score = float(score)

        if score >= 0.75:
            return "Critical"
        elif score >= 0.50:
            return "High"
        elif score >= 0.25:
            return "Medium"
        else:
            return "Low"

    expected_levels = final_risk["unified_risk"].apply(expected_level)

    check(
        (expected_levels == final_risk["unified_risk_level"]).all(),
        "Unified risk levels match the documented thresholds",
        failures,
    )

    print("\nStored risk distribution:")
    print(final_risk["unified_risk_level"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 8. REFACTORING OUTPUT CONSISTENCY
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("8. REFACTORING OUTPUT CONSISTENCY")
    print("-" * 80)

    merged = final_risk[
        [
            "file_path",
            "unified_risk",
            "unified_risk_level",
            "prioritization_eligible",
        ]
    ].merge(
        refactoring,
        on="file_path",
        how="inner",
        suffixes=("_risk", "_refactor"),
    )

    check(
        len(merged) == 284,
        "All 284 files successfully join between risk and refactoring outputs",
        failures,
    )

    check(
        compare_series(
            merged["unified_risk_risk"],
            merged["unified_risk_refactor"],
        ),
        "Unified risk values match between risk and refactoring outputs",
        failures,
    )

    check(
        (
            merged["unified_risk_level_risk"]
            == merged["unified_risk_level_refactor"]
        ).all(),
        "Unified risk levels match between risk and refactoring outputs",
        failures,
    )

    # ------------------------------------------------------------------
    # 9. REPORT CONSISTENCY
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("9. REPORT CONSISTENCY")
    print("-" * 80)

    report_merge = final_risk[
        [
            "file_path",
            "unified_risk",
            "unified_risk_level",
            "prioritization_eligible",
        ]
    ].merge(
        report,
        on="file_path",
        how="inner",
        suffixes=("_risk", "_report"),
    )

    check(
        len(report_merge) == 284,
        "All 284 files successfully join between risk and report outputs",
        failures,
    )

    # Report column names can vary depending on report generation.
    if "unified_risk_report" in report_merge.columns:

        check(
            compare_series(
                report_merge["unified_risk_risk"],
                report_merge["unified_risk_report"],
            ),
            "Unified risk values match between risk and report outputs",
            failures,
        )

    if "unified_risk_level_report" in report_merge.columns:

        check(
            (
                report_merge["unified_risk_level_risk"]
                == report_merge["unified_risk_level_report"]
            ).all(),
            "Unified risk levels match between risk and report outputs",
            failures,
        )

    # ------------------------------------------------------------------
    # 10. TEMPORAL ML DATASET CONSISTENCY
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("10. TEMPORAL ML DATASET CONSISTENCY")
    print("-" * 80)

    check(
        len(temporal) == 574,
        "Final temporal dataset contains 574 events",
        failures,
    )

    temporal_positive = int(temporal["ml_defect_target"].sum())

    check(
        temporal_positive == 26,
        "Final temporal dataset contains exactly 26 positive events",
        failures,
    )

    temporal_negative = len(temporal) - temporal_positive

    check(
        temporal_negative == 548,
        "Final temporal dataset contains exactly 548 negative events",
        failures,
    )

    split_total = len(train) + len(val) + len(test)

    check(
        split_total == len(temporal),
        "Train + validation + test rows equal the complete temporal dataset",
        failures,
    )

    split_positive = (
        int(train["ml_defect_target"].sum())
        + int(val["ml_defect_target"].sum())
        + int(test["ml_defect_target"].sum())
    )

    check(
        split_positive == temporal_positive,
        "Train + validation + test positives equal the complete dataset",
        failures,
    )

    # Commit overlap
    train_commits = set(train["commit_sha"])
    val_commits = set(val["commit_sha"])
    test_commits = set(test["commit_sha"])

    check(
        not (train_commits & val_commits),
        "Train and validation have no commit overlap",
        failures,
    )

    check(
        not (train_commits & test_commits),
        "Train and test have no commit overlap",
        failures,
    )

    check(
        not (val_commits & test_commits),
        "Validation and test have no commit overlap",
        failures,
    )

    # ------------------------------------------------------------------
    # 11. PREDICTION DATASET CONSISTENCY
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("11. ML PREDICTION CONSISTENCY")
    print("-" * 80)

    check(
        len(rf) == len(test),
        "Random Forest predictions contain exactly one row per test event",
        failures,
    )

    check(
        len(logistic) == len(test),
        "Logistic predictions contain exactly one row per test event",
        failures,
    )

    rf_pairs = set(
        zip(
            rf["commit_sha"].astype(str),
            rf["file_path"].astype(str),
        )
    )

    test_pairs = set(
        zip(
            test["commit_sha"].astype(str),
            test["file_path"].astype(str),
        )
    )

    logistic_pairs = set(
        zip(
            logistic["commit_sha"].astype(str),
            logistic["file_path"].astype(str),
        )
    )

    check(
        rf_pairs == test_pairs,
        "Random Forest predictions map exactly to test events",
        failures,
    )

    check(
        logistic_pairs == test_pairs,
        "Logistic predictions map exactly to test events",
        failures,
    )

    # ------------------------------------------------------------------
    # 12. CURRENT ML RISK SANITY
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("12. CURRENT ML RISK SANITY")
    print("-" * 80)

    check(
        len(current_ml) == 124,
        "Current ML deployment risk contains 124 scored files",
        failures,
    )

    check(
        current_ml["file_path"].nunique() == len(current_ml),
        "Current ML risk has no duplicate files",
        failures,
    )

    current_risk_values = pd.to_numeric(
        current_ml["ml_defect_risk"],
        errors="coerce",
    )

    check(
        current_risk_values.notna().all(),
        "Current ML risk contains no missing probabilities",
        failures,
    )

    check(
        ((current_risk_values >= 0) & (current_risk_values <= 1)).all(),
        "Current ML risk values are within [0,1]",
        failures,
    )

    # ------------------------------------------------------------------
    # 13. FOUR-SIGNAL OUTPUT CONSISTENCY
    # ------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("13. FOUR-SIGNAL OUTPUT CONSISTENCY")
    print("-" * 80)

    check(
        len(four_signal) == 284,
        "Four-signal fusion contains 284 files",
        failures,
    )

    check(
        len(applicability) == 284,
        "Applicability dataset contains 284 files",
        failures,
    )

    four_files = set(four_signal["file_path"])
    app_files = set(applicability["file_path"])

    check(
        four_files == app_files,
        "Four-signal and applicability datasets contain identical files",
        failures,
    )

    # ------------------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL CONSISTENCY AUDIT RESULT")
    print("=" * 80)

    print(f"Total checks failed : {len(failures)}")

    if not failures:
        print("\nRESULT: PASS")
        print("The final CodeGuard AI implementation is internally consistent.")
        print("All major pipeline stages agree on file coverage, signal")
        print("availability, fusion mathematics, temporal ML splits, and")
        print("final prioritization outputs.")
    else:
        print("\nRESULT: REVIEW REQUIRED")
        print("\nFailed checks:")

        for failure in failures:
            print(f" - {failure}")

    print("\n" + "=" * 80)
    print("END OF CONSISTENCY AUDIT")
    print("=" * 80)


if __name__ == "__main__":
    main()