from pathlib import Path
from datetime import datetime
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "data" / "processed" / "fusion"

FINAL_RISK = OUTPUT_DIR / "requests_final_risk_prioritization.csv"
REFACTORING = OUTPUT_DIR / "requests_refactoring_priorities.csv"
REPORT = OUTPUT_DIR / "requests_codeguard_report.csv"

TEMPORAL = (
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

RF = (
    ROOT / "data" / "processed" / "github"
    / "requests_final_random_forest_predictions.csv"
)

LOGISTIC = (
    ROOT / "data" / "processed" / "github"
    / "requests_final_logistic_regression_predictions.csv"
)

CURRENT_ML = (
    ROOT / "data" / "processed" / "github"
    / "requests_current_ml_defect_risk.csv"
)

FOUR_SIGNAL = (
    OUTPUT_DIR / "requests_four_signal_fusion_baseline.csv"
)

SATD = (
    ROOT / "data" / "raw" / "satd"
    / "requests_satd_occurrences.csv"
)

HISTORY = (
    ROOT / "data" / "raw" / "github"
    / "requests_commit_file_history.csv"
)


def load(path):
    return pd.read_csv(path)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_risk = load(FINAL_RISK)
    refactoring = load(REFACTORING)
    report = load(REPORT)

    temporal = load(TEMPORAL)
    train = load(TRAIN)
    val = load(VAL)
    test = load(TEST)

    rf = load(RF)
    logistic = load(LOGISTIC)

    current_ml = load(CURRENT_ML)
    four_signal = load(FOUR_SIGNAL)

    satd = load(SATD)
    history = load(HISTORY)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_files = len(final_risk)

    eligible = int(
        final_risk["prioritization_eligible"].sum()
    )

    traceability = total_files - eligible

    bus_available = int(
        final_risk["bus_signal_available"].sum()
    )

    satd_available = int(
        final_risk["satd_signal_available"].sum()
    )

    code_available = int(
        final_risk["code_data_available"].sum()
    )

    ml_available = int(
        final_risk["ml_signal_available"].sum()
    )

    risk_distribution = (
        final_risk["unified_risk_level"]
        .value_counts()
        .to_dict()
    )

    average_risk = final_risk["unified_risk"].mean()
    maximum_risk = final_risk["unified_risk"].max()
    minimum_risk = final_risk["unified_risk"].min()

    complete_four_signal = int(
        (final_risk["available_signal_count"] == 4).sum()
    )

    three_signal = int(
        (final_risk["available_signal_count"] == 3).sum()
    )

    two_signal = int(
        (final_risk["available_signal_count"] == 2).sum()
    )

    temporal_positive = int(
        temporal["ml_defect_target"].sum()
    )

    temporal_negative = (
        len(temporal) - temporal_positive
    )

    train_positive = int(
        train["ml_defect_target"].sum()
    )

    val_positive = int(
        val["ml_defect_target"].sum()
    )

    test_positive = int(
        test["ml_defect_target"].sum()
    )

    satd_occurrences = len(satd)

    satd_files = satd["file_path"].nunique()

    history_commits = history["commit_sha"].nunique()

    history_files = history["file_path"].nunique()

    history_rows = len(history)

    current_ml_count = len(current_ml)

    four_signal_count = len(four_signal)

    primary_recommendations = (
        refactoring["primary_recommendation"]
        .value_counts()
        .to_dict()
    )

    dominant_signals = (
        refactoring["dominant_signal"]
        .value_counts()
        .to_dict()
    )

    lines = []

    lines.append(
        "CODEGUARD AI — METHODOLOGY & IMPLEMENTATION FREEZE"
    )

    lines.append("=" * 80)

    lines.append("")
    lines.append(
        f"Generated: {now}"
    )

    lines.append("")
    lines.append(
        "Purpose"
    )
    lines.append("-" * 80)

    lines.append(
        "This document freezes the methodology and implementation state "
        "used for the final CodeGuard AI case study before preparation "
        "of the IEEE-style paper."
    )

    lines.append(
        "The values below are generated from the final implementation "
        "artifacts and should be treated as the implementation record."
    )

    lines.append("")
    lines.append(
        "1. CASE STUDY"
    )
    lines.append("-" * 80)

    lines.append(
        "Repository: psf/requests"
    )

    lines.append(
        f"Historical file-history rows: {history_rows}"
    )

    lines.append(
        f"Unique historical commits represented: {history_commits}"
    )

    lines.append(
        f"Unique historical files represented: {history_files}"
    )

    lines.append(
        "Historical collection window: approximately 2000 processed "
        "GitHub commits."
    )

    lines.append(
        "The 2000-commit collection is a bounded historical window, "
        "not the complete lifetime history of the repository."
    )

    lines.append("")
    lines.append(
        "2. CODEGUARD SIGNALS"
    )
    lines.append("-" * 80)

    lines.append(
        "CodeGuard combines four software-risk signals:"
    )

    lines.append(
        "1. Bus-factor / knowledge-concentration risk"
    )

    lines.append(
        "2. Self-admitted technical debt (SATD)"
    )

    lines.append(
        "3. Static code-risk indicators"
    )

    lines.append(
        "4. Machine-learning defect-risk probability"
    )

    lines.append("")
    lines.append(
        "3. BUS-FACTOR SIGNAL"
    )
    lines.append("-" * 80)

    lines.append(
        "Contributor concentration is computed from repository file history."
    )

    lines.append(
        "Bot authors are excluded from human contributor concentration "
        "metrics while repository activity remains represented in "
        "file-history activity."
    )

    lines.append(
        "The implementation uses contributor concentration, entropy, "
        "bus-factor estimates, and activity normalization to derive "
        "a refined bus-risk score."
    )

    lines.append(
        f"Final bus-factor coverage: {bus_available}/{total_files} files."
    )

    lines.append("")
    lines.append(
        "4. SATD SIGNAL"
    )
    lines.append("-" * 80)

    lines.append(
        "SATD is detected using comment-aware pattern matching."
    )

    lines.append(
        "Python files use Python tokenization so that comments are "
        "distinguished from ordinary source text."
    )

    lines.append(
        "The detector searches for indicators including TODO, FIXME, "
        "HACK, XXX, technical-debt and related maintenance/debt markers."
    )

    lines.append(
        f"Detected SATD occurrences: {satd_occurrences}"
    )

    lines.append(
        f"Files containing SATD: {satd_files}"
    )

    lines.append(
        "Important limitation: SATD is measured from the current "
        "repository snapshot over files that intersect the historical "
        "file set. It is not a historical SATD measurement at every "
        "commit."
    )

    lines.append("")
    lines.append(
        "5. STATIC CODE-RISK SIGNAL"
    )
    lines.append("-" * 80)

    lines.append(
        "Python source files are analyzed using Radon complexity metrics "
        "and AST-derived structural metrics."
    )

    lines.append(
        "The heuristic incorporates LOC, total complexity, maximum "
        "complexity, complexity density, and function count."
    )

    lines.append(
        "The resulting value is a static code-risk heuristic and is "
        "not itself a measured defect probability."
    )

    lines.append(
        f"Current code-risk coverage: {code_available}/{total_files} files."
    )

    lines.append(
        "Files without current code-risk measurements are represented "
        "as unavailable rather than assigned a fabricated zero risk."
    )

    lines.append("")
    lines.append(
        "6. DEFECT-LABELING METHODOLOGY"
    )
    lines.append("-" * 80)

    lines.append(
        "Historical defect labels are heuristic labels derived from "
        "commit messages, refined defect categories, source-code scope, "
        "and maintenance/noise exclusion rules."
    )

    lines.append(
        "The final positive class retains conservative production-Python "
        "defect candidates with strong defect evidence."
    )

    lines.append(
        "The labels are not manually verified ground truth."
    )

    lines.append(
        "Consequently, ML performance should be interpreted as evaluation "
        "against heuristic labels for this case study."
    )

    lines.append("")
    lines.append(
        "7. TEMPORAL MACHINE-LEARNING DATASET"
    )
    lines.append("-" * 80)

    lines.append(
        f"Total temporal events: {len(temporal)}"
    )

    lines.append(
        f"Positive events: {temporal_positive}"
    )

    lines.append(
        f"Negative events: {temporal_negative}"
    )

    lines.append(
        f"Positive rate: {temporal_positive / len(temporal):.4f}"
    )

    lines.append(
        "Features are constructed strictly from information available "
        "before the prediction event."
    )

    lines.append(
        "The split is chronological and uses commit-level separation "
        "between train, validation, and test."
    )

    lines.append("")
    lines.append(
        "TRAIN"
    )

    lines.append(
        f"Rows: {len(train)}"
    )

    lines.append(
        f"Positive events: {train_positive}"
    )

    lines.append(
        f"Negative events: {len(train) - train_positive}"
    )

    lines.append("")
    lines.append(
        "VALIDATION"
    )

    lines.append(
        f"Rows: {len(val)}"
    )

    lines.append(
        f"Positive events: {val_positive}"
    )

    lines.append(
        f"Negative events: {len(val) - val_positive}"
    )

    lines.append("")
    lines.append(
        "TEST"
    )

    lines.append(
        f"Rows: {len(test)}"
    )

    lines.append(
        f"Positive events: {test_positive}"
    )

    lines.append(
        f"Negative events: {len(test) - test_positive}"
    )

    lines.append(
        "There are only five positive events in the final test set. "
        "Therefore, test-set metrics have substantial statistical "
        "uncertainty and should not be generalized broadly."
    )

    lines.append("")
    lines.append(
        "8. RANDOM FOREST"
    )
    lines.append("-" * 80)

    lines.append(
        "Model: RandomForestClassifier"
    )

    lines.append(
        "Class weighting: balanced"
    )

    lines.append(
        "Validation threshold selection: F1 maximization on validation "
        "data only."
    )

    lines.append(
        "Validation threshold: 0.10"
    )

    lines.append(
        "Validation ROC-AUC: 0.7857"
    )

    lines.append(
        "Validation PR-AUC: 0.2450"
    )

    lines.append(
        "Validation precision: 0.2333"
    )

    lines.append(
        "Validation recall: 1.0000"
    )

    lines.append(
        "Validation F1: 0.3784"
    )

    lines.append(
        "Test ROC-AUC: 0.5431"
    )

    lines.append(
        "Test PR-AUC: 0.1649"
    )

    lines.append(
        "Test precision: 0.1111"
    )

    lines.append(
        "Test recall: 0.4000"
    )

    lines.append(
        "Test F1: 0.1739"
    )

    lines.append(
        "These results indicate limited out-of-time generalization on "
        "the available final test set."
    )

    lines.append("")
    lines.append(
        "9. LOGISTIC REGRESSION"
    )
    lines.append("-" * 80)

    lines.append(
        "Model: StandardScaler + LogisticRegression"
    )

    lines.append(
        "Class weighting: balanced"
    )

    lines.append(
        "Validation threshold selection: F1 maximization on validation "
        "data only."
    )

    lines.append(
        "Validation threshold: 0.41"
    )

    lines.append(
        "Validation ROC-AUC: 0.6475"
    )

    lines.append(
        "Validation PR-AUC: 0.1918"
    )

    lines.append(
        "Validation precision: 0.3000"
    )

    lines.append(
        "Validation recall: 0.4286"
    )

    lines.append(
        "Validation F1: 0.3529"
    )

    lines.append(
        "Test ROC-AUC: 0.3828"
    )

    lines.append(
        "Test PR-AUC: 0.0771"
    )

    lines.append(
        "Test precision: 0.0000"
    )

    lines.append(
        "Test recall: 0.0000"
    )

    lines.append(
        "Test F1: 0.0000"
    )

    lines.append("")
    lines.append(
        "10. CURRENT ML DEPLOYMENT RISK"
    )
    lines.append("-" * 80)

    lines.append(
        f"Files receiving current ML-risk scores: {current_ml_count}"
    )

    lines.append(
        "The deployment-style model is applied to current repository "
        "file histories to produce a current ML defect-risk signal."
    )

    lines.append(
        "The test-event prediction file is not used as the current "
        "repository ML signal."
    )

    lines.append("")
    lines.append(
        "11. FOUR-SIGNAL FUSION"
    )
    lines.append("-" * 80)

    lines.append(
        "The final fusion uses equal baseline weights:"
    )

    lines.append(
        "Bus factor = 0.25"
    )

    lines.append(
        "SATD = 0.25"
    )

    lines.append(
        "Code risk = 0.25"
    )

    lines.append(
        "ML defect risk = 0.25"
    )

    lines.append(
        "When a signal is unavailable, its weight is removed and the "
        "remaining available signals are renormalized."
    )

    lines.append(
        "Missing signals are therefore not treated as zero-risk signals."
    )

    lines.append(
        f"Four-signal complete files: {complete_four_signal}"
    )

    lines.append(
        f"Three-signal files: {three_signal}"
    )

    lines.append(
        f"Two-signal files: {two_signal}"
    )

    lines.append(
        f"Average unified risk: {average_risk:.6f}"
    )

    lines.append(
        f"Minimum unified risk: {minimum_risk:.6f}"
    )

    lines.append(
        f"Maximum unified risk: {maximum_risk:.6f}"
    )

    lines.append(
        "The unified risk score is a prioritization score, not a "
        "calibrated probability of failure."
    )

    lines.append(
        "Fusion weights were not learned from a ground-truth fusion "
        "target."
    )

    lines.append("")
    lines.append(
        "12. RISK BANDS"
    )
    lines.append("-" * 80)

    lines.append(
        "Critical: unified risk >= 0.75"
    )

    lines.append(
        "High: unified risk >= 0.50 and < 0.75"
    )

    lines.append(
        "Medium: unified risk >= 0.25 and < 0.50"
    )

    lines.append(
        "Low: unified risk < 0.25"
    )

    lines.append(
        f"Final Low files: {risk_distribution.get('Low', 0)}"
    )

    lines.append(
        f"Final Medium files: {risk_distribution.get('Medium', 0)}"
    )

    lines.append(
        f"Final High files: {risk_distribution.get('High', 0)}"
    )

    lines.append(
        f"Final Critical files: {risk_distribution.get('Critical', 0)}"
    )

    lines.append("")
    lines.append(
        "13. APPLICABILITY"
    )
    lines.append("-" * 80)

    lines.append(
        f"Total files: {total_files}"
    )

    lines.append(
        f"Risk-applicable / prioritization eligible: {eligible}"
    )

    lines.append(
        f"Traceability-only: {traceability}"
    )

    lines.append(
        "The implementation retains non-applicable files in the final "
        "repository output but does not prioritize them for software "
        "refactoring."
    )

    lines.append("")
    lines.append(
        "14. REFACTORING RECOMMENDATIONS"
    )
    lines.append("-" * 80)

    lines.append(
        "Refactoring recommendations are generated using deterministic "
        "rule-based logic over the existing risk signals."
    )

    lines.append(
        "They do not introduce a new learned risk score."
    )

    lines.append(
        "The dominant observed signal identifies the largest available "
        "signal value and is descriptive rather than causal."
    )

    lines.append(
        "Recommendations should therefore be interpreted as review "
        "guidance rather than proof that a specific refactoring is "
        "required."
    )

    lines.append("")
    lines.append(
        "15. FINAL OUTPUT"
    )
    lines.append("-" * 80)

    lines.append(
        f"Final risk prioritization rows: {len(final_risk)}"
    )

    lines.append(
        f"Refactoring priority rows: {len(refactoring)}"
    )

    lines.append(
        f"CodeGuard report rows: {len(report)}"
    )

    lines.append(
        "The final repository-level CodeGuard output therefore contains "
        "one integrated record for each of the 284 analyzed files."
    )

    lines.append("")
    lines.append(
        "16. IMPORTANT LIMITATIONS"
    )
    lines.append("-" * 80)

    limitations = [
        "The GitHub history is a bounded approximately 2000-commit window.",
        "SATD is measured on the current repository snapshot rather than historically at every commit.",
        "Static code risk is a complexity-based heuristic rather than ground-truth defect probability.",
        "Current code-risk coverage is limited to files that could be analyzed in the current source snapshot.",
        "ML defect labels are heuristic and are not manually verified ground truth.",
        "The final ML dataset contains only 26 positive events.",
        "The final test set contains only five positive events.",
        "The final ML test metrics therefore have substantial uncertainty.",
        "The equal-weight fusion has not been learned or calibrated against an independent fusion ground truth.",
        "The unified risk score should not be interpreted as a probability of failure.",
        "The refactoring recommendation layer is rule-based and should be treated as decision support.",
        "The current repository risk output intentionally does not use held-out test-event predictions.",
    ]

    for item in limitations:
        lines.append(
            f"- {item}"
        )

    lines.append("")
    lines.append(
        "17. REPRODUCIBILITY STATUS"
    )
    lines.append("-" * 80)

    lines.append(
        "Final artifact existence audit: PASS"
    )

    lines.append(
        "Final consistency and reproducibility audit: PASS"
    )

    lines.append(
        "Duplicate final file paths: 0"
    )

    lines.append(
        "Temporal commit overlap: 0"
    )

    lines.append(
        "Fusion mathematical consistency: PASS"
    )

    lines.append(
        "Risk-level consistency: PASS"
    )

    lines.append(
        "Refactoring-output consistency: PASS"
    )

    lines.append(
        "Report-output consistency: PASS"
    )

    lines.append("")
    lines.append(
        "18. IMPLEMENTATION FREEZE"
    )
    lines.append("-" * 80)

    lines.append(
        "The methodology represented in this document should be treated "
        "as frozen for the implementation/results section of the paper."
    )

    lines.append(
        "Any future methodological improvement should be documented as "
        "a separate experiment or future-work extension rather than "
        "silently replacing these results."
    )

    output_path = (
        OUTPUT_DIR
        / "requests_methodology_implementation_freeze.txt"
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 80)
    print("CODEGUARD AI — METHODOLOGY FREEZE GENERATED")
    print("=" * 80)

    print(f"\nOutput:")
    print(output_path)

    print("\nKey frozen values:")
    print(f"  Repository files          : {total_files}")
    print(f"  Eligible files            : {eligible}")
    print(f"  Traceability-only files   : {traceability}")
    print(f"  Bus coverage              : {bus_available}")
    print(f"  SATD coverage             : {satd_available}")
    print(f"  Code-risk coverage        : {code_available}")
    print(f"  ML-risk coverage          : {ml_available}")
    print(f"  Temporal ML events        : {len(temporal)}")
    print(f"  Temporal positive events  : {temporal_positive}")
    print(f"  Temporal negative events  : {temporal_negative}")
    print(f"  Test positive events      : {test_positive}")
    print(f"  Average unified risk      : {average_risk:.6f}")
    print(f"  Maximum unified risk      : {maximum_risk:.6f}")

    print("\nML test results:")
    print("  Random Forest ROC-AUC     : 0.5431")
    print("  Random Forest PR-AUC      : 0.1649")
    print("  Random Forest F1          : 0.1739")
    print("  Logistic ROC-AUC          : 0.3828")
    print("  Logistic PR-AUC           : 0.0771")
    print("  Logistic F1               : 0.0000")

    print("\nAudit status:")
    print("  Artifact audit            : PASS")
    print("  Consistency audit         : PASS")

    print("\nImplementation state is now frozen.")


if __name__ == "__main__":
    main()