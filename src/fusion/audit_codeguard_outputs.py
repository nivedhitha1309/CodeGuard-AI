from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

FILES_TO_CHECK = {
    "Repository history":
        ROOT / "data" / "raw" / "github" / "requests_commit_file_history.csv",

    "SATD occurrences":
        ROOT / "data" / "raw" / "satd" / "requests_satd_occurrences.csv",

    "SATD aggregated":
        ROOT / "data" / "processed" / "satd" / "requests_satd_aggregated.csv",

    "Bus factor refined":
        ROOT / "data" / "processed" / "bus_factor" / "requests_bus_factor_refined.csv",

    "Bus + SATD merged":
        ROOT / "data" / "processed" / "fusion" / "requests_bus_satd_merged.csv",

    "Three-signal dataset":
        ROOT / "data" / "processed" / "fusion" / "requests_three_signal.csv",

    "Current code risk":
        ROOT / "data" / "processed" / "code_risk" / "requests_code_risk.csv",

    "Clean defect targets":
        ROOT / "data" / "processed" / "github" / "requests_clean_defect_targets.csv",

    "Strict source defect targets":
        ROOT / "data" / "processed" / "github" / "requests_strict_source_defect_targets.csv",

    "Final ML defect targets":
        ROOT / "data" / "processed" / "github" / "requests_final_ml_defect_targets.csv",

    "Final temporal ML dataset":
        ROOT / "data" / "processed" / "github" / "requests_final_temporal_ml_dataset.csv",

    "Final temporal train":
        ROOT / "data" / "processed" / "github" / "requests_final_temporal_train.csv",

    "Final temporal validation":
        ROOT / "data" / "processed" / "github" / "requests_final_temporal_val.csv",

    "Final temporal test":
        ROOT / "data" / "processed" / "github" / "requests_final_temporal_test.csv",

    "RF predictions":
        ROOT / "data" / "processed" / "github" / "requests_final_random_forest_predictions.csv",

    "Logistic predictions":
        ROOT / "data" / "processed" / "github" / "requests_final_logistic_regression_predictions.csv",

    "Current ML defect risk":
        ROOT / "data" / "processed" / "github" / "requests_current_ml_defect_risk.csv",

    "Four-signal dataset":
        ROOT / "data" / "processed" / "fusion" / "requests_four_signal_dataset.csv",

    "Four-signal fusion":
        ROOT / "data" / "processed" / "fusion" / "requests_four_signal_fusion_baseline.csv",

    "Fusion applicability":
        ROOT / "data" / "processed" / "fusion" / "requests_four_signal_fusion_applicability.csv",

    "Final risk prioritization":
        ROOT / "data" / "processed" / "fusion" / "requests_final_risk_prioritization.csv",

    "Refactoring priorities":
        ROOT / "data" / "processed" / "fusion" / "requests_refactoring_priorities.csv",

    "CodeGuard report CSV":
        ROOT / "data" / "processed" / "fusion" / "requests_codeguard_report.csv",

    "CodeGuard summary":
        ROOT / "data" / "processed" / "fusion" / "requests_codeguard_summary.txt",

    "Random Forest model":
        ROOT / "models" / "final_random_forest_temporal.joblib",

    "Logistic Regression model":
        ROOT / "models" / "final_logistic_regression_temporal.joblib",
}


def inspect_csv(path):
    try:
        df = pd.read_csv(path)

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "status": "OK",
        }

    except Exception as exc:
        return {
            "rows": "-",
            "columns": "-",
            "status": f"READ ERROR: {exc}",
        }


def main():
    print("=" * 80)
    print("CODEGUARD AI — FINAL IMPLEMENTATION OUTPUT AUDIT")
    print("=" * 80)

    print(f"\nProject root: {ROOT}")

    found = 0
    missing = 0
    readable = 0

    print("\n" + "-" * 80)
    print("FILE CHECK")
    print("-" * 80)

    for name, path in FILES_TO_CHECK.items():

        if not path.exists():
            print(f"[MISSING] {name}")
            print(f"         {path}")
            missing += 1
            continue

        found += 1

        if path.suffix.lower() == ".csv":
            info = inspect_csv(path)

            if info["status"] == "OK":
                readable += 1
                print(
                    f"[OK]      {name:<30} "
                    f"rows={info['rows']:<6} "
                    f"columns={info['columns']:<4}"
                )
            else:
                print(
                    f"[ERROR]   {name:<30} "
                    f"{info['status']}"
                )

        else:
            size_kb = path.stat().st_size / 1024

            print(
                f"[OK]      {name:<30} "
                f"size={size_kb:.1f} KB"
            )
            readable += 1

    print("\n" + "-" * 80)
    print("AUDIT SUMMARY")
    print("-" * 80)

    print(f"Expected artifacts : {len(FILES_TO_CHECK)}")
    print(f"Found artifacts    : {found}")
    print(f"Missing artifacts  : {missing}")
    print(f"Readable artifacts : {readable}")

    if missing == 0:
        print("\nRESULT: PASS")
        print("All expected CodeGuard AI artifacts are present.")
    else:
        print("\nRESULT: REVIEW REQUIRED")
        print("One or more expected artifacts are missing.")

    print("\n" + "-" * 80)
    print("KEY FINAL OUTPUTS")
    print("-" * 80)

    key_files = [
        ROOT / "data" / "processed" / "fusion" /
        "requests_final_risk_prioritization.csv",

        ROOT / "data" / "processed" / "fusion" /
        "requests_refactoring_priorities.csv",

        ROOT / "data" / "processed" / "fusion" /
        "requests_codeguard_summary.txt",
    ]

    for path in key_files:
        print(f"\n{path}")

        if path.exists():
            print(f"  Exists : YES")
            print(f"  Size   : {path.stat().st_size / 1024:.1f} KB")

            if path.suffix.lower() == ".csv":
                try:
                    df = pd.read_csv(path)
                    print(f"  Rows   : {len(df)}")
                    print(f"  Columns: {len(df.columns)}")
                except Exception as exc:
                    print(f"  Read error: {exc}")
        else:
            print("  Exists : NO")

    print("\n" + "=" * 80)
    print("END OF AUDIT")
    print("=" * 80)


if __name__ == "__main__":
    main()