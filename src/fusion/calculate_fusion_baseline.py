import os
import pandas as pd


INPUT_FILE = "data/processed/fusion/requests_three_signal.csv"
OUTPUT_FILE = "data/processed/fusion/requests_fusion_baseline.csv"


def main():
    print("=" * 70)
    print("CodeGuard AI - Three-Signal Fusion Baseline")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput files : {len(df)}")

    required_columns = [
        "file_path",
        "refined_bus_risk",
        "satd_score",
        "code_analyzed",
        "code_risk_for_fusion",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns:\n" +
            "\n".join(missing)
        )

    # ------------------------------------------------------------
    # Use only files where all three signals are available.
    # ------------------------------------------------------------

    complete = df[
        df["code_analyzed"] &
        df["code_risk_for_fusion"].notna()
    ].copy()

    print(f"Complete three-signal files : {len(complete)}")
    print(f"Excluded incomplete files   : {len(df) - len(complete)}")

    if len(complete) == 0:
        raise ValueError("No complete three-signal files available.")

    # ------------------------------------------------------------
    # Clean and constrain the three signals.
    # ------------------------------------------------------------

    complete["refined_bus_risk"] = (
        pd.to_numeric(
            complete["refined_bus_risk"],
            errors="coerce"
        )
        .clip(0, 1)
    )

    complete["satd_score"] = (
        pd.to_numeric(
            complete["satd_score"],
            errors="coerce"
        )
        .clip(0, 1)
    )

    complete["code_risk_for_fusion"] = (
        pd.to_numeric(
            complete["code_risk_for_fusion"],
            errors="coerce"
        )
        .clip(0, 1)
    )

    # ------------------------------------------------------------
    # Remove rows with invalid signal values.
    # ------------------------------------------------------------

    signal_columns = [
        "refined_bus_risk",
        "satd_score",
        "code_risk_for_fusion",
    ]

    before_drop = len(complete)

    complete = complete.dropna(
        subset=signal_columns
    ).copy()

    dropped = before_drop - len(complete)

    if dropped > 0:
        print(f"Rows removed due to missing signals : {dropped}")

    # ------------------------------------------------------------
    # Baseline equal-weight fusion.
    # ------------------------------------------------------------

    BUS_WEIGHT = 0.3333
    SATD_WEIGHT = 0.3333
    CODE_WEIGHT = 0.3334

    complete["bus_weight"] = BUS_WEIGHT
    complete["satd_weight"] = SATD_WEIGHT
    complete["code_weight"] = CODE_WEIGHT

    complete["bus_component"] = (
        complete["refined_bus_risk"] * BUS_WEIGHT
    )

    complete["satd_component"] = (
        complete["satd_score"] * SATD_WEIGHT
    )

    complete["code_component"] = (
        complete["code_risk_for_fusion"] * CODE_WEIGHT
    )

    complete["unified_risk_score"] = (
        complete["bus_component"] +
        complete["satd_component"] +
        complete["code_component"]
    ).clip(0, 1)

    complete["unified_risk_score"] = (
        complete["unified_risk_score"]
        .round(6)
    )

    # ------------------------------------------------------------
    # Risk levels.
    # ------------------------------------------------------------

    def risk_level(score):
        if score >= 0.75:
            return "Critical"
        elif score >= 0.50:
            return "High"
        elif score >= 0.25:
            return "Medium"
        else:
            return "Low"

    complete["unified_risk_level"] = (
        complete["unified_risk_score"]
        .apply(risk_level)
    )

    complete["fusion_method"] = (
        "equal_weight_three_signal_baseline_v1"
    )

    # ------------------------------------------------------------
    # Add rank for easier inspection.
    # ------------------------------------------------------------

    complete["risk_rank"] = (
        complete["unified_risk_score"]
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)
    )

    # ------------------------------------------------------------
    # Validate uniqueness.
    # ------------------------------------------------------------

    if complete["file_path"].duplicated().any():
        duplicates = (
            complete[
                complete["file_path"].duplicated(keep=False)
            ]["file_path"]
            .tolist()
        )

        raise ValueError(
            "Duplicate file paths detected:\n" +
            "\n".join(duplicates)
        )

    # ------------------------------------------------------------
    # Select useful output columns.
    # ------------------------------------------------------------

    output_columns = [
        "file_path",
        "knowledge_relevant",

        # Original signals
        "refined_bus_risk",
        "satd_count",
        "satd_score",
        "code_risk_for_fusion",

        # Code metrics
        "loc",
        "function_count",
        "class_count",
        "complexity_total",
        "complexity_average",
        "complexity_max",
        "complexity_density",

        # Fusion components
        "bus_weight",
        "satd_weight",
        "code_weight",
        "bus_component",
        "satd_component",
        "code_component",

        # Final result
        "unified_risk_score",
        "unified_risk_level",
        "risk_rank",
        "fusion_method",
    ]

    output_columns = [
        c for c in output_columns
        if c in complete.columns
    ]

    result = complete[output_columns].copy()

    # ------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FUSION BASELINE SUMMARY")
    print("=" * 70)

    print(f"Complete files       : {len(result)}")
    print(
        f"Average unified risk : "
        f"{result['unified_risk_score'].mean():.4f}"
    )
    print(
        f"Maximum unified risk : "
        f"{result['unified_risk_score'].max():.4f}"
    )
    print(
        f"Minimum unified risk : "
        f"{result['unified_risk_score'].min():.4f}"
    )

    print("\nUNIFIED RISK LEVEL DISTRIBUTION")
    print("-" * 70)

    print(
        result["unified_risk_level"]
        .value_counts()
        .reindex(
            ["Critical", "High", "Medium", "Low"],
            fill_value=0
        )
        .to_string()
    )

    print("\nTOP 15 FILES BY UNIFIED RISK")
    print("-" * 70)

    display_columns = [
        "risk_rank",
        "file_path",
        "refined_bus_risk",
        "satd_score",
        "code_risk_for_fusion",
        "unified_risk_score",
        "unified_risk_level",
    ]

    print(
        result.sort_values(
            "unified_risk_score",
            ascending=False
        )[display_columns]
        .head(15)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print(f"Output saved to: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()