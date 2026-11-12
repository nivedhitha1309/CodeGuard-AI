from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

THREE_SIGNAL_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "fusion"
    / "requests_three_signal.csv"
)

RF_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_final_random_forest_predictions.csv"
)

BUS_SATD_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "fusion"
    / "requests_bus_satd_merged.csv"
)

CODE_RISK_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "code_risk"
    / "requests_code_risk.csv"
)


def inspect_file(path, name):

    print("=" * 80)
    print(name)
    print("=" * 80)

    if not path.exists():
        print(f"NOT FOUND: {path}")
        print()
        return None

    df = pd.read_csv(path)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")
    print()

    print("Column names:")
    for column in df.columns:
        print(f"  - {column}")

    print()

    print("First 5 rows:")
    print(
        df.head(5).to_string(
            index=False
        )
    )

    print()

    return df


def main():

    print("=" * 80)
    print("CODEGUARD AI - FINAL FUSION INPUT INSPECTION")
    print("=" * 80)
    print()

    three_signal = inspect_file(
        THREE_SIGNAL_FILE,
        "THREE-SIGNAL DATASET",
    )

    bus_satd = inspect_file(
        BUS_SATD_FILE,
        "BUS + SATD DATASET",
    )

    code_risk = inspect_file(
        CODE_RISK_FILE,
        "CODE-RISK DATASET",
    )

    rf = inspect_file(
        RF_FILE,
        "RANDOM FOREST PREDICTIONS",
    )

    if three_signal is not None:
        print("=" * 80)
        print("THREE-SIGNAL AVAILABILITY")
        print("=" * 80)

        availability_columns = [
            column
            for column in [
                "code_analyzed",
                "code_data_available",
                "code_risk_observed",
                "code_risk_for_fusion",
                "bus_signal_available",
                "satd_signal_available",
                "available_signal_count",
                "signal_completeness",
            ]
            if column in three_signal.columns
        ]

        if availability_columns:
            print(
                three_signal[
                    availability_columns
                ]
                .value_counts(dropna=False)
                .head(20)
                .to_string()
            )
        else:
            print(
                "Expected availability columns "
                "were not found."
            )

        print()

        if "file_path" in three_signal.columns:
            print(
                "Three-signal unique files:",
                three_signal["file_path"].nunique(),
            )

    if rf is not None:

        print()
        print("=" * 80)
        print("RANDOM FOREST FILE COVERAGE")
        print("=" * 80)

        if "file_path" in rf.columns:

            print(
                "Unique RF files:",
                rf["file_path"].nunique(),
            )

            print(
                "RF rows:",
                len(rf),
            )

            print(
                "RF positive rows:",
                int(
                    rf["ml_defect_target"].sum()
                ),
            )

            print(
                "RF probability range:",
                round(
                    rf["predicted_probability"].min(),
                    6,
                ),
                "to",
                round(
                    rf["predicted_probability"].max(),
                    6,
                ),
            )

            print()

            print(
                "RF predictions by file:"
            )

            print(
                rf.groupby("file_path")
                .agg(
                    events=("file_path", "size"),
                    max_probability=(
                        "predicted_probability",
                        "max",
                    ),
                    mean_probability=(
                        "predicted_probability",
                        "mean",
                    ),
                    positive_events=(
                        "ml_defect_target",
                        "sum",
                    ),
                )
                .sort_values(
                    "max_probability",
                    ascending=False,
                )
                .head(20)
                .to_string()
            )

    print()
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()