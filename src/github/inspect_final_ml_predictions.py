from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

TEST_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_final_temporal_test.csv"
)

RF_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_final_random_forest_predictions.csv"
)

LR_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_final_logistic_regression_predictions.csv"
)


def print_event_columns(df, title):

    print("=" * 90)
    print(title)
    print("=" * 90)

    preferred_columns = [
        "event_date",
        "commit_date",
        "commit_sha",
        "file_path",
        "commit_message",
        "clean_refined_label",
        "ml_defect_target",
        "predicted_probability",
        "predicted_label",
    ]

    columns = [
        column
        for column in preferred_columns
        if column in df.columns
    ]

    if not columns:
        print(df.to_string(index=False))
        print()
        return

    display_df = df[columns].copy()

    print(
        display_df.to_string(
            index=False
        )
    )

    print()


def main():

    print("=" * 90)
    print("CODEGUARD AI - FINAL ML PREDICTION INSPECTION")
    print("=" * 90)
    print()

    test_df = pd.read_csv(
        TEST_FILE
    )

    rf_df = pd.read_csv(
        RF_FILE
    )

    lr_df = pd.read_csv(
        LR_FILE
    )

    print("FILES LOADED")
    print("-" * 90)
    print(f"Test rows       : {len(test_df)}")
    print(f"RF predictions  : {len(rf_df)}")
    print(f"LR predictions  : {len(lr_df)}")
    print()

    # ---------------------------------------------------------------
    # RANDOM FOREST
    # ---------------------------------------------------------------

    rf_positive = rf_df[
        rf_df["ml_defect_target"] == 1
    ].copy()

    rf_detected = rf_positive[
        rf_positive["predicted_label"] == 1
    ].copy()

    rf_missed = rf_positive[
        rf_positive["predicted_label"] == 0
    ].copy()

    print_event_columns(
        rf_positive,
        "RANDOM FOREST - ALL ACTUAL TEST POSITIVES",
    )

    print_event_columns(
        rf_detected,
        "RANDOM FOREST - DETECTED TEST POSITIVES",
    )

    print_event_columns(
        rf_missed,
        "RANDOM FOREST - MISSED TEST POSITIVES",
    )

    # ---------------------------------------------------------------
    # RANDOM FOREST FALSE POSITIVES
    # ---------------------------------------------------------------

    rf_false_positive = rf_df[
        (rf_df["ml_defect_target"] == 0)
        & (rf_df["predicted_label"] == 1)
    ].copy()

    rf_false_positive = rf_false_positive.sort_values(
        "predicted_probability",
        ascending=False,
    )

    print_event_columns(
        rf_false_positive.head(20),
        "RANDOM FOREST - TOP FALSE POSITIVES",
    )

    # ---------------------------------------------------------------
    # LOGISTIC REGRESSION
    # ---------------------------------------------------------------

    lr_positive = lr_df[
        lr_df["ml_defect_target"] == 1
    ].copy()

    lr_detected = lr_positive[
        lr_positive["predicted_label"] == 1
    ].copy()

    lr_missed = lr_positive[
        lr_positive["predicted_label"] == 0
    ].copy()

    print_event_columns(
        lr_detected,
        "LOGISTIC REGRESSION - DETECTED TEST POSITIVES",
    )

    print_event_columns(
        lr_missed,
        "LOGISTIC REGRESSION - MISSED TEST POSITIVES",
    )

    # ---------------------------------------------------------------
    # PROBABILITY RANKING
    # ---------------------------------------------------------------

    print("=" * 90)
    print("RANDOM FOREST - TOP TEST EVENTS BY PREDICTED RISK")
    print("=" * 90)

    ranking_columns = [
        column
        for column in [
            "event_date",
            "commit_sha",
            "file_path",
            "commit_message",
            "ml_defect_target",
            "predicted_probability",
        ]
        if column in rf_df.columns
    ]

    ranked = rf_df.sort_values(
        "predicted_probability",
        ascending=False,
    )

    print(
        ranked[
            ranking_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    print()

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------

    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(
        f"Random Forest actual positives : "
        f"{len(rf_positive)}"
    )

    print(
        f"Random Forest detected          : "
        f"{len(rf_detected)}"
    )

    print(
        f"Random Forest missed            : "
        f"{len(rf_missed)}"
    )

    print(
        f"Random Forest false positives   : "
        f"{len(rf_false_positive)}"
    )

    print()

    print(
        f"Logistic Regression actual positives : "
        f"{len(lr_positive)}"
    )

    print(
        f"Logistic Regression detected          : "
        f"{len(lr_detected)}"
    )

    print(
        f"Logistic Regression missed            : "
        f"{len(lr_missed)}"
    )

    print()

    print("=" * 90)


if __name__ == "__main__":
    main()