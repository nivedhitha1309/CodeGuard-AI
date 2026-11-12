from pathlib import Path
import pandas as pd
import re


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_strict_source_defect_targets.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_final_ml_defect_targets.csv"
)


def contains_any(text, patterns):
    text = str(text).lower()

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


# ------------------------------------------------------------
# Strong evidence that the change addresses an actual defect
# ------------------------------------------------------------

DEFECT_PATTERNS = [
    r"\bbug\b",
    r"\bbugs\b",
    r"\bissue\b",
    r"\bissues\b",
    r"\berror\b",
    r"\berrors\b",
    r"\bfailure\b",
    r"\bfailing\b",
    r"\bfailed\b",
    r"\bregression\b",
    r"\bincorrect\b",
    r"\bwrong\b",
    r"\bbroken\b",
    r"\bbroke\b",
    r"\bunexpect(?:ed|ed)\b",
    r"\bexception\b",
    r"\bcrash\b",
    r"\bmalformed\b",
    r"\binvalid\b",
    r"\bimproper\b",
    r"\bdoesn't\b",
    r"\bdoes not\b",
    r"\bnot working\b",
]


# ------------------------------------------------------------
# Strong evidence that this is maintenance rather than a
# production defect
# ------------------------------------------------------------

MAINTENANCE_PATTERNS = [
    r"\bflake8\b",
    r"\blint(?:ing)?\b",
    r"\bpep8\b",
    r"\bformat(?:ting)?\b",
    r"\bstyle\b",
    r"\bwhitespace\b",
    r"\bindent(?:ation)?\b",
    r"\bdocstring\b",
    r"\bdoctest\b",
    r"\bappveyor\b",
    r"\btravis\b",
    r"\bcodecov\b",
    r"\bci\b",
    r"\bcontinuous integration\b",
    r"\btest(?:s|ing)?\b",
    r"\btesting\b",
    r"\bdebugging submodule\b",
]


# ------------------------------------------------------------
# Revert changes are kept separately because a revert does
# not directly identify the original defect mechanism.
# ------------------------------------------------------------

REVERT_PATTERN = r"\brevert\b"


def classify_row(row):

    text = (
        str(row["commit_subject"])
        + " "
        + str(row["commit_message"])
    ).strip().lower()

    has_defect = contains_any(
        text,
        DEFECT_PATTERNS,
    )

    has_maintenance = contains_any(
        text,
        MAINTENANCE_PATTERNS,
    )

    is_revert = bool(
        re.search(
            REVERT_PATTERN,
            text,
        )
    )

    # --------------------------------------------------------
    # Primary ML positive
    #
    # Require defect evidence and no obvious maintenance-only
    # language.
    # --------------------------------------------------------

    if has_defect and not has_maintenance and not is_revert:
        return "primary_defect"

    # --------------------------------------------------------
    # Review-needed positive
    #
    # These remain useful for sensitivity analysis but are not
    # included in the cleanest primary target set.
    # --------------------------------------------------------

    return "review_needed"


def main():

    print("=" * 70)
    print("CODEGUARD AI - FINAL ML DEFECT TARGET BUILDER")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\nInput rows:", len(df))
    print(
        "Unique commits:",
        df["commit_sha"].nunique(),
    )
    print(
        "Unique files:",
        df["file_path"].nunique(),
    )

    required_columns = [
        "commit_sha",
        "commit_date",
        "commit_subject",
        "commit_message",
        "file_path",
        "commit_refined_defect_label",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {col}"
                for col in missing
            )
        )

    df = df.copy()

    df["commit_subject"] = (
        df["commit_subject"]
        .fillna("")
        .astype(str)
    )

    df["commit_message"] = (
        df["commit_message"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # Classify
    # --------------------------------------------------------

    df["ml_target_quality"] = df.apply(
        classify_row,
        axis=1,
    )

    df["ml_defect_target"] = (
        df["ml_target_quality"]
        == "primary_defect"
    ).astype(int)

    # --------------------------------------------------------
    # Print classification
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TARGET QUALITY")
    print("=" * 70)

    print(
        df["ml_target_quality"]
        .value_counts()
        .to_string()
    )

    primary = df[
        df["ml_target_quality"]
        == "primary_defect"
    ].copy()

    review = df[
        df["ml_target_quality"]
        == "review_needed"
    ].copy()

    print("\nPrimary defect rows:", len(primary))
    print(
        "Primary defect commits:",
        primary["commit_sha"].nunique(),
    )
    print(
        "Primary defect files:",
        primary["file_path"].nunique(),
    )

    print("\nReview-needed rows:", len(review))
    print(
        "Review-needed commits:",
        review["commit_sha"].nunique(),
    )

    # --------------------------------------------------------
    # Refined-label breakdown
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PRIMARY DEFECT LABELS")
    print("=" * 70)

    if len(primary) > 0:
        print(
            primary[
                "commit_refined_defect_label"
            ]
            .value_counts()
            .to_string()
        )
    else:
        print("No primary defects.")

    # --------------------------------------------------------
    # Primary sample
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PRIMARY DEFECT SAMPLE")
    print("=" * 70)

    if len(primary) > 0:

        sample_columns = [
            "commit_date",
            "commit_sha",
            "commit_subject",
            "file_path",
            "commit_refined_defect_label",
        ]

        print(
            primary[
                sample_columns
            ]
            .sort_values(
                [
                    "commit_date",
                    "commit_sha",
                    "file_path",
                ]
            )
            .head(50)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Review-needed sample
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("REVIEW-NEEDED SAMPLE")
    print("=" * 70)

    if len(review) > 0:

        sample_columns = [
            "commit_date",
            "commit_sha",
            "commit_subject",
            "file_path",
            "commit_refined_defect_label",
        ]

        print(
            review[
                sample_columns
            ]
            .sort_values(
                [
                    "commit_date",
                    "commit_sha",
                    "file_path",
                ]
            )
            .head(50)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Save complete classification
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(f"Saved to: {OUTPUT_FILE}")

    print("=" * 70)


if __name__ == "__main__":
    main()