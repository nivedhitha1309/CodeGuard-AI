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
    / "requests_strict_source_defect_targets_inspection.csv"
)


def classify_message(text):
    text = str(text).lower()

    categories = []

    patterns = {
        "bug": r"\bbugs?\b",
        "issue": r"\bissues?\b",
        "error": r"\berrors?\b",
        "failure": r"\bfail(?:ed|ure|ing|s)?\b",
        "regression": r"\bregression\b",
        "security": r"\bsecurity\b|\bvulnerabilit(?:y|ies)\b|\bcve-\d{4}-\d+\b",
        "incorrect": r"\bincorrect\b",
        "wrong": r"\bwrong\b",
        "broken": r"\bbroken\b|\bbroke\b",
        "test": r"\btests?\b|\bpytest\b|\btesting\b",
        "ci": r"\bci\b|\btravis\b|\bappveyor\b|\bcodecov\b",
        "review": r"\breview\b",
        "python": r"\bpython\s*[23](?:\.\d+)?\b",
        "compatibility": r"\bcompatib(?:ility|le)\b|\bsupport\b",
    }

    for name, pattern in patterns.items():
        if re.search(pattern, text):
            categories.append(name)

    return ",".join(categories)


def main():

    print("=" * 70)
    print("CODEGUARD AI - STRICT TARGET INSPECTION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\nInput rows:", len(df))
    print("Input columns:", len(df.columns))

    required = [
        "commit_sha",
        "commit_date",
        "commit_subject",
        "commit_message",
        "file_path",
        "commit_refined_defect_label",
        "strict_target_reason",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {x}" for x in missing)
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

    df["combined_message"] = (
        df["commit_subject"]
        + " "
        + df["commit_message"]
    ).str.strip()

    df["message_categories"] = (
        df["combined_message"]
        .apply(classify_message)
    )

    df["has_test_language"] = (
        df["message_categories"]
        .str.contains(r"\btest\b", regex=True)
    )

    df["has_ci_language"] = (
        df["message_categories"]
        .str.contains(r"\bci\b", regex=True)
    )

    df["has_bug_language"] = (
        df["message_categories"]
        .str.contains(
            r"\bbug\b|\bissue\b|\berror\b|\bfailure\b|"
            r"\bregression\b|\bincorrect\b|\bwrong\b|"
            r"\bbroken\b",
            regex=True,
        )
    )

    df["has_security_language"] = (
        df["message_categories"]
        .str.contains(
            r"\bsecurity\b",
            regex=True,
        )
    )

    # --------------------------------------------------------
    # Print complete inspection
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL STRICT POSITIVE TARGETS")
    print("=" * 70)

    display_columns = [
        "commit_date",
        "commit_sha",
        "commit_subject",
        "file_path",
        "commit_refined_defect_label",
        "strong_match_count",
        "maintenance_match_count",
        "message_categories",
    ]

    print(
        df[display_columns]
        .sort_values(
            ["commit_date", "commit_sha", "file_path"]
        )
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MESSAGE CATEGORY SUMMARY")
    print("=" * 70)

    category_counts = {}

    for value in df["message_categories"]:
        if not value:
            continue

        for category in value.split(","):
            category_counts[category] = (
                category_counts.get(category, 0) + 1
            )

    if category_counts:
        print(
            pd.Series(category_counts)
            .sort_values(ascending=False)
            .to_string()
        )

    print("\n" + "=" * 70)
    print("POTENTIAL NOISE INDICATORS")
    print("=" * 70)

    print(
        "Targets mentioning test language :",
        int(df["has_test_language"].sum()),
    )

    print(
        "Targets mentioning CI language   :",
        int(df["has_ci_language"].sum()),
    )

    print(
        "Targets with bug/error evidence  :",
        int(df["has_bug_language"].sum()),
    )

    print(
        "Targets with security evidence   :",
        int(df["has_security_language"].sum()),
    )

    # --------------------------------------------------------
    # Save
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