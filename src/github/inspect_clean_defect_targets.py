import os
import re
import pandas as pd


INPUT_FILE = "data/processed/github/requests_clean_defect_targets.csv"
OUTPUT_FILE = "data/processed/github/requests_clean_defect_targets_inspection.csv"


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def classify_message(message):
    text = clean_text(message).lower()

    patterns = {
        "security": [
            "security",
            "vulnerability",
            "cve",
            "injection",
            "ssl",
            "tls",
            "certificate",
            "authentication",
            "authorization",
            "credential",
            "exploit",
        ],
        "regression": [
            "regression",
            "regressed",
            "break",
            "broken",
            "broke",
            "incorrect",
            "wrong",
            "unexpected",
            "failure",
            "failing",
            "failed",
            "bug",
            "fix",
            "issue",
            "error",
        ],
        "logic": [
            "logic",
            "condition",
            "conditional",
            "handle",
            "handling",
            "return",
            "parse",
            "parsing",
            "encode",
            "encoding",
            "decode",
            "decoding",
        ],
        "compatibility": [
            "compatibility",
            "compatible",
            "python 2",
            "python 3",
            "version",
            "support",
            "backward",
            "backwards",
        ],
        "maintenance": [
            "refactor",
            "cleanup",
            "rename",
            "remove",
            "update",
            "upgrade",
            "dependency",
            "dependencies",
            "packaging",
            "release",
            "version bump",
        ],
    }

    matches = []

    for category, keywords in patterns.items():
        for keyword in keywords:
            if keyword in text:
                matches.append(f"{category}:{keyword}")

    return "; ".join(dict.fromkeys(matches))


def main():
    print("=" * 70)
    print("CODEGUARD AI - CLEAN DEFECT TARGET INSPECTION")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\n" + "=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)

    print(f"Rows              : {len(df)}")
    print(f"Columns           : {len(df.columns)}")

    if "commit_defect_target" in df.columns:
        print(
            f"Target distribution:\n"
            f"{df['commit_defect_target'].value_counts(dropna=False).to_string()}"
        )

    if "file_path" in df.columns:
        print(f"Unique files      : {df['file_path'].nunique()}")

    if "commit_sha" in df.columns:
        print(f"Unique commits    : {df['commit_sha'].nunique()}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    # ------------------------------------------------------------
    # Select target rows
    # ------------------------------------------------------------

    if "commit_defect_target" in df.columns:
        positive = df[df["commit_defect_target"] == 1].copy()
    elif "target" in df.columns:
        positive = df[df["target"] == 1].copy()
    else:
        raise KeyError(
            "Could not find 'commit_defect_target' or 'target' column."
        )

    print("\n" + "=" * 70)
    print("POSITIVE TARGETS")
    print("=" * 70)

    print(f"Positive rows     : {len(positive)}")

    if "commit_sha" in positive.columns:
        print(f"Positive commits  : {positive['commit_sha'].nunique()}")

    if "file_path" in positive.columns:
        print(f"Positive files    : {positive['file_path'].nunique()}")

    # ------------------------------------------------------------
    # Add message-based inspection
    # ------------------------------------------------------------

    if "commit_message" in positive.columns:
        positive["message_inspection"] = positive[
            "commit_message"
        ].apply(classify_message)

    elif "commit_subject" in positive.columns:
        positive["message_inspection"] = positive[
            "commit_subject"
        ].apply(classify_message)

    else:
        positive["message_inspection"] = ""

    # ------------------------------------------------------------
    # Print every positive event
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL POSITIVE TARGET EVENTS")
    print("=" * 70)

    display_columns = []

    preferred_columns = [
        "commit_sha",
        "commit_date",
        "author",
        "file_path",
        "status",
        "additions",
        "deletions",
        "changes",
        "commit_subject",
        "commit_message",
        "commit_refined_defect_label",
        "commit_defect_target",
        "commit_target_confidence",
        "message_inspection",
    ]

    for column in preferred_columns:
        if column in positive.columns:
            display_columns.append(column)

    for index, (_, row) in enumerate(
        positive.iterrows(),
        start=1
    ):
        print("\n" + "-" * 70)
        print(f"POSITIVE EVENT #{index}")
        print("-" * 70)

        for column in display_columns:
            value = row[column]

            if pd.isna(value):
                value = ""

            value = clean_text(value)

            if column == "commit_message":
                # Keep output readable.
                value = re.sub(r"\s+", " ", value)

            print(f"{column:28}: {value}")

    # ------------------------------------------------------------
    # Category distribution
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("DEFECT LABEL DISTRIBUTION")
    print("=" * 70)

    for column in [
        "commit_refined_defect_label",
        "defect_label",
        "label_confidence",
        "commit_target_confidence",
    ]:
        if column in positive.columns:
            print(f"\n{column}:")
            print(
                positive[column]
                .value_counts(dropna=False)
                .to_string()
            )

    # ------------------------------------------------------------
    # Message keyword inspection
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MESSAGE INSPECTION SUMMARY")
    print("=" * 70)

    if "message_inspection" in positive.columns:
        keyword_counts = {}

        for value in positive["message_inspection"]:
            if not value:
                continue

            for item in value.split(";"):
                category, keyword = item.split(":", 1)

                key = f"{category}:{keyword}"

                keyword_counts[key] = (
                    keyword_counts.get(key, 0) + 1
                )

        if keyword_counts:
            summary = pd.Series(keyword_counts).sort_values(
                ascending=False
            )

            print(summary.to_string())
        else:
            print("No message keywords detected.")

    # ------------------------------------------------------------
    # Save inspection dataset
    # ------------------------------------------------------------

    positive.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)

    print(f"Inspection rows saved : {len(positive)}")
    print(f"Output file           : {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()