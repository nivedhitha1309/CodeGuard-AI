import os
import re
import pandas as pd


INPUT_FILE = "data/raw/github/requests_commit_file_history.csv"
OUTPUT_FILE = "data/processed/github/requests_defect_labels.csv"


# ---------------------------------------------------------------------
# Candidate defect-related keywords.
#
# These are heuristic indicators, NOT ground truth.
# The matched evidence is preserved in the output dataset.
# ---------------------------------------------------------------------

STRONG_PATTERNS = [
    r"\bbug\b",
    r"\bbugfix\b",
    r"\bfixes\b",
    r"\bfixed\b",
    r"\bfix\b",
    r"\bfixing\b",
    r"\bregression\b",
    r"\bcrash\b",
    r"\bincorrect\b",
    r"\bexception\b",
    r"\bfailure\b",
]

MEDIUM_PATTERNS = [
    r"\bpatch\b",
    r"\bissue\b",
    r"\berror\b",
    r"\bproblem\b",
    r"\bbroken\b",
    r"\bworkaround\b",
]

EXCLUSION_PATTERNS = [
    r"\bdependabot\b",
    r"\bbump\b",
    r"\bdependency\b",
    r"\bdependencies\b",
    r"\bversion\b",
    r"\bchangelog\b",
    r"\btypo\b",
    r"\bdocs?\b",
    r"\bdocumentation\b",
    r"\breadme\b",
    r"\blicen[cs]e\b",
]


def normalize_message(message):
    """Return the first non-empty line of a commit message."""
    if pd.isna(message):
        return ""

    lines = [
        line.strip()
        for line in str(message).splitlines()
        if line.strip()
    ]

    return lines[0] if lines else ""


def matched_patterns(text, patterns):
    """Return the unique matched terms for a pattern list."""
    matches = []

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            matches.append(match.group(0).lower())

    return sorted(set(matches))


def classify_commit(message):
    """
    Classify a commit message as:
      - bug_fix_candidate
      - uncertain
      - non_bug_fix

    This is a heuristic candidate label, not a ground-truth defect label.
    """
    subject = normalize_message(message)
    full_text = str(message)

    strong_matches = matched_patterns(
        full_text,
        STRONG_PATTERNS,
    )

    medium_matches = matched_patterns(
        full_text,
        MEDIUM_PATTERNS,
    )

    exclusion_matches = matched_patterns(
        full_text,
        EXCLUSION_PATTERNS,
    )

    strong = len(strong_matches) > 0
    medium = len(medium_matches) > 0
    excluded = len(exclusion_matches) > 0

    # --------------------------------------------------------------
    # Obvious maintenance/documentation commits are excluded when
    # there is no strong defect-fix keyword.
    #
    # Example:
    # "Bump dependency ..." -> non_bug_fix
    #
    # If a strong defect keyword is also present, we preserve it as
    # a candidate so it can be manually inspected later.
    # --------------------------------------------------------------

    if excluded and not strong:
        label = "non_bug_fix"
    elif strong:
        label = "bug_fix_candidate"
    elif medium:
        label = "uncertain"
    else:
        label = "non_bug_fix"

    if label == "bug_fix_candidate":
        confidence = "high"
    elif label == "uncertain":
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "commit_subject": subject,
        "defect_label": label,
        "label_confidence": confidence,
        "strong_matches": ",".join(strong_matches),
        "medium_matches": ",".join(medium_matches),
        "exclusion_matches": ",".join(exclusion_matches),
    }


def main():
    print("=" * 70)
    print("CodeGuard AI - Historical Defect Commit Labeling")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput rows : {len(df)}")

    required_columns = [
        "commit_sha",
        "commit_date",
        "commit_message",
        "file_path",
        "status",
        "additions",
        "deletions",
        "changes",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------------
    # Deduplicate commit/file records.
    # --------------------------------------------------------------

    duplicate_count = df.duplicated(
        subset=["commit_sha", "file_path"]
    ).sum()

    print(
        f"Duplicate commit/file records : "
        f"{duplicate_count}"
    )

    df = df.drop_duplicates(
        subset=["commit_sha", "file_path"]
    ).copy()

    # --------------------------------------------------------------
    # Label each file-history row.
    # --------------------------------------------------------------

    labels = df["commit_message"].apply(classify_commit)

    labels_df = pd.DataFrame(
        labels.tolist(),
        index=df.index,
    )

    result = pd.concat(
        [
            df.reset_index(drop=True),
            labels_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # --------------------------------------------------------------
    # Commit-level label.
    #
    # A commit may modify multiple files, but all rows from the same
    # commit share the same commit message. We retain both the
    # row-level evidence and a unique commit-level label.
    # --------------------------------------------------------------

    commit_labels = (
        result.groupby("commit_sha")["defect_label"]
        .apply(
            lambda values:
                "bug_fix_candidate"
                if "bug_fix_candidate" in values.values
                else (
                    "uncertain"
                    if "uncertain" in values.values
                    else "non_bug_fix"
                )
        )
        .rename("commit_defect_label")
    )

    result = result.merge(
        commit_labels,
        on="commit_sha",
        how="left",
        validate="many_to_one",
    )

    result["defect_candidate"] = (
        result["commit_defect_label"]
        == "bug_fix_candidate"
    ).astype(int)

    # --------------------------------------------------------------
    # Save.
    # --------------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------------
    # Summary.
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("DEFECT LABEL SUMMARY")
    print("=" * 70)

    print("\nFILE-HISTORY LABEL DISTRIBUTION")
    print("-" * 70)

    print(
        result["defect_label"]
        .value_counts()
        .to_string()
    )

    print("\nCOMMIT-LEVEL LABEL DISTRIBUTION")
    print("-" * 70)

    commit_summary = (
        result[
            [
                "commit_sha",
                "commit_defect_label",
            ]
        ]
        .drop_duplicates()
    )

    print(
        commit_summary["commit_defect_label"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------------
    # Calculate counts separately.
    #
    # This avoids the multi-line f-string syntax problem that caused
    # the original SyntaxError.
    # --------------------------------------------------------------

    candidate_count = (
        commit_summary["commit_defect_label"]
        == "bug_fix_candidate"
    ).sum()

    uncertain_count = (
        commit_summary["commit_defect_label"]
        == "uncertain"
    ).sum()

    non_bug_fix_count = (
        commit_summary["commit_defect_label"]
        == "non_bug_fix"
    ).sum()

    print("\nUNIQUE COMMITS")
    print("-" * 70)

    print(
        f"Unique commits : "
        f"{result['commit_sha'].nunique()}"
    )

    print(
        f"Candidate bug-fix commits : "
        f"{candidate_count}"
    )

    print(
        f"Uncertain commits : "
        f"{uncertain_count}"
    )

    print(
        f"Non-bug-fix commits : "
        f"{non_bug_fix_count}"
    )

    # --------------------------------------------------------------
    # Show candidate examples.
    # --------------------------------------------------------------

    print("\nSAMPLE BUG-FIX CANDIDATES")
    print("-" * 70)

    candidate_commits = (
        result[
            result["commit_defect_label"]
            == "bug_fix_candidate"
        ][
            [
                "commit_sha",
                "commit_date",
                "commit_subject",
                "file_path",
                "strong_matches",
            ]
        ]
        .drop_duplicates(subset=["commit_sha"])
        .head(20)
    )

    if len(candidate_commits) == 0:
        print("No candidate bug-fix commits found.")
    else:
        print(
            candidate_commits.to_string(index=False)
        )

    # --------------------------------------------------------------
    # Show uncertain examples.
    # --------------------------------------------------------------

    print("\nSAMPLE UNCERTAIN COMMITS")
    print("-" * 70)

    uncertain_commits = (
        result[
            result["commit_defect_label"]
            == "uncertain"
        ][
            [
                "commit_sha",
                "commit_date",
                "commit_subject",
                "file_path",
                "medium_matches",
            ]
        ]
        .drop_duplicates(subset=["commit_sha"])
        .head(15)
    )

    if len(uncertain_commits) == 0:
        print("No uncertain commits found.")
    else:
        print(
            uncertain_commits.to_string(index=False)
        )

    print("\n" + "=" * 70)
    print(f"Output saved to: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
