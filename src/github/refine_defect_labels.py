import os
import re
import pandas as pd


INPUT_FILE = (
    "data/processed/github/"
    "requests_defect_labels.csv"
)

OUTPUT_FILE = (
    "data/processed/github/"
    "requests_refined_defect_labels.csv"
)


# ---------------------------------------------------------------------
# Refinement rules
#
# These rules separate message-based defect candidates into more useful
# categories for later historical ML-label construction.
#
# IMPORTANT:
# These are still heuristic labels, NOT ground truth.
# ---------------------------------------------------------------------

SECURITY_PATTERNS = [
    r"\bcve\b",
    r"\bsecurity\b",
    r"\bvulnerability\b",
    r"\bvulnerabilit(?:y|ies)\b",
    r"\bsecurity issue\b",
    r"\bsecurity fix\b",
]

REGRESSION_PATTERNS = [
    r"\bregression\b",
    r"\bregressed\b",
    r"\bregressions\b",
]

TYPO_PATTERNS = [
    r"\btypo\b",
    r"\btypos\b",
    r"\bspelling\b",
    r"\bgrammar\b",
    r"\bcodespell\b",
]

DOCUMENTATION_PATTERNS = [
    r"\bdocs?\b",
    r"\bdocumentation\b",
    r"\breadme\b",
    r"\bchangelog\b",
    r"\bexample\b",
    r"\bexamples\b",
    r"\blink\b",
    r"\blinks\b",
]

CI_BUILD_PATTERNS = [
    r"\bci\b",
    r"\bbuild\b",
    r"\bworkflow\b",
    r"\bgithub actions\b",
    r"\btest infrastructure\b",
    r"\bpackaging\b",
    r"\bwheel\b",
    r"\bdist(?:ribution)?\b",
]

TEST_PATTERNS = [
    r"\btest\b",
    r"\btests\b",
    r"\btesting\b",
    r"\bpytest\b",
    r"\bregression test\b",
]

DEFECT_PATTERNS = [
    r"\bbug\b",
    r"\bbugfix\b",
    r"\bfix\b",
    r"\bfixes\b",
    r"\bfixed\b",
    r"\bfixing\b",
    r"\bcrash\b",
    r"\bincorrect\b",
    r"\bexception\b",
    r"\bfailure\b",
    r"\bbroken\b",
    r"\bmalformed\b",
    r"\bunintended\b",
    r"\bincorrectly\b",
    r"\bwrong\b",
    r"\bleak\b",
    r"\bleakage\b",
]


def matches_any(text, patterns):
    """Return matched terms from a list of regex patterns."""
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


def normalize_subject(value):
    """Normalize a commit subject for consistent rule matching."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def classify_refined_commit(row):
    """
    Refine an existing candidate label.

    Priority:
        1. Security / CVE
        2. Regression
        3. Typo
        4. Documentation
        5. CI / build
        6. Test-related
        7. Likely defect
        8. Uncertain
        9. Non-bug-fix
    """

    subject = normalize_subject(
        row["commit_subject"]
    )

    file_path = str(
        row["file_path"]
    ).strip()

    text = f"{subject} {file_path}"

    original_label = str(
        row["commit_defect_label"]
    )

    security_matches = matches_any(
        text,
        SECURITY_PATTERNS,
    )

    regression_matches = matches_any(
        text,
        REGRESSION_PATTERNS,
    )

    typo_matches = matches_any(
        text,
        TYPO_PATTERNS,
    )

    documentation_matches = matches_any(
        text,
        DOCUMENTATION_PATTERNS,
    )

    ci_build_matches = matches_any(
        text,
        CI_BUILD_PATTERNS,
    )

    test_matches = matches_any(
        text,
        TEST_PATTERNS,
    )

    defect_matches = matches_any(
        text,
        DEFECT_PATTERNS,
    )

    # --------------------------------------------------------------
    # Security and regression receive the highest refinement priority.
    # --------------------------------------------------------------

    if security_matches:
        refined_label = "security_or_vulnerability_fix"

    elif regression_matches:
        refined_label = "regression_fix"

    # --------------------------------------------------------------
    # Documentation / typo fixes should not become code defects.
    # --------------------------------------------------------------

    elif typo_matches:
        refined_label = "typo_or_style_fix"

    elif documentation_matches and not defect_matches:
        refined_label = "documentation_fix"

    # --------------------------------------------------------------
    # CI/build changes are separated from source-code defects.
    # --------------------------------------------------------------

    elif ci_build_matches and not defect_matches:
        refined_label = "ci_or_build_fix"

    # --------------------------------------------------------------
    # Test-related changes are retained as their own category.
    # --------------------------------------------------------------

    elif test_matches and not defect_matches:
        refined_label = "test_related_fix"

    # --------------------------------------------------------------
    # Strong defect evidence.
    # --------------------------------------------------------------

    elif (
        original_label == "bug_fix_candidate"
        and defect_matches
    ):
        refined_label = "likely_defect_fix"

    # --------------------------------------------------------------
    # Medium evidence remains uncertain.
    # --------------------------------------------------------------

    elif original_label == "uncertain":
        refined_label = "uncertain"

    else:
        refined_label = "non_bug_fix"

    # --------------------------------------------------------------
    # ML target candidate.
    #
    # We create a conservative target that includes:
    #   - likely defect fixes
    #   - security/vulnerability fixes
    #   - regression fixes
    #
    # Test-only, documentation, typo, and CI/build changes are NOT
    # treated as direct software-defect targets.
    # --------------------------------------------------------------

    defect_target = int(
        refined_label
        in {
            "likely_defect_fix",
            "security_or_vulnerability_fix",
            "regression_fix",
        }
    )

    if refined_label in {
        "likely_defect_fix",
        "security_or_vulnerability_fix",
        "regression_fix",
    }:
        target_confidence = "high"

    elif refined_label in {
        "test_related_fix",
        "uncertain",
    }:
        target_confidence = "medium"

    else:
        target_confidence = "low"

    return pd.Series(
        {
            "security_matches": ",".join(
                security_matches
            ),
            "regression_matches": ",".join(
                regression_matches
            ),
            "typo_matches": ",".join(
                typo_matches
            ),
            "documentation_matches": ",".join(
                documentation_matches
            ),
            "ci_build_matches": ",".join(
                ci_build_matches
            ),
            "test_matches": ",".join(
                test_matches
            ),
            "defect_matches": ",".join(
                defect_matches
            ),
            "refined_defect_label": refined_label,
            "defect_target": defect_target,
            "target_confidence": target_confidence,
        }
    )


def main():
    print("=" * 70)
    print("CodeGuard AI - Refining Historical Defect Labels")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"\nInput rows : {len(df)}"
    )

    required_columns = [
        "commit_sha",
        "commit_date",
        "commit_subject",
        "file_path",
        "commit_defect_label",
        "defect_candidate",
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
    # Apply refinement.
    # --------------------------------------------------------------

    refined = df.apply(
        classify_refined_commit,
        axis=1,
    )

    result = pd.concat(
        [
            df.reset_index(drop=True),
            refined.reset_index(drop=True),
        ],
        axis=1,
    )

    # --------------------------------------------------------------
    # Create one unique commit-level refinement table.
    #
    # A commit can affect multiple files, so we preserve the strongest
    # refined category observed for that commit.
    # --------------------------------------------------------------

    priority = {
        "security_or_vulnerability_fix": 7,
        "regression_fix": 6,
        "likely_defect_fix": 5,
        "test_related_fix": 4,
        "ci_or_build_fix": 3,
        "documentation_fix": 2,
        "typo_or_style_fix": 1,
        "uncertain": 0,
        "non_bug_fix": -1,
    }

    commit_summary = (
        result[
            [
                "commit_sha",
                "commit_date",
                "commit_subject",
                "refined_defect_label",
                "defect_target",
                "target_confidence",
            ]
        ]
        .drop_duplicates(
            subset=["commit_sha"]
        )
        .copy()
    )

    # Because the row-level classification includes file-path context,
    # calculate the strongest category per commit directly from all rows.
    result["_priority"] = result[
        "refined_defect_label"
    ].map(priority)

    strongest_labels = (
        result.sort_values(
            [
                "commit_sha",
                "_priority",
            ],
            ascending=[True, False],
        )
        .drop_duplicates(
            subset=["commit_sha"]
        )[
            [
                "commit_sha",
                "refined_defect_label",
                "defect_target",
                "target_confidence",
            ]
        ]
    )

    commit_summary = (
        commit_summary.drop(
            columns=[
                "refined_defect_label",
                "defect_target",
                "target_confidence",
            ]
        )
        .merge(
            strongest_labels,
            on="commit_sha",
            how="left",
            validate="one_to_one",
        )
    )

    # --------------------------------------------------------------
    # Propagate the strongest commit-level label to every affected file.
    # --------------------------------------------------------------

    commit_summary_for_merge = commit_summary[
        [
            "commit_sha",
            "refined_defect_label",
            "defect_target",
            "target_confidence",
        ]
    ].rename(
        columns={
            "refined_defect_label":
                "commit_refined_defect_label",
            "defect_target":
                "commit_defect_target",
            "target_confidence":
                "commit_target_confidence",
        }
    )

    result = result.drop(
        columns=[
            "_priority",
            "refined_defect_label",
            "defect_target",
            "target_confidence",
        ]
    )

    result = result.merge(
        commit_summary_for_merge,
        on="commit_sha",
        how="left",
        validate="many_to_one",
    )

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
    print("REFINED DEFECT LABEL SUMMARY")
    print("=" * 70)

    print("\nFILE-HISTORY REFINED LABEL DISTRIBUTION")
    print("-" * 70)

    print(
        result["commit_refined_defect_label"]
        .value_counts()
        .to_string()
    )

    print("\nUNIQUE COMMIT REFINED LABEL DISTRIBUTION")
    print("-" * 70)

    print(
        commit_summary[
            "refined_defect_label"
        ]
        .value_counts()
        .to_string()
    )

    print("\nDEFECT TARGET SUMMARY")
    print("-" * 70)

    target_count = (
        commit_summary["defect_target"]
        == 1
    ).sum()

    non_target_count = (
        commit_summary["defect_target"]
        == 0
    ).sum()

    print(
        f"Defect-target commits : "
        f"{target_count}"
    )

    print(
        f"Non-target commits    : "
        f"{non_target_count}"
    )

    print("\nTARGET CATEGORIES")
    print("-" * 70)

    target_categories = commit_summary[
        commit_summary["defect_target"] == 1
    ]

    if len(target_categories) == 0:
        print("No defect-target commits found.")
    else:
        print(
            target_categories[
                "refined_defect_label"
            ]
            .value_counts()
            .to_string()
        )

    print("\nSAMPLE DEFECT-TARGET COMMITS")
    print("-" * 70)

    samples = (
        target_categories[
            [
                "commit_sha",
                "commit_date",
                "commit_subject",
                "refined_defect_label",
            ]
        ]
        .head(20)
    )

    if len(samples) == 0:
        print("No samples available.")
    else:
        print(
            samples.to_string(
                index=False
            )
        )

    print("\nSAMPLE EXCLUDED FIXES")
    print("-" * 70)

    excluded = commit_summary[
        commit_summary["refined_defect_label"].isin(
            [
                "typo_or_style_fix",
                "documentation_fix",
                "ci_or_build_fix",
                "test_related_fix",
            ]
        )
    ][
        [
            "commit_sha",
            "commit_date",
            "commit_subject",
            "refined_defect_label",
        ]
    ].head(20)

    if len(excluded) == 0:
        print("No excluded fix examples found.")
    else:
        print(
            excluded.to_string(
                index=False
            )
        )

    print("\n" + "=" * 70)
    print(
        f"Output saved to: {OUTPUT_FILE}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
