from pathlib import Path
import re
import pandas as pd


# ============================================================
# CODEGUARD AI
# Strict Production-Source Defect Target Builder
#
# Purpose:
#   Build a conservative positive defect-target dataset from
#   the existing refined GitHub defect labels.
#
# Important:
#   This is still a heuristic labeling process, not ground truth.
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_clean_defect_targets.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "github"
    / "requests_strict_source_defect_targets.csv"
)


# ------------------------------------------------------------
# Existing refined labels that can represent a defect
# ------------------------------------------------------------

POSITIVE_REFINED_LABELS = {
    "likely_defect_fix",
    "security_or_vulnerability_fix",
    "regression_fix",
}


# ------------------------------------------------------------
# Strong defect evidence in commit messages
# ------------------------------------------------------------

STRONG_DEFECT_PATTERNS = {
    "bug": r"\bbugs?\b",
    "fix": r"\bfix(?:es|ed|ing)?\b",
    "issue": r"\bissues?\b",
    "regression": r"\bregress(?:ion|ed|es|ing)?\b",
    "broken": r"\bbroken\b",
    "broke": r"\bbroke\b",
    "failing": r"\bfailing\b",
    "failure": r"\bfail(?:ed|ure|ing|s)?\b",
    "error": r"\berrors?\b",
    "incorrect": r"\bincorrect\b",
    "wrong": r"\bwrong\b",
    "unexpected": r"\bunexpected\b",
    "vulnerability": r"\bvulnerabilit(?:y|ies)\b",
    "security": r"\bsecurity\b",
    "cve": r"\bcve-\d{4}-\d+\b",
}


# ------------------------------------------------------------
# Maintenance / non-defect evidence
# ------------------------------------------------------------

MAINTENANCE_PATTERNS = {
    "refactor": r"\brefactor(?:ed|ing)?\b",
    "cleanup": r"\bcleanup\b|\bclean[- ]?up\b",
    "rename": r"\brename(?:d|s|ing)?\b",
    "format": r"\bformat(?:ted|ting)?\b",
    "style": r"\bstyle\b",
    "lint": r"\blint(?:ing)?\b",
    "typing": r"\btyping\b|\btype hints?\b|\btypehint\b",
    "dependency": r"\bdependenc(?:y|ies)\b",
    "bump": r"\bbump(?:ed|ing)?\b",
    "release": r"\brelease\b",
    "packaging": r"\bpackaging\b|\bpackage\b",
    "build": r"\bbuild\b",
    "ci": r"\bci\b|\bcontinuous integration\b",
    "docs": r"\bdocs?\b|\bdocumentation\b",
    "changelog": r"\bchangelog\b",
    "version": r"\bversion\b",
    "upgrade": r"\bupgrade(?:d|s|ing)?\b",
    "update": r"\bupdate(?:d|s|ing)?\b",
}


# ------------------------------------------------------------
# Paths that should NOT be treated as production source
# ------------------------------------------------------------

NON_SOURCE_EXACT = {
    "setup.py",
    "setup.cfg",
    "tox.ini",
    "pytest.ini",
    "pyproject.toml",
}

NON_SOURCE_NAME_PATTERNS = [
    r"(^|/)tests?(/|$)",
    r"(^|/)test_.*\.py$",
    r"(^|/).*_test\.py$",
    r"(^|/)\.github(/|$)",
    r"(^|/)docs?(/|$)",
    r"(^|/)examples?(/|$)",
    r"(^|/)benchmarks?(/|$)",
    r"(^|/)scripts?(/|$)",
    r"(^|/)tools?(/|$)",
    r"(^|/)contrib(/|$)",
    r"(^|/)vendor(/|$)",
    r"(^|/)fixtures?(/|$)",
    r"(^|/)mocks?(/|$)",
    r"(^|/)requirements[^/]*$",
    r"(^|/)setup\.py$",
    r"(^|/)setup\.cfg$",
    r"(^|/)tox\.ini$",
    r"(^|/)pytest\.ini$",
    r"(^|/)pyproject\.toml$",
]


# ------------------------------------------------------------
# Hard maintenance messages
# ------------------------------------------------------------

HARD_MAINTENANCE_PATTERNS = [
    r"\bfix(?:es|ed|ing)?\s+(?:the\s+)?docs?\b",
    r"\bfix(?:es|ed|ing)?\s+(?:the\s+)?documentation\b",
    r"\bfix(?:es|ed|ing)?\s+(?:the\s+)?build\b",
    r"\bfix(?:es|ed|ing)?\s+(?:the\s+)?ci\b",
    r"\bfix(?:es|ed|ing)?\s+(?:the\s+)?travis\b",
    r"\bfix(?:es|ed|ing)?\s+(?:the\s+)?release\b",
    r"\bfix(?:es|ed|ing)?\s+(?:the\s+)?packaging\b",
    r"\bupdate(?:d|s|ing)?\s+(?:the\s+)?docs?\b",
    r"\bupdate(?:d|s|ing)?\s+(?:the\s+)?documentation\b",
    r"\bupdate(?:d|s|ing)?\s+(?:the\s+)?dependenc(?:y|ies)\b",
    r"\bversion\s+bump\b",
    r"\bbump\s+(?:the\s+)?version\b",
    r"\bpin\b.*\b(?:ci|build)\b",
]


def normalise_path(value):
    """Normalise repository paths."""
    if pd.isna(value):
        return ""

    value = str(value).strip().replace("\\", "/")

    while value.startswith("./"):
        value = value[2:]

    return value


def is_merge_commit(subject, message):
    """Detect common merge commits."""
    text = f"{subject} {message}".strip().lower()

    if text.startswith("merge "):
        return True

    merge_patterns = [
        r"\bmerge pull request\b",
        r"\bmerge branch\b",
        r"\bmerged pull request\b",
    ]

    return any(re.search(pattern, text) for pattern in merge_patterns)


def is_production_python(path):
    """
    Keep ordinary production Python source files.

    Exclude:
      - tests
      - documentation
      - CI/configuration
      - examples
      - scripts/tools
      - requirements/setup files
    """
    path = normalise_path(path)

    if not path:
        return False

    lower_path = path.lower()

    if not lower_path.endswith(".py"):
        return False

    if lower_path in NON_SOURCE_EXACT:
        return False

    for pattern in NON_SOURCE_NAME_PATTERNS:
        if re.search(pattern, lower_path):
            return False

    return True


def count_pattern_matches(text, patterns):
    """Count how many distinct named patterns occur."""
    counts = {}

    for name, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)

        if matches:
            counts[name] = len(matches)

    return counts


def has_hard_maintenance_message(text):
    """Return True when the message strongly indicates maintenance."""
    for pattern in HARD_MAINTENANCE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True

    return False


def main():
    print("=" * 70)
    print("CODEGUARD AI - STRICT SOURCE DEFECT TARGET BUILDER")
    print("=" * 70)

    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\n" + "=" * 70)
    print("INPUT DATA")
    print("=" * 70)

    print(f"Input rows           : {len(df)}")
    print(f"Input columns        : {len(df.columns)}")

    required_columns = [
        "commit_sha",
        "commit_date",
        "commit_message",
        "commit_subject",
        "file_path",
        "commit_refined_defect_label",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )

    # --------------------------------------------------------
    # Normalise
    # --------------------------------------------------------

    df = df.copy()

    df["file_path"] = df["file_path"].apply(normalise_path)

    df["commit_subject"] = (
        df["commit_subject"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["commit_message"] = (
        df["commit_message"]
        .fillna("")
        .astype(str)
    )

    df["commit_refined_defect_label"] = (
        df["commit_refined_defect_label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["commit_sha"] = (
        df["commit_sha"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove duplicate commit/file events
    # --------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["commit_sha", "file_path"],
        keep="first",
    ).copy()

    duplicates_removed = before_duplicates - len(df)

    print(
        f"Duplicate commit/file rows removed : "
        f"{duplicates_removed}"
    )

    # --------------------------------------------------------
    # Show actual refined labels
    # --------------------------------------------------------

    print("\nOriginal refined labels:")

    print(
        df["commit_refined_defect_label"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Identify positive candidates correctly
    # --------------------------------------------------------

    positive_candidate_mask = (
        df["commit_refined_defect_label"]
        .isin(POSITIVE_REFINED_LABELS)
    )

    positive_candidates = df[
        positive_candidate_mask
    ].copy()

    print("\n" + "=" * 70)
    print("POSITIVE CANDIDATES")
    print("=" * 70)

    print(
        f"Positive candidate rows : "
        f"{int(positive_candidate_mask.sum())}"
    )

    # --------------------------------------------------------
    # Prepare message
    # --------------------------------------------------------

    positive_candidates["combined_message"] = (
        positive_candidates["commit_subject"]
        + " "
        + positive_candidates["commit_message"]
    ).str.strip()

    # --------------------------------------------------------
    # Merge exclusion
    # --------------------------------------------------------

    merge_mask = positive_candidates.apply(
        lambda row: is_merge_commit(
            row["commit_subject"],
            row["commit_message"],
        ),
        axis=1,
    )

    excluded_merge_count = int(merge_mask.sum())

    positive_candidates = positive_candidates[
        ~merge_mask
    ].copy()

    print(
        f"Excluded merge rows     : "
        f"{excluded_merge_count}"
    )

    # --------------------------------------------------------
    # Production Python filter
    # --------------------------------------------------------

    production_mask = positive_candidates[
        "file_path"
    ].apply(is_production_python)

    non_source_count = int((~production_mask).sum())

    print(
        f"Excluded non-production : "
        f"{non_source_count}"
    )

    positive_candidates = positive_candidates[
        production_mask
    ].copy()

    # --------------------------------------------------------
    # Strong defect evidence
    # --------------------------------------------------------

    positive_candidates[
        "strong_pattern_matches"
    ] = positive_candidates[
        "combined_message"
    ].apply(
        lambda text: count_pattern_matches(
            text,
            STRONG_DEFECT_PATTERNS,
        )
    )

    positive_candidates[
        "maintenance_pattern_matches"
    ] = positive_candidates[
        "combined_message"
    ].apply(
        lambda text: count_pattern_matches(
            text,
            MAINTENANCE_PATTERNS,
        )
    )

    positive_candidates[
        "strong_match_count"
    ] = positive_candidates[
        "strong_pattern_matches"
    ].apply(
        lambda d: sum(d.values())
    )

    positive_candidates[
        "maintenance_match_count"
    ] = positive_candidates[
        "maintenance_pattern_matches"
    ].apply(
        lambda d: sum(d.values())
    )

    strong_mask = (
        positive_candidates["strong_match_count"] > 0
    )

    no_strong_evidence_count = int(
        (~strong_mask).sum()
    )

    positive_candidates = positive_candidates[
        strong_mask
    ].copy()

    print(
        f"Excluded without strong evidence : "
        f"{no_strong_evidence_count}"
    )

    # --------------------------------------------------------
    # Maintenance-dominant filter
    # --------------------------------------------------------

    maintenance_dominant_mask = (
        positive_candidates[
            "maintenance_match_count"
        ]
        >
        positive_candidates[
            "strong_match_count"
        ]
    )

    maintenance_dominant_count = int(
        maintenance_dominant_mask.sum()
    )

    positive_candidates = positive_candidates[
        ~maintenance_dominant_mask
    ].copy()

    print(
        f"Excluded maintenance-dominant : "
        f"{maintenance_dominant_count}"
    )

    # --------------------------------------------------------
    # Hard maintenance filter
    # --------------------------------------------------------

    hard_maintenance_mask = positive_candidates[
        "combined_message"
    ].apply(has_hard_maintenance_message)

    hard_maintenance_count = int(
        hard_maintenance_mask.sum()
    )

    positive_candidates = positive_candidates[
        ~hard_maintenance_mask
    ].copy()

    print(
        f"Excluded hard-maintenance messages : "
        f"{hard_maintenance_count}"
    )

    # --------------------------------------------------------
    # Create final strict target
    # --------------------------------------------------------

    positive_candidates[
        "strict_defect_target"
    ] = 1

    positive_candidates[
        "strict_target_method"
    ] = (
        "production_python_strong_defect_evidence_v2"
    )

    positive_candidates[
        "strict_target_reason"
    ] = positive_candidates.apply(
        lambda row: (
            f"refined_label={row['commit_refined_defect_label']}; "
            f"strong_matches="
            f"{','.join(row['strong_pattern_matches'].keys())}"
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    positive_candidates["commit_date"] = pd.to_datetime(
        positive_candidates["commit_date"],
        errors="coerce",
        utc=True,
    )

    positive_candidates = positive_candidates.sort_values(
        ["commit_date", "commit_sha", "file_path"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STRICT SOURCE TARGET SUMMARY")
    print("=" * 70)

    print(
        f"Strict positive rows    : "
        f"{len(positive_candidates)}"
    )

    print(
        f"Strict positive commits : "
        f"{positive_candidates['commit_sha'].nunique()}"
    )

    print(
        f"Strict positive files   : "
        f"{positive_candidates['file_path'].nunique()}"
    )

    if len(positive_candidates) > 0:

        print("\nRefined labels retained:")

        print(
            positive_candidates[
                "commit_refined_defect_label"
            ]
            .value_counts()
            .to_string()
        )

        print("\nStrong evidence frequencies:")

        strong_frequency = {}

        for matches in positive_candidates[
            "strong_pattern_matches"
        ]:
            for name, count in matches.items():
                strong_frequency[name] = (
                    strong_frequency.get(name, 0)
                    + count
                )

        if strong_frequency:
            print(
                pd.Series(
                    strong_frequency
                )
                .sort_values(
                    ascending=False
                )
                .to_string()
            )
        else:
            print("No strong evidence patterns.")

        print("\nSample strict targets:")

        sample_columns = [
            "commit_date",
            "commit_sha",
            "commit_subject",
            "file_path",
            "commit_refined_defect_label",
            "strong_match_count",
            "maintenance_match_count",
            "strict_target_reason",
        ]

        print(
            positive_candidates[
                sample_columns
            ]
            .head(30)
            .to_string(index=False)
        )

    else:
        print("\nWARNING: No strict positive targets survived.")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    positive_candidates.to_csv(
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