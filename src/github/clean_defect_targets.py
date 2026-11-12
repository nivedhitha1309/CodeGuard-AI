from pathlib import Path
import re
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/github/requests_refined_defect_labels.csv"
)

OUTPUT_FILE = Path(
    "data/processed/github/requests_clean_defect_targets.csv"
)


# ============================================================
# PATH PATTERNS
# ============================================================

DOC_PATH_PATTERN = re.compile(
    r"(^|/)"
    r"(docs?|documentation|examples?)(/|$)"
    r"|"
    r"\.(md|rst|txt)$",
    re.IGNORECASE,
)

DOC_FILE_PATTERN = re.compile(
    r"(CONTRIBUTING|HISTORY|AUTHORS|CHANGELOG|README)",
    re.IGNORECASE,
)

CI_PATH_PATTERN = re.compile(
    r"(^|/)"
    r"(\.github|ci|\.circleci|\.travis)(/|$)"
    r"|"
    r"(^|/)(Makefile|tox\.ini|\.pre-commit-config\.yaml)$",
    re.IGNORECASE,
)

TEST_PATH_PATTERN = re.compile(
    r"(^|/)(tests?|testing)(/|$)"
    r"|"
    r"(^|/)test_[^/]+$"
    r"|"
    r"(^|/)[^/]+_test\.py$",
    re.IGNORECASE,
)

SOURCE_FILE_PATTERN = re.compile(
    r"\.(py|pyi)$",
    re.IGNORECASE,
)


# ============================================================
# COMMIT MESSAGE PATTERNS
# ============================================================

MERGE_PATTERN = re.compile(
    r"^\s*merge\b",
    re.IGNORECASE,
)

SECURITY_PATTERN = re.compile(
    r"\b("
    r"security|"
    r"vulnerability|"
    r"vulnerable|"
    r"cve-\d{4}-\d+|"
    r"security\s+issue|"
    r"security\s+fix"
    r")\b",
    re.IGNORECASE,
)

REGRESSION_PATTERN = re.compile(
    r"\b("
    r"regression|"
    r"regressed|"
    r"regressions"
    r")\b",
    re.IGNORECASE,
)

STRONG_DEFECT_PATTERN = re.compile(
    r"\b("
    r"bug|"
    r"bugfix|"
    r"defect|"
    r"crash|"
    r"incorrect|"
    r"incorrectly|"
    r"broken|"
    r"breaks|"
    r"failure|"
    r"fails|"
    r"malformed|"
    r"wrong|"
    r"invalid|"
    r"unexpected|"
    r"unintended|"
    r"exception|"
    r"fix\s+#?\d+|"
    r"issue\s+#?\d+"
    r")\b",
    re.IGNORECASE,
)

# Explicit defect-fix verbs.
FIX_PATTERN = re.compile(
    r"\b("
    r"fix|"
    r"fixes|"
    r"fixed|"
    r"fixing|"
    r"resolve|"
    r"resolves|"
    r"resolved|"
    r"correct|"
    r"corrects|"
    r"corrected|"
    r"prevent|"
    r"prevents|"
    r"prevented"
    r")\b",
    re.IGNORECASE,
)

TYPO_PATTERN = re.compile(
    r"\b("
    r"typo|"
    r"typos|"
    r"spelling|"
    r"grammar|"
    r"wording|"
    r"punctuation|"
    r"cosmetic"
    r")\b",
    re.IGNORECASE,
)

STYLE_PATTERN = re.compile(
    r"\b("
    r"style|"
    r"formatting|"
    r"format|"
    r"lint|"
    r"linting|"
    r"flake8|"
    r"black|"
    r"isort|"
    r"pre-commit"
    r")\b",
    re.IGNORECASE,
)

REFACTOR_PATTERN = re.compile(
    r"\b("
    r"refactor|"
    r"refactoring|"
    r"cleanup|"
    r"clean-up|"
    r"reorganize|"
    r"restructure"
    r")\b",
    re.IGNORECASE,
)

TYPING_PATTERN = re.compile(
    r"\b("
    r"typing|"
    r"type hints?|"
    r"static typing|"
    r"type checking|"
    r"mypy"
    r")\b",
    re.IGNORECASE,
)

CI_PATTERN = re.compile(
    r"\b("
    r"ci|"
    r"build|"
    r"workflow|"
    r"github actions|"
    r"pipeline|"
    r"tox"
    r")\b",
    re.IGNORECASE,
)

DEPENDENCY_PATTERN = re.compile(
    r"\b("
    r"dependabot|"
    r"dependency|"
    r"dependencies|"
    r"renovate|"
    r"requirements|"
    r"upgrade"
    r")\b",
    re.IGNORECASE,
)

BUMP_PATTERN = re.compile(
    r"\bbump\b",
    re.IGNORECASE,
)

DOC_LANGUAGE_PATTERN = re.compile(
    r"\b("
    r"docs?|"
    r"documentation|"
    r"readme|"
    r"dead link|"
    r"broken link|"
    r"example|"
    r"examples"
    r")\b",
    re.IGNORECASE,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_path(value):
    """Normalize repository paths."""

    if pd.isna(value):
        return ""

    value = str(value).strip().replace("\\", "/")

    value = re.sub(
        r"^/+|/+$",
        "",
        value,
    )

    return value


def is_documentation_file(path):
    """Return True for documentation files."""

    if not path:
        return False

    return bool(
        DOC_PATH_PATTERN.search(path)
        or DOC_FILE_PATTERN.search(path)
    )


def is_ci_file(path):
    """Return True for CI/build files."""

    if not path:
        return False

    return bool(
        CI_PATH_PATTERN.search(path)
    )


def is_test_file(path):
    """Return True for test files."""

    if not path:
        return False

    return bool(
        TEST_PATH_PATTERN.search(path)
    )


def is_source_file(path):
    """Return True for Python source files."""

    if not path:
        return False

    return bool(
        SOURCE_FILE_PATTERN.search(path)
    )


def is_merge_commit(subject):
    """Return True for merge commits."""

    if not subject:
        return False

    return bool(
        MERGE_PATTERN.search(subject)
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_commit(row):
    """
    Conservative defect-target classification.

    Positive categories are intentionally strict.

    This creates candidate historical labels,
    not manually verified ground truth.
    """

    subject = str(
        row.get("commit_subject", "") or ""
    ).strip()

    message = str(
        row.get("commit_message", "") or ""
    ).strip()

    path = normalize_path(
        row.get("file_path", "")
    )

    text = f"{subject} {message}".strip()

    # --------------------------------------------------------
    # File characteristics
    # --------------------------------------------------------

    source_file = is_source_file(path)
    test_file = is_test_file(path)
    documentation_file = is_documentation_file(path)
    ci_file = is_ci_file(path)

    # --------------------------------------------------------
    # 1. Merge commits
    # --------------------------------------------------------

    if is_merge_commit(subject):
        return "excluded_merge_commit"

    # --------------------------------------------------------
    # 2. Documentation files
    # --------------------------------------------------------

    if documentation_file:
        return "excluded_documentation"

    # --------------------------------------------------------
    # 3. CI/build files
    # --------------------------------------------------------

    if ci_file:
        return "excluded_ci_build"

    # --------------------------------------------------------
    # 4. Non-Python files
    # --------------------------------------------------------

    if not source_file:
        return "excluded_non_source"

    # --------------------------------------------------------
    # 5. Test files
    # --------------------------------------------------------

    if test_file:

        if SECURITY_PATTERN.search(text):
            return "test_only_security_evidence"

        if REGRESSION_PATTERN.search(text):
            return "test_only_regression_evidence"

        if STRONG_DEFECT_PATTERN.search(text):
            return "test_only_defect_evidence"

        return "excluded_test_only"

    # --------------------------------------------------------
    # 6. Typo / style / formatting
    # --------------------------------------------------------

    if TYPO_PATTERN.search(text):
        return "excluded_typo_or_style"

    if STYLE_PATTERN.search(text):
        return "excluded_style_or_lint"

    # --------------------------------------------------------
    # 7. Refactoring / cleanup
    # --------------------------------------------------------

    if REFACTOR_PATTERN.search(text):
        if not (
            SECURITY_PATTERN.search(text)
            or REGRESSION_PATTERN.search(text)
            or STRONG_DEFECT_PATTERN.search(text)
        ):
            return "excluded_refactor_or_cleanup"

    # --------------------------------------------------------
    # 8. Typing-only changes
    # --------------------------------------------------------

    if TYPING_PATTERN.search(text):

        if not (
            SECURITY_PATTERN.search(text)
            or REGRESSION_PATTERN.search(text)
            or STRONG_DEFECT_PATTERN.search(text)
        ):
            return "excluded_typing_maintenance"

    # --------------------------------------------------------
    # 9. CI/build language inside source commit
    # --------------------------------------------------------

    if CI_PATTERN.search(text):

        if not (
            SECURITY_PATTERN.search(text)
            or REGRESSION_PATTERN.search(text)
            or STRONG_DEFECT_PATTERN.search(text)
        ):
            return "excluded_ci_language"

    # --------------------------------------------------------
    # 10. Dependency maintenance
    # --------------------------------------------------------

    if DEPENDENCY_PATTERN.search(text):
        return "excluded_dependency_maintenance"

    if BUMP_PATTERN.search(text):
        return "excluded_version_or_dependency_bump"

    # --------------------------------------------------------
    # 11. Documentation language
    # --------------------------------------------------------

    if DOC_LANGUAGE_PATTERN.search(text):

        if not (
            SECURITY_PATTERN.search(text)
            or REGRESSION_PATTERN.search(text)
            or STRONG_DEFECT_PATTERN.search(text)
        ):
            return "excluded_documentation_language"

    # --------------------------------------------------------
    # 12. SECURITY DEFECT
    # --------------------------------------------------------

    if SECURITY_PATTERN.search(text):

        return "security_defect_candidate"

    # --------------------------------------------------------
    # 13. REGRESSION DEFECT
    # --------------------------------------------------------

    if REGRESSION_PATTERN.search(text):

        return "regression_defect_candidate"

    # --------------------------------------------------------
    # 14. STRONG DEFECT EVIDENCE
    # --------------------------------------------------------

    if STRONG_DEFECT_PATTERN.search(text):

        return "likely_defect_candidate"

    # --------------------------------------------------------
    # 15. Explicit "fix" language
    #
    # We only accept generic "fix" when the subject also
    # describes a concrete software behavior/problem.
    # --------------------------------------------------------

    if FIX_PATTERN.search(subject):

        subject_without_fix = FIX_PATTERN.sub(
            "",
            subject,
        ).strip()

        # Reject vague maintenance subjects.
        if len(subject_without_fix) < 8:
            return "uncertain_fix"

        # Reject clearly non-defect concepts.
        if TYPO_PATTERN.search(subject):
            return "excluded_typo_or_style"

        if STYLE_PATTERN.search(subject):
            return "excluded_style_or_lint"

        if REFACTOR_PATTERN.search(subject):
            return "excluded_refactor_or_cleanup"

        if TYPING_PATTERN.search(subject):
            return "excluded_typing_maintenance"

        if CI_PATTERN.search(subject):
            return "excluded_ci_language"

        if DOC_LANGUAGE_PATTERN.search(subject):
            return "excluded_documentation_language"

        return "likely_defect_candidate"

    # --------------------------------------------------------
    # 16. No sufficient defect evidence
    # --------------------------------------------------------

    return "non_defect"


def assign_target(category):
    """Convert category into binary target."""

    positive_categories = {
        "security_defect_candidate",
        "regression_defect_candidate",
        "likely_defect_candidate",
    }

    return int(
        category in positive_categories
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CODEGUARD AI - CONSERVATIVE DEFECT TARGET CLEANING")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(
        f"\nInput rows    : {len(df)}"
    )

    print(
        f"Input columns : {len(df.columns)}"
    )

    required_columns = {
        "commit_sha",
        "commit_subject",
        "commit_message",
        "author",
        "file_path",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    # --------------------------------------------------------
    # Normalize paths
    # --------------------------------------------------------

    df["file_path"] = (
        df["file_path"]
        .apply(normalize_path)
    )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "commit_sha",
            "file_path",
        ]
    ).copy()

    removed = before - len(df)

    print(
        "Duplicate commit/file rows removed : "
        f"{removed}"
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    df["clean_defect_category"] = (
        df.apply(
            classify_commit,
            axis=1,
        )
    )

    df["defect_target"] = (
        df["clean_defect_category"]
        .apply(assign_target)
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    df["is_source_file"] = (
        df["file_path"]
        .apply(is_source_file)
    )

    df["is_test_file"] = (
        df["file_path"]
        .apply(is_test_file)
    )

    df["is_documentation_file"] = (
        df["file_path"]
        .apply(is_documentation_file)
    )

    df["is_ci_file"] = (
        df["file_path"]
        .apply(is_ci_file)
    )

    df["is_merge_commit"] = (
        df["commit_subject"]
        .fillna("")
        .apply(is_merge_commit)
    )

    # --------------------------------------------------------
    # Commit-level summary
    # --------------------------------------------------------

    commit_summary = (
        df.groupby(
            "commit_sha",
            as_index=False,
        )
        .agg(
            commit_subject=(
                "commit_subject",
                "first",
            ),
            author=(
                "author",
                "first",
            ),
            commit_date=(
                "commit_date",
                "first",
            ),
            file_count=(
                "file_path",
                "nunique",
            ),
            target_file_count=(
                "defect_target",
                "sum",
            ),
        )
    )

    commit_summary[
        "commit_defect_target"
    ] = (
        commit_summary[
            "target_file_count"
        ] > 0
    ).astype(int)

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

    # ========================================================
    # REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("FILE-HISTORY CLEAN CATEGORY DISTRIBUTION")
    print("=" * 70)

    print(
        df[
            "clean_defect_category"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FILE-HISTORY TARGET DISTRIBUTION")
    print("=" * 70)

    target_distribution = (
        df["defect_target"]
        .value_counts()
        .sort_index()
        .rename(
            index={
                0: "non_target",
                1: "defect_target",
            }
        )
    )

    print(
        target_distribution.to_string()
    )

    # --------------------------------------------------------
    # Commit distribution
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMMIT-LEVEL TARGET DISTRIBUTION")
    print("=" * 70)

    commit_distribution = (
        commit_summary[
            "commit_defect_target"
        ]
        .value_counts()
        .sort_index()
        .rename(
            index={
                0: "non_target_commit",
                1: "defect_target_commit",
            }
        )
    )

    print(
        commit_distribution.to_string()
    )

    # --------------------------------------------------------
    # Target summary
    # --------------------------------------------------------

    target_rows = df[
        df["defect_target"] == 1
    ].copy()

    print("\n" + "=" * 70)
    print("TARGET SUMMARY")
    print("=" * 70)

    print(
        "Defect-target file-history rows : "
        f"{len(target_rows)}"
    )

    print(
        "Defect-target unique commits     : "
        f"{target_rows['commit_sha'].nunique()}"
    )

    print(
        "Defect-target unique files       : "
        f"{target_rows['file_path'].nunique()}"
    )

    print("\nTarget categories:")

    if len(target_rows) > 0:
        print(
            target_rows[
                "clean_defect_category"
            ]
            .value_counts()
            .to_string()
        )
    else:
        print("No target rows.")

    # --------------------------------------------------------
    # Sample targets
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAMPLE DEFECT TARGETS")
    print("=" * 70)

    if len(target_rows) > 0:

        sample = target_rows[
            [
                "commit_sha",
                "commit_subject",
                "file_path",
                "clean_defect_category",
                "defect_target",
            ]
        ].head(20)

        for _, row in sample.iterrows():

            print(
                f"\nCommit : "
                f"{row['commit_sha']}"
            )

            print(
                f"Subject: "
                f"{row['commit_subject']}"
            )

            print(
                f"File   : "
                f"{row['file_path']}"
            )

            print(
                f"Class  : "
                f"{row['clean_defect_category']}"
            )

    else:
        print(
            "No defect-target rows found."
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"Saved file: {OUTPUT_FILE}"
    )

    print(
        "\nThese are conservative candidate "
        "defect labels, not manually verified "
        "ground truth."
    )

    print(
        "Historical ML features will later "
        "be constructed using only information "
        "available before each defect event."
    )


if __name__ == "__main__":
    main()