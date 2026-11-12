from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/fusion/requests_four_signal_fusion_baseline.csv"
)

OUTPUT_FILE = Path(
    "data/processed/fusion/requests_four_signal_fusion_applicability.csv"
)


def classify_file(file_path):
    """
    Classify whether a repository file is relevant to software-risk
    prioritisation.

    The classification is intentionally conservative:
      - production_source: production Python/source code
      - test_code: automated tests
      - configuration: project/build/tool configuration
      - documentation: documentation and explanatory text
      - binary_asset: images/binary assets
      - other: anything not covered above
    """

    path = str(file_path).replace("\\", "/").lower()
    name = path.split("/")[-1]

    # ------------------------------------------------------------
    # Binary / image assets
    # ------------------------------------------------------------
    binary_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".webp",
        ".bmp",
        ".tiff",
        ".pdf",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".zip",
        ".tar",
        ".gz",
        ".mp3",
        ".mp4",
    }

    extension = ""

    if "." in name:
        extension = "." + name.rsplit(".", 1)[1]

    if extension in binary_extensions:
        return "binary_asset"

    # ------------------------------------------------------------
    # Documentation
    # ------------------------------------------------------------
    documentation_directories = (
        "docs/",
        "doc/",
        "documentation/",
    )

    documentation_extensions = {
        ".md",
        ".rst",
        ".txt",
        ".adoc",
        ".asciidoc",
    }

    if path.startswith(documentation_directories):
        return "documentation"

    if extension in documentation_extensions:
        return "documentation"

    # Common documentation files
    documentation_names = {
        "readme",
        "readme.md",
        "readme.rst",
        "changelog",
        "changelog.md",
        "changelog.rst",
        "history.md",
        "history.rst",
        "authors",
        "authors.rst",
        "contributing.md",
        "contributing.rst",
        "license",
        "license.md",
        "license.rst",
    }

    if name in documentation_names:
        return "documentation"

    # ------------------------------------------------------------
    # Test code
    # ------------------------------------------------------------
    test_directories = (
        "tests/",
        "test/",
    )

    if path.startswith(test_directories):
        return "test_code"

    if "/tests/" in path or "/test/" in path:
        return "test_code"

    if name.startswith("test_") and extension in {
        ".py",
        ".pyi",
    }:
        return "test_code"

    if name.endswith("_test.py"):
        return "test_code"

    # ------------------------------------------------------------
    # Configuration / build / CI
    # ------------------------------------------------------------
    configuration_directories = (
        ".github/",
        ".gitlab/",
        ".circleci/",
        "ci/",
        "build/",
        "tools/",
        "scripts/",
    )

    if path.startswith(configuration_directories):
        return "configuration"

    configuration_names = {
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "tox.ini",
        "pytest.ini",
        "mypy.ini",
        "ruff.toml",
        "flake8",
        ".flake8",
        "makefile",
        "dockerfile",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-to-freeze.txt",
        "pipfile",
        "pipfile.lock",
        "poetry.lock",
    }

    if name in configuration_names:
        return "configuration"

    configuration_extensions = {
        ".ini",
        ".cfg",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
    }

    if extension in configuration_extensions:
        return "configuration"

    # ------------------------------------------------------------
    # Production Python/source code
    # ------------------------------------------------------------
    if extension in {".py", ".pyi"}:

        # Exclude obvious development/support directories
        excluded_source_directories = (
            "examples/",
            "example/",
            "benchmarks/",
            "benchmark/",
            "scripts/",
            "tools/",
        )

        if path.startswith(excluded_source_directories):
            return "other"

        return "production_source"

    # ------------------------------------------------------------
    # Other source-code extensions
    # ------------------------------------------------------------
    source_extensions = {
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".kts",
    }

    if extension in source_extensions:
        return "production_source"

    # ------------------------------------------------------------
    # Everything else
    # ------------------------------------------------------------
    return "other"


def main():

    print("=" * 70)
    print("CODEGUARD AI - FUSION APPLICABILITY CLASSIFICATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load fusion dataset
    # ------------------------------------------------------------
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput file: {INPUT_FILE}")
    print(f"Input rows: {len(df)}")
    print(f"Input columns: {len(df.columns)}")

    # ------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------
    required_columns = {
        "file_path",
        "unified_risk",
        "unified_risk_level",
        "fusion_signal_count",
        "fusion_signal_completeness",
        "signals_used",
    }

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    duplicate_files = df["file_path"].duplicated().sum()

    print(f"Duplicate files: {duplicate_files}")

    if duplicate_files != 0:
        raise ValueError(
            "Duplicate file paths detected."
        )

    # ------------------------------------------------------------
    # Classify every file
    # ------------------------------------------------------------
    df["applicability_category"] = df["file_path"].apply(
        classify_file
    )

    # ------------------------------------------------------------
    # Knowledge relevance
    # ------------------------------------------------------------
    knowledge_relevant_categories = {
        "production_source",
        "test_code",
        "configuration",
    }

    df["risk_applicable"] = (
        df["applicability_category"]
        .isin(knowledge_relevant_categories)
    )

    # ------------------------------------------------------------
    # Applicability reason
    # ------------------------------------------------------------
    reason_map = {
        "production_source": "Production source code can directly contain software defects.",
        "test_code": "Test code affects defect detection and test maintenance risk.",
        "configuration": "Configuration/build/CI files can affect software behaviour or delivery.",
        "documentation": "Documentation is retained for traceability but is outside primary software-risk scoring.",
        "binary_asset": "Binary assets are retained for repository completeness but are outside primary software-risk scoring.",
        "other": "File type is not classified as a primary software-risk target.",
    }

    df["applicability_reason"] = (
        df["applicability_category"]
        .map(reason_map)
    )

    # ------------------------------------------------------------
    # Applicability-aware priority
    # ------------------------------------------------------------
    #
    # We DO NOT delete non-applicable files.
    #
    # Instead:
    #   - risk_applicable=True  -> eligible for risk prioritisation
    #   - risk_applicable=False -> retained for repository traceability
    #
    df["prioritization_eligible"] = df["risk_applicable"]

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("APPLICABILITY SUMMARY")
    print("=" * 70)

    print("\nCategory counts:")

    category_counts = (
        df["applicability_category"]
        .value_counts()
        .sort_index()
    )

    print(category_counts.to_string())

    print("\nRisk applicability:")

    applicable_count = int(df["risk_applicable"].sum())
    not_applicable_count = int(
        (~df["risk_applicable"]).sum()
    )

    print(
        f"Risk-applicable files     : {applicable_count}"
    )
    print(
        f"Not primarily applicable  : {not_applicable_count}"
    )

    # ------------------------------------------------------------
    # Risk distribution among applicable files
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RISK DISTRIBUTION - APPLICABLE FILES")
    print("=" * 70)

    applicable_df = df[df["risk_applicable"]].copy()

    if len(applicable_df) > 0:

        risk_distribution = (
            applicable_df["unified_risk_level"]
            .value_counts()
            .reindex(
                ["Low", "Medium", "High", "Critical"],
                fill_value=0,
            )
        )

        print(risk_distribution.to_string())

        print(
            f"\nApplicable files: {len(applicable_df)}"
        )

        print(
            f"Average unified risk: "
            f"{applicable_df['unified_risk'].mean():.4f}"
        )

        print(
            f"Minimum unified risk: "
            f"{applicable_df['unified_risk'].min():.4f}"
        )

        print(
            f"Maximum unified risk: "
            f"{applicable_df['unified_risk'].max():.4f}"
        )

    # ------------------------------------------------------------
    # Top applicable files
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TOP APPLICABLE FILES BY UNIFIED RISK")
    print("=" * 70)

    top_columns = [
        "file_path",
        "applicability_category",
        "risk_applicable",
        "unified_risk",
        "unified_risk_level",
        "fusion_signal_count",
        "signals_used",
    ]

    top_applicable = (
        applicable_df[top_columns]
        .sort_values(
            "unified_risk",
            ascending=False,
        )
        .head(20)
    )

    if len(top_applicable) > 0:
        print(
            top_applicable.to_string(
                index=False
            )
        )

    # ------------------------------------------------------------
    # Non-applicable high-scoring files
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("NON-APPLICABLE FILES WITH HIGHEST FUSION SCORES")
    print("=" * 70)

    non_applicable_df = df[
        ~df["risk_applicable"]
    ].copy()

    non_applicable_columns = [
        "file_path",
        "applicability_category",
        "unified_risk",
        "unified_risk_level",
        "fusion_signal_count",
        "signals_used",
    ]

    top_non_applicable = (
        non_applicable_df[non_applicable_columns]
        .sort_values(
            "unified_risk",
            ascending=False,
        )
        .head(15)
    )

    if len(top_non_applicable) > 0:
        print(
            top_non_applicable.to_string(
                index=False
            )
        )

    # ------------------------------------------------------------
    # Signal completeness by applicability
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SIGNAL COMPLETENESS BY APPLICABILITY")
    print("=" * 70)

    completeness_summary = (
        df.groupby("risk_applicable")[
            "fusion_signal_completeness"
        ]
        .agg(
            ["count", "mean", "min", "max"]
        )
    )

    print(completeness_summary.to_string())

    # ------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------
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

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Final shape: {df.shape}")

    print("\nAdded columns:")
    print("  applicability_category")
    print("  risk_applicable")
    print("  applicability_reason")
    print("  prioritization_eligible")

    print("\nImportant:")
    print(
        "All repository files are retained. "
        "Only risk-applicable files are marked eligible "
        "for primary risk prioritisation."
    )

    print("\nClassification complete.")


if __name__ == "__main__":
    main()