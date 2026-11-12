import os
import math
import pandas as pd


INPUT_FILE = "data/processed/bus_factor/requests_bus_factor.csv"
OUTPUT_FILE = "data/processed/bus_factor/requests_bus_factor_refined.csv"


def classify_file(file_path):
    """
    Classify repository files.

    Classification is used for analysis and downstream modeling.
    It is not directly multiplied into the bus-risk formula.
    """

    path = str(file_path).replace("\\", "/").lower()
    filename = path.split("/")[-1]

    _, extension = os.path.splitext(filename)

    # ---------------------------------------------------------------
    # Binary assets
    # ---------------------------------------------------------------

    binary_extensions = {
        ".png", ".jpg", ".jpeg", ".gif",
        ".bmp", ".ico", ".webp",
        ".zip", ".tar", ".gz",
        ".woff", ".woff2", ".ttf"
    }

    if extension in binary_extensions:
        return "binary_asset"

    # ---------------------------------------------------------------
    # Tests
    # ---------------------------------------------------------------

    if path.startswith("tests/") and extension == ".py":
        return "test_code"

    # ---------------------------------------------------------------
    # Configuration / dependency management
    # ---------------------------------------------------------------

    configuration_files = {
        "makefile",
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "tox.ini",
        "pytest.ini",
        ".coveragerc",
        ".gitignore",
        ".gitattributes",
        ".editorconfig",
        "requirements.txt",
        "requirements-to-freeze.txt",
        "circle.yml",
    }

    if (
        filename in configuration_files
        or path.startswith(".github/")
        or path.startswith(".vscode/")
        or path.startswith(".idea/")
    ):
        return "configuration"

    # ---------------------------------------------------------------
    # Production source
    # ---------------------------------------------------------------

    if (
        extension == ".py"
        and (
            path.startswith("src/requests/")
            or path.startswith("requests/")
        )
    ):
        return "production_source"

    # ---------------------------------------------------------------
    # Documentation
    # ---------------------------------------------------------------

    documentation_extensions = {
        ".md", ".rst", ".txt"
    }

    if path.startswith("docs/"):
        return "documentation"

    if extension in documentation_extensions:
        return "documentation"

    # ---------------------------------------------------------------
    # Other Python
    # ---------------------------------------------------------------

    if extension == ".py":
        return "python_code"

    return "other"


def normalize_entropy(row):
    """
    Normalize Shannon entropy.

    H_norm = H / log(N)

    N = number of contributors.
    """

    contributor_count = max(
        int(row["contributor_count"]),
        1
    )

    entropy = float(
        row["contribution_entropy"]
    )

    if contributor_count <= 1:
        return 0.0

    denominator = math.log(
        contributor_count
    )

    if denominator <= 0:
        return 0.0

    value = entropy / denominator

    return max(
        0.0,
        min(value, 1.0)
    )


def bus_factor_risk(bus_factor):
    """
    Convert bus factor into a concentration-risk signal.

    BF = 1 -> 1.000
    BF = 2 -> 0.500
    BF = 3 -> 0.333
    ...
    """

    value = max(
        float(bus_factor),
        1.0
    )

    return 1.0 / value


def activity_score(total_commits, maximum_commits):
    """
    Log-normalized file activity.
    """

    commits = max(
        float(total_commits),
        1.0
    )

    maximum = max(
        float(maximum_commits),
        1.0
    )

    denominator = math.log1p(maximum)

    if denominator <= 0:
        return 0.0

    value = (
        math.log1p(commits)
        / denominator
    )

    return max(
        0.0,
        min(value, 1.0)
    )


def risk_level(score):
    """
    Operational risk thresholds.

    These are heuristic thresholds and will later be
    validated against project-specific evidence.
    """

    if score >= 0.80:
        return "Critical"

    if score >= 0.60:
        return "High"

    if score >= 0.40:
        return "Medium"

    return "Low"


def main():

    print("=" * 70)
    print("CodeGuard AI - Refined Bus Factor Risk Analysis v3")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError(
            "Input CSV is empty."
        )

    required_columns = [
        "file_path",
        "total_commits",
        "contributor_count",
        "top_contributor_share",
        "contribution_entropy",
        "bus_factor",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    print(f"\nInput file : {INPUT_FILE}")
    print(f"Rows       : {len(df)}")

    # ===============================================================
    # 1. File category
    # ===============================================================

    df["file_category"] = (
        df["file_path"]
        .apply(classify_file)
    )

    # ===============================================================
    # 2. Clean numeric values
    # ===============================================================

    numeric_columns = [
        "total_commits",
        "contributor_count",
        "top_contributor_share",
        "contribution_entropy",
        "bus_factor",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df[numeric_columns] = (
        df[numeric_columns]
        .fillna(0)
    )

    df["total_commits"] = (
        df["total_commits"]
        .clip(lower=1)
    )

    df["contributor_count"] = (
        df["contributor_count"]
        .clip(lower=1)
    )

    df["bus_factor"] = (
        df["bus_factor"]
        .clip(lower=1)
    )

    # ===============================================================
    # 3. Correct contributor share
    # ===============================================================

    df["top_contributor_share"] = (
    df["top_contributor_share"]
    .clip(lower=0, upper=1)
    .round(4)
)

    # ===============================================================
    # 4. Normalized entropy
    # ===============================================================

    df["normalized_entropy"] = (
        df.apply(
            normalize_entropy,
            axis=1
        )
    )

    # ===============================================================
    # 5. Entropy concentration
    # ===============================================================

    df["entropy_concentration"] = (
        1.0
        - df["normalized_entropy"]
    )

    # ===============================================================
    # 6. Bus factor risk
    # ===============================================================

    df["bus_factor_risk"] = (
        df["bus_factor"]
        .apply(bus_factor_risk)
    )

    # ===============================================================
    # 7. Knowledge concentration
    # ===============================================================

    df["knowledge_concentration"] = (
        0.50 * df["top_contributor_share"]
        + 0.25 * df["entropy_concentration"]
        + 0.25 * df["bus_factor_risk"]
    )

    df["knowledge_concentration"] = (
        df["knowledge_concentration"]
        .clip(lower=0, upper=1)
    )

    # ===============================================================
    # 8. Activity
    # ===============================================================

    maximum_commits = (
        df["total_commits"].max()
    )

    df["activity_score"] = (
        df["total_commits"]
        .apply(
            lambda x:
            activity_score(
                x,
                maximum_commits
            )
        )
    )

    # ===============================================================
    # 9. Activity-adjusted bus risk
    # ===============================================================

    df["refined_bus_risk"] = (
        df["knowledge_concentration"]
        *
        (
            0.50
            + 0.50 * df["activity_score"]
        )
    )

    df["refined_bus_risk"] = (
        df["refined_bus_risk"]
        .clip(lower=0, upper=1)
    )

    # ===============================================================
    # 10. Risk level
    # ===============================================================

    df["refined_risk_level"] = (
        df["refined_bus_risk"]
        .apply(risk_level)
    )

    # ===============================================================
    # 11. Knowledge relevance
    # ===============================================================

    df["knowledge_relevant"] = (
        df["file_category"].isin(
            [
                "production_source",
                "test_code",
                "configuration",
            ]
        )
    )

    # ===============================================================
    # 12. Risk method
    # ===============================================================

    df["risk_method"] = (
        "knowledge_concentration_v3"
    )

    # ===============================================================
    # 13. Arrange columns
    # ===============================================================

    preferred_columns = [
        "file_path",
        "file_category",
        "knowledge_relevant",

        "total_commits",
        "contributor_count",
        "total_contributions",

        "top_contributor_share",
        "contribution_entropy",
        "normalized_entropy",
        "entropy_concentration",

        "bus_factor",
        "bus_factor_risk",

        "knowledge_concentration",
        "activity_score",

        "refined_bus_risk",
        "refined_risk_level",

        "risk_method",

        "total_additions",
        "total_deletions",

        "bus_risk",
        "bus_risk_level",
    ]

    existing = [
        column
        for column in preferred_columns
        if column in df.columns
    ]

    remaining = [
        column
        for column in df.columns
        if column not in existing
    ]

    df = df[
        existing + remaining
    ]

    # ===============================================================
    # 14. Save
    # ===============================================================

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ===============================================================
    # 15. Summary
    # ===============================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Files analysed       : {len(df)}"
    )

    print(
        f"Average contributors : "
        f"{df['contributor_count'].mean():.2f}"
    )

    print(
        f"Average bus factor   : "
        f"{df['bus_factor'].mean():.2f}"
    )

    print(
        f"Average knowledge concentration : "
        f"{df['knowledge_concentration'].mean():.4f}"
    )

    print(
        f"Average activity score           : "
        f"{df['activity_score'].mean():.4f}"
    )

    print(
        f"Average refined bus risk         : "
        f"{df['refined_bus_risk'].mean():.4f}"
    )

    # ===============================================================
    # 16. Category distribution
    # ===============================================================

    print("\n" + "=" * 70)
    print("FILE CATEGORY DISTRIBUTION")
    print("=" * 70)

    print(
        df["file_category"]
        .value_counts()
        .to_string()
    )

    # ===============================================================
    # 17. Knowledge relevance
    # ===============================================================

    print("\n" + "=" * 70)
    print("KNOWLEDGE RELEVANCE")
    print("=" * 70)

    print(
        df["knowledge_relevant"]
        .value_counts()
        .to_string()
    )

    # ===============================================================
    # 18. Risk distribution
    # ===============================================================

    print("\n" + "=" * 70)
    print("REFINED RISK LEVEL DISTRIBUTION")
    print("=" * 70)

    print(
        df["refined_risk_level"]
        .value_counts()
        .to_string()
    )

    # ===============================================================
    # 19. Top risky files
    # ===============================================================

    print("\n" + "=" * 70)
    print("TOP 20 KNOWLEDGE-RELEVANT FILES")
    print("=" * 70)

    relevant = df[
        df["knowledge_relevant"] == True
    ].copy()

    top_risk = (
        relevant
        .sort_values(
            "refined_bus_risk",
            ascending=False
        )
        .head(20)
    )

    display_columns = [
        "file_path",
        "file_category",
        "total_commits",
        "contributor_count",
        "top_contributor_share",
        "bus_factor",
        "knowledge_concentration",
        "activity_score",
        "refined_bus_risk",
        "refined_risk_level",
    ]

    print(
        top_risk[
            display_columns
        ].to_string(index=False)
    )

    # ===============================================================
    # 20. Save confirmation
    # ===============================================================

    print("\n" + "=" * 70)
    print(
        f"Output saved to: {OUTPUT_FILE}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()