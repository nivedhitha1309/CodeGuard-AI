import os
import numpy as np
import pandas as pd


INPUT_FILE = "data/raw/github/requests_commit_file_history.csv"
OUTPUT_FILE = "data/processed/bus_factor/requests_bus_factor.csv"


def calculate_entropy(shares):
    """
    Shannon entropy of contributor contribution shares.

    Higher entropy = knowledge is more distributed.
    Lower entropy = knowledge is more concentrated.
    """
    shares = np.array(shares, dtype=float)
    shares = shares[shares > 0]

    if len(shares) <= 1:
        return 0.0

    return float(-np.sum(shares * np.log2(shares)))


def calculate_bus_factor(contribution_counts):
    """
    Contribution-coverage bus factor.

    Sort contributors by contribution count in descending order.
    The bus factor is the minimum number of human contributors
    required to account for at least 50% of contributions.
    """
    counts = sorted(contribution_counts, reverse=True)

    total = sum(counts)

    if total == 0:
        return 0

    cumulative = 0

    for index, count in enumerate(counts, start=1):
        cumulative += count

        if cumulative / total >= 0.50:
            return index

    return len(counts)


def calculate_bus_risk(
    bus_factor,
    contributor_count,
    top_contributor_share
):
    """
    Operational bus-factor risk score.

    Higher concentration and lower bus factor indicate
    higher knowledge concentration risk.

    This is a heuristic research implementation and is
    not a validated production threshold.
    """

    if contributor_count <= 0:
        return 0.0

    if contributor_count == 1:
        return 1.0

    # Safety: contribution share must remain in [0, 1].
    concentration_risk = min(
        max(float(top_contributor_share), 0.0),
        1.0
    )

    bus_factor_risk = 1 - (
        min(bus_factor, contributor_count)
        / contributor_count
    )

    risk = (
        0.70 * concentration_risk
        + 0.30 * bus_factor_risk
    )

    return round(
        min(max(float(risk), 0.0), 1.0),
        4
    )


def is_bot_author(author):
    """
    Detect GitHub-style bot accounts.

    Example:
        dependabot[bot] -> True

    Bots are retained in the raw dataset but are excluded
    from human knowledge-holder calculations.
    """
    if pd.isna(author):
        return False

    author = str(author).strip().lower()

    return author.endswith("[bot]")


def main():

    print("=" * 70)
    print("CodeGuard AI - Bus Factor Analysis")
    print("=" * 70)

    print("\nLoading dataset:")
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput records      : {len(df)}")
    print(f"Unique files       : {df['file_path'].nunique()}")
    print(f"Unique commits     : {df['commit_sha'].nunique()}")
    print(f"Unique authors     : {df['author'].nunique()}")

    # ---------------------------------------------------------------
    # Identify bot accounts
    # ---------------------------------------------------------------

    df["is_bot"] = df["author"].apply(is_bot_author)

    bot_records = int(df["is_bot"].sum())
    human_records = int((~df["is_bot"]).sum())

    print(f"\nBot records        : {bot_records}")
    print(f"Human records      : {human_records}")

    # ---------------------------------------------------------------
    # Total file activity
    #
    # This remains based on ALL commits, including automated commits.
    # It is useful later when measuring file activity.
    # ---------------------------------------------------------------

    file_activity = (
        df.groupby("file_path")
        .agg(
            total_commits=("commit_sha", "nunique"),
            total_additions=("additions", "sum"),
            total_deletions=("deletions", "sum")
        )
        .reset_index()
    )

    # ---------------------------------------------------------------
    # Human knowledge contribution
    #
    # Bots are excluded from contributor/knowledge calculations.
    # One row represents one author-file-commit relationship.
    # We count unique commits per human author per file.
    # ---------------------------------------------------------------

    human_df = df[~df["is_bot"]].copy()

    author_file = (
        human_df
        .groupby(["file_path", "author"])["commit_sha"]
        .nunique()
        .reset_index(name="contribution_count")
    )

    # ---------------------------------------------------------------
    # Calculate file-level bus-factor statistics
    # ---------------------------------------------------------------

    records = []

    all_files = df["file_path"].drop_duplicates()

    for file_path in all_files:

        group = author_file[
            author_file["file_path"] == file_path
        ]

        contribution_counts = (
            group["contribution_count"].tolist()
        )

        total_human_contributions = sum(
            contribution_counts
        )

        contributor_count = len(group)

        # -----------------------------------------------------------
        # No human contributors
        # -----------------------------------------------------------

        if total_human_contributions == 0:

            top_contributor_share = 0.0
            entropy = 0.0
            bus_factor = 0
            bus_risk = 0.0

        else:

            shares = [
                count / total_human_contributions
                for count in contribution_counts
            ]

            shares_sorted = sorted(
                shares,
                reverse=True
            )

            # Numerical safety: share must never exceed 1.
            top_contributor_share = min(
                max(float(shares_sorted[0]), 0.0),
                1.0
            )

            entropy = calculate_entropy(shares)

            bus_factor = calculate_bus_factor(
                contribution_counts
            )

            bus_risk = calculate_bus_risk(
                bus_factor,
                contributor_count,
                top_contributor_share
            )

        # -----------------------------------------------------------
        # Complete file activity
        # -----------------------------------------------------------

        activity_row = file_activity[
            file_activity["file_path"] == file_path
        ].iloc[0]

        total_commits = int(
            activity_row["total_commits"]
        )

        total_additions = int(
            activity_row["total_additions"]
        )

        total_deletions = int(
            activity_row["total_deletions"]
        )

        records.append({
            "file_path": file_path,

            # Complete repository activity
            "total_commits": total_commits,

            # Human knowledge holders
            "contributor_count": contributor_count,
            "total_contributions": total_human_contributions,

            "top_contributor_share": round(
                top_contributor_share,
                4
            ),

            "contribution_entropy": round(
                entropy,
                4
            ),

            "bus_factor": bus_factor,

            "bus_risk": bus_risk,

            # Complete file change activity
            "total_additions": total_additions,
            "total_deletions": total_deletions
        })

    result = pd.DataFrame(records)

    # ---------------------------------------------------------------
    # Risk categories
    # ---------------------------------------------------------------

    def risk_category(score):

        if score < 0.30:
            return "Low"

        elif score < 0.60:
            return "Medium"

        elif score < 0.80:
            return "High"

        else:
            return "Critical"

    result["bus_risk_level"] = result["bus_risk"].apply(
        risk_category
    )

    # ---------------------------------------------------------------
    # Final numerical safety checks
    # ---------------------------------------------------------------

    result["top_contributor_share"] = (
        result["top_contributor_share"]
        .clip(0.0, 1.0)
    )

    result["bus_risk"] = (
        result["bus_risk"]
        .clip(0.0, 1.0)
    )

    # ---------------------------------------------------------------
    # Sort highest-risk files first
    # ---------------------------------------------------------------

    result = result.sort_values(
        by="bus_risk",
        ascending=False
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------------
    # Output summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUS FACTOR ANALYSIS COMPLETED")
    print("=" * 70)

    print(f"\nFiles analysed       : {len(result)}")

    print(
        f"Average contributors : "
        f"{result['contributor_count'].mean():.2f}"
    )

    print(
        f"Average bus factor  : "
        f"{result['bus_factor'].mean():.2f}"
    )

    print(
        f"Average top share   : "
        f"{result['top_contributor_share'].mean():.4f}"
    )

    print(
        f"Average bus risk    : "
        f"{result['bus_risk'].mean():.4f}"
    )

    print("\nRISK LEVEL DISTRIBUTION")
    print("-" * 70)
    print(
        result["bus_risk_level"]
        .value_counts()
    )

    print("\nHIGHEST-RISK FILES")
    print("-" * 70)

    print(
        result[
            [
                "file_path",
                "contributor_count",
                "top_contributor_share",
                "contribution_entropy",
                "bus_factor",
                "bus_risk",
                "bus_risk_level"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nOUTPUT COLUMNS")
    print("-" * 70)
    print(list(result.columns))

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()