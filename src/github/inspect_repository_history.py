import pandas as pd

INPUT_FILE = "data/raw/github/requests_commit_file_history.csv"


def main():
    print("=" * 70)
    print("CodeGuard AI - GitHub History Inspection")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    print("\nDATASET SHAPE")
    print("-" * 70)
    print(f"Rows       : {len(df)}")
    print(f"Columns    : {len(df.columns)}")
    print(f"Files      : {df['file_path'].nunique()}")
    print(f"Commits    : {df['commit_sha'].nunique()}")
    print(f"Authors    : {df['author'].nunique()}")

    print("\nMISSING VALUES")
    print("-" * 70)
    print(df.isnull().sum())

    print("\nDUPLICATE RECORDS")
    print("-" * 70)
    print(f"Exact duplicate rows : {df.duplicated().sum()}")

    print("\nFILE STATUS")
    print("-" * 70)
    print(df["status"].value_counts())

    print("\nCHANGE STATISTICS")
    print("-" * 70)
    print(df[["additions", "deletions", "changes"]].describe())

    print("\nTOP AUTHORS BY FILE-HISTORY RECORDS")
    print("-" * 70)
    print(df["author"].value_counts().head(15))

    print("\nTOP FILES BY NUMBER OF CONTRIBUTORS")
    print("-" * 70)

    contributors_per_file = (
        df.groupby("file_path")["author"]
        .nunique()
        .sort_values(ascending=False)
    )

    print(contributors_per_file.head(15))

    print("\nTOP FILES BY NUMBER OF COMMITS")
    print("-" * 70)

    commits_per_file = (
        df.groupby("file_path")["commit_sha"]
        .nunique()
        .sort_values(ascending=False)
    )

    print(commits_per_file.head(15))

    print("\nFILES WITH MULTIPLE CONTRIBUTORS")
    print("-" * 70)

    multi_contributor = contributors_per_file[contributors_per_file > 1]

    print(f"Files with >1 contributor : {len(multi_contributor)}")
    print(f"Files with 1 contributor   : {len(contributors_per_file) - len(multi_contributor)}")

    print("\nTOP MULTI-CONTRIBUTOR FILES")
    print("-" * 70)
    print(multi_contributor.head(20))

    print("\nSAMPLE RECORDS")
    print("-" * 70)
    print(
        df[
            [
                "commit_sha",
                "author",
                "commit_date",
                "file_path",
                "status",
                "additions",
                "deletions",
                "changes",
            ]
        ].head(10).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()