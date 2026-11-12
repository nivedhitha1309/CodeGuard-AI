import os
import re
from collections import defaultdict, deque

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

HISTORY_FILE = (
    "data/raw/github/requests_commit_file_history.csv"
)

FINAL_TARGET_FILE = (
    "data/processed/github/requests_final_ml_defect_targets.csv"
)

CLEAN_TARGET_FILE = (
    "data/processed/github/requests_clean_defect_targets.csv"
)

OUTPUT_FILE = (
    "data/processed/github/requests_final_temporal_ml_dataset.csv"
)


# ============================================================
# PRODUCTION PYTHON FILTER
# ============================================================

EXCLUDED_PATH_PATTERNS = [
    r"(^|/)tests?(/|$)",
    r"(^|/)test_[^/]*\.py$",
    r"(^|/)conftest\.py$",
    r"(^|/)docs?(/|$)",
    r"(^|/)examples?(/|$)",
    r"(^|/)scripts?(/|$)",
    r"(^|/)tools?(/|$)",
    r"(^|/)vendor(/|$)",
    r"(^|/)\.github(/|$)",
    r"(^|/)tox\.ini$",
    r"(^|/)setup\.cfg$",
    r"(^|/)setup\.py$",
    r"(^|/)pyproject\.toml$",
    r"(^|/)requirements[^/]*$",
]


def normalize_path(path):
    if pd.isna(path):
        return ""

    return str(path).replace("\\", "/").strip()


def is_production_python(path):

    if pd.isna(path):
        return False

    path = normalize_path(path)

    if not path.endswith(".py"):
        return False

    for pattern in EXCLUDED_PATH_PATTERNS:

        if re.search(
            pattern,
            path,
            flags=re.IGNORECASE,
        ):
            return False

    return True


def safe_datetime(series):

    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
    )


def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)


# ============================================================
# ENTROPY
# ============================================================

def normalized_entropy_from_counts(counts):

    counts = np.asarray(
        counts,
        dtype=float,
    )

    counts = counts[counts > 0]

    if len(counts) <= 1:
        return 0.0

    probabilities = (
        counts / counts.sum()
    )

    entropy = -np.sum(
        probabilities
        * np.log(probabilities)
    )

    return float(
        entropy / np.log(len(counts))
    )


# ============================================================
# TEMPORAL FEATURE BUILDER
# ============================================================

def build_temporal_features(history):

    history = (
        history
        .sort_values(
            [
                "commit_date",
                "commit_sha",
                "file_path",
            ]
        )
        .reset_index(drop=True)
    )

    file_state = defaultdict(
        lambda: {
            "commit_count": 0,
            "additions": [],
            "deletions": [],
            "changes": [],
            "contributors": defaultdict(int),
            "first_date": None,
            "last_date": None,
            "events": deque(),
        }
    )

    output_rows = []

    for _, row in history.iterrows():

        file_path = row["file_path"]
        commit_sha = row["commit_sha"]
        commit_date = row["commit_date"]
        author = str(row["author"])

        additions = float(
            row["additions"]
        )

        deletions = float(
            row["deletions"]
        )

        changes = float(
            row["changes"]
        )

        state = file_state[file_path]

        # ----------------------------------------------------
        # PRIOR COMMIT COUNT
        # ----------------------------------------------------

        prior_commit_count = (
            state["commit_count"]
        )

        # ----------------------------------------------------
        # PRIOR CHANGE STATISTICS
        # ----------------------------------------------------

        if state["changes"]:

            prior_additions = float(
                np.sum(
                    state["additions"]
                )
            )

            prior_deletions = float(
                np.sum(
                    state["deletions"]
                )
            )

            prior_changes = float(
                np.sum(
                    state["changes"]
                )
            )

            mean_prior_changes = float(
                np.mean(
                    state["changes"]
                )
            )

            median_prior_changes = float(
                np.median(
                    state["changes"]
                )
            )

            max_prior_changes = float(
                np.max(
                    state["changes"]
                )
            )

        else:

            prior_additions = 0.0
            prior_deletions = 0.0
            prior_changes = 0.0

            mean_prior_changes = 0.0
            median_prior_changes = 0.0
            max_prior_changes = 0.0

        # ----------------------------------------------------
        # CHURN
        # ----------------------------------------------------

        if prior_commit_count > 0:

            prior_churn_per_commit = (
                prior_changes
                / prior_commit_count
            )

        else:

            prior_churn_per_commit = 0.0

        # ----------------------------------------------------
        # FILE AGE
        # ----------------------------------------------------

        if state["first_date"] is None:

            file_age_days = 0.0

        else:

            file_age_days = max(
                0.0,
                (
                    commit_date
                    - state["first_date"]
                ).total_seconds()
                / 86400.0,
            )

        # ----------------------------------------------------
        # DAYS SINCE LAST CHANGE
        # ----------------------------------------------------

        if state["last_date"] is None:

            days_since_last_change = 0.0

        else:

            days_since_last_change = max(
                0.0,
                (
                    commit_date
                    - state["last_date"]
                ).total_seconds()
                / 86400.0,
            )

        # ----------------------------------------------------
        # CONTRIBUTOR FEATURES
        # ----------------------------------------------------

        contributor_counts = list(
            state["contributors"].values()
        )

        prior_contributor_count = len(
            contributor_counts
        )

        if prior_commit_count > 0:

            prior_top_contributor_share = (
                max(contributor_counts)
                / prior_commit_count
            )

        else:

            prior_top_contributor_share = 0.0

        prior_contribution_entropy = (
            normalized_entropy_from_counts(
                contributor_counts
            )
        )

        # ----------------------------------------------------
        # BUS FACTOR
        # ----------------------------------------------------

        if contributor_counts:

            sorted_counts = sorted(
                contributor_counts,
                reverse=True,
            )

            cumulative = 0
            prior_bus_factor = 0

            threshold = (
                0.50
                * prior_commit_count
            )

            for count in sorted_counts:

                cumulative += count
                prior_bus_factor += 1

                if cumulative >= threshold:
                    break

        else:

            prior_bus_factor = 0

        # ----------------------------------------------------
        # ROLLING WINDOWS
        # ----------------------------------------------------

        rolling = {}

        for window_days in [
            30,
            90,
            180,
        ]:

            cutoff = (
                commit_date
                - pd.Timedelta(
                    days=window_days
                )
            )

            while (
                state["events"]
                and state["events"][0][0]
                < cutoff
            ):

                state["events"].popleft()

            events = list(
                state["events"]
            )

            rolling[window_days] = {
                "commits": len(events),

                "changes": sum(
                    event[3]
                    for event in events
                ),

                "additions": sum(
                    event[1]
                    for event in events
                ),

                "deletions": sum(
                    event[2]
                    for event in events
                ),
            }

        # ----------------------------------------------------
        # CREATE FEATURE ROW
        #
        # IMPORTANT:
        # Current event values are NOT included.
        # ----------------------------------------------------

        output_rows.append(
            {
                "file_path": file_path,
                "commit_sha": commit_sha,
                "commit_date": commit_date,
                "author": author,

                "prior_commit_count":
                    prior_commit_count,

                "prior_additions":
                    prior_additions,

                "prior_deletions":
                    prior_deletions,

                "prior_changes":
                    prior_changes,

                "mean_prior_changes":
                    mean_prior_changes,

                "median_prior_changes":
                    median_prior_changes,

                "max_prior_changes":
                    max_prior_changes,

                "prior_churn_per_commit":
                    prior_churn_per_commit,

                "file_age_days":
                    file_age_days,

                "days_since_last_change":
                    days_since_last_change,

                "prior_contributor_count":
                    prior_contributor_count,

                "prior_top_contributor_share":
                    prior_top_contributor_share,

                "prior_contribution_entropy":
                    prior_contribution_entropy,

                "prior_bus_factor":
                    prior_bus_factor,

                "rolling_30d_commits":
                    rolling[30]["commits"],

                "rolling_30d_changes":
                    rolling[30]["changes"],

                "rolling_30d_additions":
                    rolling[30]["additions"],

                "rolling_30d_deletions":
                    rolling[30]["deletions"],

                "rolling_90d_commits":
                    rolling[90]["commits"],

                "rolling_90d_changes":
                    rolling[90]["changes"],

                "rolling_90d_additions":
                    rolling[90]["additions"],

                "rolling_90d_deletions":
                    rolling[90]["deletions"],

                "rolling_180d_commits":
                    rolling[180]["commits"],

                "rolling_180d_changes":
                    rolling[180]["changes"],

                "rolling_180d_additions":
                    rolling[180]["additions"],

                "rolling_180d_deletions":
                    rolling[180]["deletions"],
            }
        )

        # ----------------------------------------------------
        # UPDATE STATE AFTER FEATURE CALCULATION
        # ----------------------------------------------------

        state["commit_count"] += 1

        state["additions"].append(
            additions
        )

        state["deletions"].append(
            deletions
        )

        state["changes"].append(
            changes
        )

        state["contributors"][author] += 1

        if state["first_date"] is None:

            state["first_date"] = commit_date

        state["last_date"] = commit_date

        state["events"].append(
            (
                commit_date,
                additions,
                deletions,
                changes,
            )
        )

    return pd.DataFrame(
        output_rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CODEGUARD AI - FINAL TEMPORAL ML DATASET"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD FILES
    # --------------------------------------------------------

    history = pd.read_csv(
        HISTORY_FILE
    )

    final_targets = pd.read_csv(
        FINAL_TARGET_FILE
    )

    clean_targets = pd.read_csv(
        CLEAN_TARGET_FILE
    )

    print(
        f"\nHistory rows: "
        f"{len(history):,}"
    )

    print(
        f"Final target rows: "
        f"{len(final_targets):,}"
    )

    print(
        f"Clean target rows: "
        f"{len(clean_targets):,}"
    )

    # --------------------------------------------------------
    # NORMALIZE HISTORY
    # --------------------------------------------------------

    history["file_path"] = (
        history["file_path"]
        .map(normalize_path)
    )

    history["commit_date"] = (
        safe_datetime(
            history["commit_date"]
        )
    )

    history["additions"] = (
        safe_numeric(
            history["additions"]
        )
    )

    history["deletions"] = (
        safe_numeric(
            history["deletions"]
        )
    )

    history["changes"] = (
        safe_numeric(
            history["changes"]
        )
    )

    history = history.dropna(
        subset=[
            "commit_sha",
            "commit_date",
            "file_path",
            "author",
        ]
    ).copy()

    # --------------------------------------------------------
    # NORMALIZE TARGETS
    # --------------------------------------------------------

    for df in [
        final_targets,
        clean_targets,
    ]:

        df["file_path"] = (
            df["file_path"]
            .map(normalize_path)
        )

        df["commit_date"] = (
            safe_datetime(
                df["commit_date"]
            )
        )

    # --------------------------------------------------------
    # PRODUCTION PYTHON HISTORY
    # --------------------------------------------------------

    history["production_python"] = (
        history["file_path"]
        .map(is_production_python)
    )

    production_history = (
        history[
            history["production_python"]
        ]
        .copy()
    )

    print(
        f"\nProduction Python rows: "
        f"{len(production_history):,}"
    )

    print(
        f"Production Python files: "
        f"{production_history['file_path'].nunique():,}"
    )

    print(
        f"Production Python commits: "
        f"{production_history['commit_sha'].nunique():,}"
    )

    # --------------------------------------------------------
    # PRIMARY POSITIVE EVENTS
    # --------------------------------------------------------

    primary_targets = (
        final_targets[
            final_targets["ml_defect_target"] == 1
        ]
        .copy()
    )

    primary_keys = set(
        primary_targets[
            ["commit_sha", "file_path"]
        ]
        .astype(str)
        .agg(
            "||".join,
            axis=1,
        )
    )

    positive_commits = set(
        primary_targets[
            "commit_sha"
        ]
    )

    print(
        f"\nPrimary positive rows: "
        f"{len(primary_targets):,}"
    )

    print(
        f"Primary positive commits: "
        f"{len(positive_commits):,}"
    )

    print(
        f"Primary positive files: "
        f"{primary_targets['file_path'].nunique():,}"
    )

    # --------------------------------------------------------
    # CLEAN TARGET LABEL MAP
    # --------------------------------------------------------

    clean_targets = (
        clean_targets
        .drop_duplicates(
            subset=[
                "commit_sha",
                "file_path",
            ],
            keep="first",
        )
        .copy()
    )

    clean_label_map = (
        clean_targets
        .set_index(
            [
                "commit_sha",
                "file_path",
            ]
        )[
            "commit_refined_defect_label"
        ]
        .to_dict()
    )

    # --------------------------------------------------------
    # CREATE EVENT KEYS
    # --------------------------------------------------------

    production_history["event_key"] = (
        production_history[
            "commit_sha"
        ].astype(str)
        + "||"
        + production_history[
            "file_path"
        ].astype(str)
    )

    production_history["is_primary_positive"] = (
        production_history["event_key"]
        .isin(primary_keys)
    )

    production_history["is_positive_commit"] = (
        production_history[
            "commit_sha"
        ].isin(positive_commits)
    )

    # --------------------------------------------------------
    # LOOK UP CLEAN LABEL
    # --------------------------------------------------------

    def lookup_clean_label(row):

        return clean_label_map.get(
            (
                row["commit_sha"],
                row["file_path"],
            ),
            None,
        )

    production_history[
        "clean_refined_label"
    ] = production_history.apply(
        lookup_clean_label,
        axis=1,
    )

    # --------------------------------------------------------
    # POSITIVE EVENTS
    # --------------------------------------------------------

    positive_events = (
        production_history[
            production_history[
                "is_primary_positive"
            ]
        ]
        .copy()
    )

    positive_events[
        "ml_defect_target"
    ] = 1

    # --------------------------------------------------------
    # NEGATIVE EVENTS
    #
    # IMPORTANT:
    #
    # A negative must be:
    #
    #   - production Python
    #   - non_bug_fix according to the
    #     broader clean target set
    #   - not a primary positive
    #   - not from a commit containing
    #     a primary positive
    #
    # This avoids assigning negative labels to
    # other files touched by a defect commit.
    # --------------------------------------------------------

    negative_events = (
        production_history[
            (~production_history[
                "is_primary_positive"
            ])
            &
            (~production_history[
                "is_positive_commit"
            ])
            &
            (
                production_history[
                    "clean_refined_label"
                ]
                == "non_bug_fix"
            )
        ]
        .copy()
    )

    negative_events[
        "ml_defect_target"
    ] = 0

    print(
        f"\nPositive events: "
        f"{len(positive_events):,}"
    )

    print(
        f"Negative events: "
        f"{len(negative_events):,}"
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    selected_events = pd.concat(
        [
            positive_events,
            negative_events,
        ],
        ignore_index=True,
    )

    selected_events = (
        selected_events
        .sort_values(
            [
                "commit_date",
                "commit_sha",
                "file_path",
            ]
        )
        .drop_duplicates(
            subset=[
                "commit_sha",
                "file_path",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        f"Selected dataset events: "
        f"{len(selected_events):,}"
    )

    # --------------------------------------------------------
    # BUILD FEATURES FROM COMPLETE HISTORY
    # --------------------------------------------------------

    print(
        "\nBuilding strictly prior-history features..."
    )

    temporal_features = (
        build_temporal_features(
            production_history[
                [
                    "file_path",
                    "commit_sha",
                    "commit_date",
                    "author",
                    "additions",
                    "deletions",
                    "changes",
                ]
            ].copy()
        )
    )

    # --------------------------------------------------------
    # SELECT LABEL COLUMNS
    # --------------------------------------------------------

    selected_events = (
        selected_events[
            [
                "file_path",
                "commit_sha",
                "commit_date",
                "author",
                "ml_defect_target",
                "clean_refined_label",
            ]
        ]
        .copy()
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    dataset = selected_events.merge(
        temporal_features,
        on=[
            "file_path",
            "commit_sha",
            "commit_date",
            "author",
        ],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    dataset = (
        dataset
        .sort_values(
            [
                "commit_date",
                "commit_sha",
                "file_path",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # TEMPORAL COMMIT SEQUENCE
    # --------------------------------------------------------

    commit_order = (
        dataset[
            [
                "commit_sha",
                "commit_date",
            ]
        ]
        .drop_duplicates(
            subset=[
                "commit_sha",
            ]
        )
        .sort_values(
            [
                "commit_date",
                "commit_sha",
            ]
        )
        .reset_index(drop=True)
    )

    commit_order[
        "temporal_commit_sequence"
    ] = np.arange(
        len(commit_order)
    )

    dataset = dataset.merge(
        commit_order,
        on=[
            "commit_sha",
            "commit_date",
        ],
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    feature_columns = [
        "prior_commit_count",
        "prior_additions",
        "prior_deletions",
        "prior_changes",
        "mean_prior_changes",
        "median_prior_changes",
        "max_prior_changes",
        "prior_churn_per_commit",
        "file_age_days",
        "days_since_last_change",
        "prior_contributor_count",
        "prior_top_contributor_share",
        "prior_contribution_entropy",
        "prior_bus_factor",
        "rolling_30d_commits",
        "rolling_30d_changes",
        "rolling_30d_additions",
        "rolling_30d_deletions",
        "rolling_90d_commits",
        "rolling_90d_changes",
        "rolling_90d_additions",
        "rolling_90d_deletions",
        "rolling_180d_commits",
        "rolling_180d_changes",
        "rolling_180d_additions",
        "rolling_180d_deletions",
    ]

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    missing = (
        dataset[
            feature_columns
        ]
        .isna()
        .sum()
    )

    missing = missing[
        missing > 0
    ]

    print(
        "\nMissing feature values:"
    )

    if len(missing) == 0:
        print("None")
    else:
        print(missing)

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicate_count = (
        dataset
        .duplicated(
            subset=[
                "commit_sha",
                "file_path",
            ]
        )
        .sum()
    )

    print(
        f"\nDuplicate commit/file events: "
        f"{duplicate_count}"
    )

    # --------------------------------------------------------
    # CHRONOLOGICAL CHECK
    # --------------------------------------------------------

    chronological = (
        dataset[
            "commit_date"
        ]
        .is_monotonic_increasing
    )

    print(
        "Chronological order: "
        f"{'PASS' if chronological else 'FAIL'}"
    )

    # --------------------------------------------------------
    # LEAKAGE CHECK
    #
    # Verify that the prior commit count for each selected
    # event equals the number of earlier history events for
    # that file.
    # --------------------------------------------------------

    history_sorted = (
        production_history
        .sort_values(
            [
                "commit_date",
                "commit_sha",
                "file_path",
            ]
        )
    )

    expected_counts = defaultdict(int)

    leakage_failures = 0

    for _, row in history_sorted.iterrows():

        file_path = row["file_path"]
        commit_sha = row["commit_sha"]

        matching = dataset[
            (dataset[
                "commit_sha"
            ] == commit_sha)
            &
            (dataset[
                "file_path"
            ] == file_path)
        ]

        if not matching.empty:

            expected = (
                expected_counts[
                    file_path
                ]
            )

            actual = int(
                matching.iloc[0][
                    "prior_commit_count"
                ]
            )

            if actual != expected:
                leakage_failures += 1

        expected_counts[
            file_path
        ] += 1

    print(
        f"Temporal leakage failures: "
        f"{leakage_failures}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "FINAL TEMPORAL DATASET SUMMARY"
    )
    print("=" * 70)

    print(
        f"Rows: "
        f"{len(dataset):,}"
    )

    print(
        f"Commits: "
        f"{dataset['commit_sha'].nunique():,}"
    )

    print(
        f"Files: "
        f"{dataset['file_path'].nunique():,}"
    )

    positives = int(
        (
            dataset[
                "ml_defect_target"
            ] == 1
        ).sum()
    )

    negatives = int(
        (
            dataset[
                "ml_defect_target"
            ] == 0
        ).sum()
    )

    print(
        f"Positive rows: "
        f"{positives:,}"
    )

    print(
        f"Negative rows: "
        f"{negatives:,}"
    )

    if len(dataset) > 0:

        positive_rate = (
            dataset[
                "ml_defect_target"
            ].mean()
        )

    else:

        positive_rate = 0.0

    print(
        f"Positive rate: "
        f"{positive_rate:.4f}"
    )

    print(
        f"Date range: "
        f"{dataset['commit_date'].min()} "
        f"to "
        f"{dataset['commit_date'].max()}"
    )

    print(
        f"Temporal commit range: "
        f"{dataset['temporal_commit_sequence'].min()} "
        f"to "
        f"{dataset['temporal_commit_sequence'].max()}"
    )

    print(
        "\nPositive label breakdown:"
    )

    print(
        dataset[
            dataset[
                "ml_defect_target"
            ] == 1
        ][
            "clean_refined_label"
        ].value_counts()
    )

    print(
        "\nNegative label breakdown:"
    )

    print(
        dataset[
            dataset[
                "ml_defect_target"
            ] == 0
        ][
            "clean_refined_label"
        ].value_counts()
    )

    print(
        f"\nFeature count: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True,
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved to: "
        f"{os.path.abspath(OUTPUT_FILE)}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()