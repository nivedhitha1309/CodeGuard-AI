import os
import ast
import math
import requests
import pandas as pd

from dotenv import load_dotenv
from radon.complexity import cc_visit
from radon.raw import analyze


# ============================================================
# CodeGuard AI - Code Risk Analysis
# Current GitHub Snapshot
# ============================================================

INPUT_FILE = "data/processed/fusion/requests_bus_satd_merged.csv"
OUTPUT_FILE = "data/processed/code_risk/requests_code_risk.csv"

REPO_OWNER = "psf"
REPO_NAME = "requests"
DEFAULT_BRANCH = "main"

API_BASE = "https://api.github.com"


# ============================================================
# Load GitHub token
# ============================================================

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise RuntimeError(
        "GITHUB_TOKEN was not found in the .env file."
    )


HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


# ============================================================
# Helper functions
# ============================================================

def normalize_path(path):
    """Normalize repository paths."""
    return str(path).replace("\\", "/").lstrip("./")


def is_python_file(file_path):
    """Return True for Python source files."""
    return file_path.lower().endswith((".py", ".pyi"))


def safe_get(url, timeout=30, retries=3):
    """GET request with simple retry handling."""

    last_error = None

    for attempt in range(1, retries + 1):

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=timeout,
            )

            if response.status_code == 200:
                return response

            if response.status_code == 404:
                return None

            if response.status_code == 403:
                raise RuntimeError(
                    "GitHub API returned 403. "
                    "Check your GITHUB_TOKEN or API rate limit."
                )

            response.raise_for_status()

        except requests.RequestException as exc:

            last_error = exc

            if attempt < retries:
                continue

    raise RuntimeError(
        f"GitHub request failed: {url}\n"
        f"Last error: {last_error}"
    )


def get_current_tree():
    """
    Retrieve the complete current repository tree.
    """

    print("\nRetrieving current GitHub repository...")

    branch_url = (
        f"{API_BASE}/repos/"
        f"{REPO_OWNER}/{REPO_NAME}"
    )

    response = safe_get(branch_url)

    if response is None:
        raise RuntimeError(
            "Could not access GitHub repository."
        )

    repo_data = response.json()

    branch = repo_data.get(
        "default_branch",
        DEFAULT_BRANCH,
    )

    print(
        f"Current default branch : {branch}"
    )

    commit_url = (
        f"{API_BASE}/repos/"
        f"{REPO_OWNER}/{REPO_NAME}"
        f"/git/refs/heads/{branch}"
    )

    response = safe_get(commit_url)

    if response is None:
        raise RuntimeError(
            "Could not retrieve branch information."
        )

    commit_data = response.json()

    commit_sha = commit_data["object"]["sha"]

    tree_url = (
        f"{API_BASE}/repos/"
        f"{REPO_OWNER}/{REPO_NAME}"
        f"/git/trees/{commit_sha}"
        f"?recursive=1"
    )

    print(
        "Retrieving current file tree..."
    )

    response = safe_get(tree_url)

    if response is None:
        raise RuntimeError(
            "Could not retrieve repository tree."
        )

    tree_data = response.json()

    if tree_data.get("truncated"):
        print(
            "WARNING: GitHub returned a truncated tree."
        )

    files = set()

    for item in tree_data.get("tree", []):

        if item.get("type") == "blob":

            files.add(
                normalize_path(
                    item.get("path", "")
                )
            )

    print(
        f"Current repository files : {len(files)}"
    )

    return branch, files


def download_file(file_path):
    """
    Download one current file from GitHub.
    """

    url = (
        f"{API_BASE}/repos/"
        f"{REPO_OWNER}/{REPO_NAME}"
        f"/contents/{file_path}"
    )

    response = safe_get(url)

    if response is None:
        return None

    data = response.json()

    if data.get("type") != "file":
        return None

    content = data.get("content")

    if not content:
        return None

    import base64

    try:

        decoded = base64.b64decode(
            content
        )

        return decoded.decode(
            "utf-8",
            errors="ignore",
        )

    except Exception:
        return None


def count_ast_features(source):
    """
    Count functions, classes and methods.
    """

    try:
        tree = ast.parse(source)

    except SyntaxError:
        return 0, 0, 0

    function_count = 0
    class_count = 0
    method_count = 0

    class_stack = 0

    class Visitor(ast.NodeVisitor):

        def __init__(self):
            self.functions = 0
            self.classes = 0
            self.methods = 0
            self.class_depth = 0

        def visit_ClassDef(self, node):

            self.classes += 1
            self.class_depth += 1

            self.generic_visit(node)

            self.class_depth -= 1

        def visit_FunctionDef(self, node):

            self.functions += 1

            if self.class_depth > 0:
                self.methods += 1

            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):

            self.functions += 1

            if self.class_depth > 0:
                self.methods += 1

            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)

    return (
        visitor.functions,
        visitor.classes,
        visitor.methods,
    )


def calculate_complexity(source):

    try:

        blocks = cc_visit(source)

        complexities = [
            block.complexity
            for block in blocks
        ]

        if not complexities:

            return {
                "complexity_total": 0,
                "complexity_average": 0.0,
                "complexity_max": 0,
                "complexity_blocks": 0,
            }

        return {
            "complexity_total": sum(complexities),
            "complexity_average": (
                sum(complexities)
                / len(complexities)
            ),
            "complexity_max": max(
                complexities
            ),
            "complexity_blocks": len(
                complexities
            ),
        }

    except Exception:

        return {
            "complexity_total": 0,
            "complexity_average": 0.0,
            "complexity_max": 0,
            "complexity_blocks": 0,
        }


def calculate_raw_metrics(source):

    try:

        metrics = analyze(source)

        return {
            "loc": metrics.loc,
            "lloc": metrics.lloc,
            "sloc": metrics.sloc,
            "comments": metrics.comments,
            "multi": metrics.multi,
            "blank": metrics.blank,
        }

    except Exception:

        return {
            "loc": 0,
            "lloc": 0,
            "sloc": 0,
            "comments": 0,
            "multi": 0,
            "blank": 0,
        }


def normalize_series(series):

    series = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)

    minimum = series.min()
    maximum = series.max()

    if maximum <= minimum:

        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    ).clip(0, 1)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("CodeGuard AI - Code Risk Analysis")
    print("=" * 70)

    # --------------------------------------------------------
    # Load merged Bus + SATD dataset
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["file_path"] = (
        df["file_path"]
        .astype(str)
        .map(normalize_path)
    )

    print(
        f"\nHistorical files available : {len(df)}"
    )

    # --------------------------------------------------------
    # Get current GitHub tree
    # --------------------------------------------------------

    branch, current_files = (
        get_current_tree()
    )

    python_historical = [
        path
        for path in df["file_path"]
        if is_python_file(path)
    ]

    current_python_files = [
        path
        for path in python_historical
        if path in current_files
    ]

    missing_python_files = [
        path
        for path in python_historical
        if path not in current_files
    ]

    print(
        f"Historical Python files    : "
        f"{len(python_historical)}"
    )

    print(
        f"Current Python files       : "
        f"{len(current_python_files)}"
    )

    print(
        f"Historical Python files no longer present : "
        f"{len(missing_python_files)}"
    )

    # --------------------------------------------------------
    # Analyze current Python files
    # --------------------------------------------------------

    results = []

    analyzed_files = 0
    failed_files = 0

    for index, row in df.iterrows():

        file_path = row["file_path"]

        result = {
            "file_path": file_path,

            "code_analyzed": False,

            "code_language": (
                "python"
                if is_python_file(file_path)
                else "other"
            ),

            "loc": 0,
            "lloc": 0,
            "sloc": 0,
            "comments": 0,
            "multi": 0,
            "blank": 0,

            "function_count": 0,
            "class_count": 0,
            "method_count": 0,

            "complexity_total": 0,
            "complexity_average": 0.0,
            "complexity_max": 0,
            "complexity_blocks": 0,
        }

        # Only analyze Python files.
        if not is_python_file(file_path):

            results.append(result)
            continue

        # File is historical but no longer exists.
        if file_path not in current_files:

            results.append(result)
            continue

        print(
            f"[{index + 1:03d}/{len(df)}] "
            f"Analyzing {file_path}"
        )

        source = download_file(
            file_path
        )

        if source is None:

            failed_files += 1
            results.append(result)
            continue

        try:

            raw_metrics = (
                calculate_raw_metrics(
                    source
                )
            )

            complexity_metrics = (
                calculate_complexity(
                    source
                )
            )

            (
                function_count,
                class_count,
                method_count,
            ) = count_ast_features(
                source
            )

            result.update(
                raw_metrics
            )

            result.update(
                complexity_metrics
            )

            result[
                "function_count"
            ] = function_count

            result[
                "class_count"
            ] = class_count

            result[
                "method_count"
            ] = method_count

            result[
                "code_analyzed"
            ] = True

            analyzed_files += 1

        except Exception as exc:

            print(
                f"    Analysis failed: {exc}"
            )

            failed_files += 1

        results.append(result)

    metrics_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Normalize risk features
    # --------------------------------------------------------

    metrics_df[
        "loc_score"
    ] = normalize_series(
        metrics_df["loc"]
    )

    metrics_df[
        "complexity_score"
    ] = normalize_series(
        metrics_df[
            "complexity_average"
        ]
    )

    metrics_df[
        "max_complexity_score"
    ] = normalize_series(
        metrics_df[
            "complexity_max"
        ]
    )

    metrics_df[
        "function_score"
    ] = normalize_series(
        metrics_df[
            "function_count"
        ]
    )

    # Complexity relative to file size.
    metrics_df[
        "complexity_density"
    ] = (
        metrics_df[
            "complexity_total"
        ]
        /
        metrics_df[
            "loc"
        ].replace(0, pd.NA)
    )

    metrics_df[
        "complexity_density"
    ] = (
        metrics_df[
            "complexity_density"
        ]
        .fillna(0)
        .clip(lower=0)
    )

    metrics_df[
        "complexity_density_score"
    ] = normalize_series(
        metrics_df[
            "complexity_density"
        ]
    )

    # --------------------------------------------------------
    # Code risk
    #
    # Transparent heuristic v1.
    # This is NOT the final ML risk model.
    # --------------------------------------------------------

    metrics_df[
        "code_risk"
    ] = (
        0.30
        * metrics_df[
            "complexity_score"
        ]

        + 0.25
        * metrics_df[
            "max_complexity_score"
        ]

        + 0.25
        * metrics_df[
            "complexity_density_score"
        ]

        + 0.10
        * metrics_df[
            "function_score"
        ]

        + 0.10
        * metrics_df[
            "loc_score"
        ]
    )

    metrics_df[
        "code_risk"
    ] = (
        metrics_df[
            "code_risk"
        ]
        .clip(0, 1)
        .round(6)
    )

    # --------------------------------------------------------
    # Risk levels
    # --------------------------------------------------------

    def risk_level(score):

        if score >= 0.75:
            return "Critical"

        if score >= 0.50:
            return "High"

        if score >= 0.25:
            return "Medium"

        return "Low"

    metrics_df[
        "code_risk_level"
    ] = (
        metrics_df[
            "code_risk"
        ].map(risk_level)
    )

    metrics_df[
        "code_risk_method"
    ] = (
        "radon_complexity_heuristic_v1"
    )

    # --------------------------------------------------------
    # Merge with Bus + SATD dataset
    # --------------------------------------------------------

    final_df = df.merge(
        metrics_df,
        on="file_path",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True,
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CODE RISK ANALYSIS SUMMARY")
    print("=" * 70)

    print(
        f"Input files            : "
        f"{len(df)}"
    )

    print(
        f"Historical Python files: "
        f"{len(python_historical)}"
    )

    print(
        f"Current Python files   : "
        f"{len(current_python_files)}"
    )

    print(
        f"Successfully analyzed  : "
        f"{analyzed_files}"
    )

    print(
        f"Missing historical     : "
        f"{len(missing_python_files)}"
    )

    print(
        f"Failed downloads       : "
        f"{failed_files}"
    )

    analyzed = metrics_df[
        metrics_df[
            "code_analyzed"
        ] == True
    ]

    if len(analyzed) > 0:

        print(
            f"\nAverage LOC            : "
            f"{analyzed['loc'].mean():.2f}"
        )

        print(
            f"Average complexity     : "
            f"{analyzed['complexity_average'].mean():.4f}"
        )

        print(
            f"Average code risk      : "
            f"{analyzed['code_risk'].mean():.4f}"
        )

        print(
            f"Maximum code risk      : "
            f"{analyzed['code_risk'].max():.4f}"
        )

    print("\nCODE RISK LEVEL DISTRIBUTION")
    print("-" * 70)

    print(
        metrics_df[
            "code_risk_level"
        ].value_counts()
        .to_string()
    )

    print("\nTOP 15 CODE-RISK FILES")
    print("-" * 70)

    display_columns = [
        "file_path",
        "loc",
        "function_count",
        "complexity_average",
        "complexity_max",
        "complexity_density",
        "code_risk",
        "code_risk_level",
    ]

    print(
        metrics_df
        .sort_values(
            "code_risk",
            ascending=False,
        )[display_columns]
        .head(15)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)

    print(
        f"Output saved to: "
        f"{OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()