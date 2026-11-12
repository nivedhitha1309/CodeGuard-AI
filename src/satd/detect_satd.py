import io
import os
import re
import time
import tokenize

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ===============================================================
# Configuration
# ===============================================================

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")

if not TOKEN:
    raise RuntimeError(
        "GITHUB_TOKEN not found. Check your .env file."
    )

OWNER = "psf"
REPO = "requests"

BASE_URL = (
    f"https://api.github.com/repos/{OWNER}/{REPO}"
)

INPUT_FILE = (
    "data/processed/bus_factor/"
    "requests_bus_factor_refined.csv"
)

OUTPUT_FILE = (
    "data/raw/satd/"
    "requests_satd_occurrences.csv"
)


# ===============================================================
# HTTP session with retry handling
# ===============================================================

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

session = requests.Session()

retry_strategy = Retry(
    total=5,
    connect=5,
    read=5,
    status=5,
    backoff_factor=1,
    status_forcelist=[
        429,
        500,
        502,
        503,
        504,
    ],
    allowed_methods=frozenset(["GET"]),
    respect_retry_after_header=True,
)

adapter = HTTPAdapter(
    max_retries=retry_strategy
)

session.mount("https://", adapter)
session.mount("http://", adapter)


def safe_get(url, params=None):
    """
    Perform a GET request with retry and GitHub
    rate-limit handling.

    A 404 is returned to the caller so that permanent
    missing-file conditions can be handled explicitly.
    """

    for attempt in range(5):

        try:

            response = session.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=30,
            )

            # ---------------------------------------------------
            # Permanent missing resource
            # ---------------------------------------------------

            if response.status_code == 404:
                return None

            # ---------------------------------------------------
            # GitHub rate limit
            # ---------------------------------------------------

            if response.status_code == 403:

                remaining = response.headers.get(
                    "X-RateLimit-Remaining"
                )

                if remaining == "0":

                    reset_time = response.headers.get(
                        "X-RateLimit-Reset"
                    )

                    if reset_time:

                        wait_time = max(
                            int(reset_time)
                            - int(time.time()),
                            1,
                        )

                    else:
                        wait_time = 10

                    print(
                        "GitHub rate limit reached. "
                        f"Waiting {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

            response.raise_for_status()

            return response

        except requests.RequestException as exc:

            if attempt == 4:
                raise

            wait_time = 2 ** attempt

            print(
                f"Request failed: {exc}"
            )

            print(
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

    return None


# ===============================================================
# SATD patterns
# ===============================================================

# More specific patterns are intentionally placed before
# generic patterns such as DEBT.
#
# This prevents:
#
#     TECHNICAL DEBT
#
# from being counted twice as:
#
#     TECHNICAL_DEBT
#     DEBT
#
SATD_PATTERNS = {
    "TECHNICAL_DEBT": re.compile(
        r"\bTECHNICAL[\s_-]+DEBT\b",
        re.IGNORECASE,
    ),

    "TECHDEBT": re.compile(
        r"\bTECH[\s_-]?DEBT\b",
        re.IGNORECASE,
    ),

    "WORKAROUND": re.compile(
        r"\bWORKAROUND\b",
        re.IGNORECASE,
    ),

    "FIXME": re.compile(
        r"\bFIXME\b",
        re.IGNORECASE,
    ),

    "TODO": re.compile(
        r"\bTODO\b",
        re.IGNORECASE,
    ),

    "HACK": re.compile(
        r"\bHACK\b",
        re.IGNORECASE,
    ),

    "XXX": re.compile(
        r"\bXXX\b",
        re.IGNORECASE,
    ),

    "TEMPORARY": re.compile(
        r"\bTEMP(?:ORARY)?\b",
        re.IGNORECASE,
    ),

    "DEBT": re.compile(
        r"\bDEBT\b",
        re.IGNORECASE,
    ),

    "REFACTOR": re.compile(
        r"\bREFACTOR(?:ING)?\b",
        re.IGNORECASE,
    ),

    "REVISIT": re.compile(
        r"\bREVISIT\b",
        re.IGNORECASE,
    ),

    "CLEANUP": re.compile(
        r"\bCLEAN[\s_-]?UP\b",
        re.IGNORECASE,
    ),
}


# ===============================================================
# File extensions to inspect
# ===============================================================

TEXT_EXTENSIONS = {
    ".py",
    ".pyi",
    ".txt",
    ".md",
    ".rst",
    ".toml",
    ".ini",
    ".cfg",
    ".yml",
    ".yaml",
    ".json",
    ".xml",
    ".html",
    ".css",
    ".js",
    ".sh",
    ".bat",
    ".make",
}


def is_text_file(file_path):
    """
    Determine whether a repository file should be scanned.
    """

    path = (
        str(file_path)
        .replace("\\", "/")
        .lower()
    )

    filename = path.split("/")[-1]

    _, extension = os.path.splitext(filename)

    if extension in TEXT_EXTENSIONS:
        return True

    # Files without extensions that may contain comments.
    if filename in {
        "makefile",
        "dockerfile",
        ".gitignore",
        ".gitattributes",
    }:
        return True

    return False


# ===============================================================
# GitHub repository metadata
# ===============================================================

def get_default_branch():
    """
    Retrieve the repository's current default branch.
    """

    response = safe_get(BASE_URL)

    if response is None:
        raise RuntimeError(
            "Unable to retrieve repository metadata."
        )

    data = response.json()

    default_branch = data.get("default_branch")

    if not default_branch:
        raise RuntimeError(
            "GitHub repository default branch not found."
        )

    return default_branch


# ===============================================================
# Current repository tree
# ===============================================================

def get_current_repository_files(default_branch):
    """
    Retrieve the current repository file tree.

    This is used to distinguish historical file paths from
    files that are still present in the current repository.
    """

    url = (
        f"{BASE_URL}/git/trees/"
        f"{default_branch}"
    )

    response = safe_get(
        url,
        params={
            "recursive": "1",
        },
    )

    if response is None:
        raise RuntimeError(
            "Unable to retrieve the current "
            "GitHub repository tree."
        )

    data = response.json()

    if data.get("truncated"):
        print(
            "WARNING: GitHub returned a truncated "
            "repository tree."
        )

    tree = data.get("tree", [])

    current_files = set()

    for item in tree:

        if item.get("type") != "blob":
            continue

        path = item.get("path")

        if path:
            current_files.add(
                path.replace("\\", "/")
            )

    return current_files


# ===============================================================
# GitHub file content
# ===============================================================

def get_file_content(file_path):
    """
    Retrieve the current file contents from GitHub
    using the Contents API.

    Returns:
        str  -> decoded file content
        None -> unavailable or non-file resource
    """

    url = (
        f"{BASE_URL}/contents/"
        f"{file_path}"
    )

    response = safe_get(url)

    if response is None:
        return None

    data = response.json()

    if data.get("type") != "file":
        return None

    encoding = data.get("encoding")
    content = data.get("content")

    if encoding != "base64" or content is None:
        return None

    try:

        import base64

        decoded = base64.b64decode(
            content
        )

        return decoded.decode(
            "utf-8",
            errors="replace",
        )

    except Exception:

        return None


# ===============================================================
# Python comment extraction
# ===============================================================

def extract_python_comments(content):
    """
    Extract only real Python comments using Python's tokenizer.

    This prevents SATD markers inside strings such as:

        message = "TODO: this is not a comment"

    from being detected as SATD.
    """

    comments = []

    if not content:
        return comments

    try:

        tokens = tokenize.generate_tokens(
            io.StringIO(content).readline
        )

        for token in tokens:

            token_type = token.type
            token_string = token.string
            start_line = token.start[0]

            if token_type == tokenize.COMMENT:

                comments.append(
                    (
                        start_line,
                        token_string.strip(),
                    )
                )

    except (
        tokenize.TokenError,
        IndentationError,
        SyntaxError,
    ):

        # If a Python file cannot be tokenized completely,
        # return an empty list rather than scanning potentially
        # misleading string content as comments.
        return []

    return comments


# ===============================================================
# Generic comment extraction
# ===============================================================

def extract_generic_comments(content):
    """
    Extract likely comments from non-Python text files.

    This is intentionally lightweight because different
    languages use different comment grammars.
    """

    comments = []

    if not content:
        return comments

    lines = content.splitlines()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):

        stripped = line.strip()

        if not stripped:
            continue

        # -------------------------------------------------------
        # Full-line comments
        # -------------------------------------------------------

        full_line_markers = (
            "#",
            "//",
            "/*",
            "*",
            "<!--",
            ";",
        )

        if stripped.startswith(
            full_line_markers
        ):

            comments.append(
                (
                    line_number,
                    stripped,
                )
            )

            continue

        # -------------------------------------------------------
        # Inline comments
        # -------------------------------------------------------

        # Avoid treating URL fragments such as:
        #
        #     https://example.com
        #
        # as comments.

        for marker in (
            "#",
            "//",
        ):

            position = line.find(marker)

            if position < 0:
                continue

            if marker == "//":

                # Ignore URLs such as http:// and https://
                prefix = line[:position].lower()

                if (
                    prefix.endswith("http:")
                    or prefix.endswith("https:")
                ):
                    continue

            comment = line[position:].strip()

            if comment:

                comments.append(
                    (
                        line_number,
                        comment,
                    )
                )

            break

    return comments


# ===============================================================
# Comment extraction dispatcher
# ===============================================================

def extract_comments(file_path, content):
    """
    Select the appropriate comment extraction strategy.
    """

    normalized_path = (
        str(file_path)
        .replace("\\", "/")
        .lower()
    )

    if normalized_path.endswith(
        (".py", ".pyi")
    ):

        return extract_python_comments(
            content
        )

    return extract_generic_comments(
        content
    )


# ===============================================================
# False-positive filtering
# ===============================================================

def is_known_false_positive(
    file_path,
    comment_text,
):
    """
    Identify known non-SATD matches.

    The filter is content-based rather than dependent on a
    fixed line number, making it more stable when source files
    change.
    """

    normalized_path = (
        str(file_path)
        .replace("\\", "/")
        .lower()
    )

    normalized_comment = (
        str(comment_text)
        .strip()
        .lower()
    )

    # -----------------------------------------------------------
    # Sphinx documentation configuration
    # -----------------------------------------------------------

    # In docs/conf.py, "todolist" refers to a documentation
    # configuration option and not to technical debt.
    if (
        normalized_path == "docs/conf.py"
        and "todolist" in normalized_comment
        and "produce output" in normalized_comment
    ):
        return True

    return False


# ===============================================================
# SATD match extraction
# ===============================================================

def find_satd_matches(comment):
    """
    Find SATD markers in a comment.

    Matches are returned in source order and overlapping
    matches are removed.

    Example:

        TECHNICAL DEBT

    is counted once as TECHNICAL_DEBT rather than once as
    TECHNICAL_DEBT and again as DEBT.
    """

    matches = []

    if not comment:
        return matches

    for satd_type, pattern in SATD_PATTERNS.items():

        for match in pattern.finditer(comment):

            matches.append(
                {
                    "satd_type": satd_type,
                    "start": match.start(),
                    "end": match.end(),
                    "matched_text": match.group(0),
                }
            )

    # -----------------------------------------------------------
    # Sort:
    #
    # 1. earliest position
    # 2. longest match first
    #
    # This gives specific patterns priority over generic ones.
    # -----------------------------------------------------------

    matches.sort(
        key=lambda item: (
            item["start"],
            -(item["end"] - item["start"]),
        )
    )

    accepted = []

    occupied_ranges = []

    for match in matches:

        overlaps = False

        for start, end in occupied_ranges:

            if (
                match["start"] < end
                and match["end"] > start
            ):

                overlaps = True
                break

        if overlaps:
            continue

        accepted.append(match)

        occupied_ranges.append(
            (
                match["start"],
                match["end"],
            )
        )

    # Restore source order.
    accepted.sort(
        key=lambda item: item["start"]
    )

    return accepted


# ===============================================================
# SATD detection
# ===============================================================

def detect_satd_in_file(
    file_path,
    content,
):
    """
    Detect SATD candidates in a single file.
    """

    occurrences = []

    if content is None:
        return occurrences

    comments = extract_comments(
        file_path,
        content,
    )

    for line_number, comment in comments:

        if not comment:
            continue

        if is_known_false_positive(
            file_path,
            comment,
        ):
            continue

        matches = find_satd_matches(
            comment
        )

        for match in matches:

            occurrences.append(
                {
                    "file_path": file_path,
                    "line_number": line_number,
                    "satd_type": match[
                        "satd_type"
                    ],
                    "matched_text": match[
                        "matched_text"
                    ],
                    "comment_text": comment.strip(),
                }
            )

    return occurrences


# ===============================================================
# Main
# ===============================================================

def main():

    print("=" * 70)
    print("CodeGuard AI - SATD Detection")
    print("=" * 70)

    # -----------------------------------------------------------
    # Validate input
    # -----------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    if df.empty:

        raise ValueError(
            "Input CSV is empty."
        )

    if "file_path" not in df.columns:

        raise ValueError(
            "file_path column not found."
        )

    # -----------------------------------------------------------
    # Historical file universe
    # -----------------------------------------------------------

    files = (
        df["file_path"]
        .dropna()
        .astype(str)
        .str.replace(
            "\\",
            "/",
            regex=False,
        )
        .drop_duplicates()
        .tolist()
    )

    print(
        "\nHistorical files available : "
        f"{len(files)}"
    )

    # -----------------------------------------------------------
    # Select text files
    # -----------------------------------------------------------

    text_files = [
        file_path
        for file_path in files
        if is_text_file(file_path)
    ]

    print(
        "Historical text files      : "
        f"{len(text_files)}"
    )

    # -----------------------------------------------------------
    # Determine current branch
    # -----------------------------------------------------------

    print(
        "\nRetrieving current GitHub repository..."
    )

    default_branch = get_default_branch()

    print(
        f"Current default branch     : "
        f"{default_branch}"
    )

    # -----------------------------------------------------------
    # Retrieve current repository tree
    # -----------------------------------------------------------

    print(
        "Retrieving current file tree..."
    )

    current_files = (
        get_current_repository_files(
            default_branch
        )
    )

    print(
        "Current repository files   : "
        f"{len(current_files)}"
    )

    # -----------------------------------------------------------
    # Historical vs current file paths
    # -----------------------------------------------------------

    existing_text_files = []
    missing_text_files = []

    current_file_set = {
        path.replace("\\", "/")
        for path in current_files
    }

    for file_path in text_files:

        normalized_path = (
            file_path
            .replace("\\", "/")
        )

        if normalized_path in current_file_set:

            existing_text_files.append(
                normalized_path
            )

        else:

            missing_text_files.append(
                normalized_path
            )

    print(
        "\nCurrent text files to scan : "
        f"{len(existing_text_files)}"
    )

    print(
        "Historical text files no "
        "longer present             : "
        f"{len(missing_text_files)}"
    )

    # -----------------------------------------------------------
    # Scan current files
    # -----------------------------------------------------------

    all_occurrences = []

    skipped_files = []

    total_files = len(
        existing_text_files
    )

    for index, file_path in enumerate(
        existing_text_files,
        start=1,
    ):

        print(
            f"[{index}/{total_files}] "
            f"Scanning: {file_path}"
        )

        try:

            content = get_file_content(
                file_path
            )

            if content is None:

                print(
                    "  SKIPPED: "
                    "Unable to retrieve file content."
                )

                skipped_files.append(
                    {
                        "file_path": file_path,
                        "reason": "content_unavailable",
                    }
                )

                continue

            occurrences = (
                detect_satd_in_file(
                    file_path,
                    content,
                )
            )

            all_occurrences.extend(
                occurrences
            )

            if occurrences:

                print(
                    f"  SATD found: "
                    f"{len(occurrences)}"
                )

        except requests.RequestException as exc:

            print(
                f"  REQUEST ERROR: {exc}"
            )

            skipped_files.append(
                {
                    "file_path": file_path,
                    "reason": "request_error",
                }
            )

        except Exception as exc:

            print(
                f"  ERROR: {exc}"
            )

            skipped_files.append(
                {
                    "file_path": file_path,
                    "reason": "unexpected_error",
                }
            )

    # -----------------------------------------------------------
    # Create occurrence output
    # -----------------------------------------------------------

    output_columns = [
        "file_path",
        "line_number",
        "satd_type",
        "matched_text",
        "comment_text",
    ]

    result = pd.DataFrame(
        all_occurrences,
        columns=output_columns,
    )

    # -----------------------------------------------------------
    # Defensive deduplication
    # -----------------------------------------------------------

    if not result.empty:

        result = (
            result
            .drop_duplicates(
                subset=[
                    "file_path",
                    "line_number",
                    "satd_type",
                    "matched_text",
                    "comment_text",
                ]
            )
            .sort_values(
                by=[
                    "file_path",
                    "line_number",
                    "satd_type",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # -----------------------------------------------------------
    # Save output
    # -----------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    # -----------------------------------------------------------
    # Summary
    # -----------------------------------------------------------

    print("\n" + "=" * 70)
    print("SATD DETECTION SUMMARY")
    print("=" * 70)

    print(
        f"Historical files available        : "
        f"{len(files)}"
    )

    print(
        f"Historical text files             : "
        f"{len(text_files)}"
    )

    print(
        f"Current repository files           : "
        f"{len(current_files)}"
    )

    print(
        f"Current text files scanned         : "
        f"{len(existing_text_files)}"
    )

    print(
        f"Historical files no longer present : "
        f"{len(missing_text_files)}"
    )

    print(
        f"Files skipped during scan          : "
        f"{len(skipped_files)}"
    )

    print(
        f"SATD occurrences                   : "
        f"{len(result)}"
    )

    if not result.empty:

        print(
            f"Files with SATD                    : "
            f"{result['file_path'].nunique()}"
        )

        print("\nSATD TYPE DISTRIBUTION")
        print("-" * 70)

        print(
            result[
                "satd_type"
            ]
            .value_counts()
            .to_string()
        )

        print("\nTOP FILES BY SATD COUNT")
        print("-" * 70)

        top_files = (
            result[
                "file_path"
            ]
            .value_counts()
            .head(20)
        )

        print(
            top_files.to_string()
        )

        print("\nSATD OCCURRENCES")
        print("-" * 70)

        print(
            result[
                [
                    "file_path",
                    "line_number",
                    "satd_type",
                    "matched_text",
                    "comment_text",
                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "\nNo SATD candidates detected."
        )

    print("\n" + "=" * 70)

    print(
        f"Output saved to: {OUTPUT_FILE}"
    )

    print("=" * 70)


# ===============================================================
# Entry point
# ===============================================================

if __name__ == "__main__":
    main()