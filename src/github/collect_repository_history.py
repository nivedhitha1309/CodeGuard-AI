import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CodeGuard AI - GitHub Repository History Collector
# ============================================================

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")

if not TOKEN:
    raise RuntimeError(
        "GITHUB_TOKEN not found. Check your .env file."
    )


# ------------------------------------------------------------
# Repository configuration
# ------------------------------------------------------------

OWNER = "psf"
REPO = "requests"

BASE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}"

OUTPUT_FILE = (
    "data/raw/github/requests_commit_file_history.csv"
)

PER_PAGE = 100

# IMPORTANT:
# Collect 2000 commits for a stronger historical analysis.
MAX_COMMITS = 2000


# ------------------------------------------------------------
# HTTP headers
# ------------------------------------------------------------

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "User-Agent": "CodeGuard-AI",
}


# ------------------------------------------------------------
# HTTP session with automatic retry
# ------------------------------------------------------------

def create_session():

    session = requests.Session()

    retry_strategy = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=2,

        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],

        allowed_methods=[
            "GET",
        ],

        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )

    session.mount(
        "https://",
        adapter
    )

    session.mount(
        "http://",
        adapter
    )

    session.headers.update(HEADERS)

    return session


SESSION = create_session()


# ------------------------------------------------------------
# Safe GET request
# ------------------------------------------------------------

def safe_get(url, params=None, timeout=30):

    max_attempts = 5

    for attempt in range(1, max_attempts + 1):

        try:

            response = SESSION.get(
                url,
                params=params,
                timeout=timeout,
            )

            # Successful response
            if response.status_code == 200:
                return response

            # Rate limit
            if response.status_code == 403:

                remaining = response.headers.get(
                    "X-RateLimit-Remaining"
                )

                reset_time = response.headers.get(
                    "X-RateLimit-Reset"
                )

                if remaining == "0" and reset_time:

                    try:
                        reset_timestamp = int(reset_time)

                        wait_seconds = max(
                            reset_timestamp - int(time.time()),
                            1,
                        )

                        print(
                            f"GitHub rate limit reached. "
                            f"Waiting {wait_seconds} seconds..."
                        )

                        time.sleep(
                            wait_seconds + 2
                        )

                        continue

                    except ValueError:
                        pass

            # Temporary server/network-related errors
            if response.status_code in [
                429,
                500,
                502,
                503,
                504,
            ]:

                wait_seconds = 2 ** attempt

                print(
                    f"HTTP {response.status_code}. "
                    f"Retrying in {wait_seconds} seconds "
                    f"(attempt {attempt}/{max_attempts})..."
                )

                time.sleep(wait_seconds)

                continue

            # Other HTTP errors
            print(
                f"GitHub API error: "
                f"HTTP {response.status_code}"
            )

            return None

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.SSLError,
        ) as error:

            wait_seconds = 2 ** attempt

            print()
            print(
                f"Network/SSL error while contacting GitHub."
            )

            print(
                f"Retrying in {wait_seconds} seconds "
                f"(attempt {attempt}/{max_attempts})..."
            )

            print(
                f"Error: {error}"
            )

            time.sleep(wait_seconds)

    print()
    print(
        f"Request failed after {max_attempts} attempts:"
    )

    print(url)

    return None


# ------------------------------------------------------------
# Get commit list
# ------------------------------------------------------------

def get_commits():

    print("=" * 70)
    print("CodeGuard AI - GitHub Data Collection")
    print("=" * 70)

    print(f"Repository : {OWNER}/{REPO}")
    print(f"Maximum commits : {MAX_COMMITS}")
    print()

    commits = []

    page = 1

    while len(commits) < MAX_COMMITS:

        print(
            f"Fetching commit page {page}..."
        )

        url = f"{BASE_URL}/commits"

        params = {
            "per_page": PER_PAGE,
            "page": page,
        }

        response = safe_get(
            url,
            params=params,
            timeout=30,
        )

        if response is None:

            print(
                f"Unable to retrieve commit page {page}."
            )

            break

        batch = response.json()

        if not batch:
            break

        commits.extend(batch)

        print(
            f"  Retrieved: {len(batch)} commits"
        )

        if len(batch) < PER_PAGE:
            break

        page += 1

        time.sleep(0.5)

    return commits[:MAX_COMMITS]


# ------------------------------------------------------------
# Get details for each commit
# ------------------------------------------------------------

def collect_file_history(commits):

    records = []

    total = len(commits)

    failed_commits = []

    print()
    print(
        f"Collecting file information from "
        f"{total} commits..."
    )
    print()

    for index, commit in enumerate(
        commits,
        start=1
    ):

        sha = commit.get("sha")

        commit_url = (
            f"{BASE_URL}/commits/{sha}"
        )

        response = safe_get(
            commit_url,
            timeout=30,
        )

        if response is None:

            print(
                f"Skipping commit {sha} "
                f"after repeated request failures."
            )

            failed_commits.append(sha)

            continue

        data = response.json()

        commit_info = data.get(
            "commit",
            {}
        )

        author_info = (
            commit_info.get("author")
            or {}
        )

        author_name = author_info.get(
            "name"
        )

        author_email = author_info.get(
            "email"
        )

        commit_date = author_info.get(
            "date"
        )

        message = commit_info.get(
            "message",
            ""
        )

        files = data.get(
            "files",
            []
        )

        for file_info in files:

            records.append({
                "commit_sha": sha,

                "author": author_name,

                "author_email": author_email,

                "commit_date": commit_date,

                "commit_message": message,

                "file_path": file_info.get(
                    "filename"
                ),

                "status": file_info.get(
                    "status"
                ),

                "additions": file_info.get(
                    "additions",
                    0
                ),

                "deletions": file_info.get(
                    "deletions",
                    0
                ),

                "changes": file_info.get(
                    "changes",
                    0
                ),
            })

        if (
            index % 25 == 0
            or index == total
        ):

            print(
                f"Processed {index}/{total} commits "
                f"| file records: {len(records)}"
            )

        time.sleep(0.15)

    print()

    if failed_commits:

        print(
            f"Commits skipped because of "
            f"repeated network/API failures: "
            f"{len(failed_commits)}"
        )

    else:

        print(
            "No commits were skipped."
        )

    return records


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    commits = get_commits()

    if not commits:

        raise RuntimeError(
            "No commits were retrieved."
        )

    print()

    print(
        f"Total commits collected: "
        f"{len(commits)}"
    )

    records = collect_file_history(
        commits
    )

    if not records:

        raise RuntimeError(
            "No file history records were collected."
        )

    df = pd.DataFrame(records)

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print()

    print("=" * 70)
    print("COLLECTION COMPLETED")
    print("=" * 70)

    print(
        f"Commits : "
        f"{df['commit_sha'].nunique()}"
    )

    print(
        f"Files   : "
        f"{df['file_path'].nunique()}"
    )

    print(
        f"Records : "
        f"{len(df)}"
    )

    print()

    print("Columns:")

    print(
        list(df.columns)
    )

    print()

    print("Saved to:")

    print(
        OUTPUT_FILE
    )

    print("=" * 70)


if __name__ == "__main__":

    main()