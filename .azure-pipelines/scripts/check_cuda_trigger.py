import json
import os
import sys
from urllib.request import Request, urlopen


def main():
    repo_path = os.environ.get("REPO_PATH", "")
    pr_number_str = os.environ.get("PR_NUMBER", "")
    sha = os.environ.get("CURRENT_SHA", "")
    token = os.environ.get("GITHUB_TOKEN", "")

    if not repo_path or "/" not in repo_path or not pr_number_str:
        print("Missing required environment variables (REPO_PATH, PR_NUMBER).", file=sys.stderr)
        sys.exit(1)

    owner, name = repo_path.split("/")
    pr = int(pr_number_str)

    query = """
    query($owner: String!, $name: String!, $pr: Int!) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $pr) {
          timelineItems(last: 50, itemTypes: [ISSUE_COMMENT, PULL_REQUEST_COMMIT]) {
            nodes {
              __typename
              ... on IssueComment {
                 body
              }
              ... on PullRequestCommit {
                 commit {
                   oid
                 }
              }
            }
          }
        }
      }
    }
    """

    req = Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": query, "variables": {"owner": owner, "name": name, "pr": pr}}).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )

    try:
        with urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            nodes = data["data"]["repository"]["pullRequest"]["timelineItems"]["nodes"]
    except Exception as e:
        print("GraphQL Error:", e, file=sys.stderr)
        sys.exit(1)

    target = "/azp run Unit-Test-CUDA-AutoRound"

    # Traverse nodes in reverse (from newest to oldest)
    for node in reversed(nodes):
        if node.get("__typename") == "IssueComment":
            if target in node.get("body", ""):
                # Target comment found before we reached the current commit.
                # It means a comment has already been posted to trigger CI for this commit state.
                sys.exit(0)
        elif node.get("__typename") == "PullRequestCommit":
            if node.get("commit", {}).get("oid") == sha:
                # We reached the tip commit of this trigger before finding any trigger comment.
                # Therefore, we need to post a new comment to trigger CI.
                break

    # Exit 1 means we break the loop (commit reached) without calling sys.exit(0)
    # The bash script catching exit 1 means "proceed to post the comment".
    sys.exit(1)


if __name__ == "__main__":
    main()
