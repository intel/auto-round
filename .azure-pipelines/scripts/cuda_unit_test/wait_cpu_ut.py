"""Gate the CUDA unit tests on the CPU unit-test result of the same PR commit.

Only automatic PR builds are gated. Manual runs (Azure UI) and runs requested by
a ``/azp run`` PR comment pass immediately without looking at the CPU pipeline.

For an automatic PR build the CPU build of the same PR head commit is polled:
  * not queued within ``--appear-timeout`` -> CPU was skipped by its path
    filter, run the GPU tests;
  * succeeded / partiallySucceeded          -> run the GPU tests;
  * failed / canceled                       -> skip the GPU tests and fail this
    gate, so the CUDA check is reported as failed on GitHub.

The decision is exported as the output variable ``RUN_GPU`` (true/false).
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

CUDA_PIPELINE = "Unit-Test-CUDA-AutoRound"
AZP_COMMENT = re.compile(r"^\s*/(?:azp|AzurePipelines)\s+run\b(.*)$", re.IGNORECASE | re.MULTILINE)


def log(msg):
    print(msg, flush=True)


def http_json(url, headers=None, data=None):
    req = Request(url, data=data, headers=headers or {})
    with urlopen(req, timeout=30) as resp:  # nosec B310 - fixed https endpoints
        return json.loads(resp.read().decode("utf-8"))


def parse_time(value):
    # Azure returns up to 7 fractional digits, which fromisoformat rejects before 3.11.
    value = re.sub(r"(\.\d{6})\d+", r"\1", value.replace("Z", "+00:00"))
    return datetime.fromisoformat(value)


def requests_cuda(body):
    """Whether a comment body contains an /azp run command that covers the CUDA pipeline."""
    for match in AZP_COMMENT.finditer(body or ""):
        args = match.group(1).strip()
        # A bare "/azp run" runs every comment-triggered pipeline.
        if not args or CUDA_PIPELINE.lower() in args.lower():
            return True
    return False


def find_manual_comment(repo_path, pr_number, head_sha, queue_time, token):
    """Return True when a /azp run comment for CUDA was posted after the PR head commit.

    Comments posted after this build was queued are ignored since they cannot
    have triggered it.
    """
    owner, name = repo_path.split("/", 1)
    query = """
    query($owner: String!, $name: String!, $pr: Int!) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $pr) {
          timelineItems(last: 100, itemTypes: [ISSUE_COMMENT, PULL_REQUEST_COMMIT]) {
            nodes {
              __typename
              ... on IssueComment { body createdAt }
              ... on PullRequestCommit { commit { oid } }
            }
          }
        }
      }
    }
    """
    payload = json.dumps({"query": query, "variables": {"owner": owner, "name": name, "pr": pr_number}})
    data = http_json(
        "https://api.github.com/graphql",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        data=payload.encode("utf-8"),
    )
    nodes = data["data"]["repository"]["pullRequest"]["timelineItems"]["nodes"]

    # Walk from newest to oldest until the head commit is reached.
    for node in reversed(nodes):
        if node.get("__typename") == "PullRequestCommit":
            if node.get("commit", {}).get("oid") == head_sha:
                break
        elif node.get("__typename") == "IssueComment" and requests_cuda(node.get("body")):
            if queue_time is None or parse_time(node["createdAt"]) <= queue_time:
                return True
    return False


class AzureBuilds:
    def __init__(self, collection_uri, project, token):
        self.base = f"{collection_uri.rstrip('/')}/{project}/_apis/build"
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def get(self, path, **params):
        params["api-version"] = "7.1"
        return http_json(f"{self.base}/{path}?{urlencode(params)}", headers=self.headers)

    def definition_id(self, name):
        defs = self.get("definitions", name=name)["value"]
        if not defs:
            raise RuntimeError(f"pipeline definition '{name}' not found")
        return defs[0]["id"]

    def queue_time(self, build_id):
        return parse_time(self.get(f"builds/{build_id}")["queueTime"])

    def latest_pr_build(self, definition_id, pr_number, head_sha):
        builds = self.get(
            "builds",
            definitions=definition_id,
            branchName=f"refs/pull/{pr_number}/merge",
            reasonFilter="pullRequest",
            queryOrder="queueTimeDescending",
            **{"$top": 20},
        )["value"]
        for build in builds:
            if build.get("triggerInfo", {}).get("pr.sourceSha") == head_sha:
                return build
        return None


def set_output(run_gpu):
    value = "true" if run_gpu else "false"
    log(f"RUN_GPU={value}")
    log(f"##vso[task.setvariable variable=RUN_GPU;isOutput=true]{value}")


def wait_for_cpu(builds, definition_id, pr_number, head_sha, appear_timeout, interval):
    """Return True to run the GPU tests, False to skip them."""
    start = time.monotonic()
    while True:
        build = None
        try:
            build = builds.latest_pr_build(definition_id, pr_number, head_sha)
        except Exception as e:  # transient API error, keep polling
            log(f"##[warning]Failed to query CPU builds: {e}")

        elapsed = time.monotonic() - start
        if build is None:
            if elapsed >= appear_timeout:
                log(f"No CPU build for {head_sha} within {appear_timeout}s, assume it was skipped by path filter.")
                return True
            log(f"[{int(elapsed)}s] CPU build for {head_sha} not found yet, retry in {interval}s.")
        else:
            status, result = build.get("status"), build.get("result")
            log(f"[{int(elapsed)}s] CPU build {build['id']}: status={status}, result={result}")
            if status == "completed":
                return result in ("succeeded", "partiallySucceeded")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu-pipeline", default="Unit-Test-AutoRound")
    parser.add_argument("--appear-timeout", type=int, default=600, help="seconds to wait for the CPU build to appear")
    parser.add_argument("--interval", type=int, default=30, help="polling interval in seconds")
    args = parser.parse_args()

    env = os.environ
    if env.get("BUILD_REASON") != "PullRequest":
        log(f"BUILD_REASON={env.get('BUILD_REASON')} is not an automatic PR build, skip CPU gate.")
        set_output(True)
        return

    pr_number = int(env["SYSTEM_PULLREQUEST_PULLREQUESTNUMBER"])
    head_sha = env["SYSTEM_PULLREQUEST_SOURCECOMMITID"]
    repo_path = env["BUILD_REPOSITORY_NAME"]
    builds = AzureBuilds(env["SYSTEM_COLLECTIONURI"], env["SYSTEM_TEAMPROJECT"], env.get("SYSTEM_ACCESSTOKEN", ""))

    queue_time = None
    try:
        queue_time = builds.queue_time(env["BUILD_BUILDID"])
    except Exception as e:
        log(f"##[warning]Failed to read build queue time: {e}")

    try:
        if find_manual_comment(repo_path, pr_number, head_sha, queue_time, env.get("GITHUB_TOKEN", "")):
            log("Triggered by a /azp run comment, skip CPU gate.")
            set_output(True)
            return
    except Exception as e:
        log(f"##[warning]Failed to check PR comments, treat as automatic trigger: {e}")

    definition_id = builds.definition_id(args.cpu_pipeline)
    log(f"Waiting for {args.cpu_pipeline} (definition {definition_id}) on PR #{pr_number} commit {head_sha}...")
    run_gpu = wait_for_cpu(builds, definition_id, pr_number, head_sha, args.appear_timeout, args.interval)
    set_output(run_gpu)
    if not run_gpu:
        # Fail the gate so the CUDA check shows as failed on GitHub; the GPU stages are skipped.
        log(
            f"##vso[task.logissue type=error]CUDA unit tests skipped because {args.cpu_pipeline} failed "
            f"for commit {head_sha}. Fix the CPU unit tests first, or comment "
            f"'/azp run {CUDA_PIPELINE}' to run the CUDA tests anyway."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
