"""CI checks: reset labels, gate on author association/labels, poll check runs, save PR payload."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_PASSING_CONCLUSIONS = frozenset({"success", "neutral", "skipped"})
_TRUSTED_ASSOCIATIONS = frozenset({"MEMBER", "OWNER", "COLLABORATOR"})


class ChecksError(Exception):
    """Raised when CI checks fail or time out."""


class GhClient:
    """Wraps subprocess calls to the gh CLI."""

    def remove_label(self, repo: str, pr_number: int, label: str) -> None:
        """Remove a label from a PR via ``gh pr edit``."""
        subprocess.run(
            ["gh", "pr", "edit", str(pr_number), "--remove-label", label, "--repo", repo],
            check=True,
        )

    def get_check_runs(self, repo: str, head_sha: str) -> dict:
        """Return parsed JSON from the GitHub check-runs API."""
        result = subprocess.run(
            ["gh", "api", "--paginate", f"repos/{repo}/commits/{head_sha}/check-runs"],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    def get_own_check_run_id(self, repo: str, head_sha: str, check_name: str) -> int:
        """Return the ID of the check run matching *check_name*."""
        data = self.get_check_runs(repo, head_sha)
        for cr in data.get("check_runs", []):
            if cr["name"] == check_name:
                return cr["id"]
        raise ChecksError(f"Check run '{check_name}' not found")


def is_trusted_association(author_association: str) -> bool:
    """Return True if *author_association* represents a trusted contributor."""
    return author_association in _TRUSTED_ASSOCIATIONS


def should_run_checks(labels: list[str], *, author_association: str) -> bool:
    """Determine whether CI checks should run based on author association and PR labels."""
    if is_trusted_association(author_association):
        return True
    return "ok-to-test" in labels


def reset_label(gh: GhClient, repo: str, pr_number: int, labels: list[str]) -> None:
    """Remove the ci-passed label from a PR if it is present."""
    if "ci-passed" in labels:
        gh.remove_label(repo, pr_number, "ci-passed")


def wait_for_checks(
    gh: GhClient,
    repo: str,
    head_sha: str,
    *,
    check_run_id: int,
    delay: int,
    retries: int,
    interval: int,
) -> None:
    """Poll check runs until all pass or retries are exhausted."""
    if delay > 0:
        logger.info("Waiting %d seconds before first poll...", delay)
        time.sleep(delay)

    for attempt in range(retries):
        if attempt > 0:
            time.sleep(interval)

        logger.info("Poll %d/%d for commit %s", attempt + 1, retries, head_sha[:12])
        data = gh.get_check_runs(repo, head_sha)
        all_runs = data.get("check_runs", [])
        check_runs = [cr for cr in all_runs if cr["id"] != check_run_id]

        if not check_runs:
            if all_runs:
                logger.info("No other check runs found (only self). Done.")
                return
            logger.info("No check runs registered yet. Retrying...")
            continue

        pending = [cr for cr in check_runs if cr["status"] != "completed"]
        if pending:
            names = ", ".join(cr["name"] for cr in pending)
            logger.info("%d pending: %s", len(pending), names)
            continue

        failed = [cr for cr in check_runs if cr.get("conclusion") not in _PASSING_CONCLUSIONS]
        if failed:
            names = ", ".join(cr["name"] for cr in failed)
            raise ChecksError(f"Check(s) failed: {names}")

        logger.info("All %d check(s) passed.", len(check_runs))
        return

    raise ChecksError("Checks did not complete within the retry limit")


def save_pr_payload(output_dir: str, pr_number: int, event_action: str) -> None:
    """Save PR number and event action to files."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "pr_number").write_text(f"{pr_number}\n")
    (path / "event_action").write_text(f"{event_action}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CI check orchestration for pull requests.")
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--event-action", required=True)
    parser.add_argument("--labels", required=True, help="Comma-separated list of PR labels")
    parser.add_argument("--author-association", required=True, help="GitHub author_association value")
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--check-name", required=True)
    parser.add_argument("--delay", type=int, required=True, help="Seconds to wait before first poll")
    parser.add_argument("--retries", type=int, required=True)
    parser.add_argument("--polling-interval", type=int, required=True, help="Seconds between polls")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    labels = [label for label in args.labels.split(",") if label]
    gh = GhClient()

    if args.event_action in ("synchronize", "reopened"):
        reset_label(gh, args.repo, args.pr_number, labels)

    if not should_run_checks(labels, author_association=args.author_association):
        logger.info("PR requires '/ok-to-test' approval. Skipping CI checks.")
        return 0

    check_run_id = gh.get_own_check_run_id(args.repo, args.head_sha, args.check_name)

    try:
        wait_for_checks(
            gh,
            args.repo,
            args.head_sha,
            check_run_id=check_run_id,
            delay=args.delay,
            retries=args.retries,
            interval=args.polling_interval,
        )
    except ChecksError as exc:
        logger.error("CI checks failed: %s", exc)
        return 1

    save_pr_payload(args.output_dir, args.pr_number, args.event_action)
    return 0


if __name__ == "__main__":
    sys.exit(main())
