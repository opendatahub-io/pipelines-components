#!/usr/bin/env python3
"""Command line entry point for the Agentic RAG starter kit.

The Makefile forwards every cluster-facing target here, so the shell logic is
readable, testable, and free of Make's dollar escaping. Run a command directly
with ``python3 openshell/cli.py <command>`` or preview it with
``python3 openshell/cli.py <command> --dry-run``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openshell.core import cluster, log, sandbox
from openshell.core.config import KitConfig
from openshell.core.log import KitError

_COMMANDS = {
    "config": "Print the resolved deployment inputs",
    "refresh-certs": "Refresh openShell mTLS client certificates from the cluster",
    "build": "Build the openShell sandbox image via an OpenShift BuildConfig",
    "create": "Delete and recreate the openShell sandbox",
    "wait": "Wait until the sandbox reports Ready",
    "egress": "Add MaaS, K8s API, and vector database egress policies",
    "start": "Start the agent with ServiceAccount auth inside the sandbox",
    "expose": "Expose the agent service and create the Route",
    "rotate-tokens": "Rotate ServiceAccount tokens and restart the agent",
    "deploy": "Full deployment: create, wait, egress, start, expose",
    "undeploy": "Remove the openShell sandbox and Route",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Agentic RAG starter kit helper (pass -n/--dry-run anywhere to preview commands)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log every command executed")
    parser.add_argument("-C", "--directory", help="kit directory (defaults to the script's parent)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, description in _COMMANDS.items():
        subparsers.add_parser(name, help=description, description=description)
    return parser


def _dispatch(command: str, config: KitConfig) -> None:
    if command == "config":
        log.info(json.dumps(config.as_dict(), indent=2))
        return
    if command == "build":
        cluster.build_openshell(config)
        return
    cli = sandbox.OpenShell(config)
    if command == "refresh-certs":
        sandbox.refresh_certs(config, cli)
        return
    if command == "rotate-tokens":
        sandbox.rotate_tokens(config, cli)
        return
    if command == "undeploy":
        sandbox.undeploy(config, cli)
        return
    if command == "deploy":
        sandbox.deploy(config, cli)
        return
    context = sandbox.prepare(config)
    handlers = {
        "create": sandbox.create,
        "wait": sandbox.wait,
        "egress": sandbox.egress,
        "start": sandbox.start,
        "expose": sandbox.expose,
    }
    if command not in handlers:
        raise KitError(f"ERROR: unknown command '{command}'")
    handlers[command](config, cli, context)


def _preprocess(argv: list[str]) -> tuple[list[str], dict[str, object]]:
    """Lift global flags out of argv so they work before or after the command."""
    options: dict[str, object] = {"dry_run": False, "verbose": False, "directory": None}
    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in ("-n", "--dry-run"):
            options["dry_run"] = True
        elif item in ("-v", "--verbose"):
            options["verbose"] = True
        elif item in ("-C", "--directory") and index + 1 < len(argv):
            index += 1
            options["directory"] = argv[index]
        elif item.startswith("--directory="):
            options["directory"] = item.split("=", 1)[1]
        else:
            cleaned.append(item)
        index += 1
    return cleaned, options


def main(argv: list[str] | None = None) -> int:
    """Run one kit command and return a process exit code."""
    cleaned, options = _preprocess(list(sys.argv[1:] if argv is None else argv))
    if options["dry_run"]:
        os.environ["KIT_DRY_RUN"] = "1"
    args = _parser().parse_args(cleaned)
    log.configure(verbose=bool(options["verbose"]) or args.verbose)
    directory = options["directory"] or args.directory
    root = Path(str(directory)).resolve() if directory else Path(__file__).resolve().parents[1]
    try:
        config = KitConfig.load(root=root, lookup_namespace=cluster.current_namespace)
        _dispatch(args.command, config)
    except (KitError, OSError) as error:
        log.error(str(error))
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
