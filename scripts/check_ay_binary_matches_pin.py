#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Check the AyProc executable's successful --version against the ay-milp pin.

Like AyProc, use NY_AY whenever set, otherwise `ay` on PATH. Bare commands use
PATH. Relative paths with separators use the invoking working directory. An
empty or unresolvable explicit NY_AY is an error, never a fallback or skip.
There is no Trust toolchain fallback. Only an unset NY_AY with no `ay` on PATH
is an optional-lane skip.

Usage:
    python3 scripts/check_ay_binary_matches_pin.py           # report, exit 1 on mismatch
    python3 scripts/check_ay_binary_matches_pin.py --json
    python3 scripts/check_ay_binary_matches_pin.py --warn-only   # always exit 0

Build the reported pinned_rev in a persistent worktree to fix a mismatch:

    git -C <ay-repo> worktree add --detach "$HOME/ay-pinned" <pinned_rev>
    targo --unverified build --locked --release --manifest-path "$HOME/ay-pinned/Cargo.toml" \\
        --bin ay --features cli
    export NY_AY="$HOME/ay-pinned/target/release/ay"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from git_dep_pins import workspace_git_dependency_pins

REPO_ROOT = Path(__file__).resolve().parent.parent

# `ay --version` embeds its full source commit in the build metadata.
SHA_IN_VERSION = re.compile(r"\b([0-9a-f]{40})\b")


def resolve_ay_binary() -> str | None:
    """Preflight AyProc's explicit command or default PATH lookup."""
    explicit = os.environ.get("NY_AY")
    return shutil.which(explicit if explicit is not None else "ay")


def binary_sha(binary: str) -> tuple[str | None, str]:
    """Return (successful 40-hex build sha or None, raw --version first line)."""
    try:
        out = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return None, f"<failed to run: {exc}>"
    raw = (out.stdout or out.stderr or "").strip().splitlines()
    first = raw[0] if raw else ""
    found = SHA_IN_VERSION.search(first) if out.returncode == 0 else None
    return (found.group(1) if found else None), first


def pinned_ay_rev() -> str | None:
    """The rev `ay-milp` is pinned to, from the workspace manifests."""
    for pin in workspace_git_dependency_pins(REPO_ROOT / "Cargo.toml"):
        if pin.name == "ay-milp":
            return pin.rev
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument(
        "--warn-only",
        action="store_true",
        help="report a mismatch but always exit 0",
    )
    args = ap.parse_args()

    pinned = pinned_ay_rev()
    requested = os.environ.get("NY_AY")
    binary = resolve_ay_binary()
    sha, version = (None, "") if binary is None else binary_sha(binary)

    if pinned is None:
        status, detail = "error", "no ay-milp git pin found in the workspace"
    elif binary is None and requested is not None:
        status, detail = (
            "error",
            f"$NY_AY={requested!r} does not resolve to an executable; "
            "the AyProc lane cannot launch it",
        )
    elif binary is None:
        # Not fatal on its own: the in-process lane is the production default,
        # and AyProc is debug/bootstrap plus the mip-diff gate.
        status, detail = (
            "absent",
            "$NY_AY unset and no `ay` on $PATH; AyProc lane unavailable",
        )
    elif sha is None:
        status, detail = (
            "unknown",
            f"--version did not establish a successful build sha: {version!r}",
        )
    elif sha == pinned:
        status, detail = "ok", "binary matches the ay-milp pin"
    else:
        status, detail = "mismatch", f"binary {sha[:12]} != pin {pinned[:12]}"

    if args.json:
        print(
            json.dumps(
                {
                    "status": status,
                    "detail": detail,
                    "pinned_rev": pinned,
                    "ny_ay_requested": requested,
                    "binary": binary,
                    "binary_sha": sha,
                    "binary_version": version,
                },
                indent=2,
            )
        )
    else:
        print("AyProc binary vs ay-milp pin:")
        print(f"  pin:    {pinned}")
        print(f"  binary: {binary or '<none>'}")
        if version:
            print(f"  version: {version}")
        tag = {"ok": "[OK]", "absent": "[SKIP]"}.get(status, "[FAIL]")
        print(f"  {tag}   {detail}")
        if status == "mismatch":
            print()
            print("  The two AY lanes are different solvers. Any mip-diff")
            print("  lib-vs-proc result is measuring the version delta, not the")
            print("  lanes. See the module docstring for the rebuild recipe.")

    if args.warn_only or status in {"ok", "absent"}:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
