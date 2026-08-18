#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Check that the AyProc `ay` BINARY matches the `ay-milp` Cargo rev pin.

NY solves AY on two lanes that are supposed to be the same solver:

    ay-milp   the in-process library, rev-pinned in crates/ny-mip/Cargo.toml
    ay-proc   the frozen P0 subprocess lane, which runs an EXTERNAL `ay`
              binary found via $NY_AY, else `ay` on $PATH

`scripts/check_git_dep_freshness.py` checks the first against AY's remote HEAD.
NOTHING checked the second against anything -- so the binary can drift
arbitrarily far from the pin while every gate stays green.

WHY THIS MATTERS (measured, 2026-07-28). The binary on PATH was 46 days older
than the pin. Consequences:

  * `tests::backends_agree_on_random_tiny_nets` failed reproducibly (2/2) with
    "AyProc: tiny net must decide, got Timeout" -- a 3x2 network, at the 300 s
    default budget. With a binary built from the pin it passes in 2.66 s, and
    the ny-mip lib suite drops from 2550 s to 5.13 s.
  * More seriously, `docs/SOLVER_POLICY.md` names the `mip-diff` lib-vs-proc
    differential as the cross-check that survived the HiGHS deletion. A
    differential is only evidence when both sides are the SAME solver. Against a
    stale binary it measures the delta between two AY versions, and a
    disagreement cannot distinguish "the in-process lane is wrong" from "the
    subprocess lane is old".

"Frozen" in that policy means NY's P0 subprocess CODE PATH is frozen. It does
not mean the binary should be pinned in the past.

Usage:
    python3 scripts/check_ay_binary_matches_pin.py           # report, exit 1 on mismatch
    python3 scripts/check_ay_binary_matches_pin.py --json
    python3 scripts/check_ay_binary_matches_pin.py --warn-only   # always exit 0

To fix a mismatch, build the binary from the pinned rev and point $NY_AY at it:

    git -C <ay-repo> worktree add --detach /tmp/ay-pinned <REV>
    cargo build --release --manifest-path /tmp/ay-pinned/Cargo.toml \\
        --bin ay --features cli
    export NY_AY=/tmp/ay-pinned/target/release/ay
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

# `ay --version` prints e.g.
#   ay 0.5.0+build.6135.fc32ad715d8e250590a1c1bbc3e74622faff5e4b@2026-07-29T...
SHA_IN_VERSION = re.compile(r"\b([0-9a-f]{40})\b")


def resolve_ay_binary() -> str | None:
    """Resolve the binary the AyProc lane would run: $NY_AY, else `ay` on PATH."""
    explicit = os.environ.get("NY_AY")
    if explicit:
        return explicit if Path(explicit).is_file() else None
    return shutil.which("ay")


def binary_sha(binary: str) -> tuple[str | None, str]:
    """Return (40-hex build sha or None, raw first line of --version)."""
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
    found = SHA_IN_VERSION.search(first)
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
    binary = resolve_ay_binary()
    sha, version = (None, "") if binary is None else binary_sha(binary)

    if pinned is None:
        status, detail = "error", "no ay-milp git pin found in the workspace"
    elif binary is None:
        # Not fatal on its own: the in-process lane is the production default,
        # and AyProc is debug/bootstrap plus the mip-diff gate.
        status, detail = "absent", "no ay binary on $NY_AY or $PATH; AyProc lane unavailable"
    elif sha is None:
        status, detail = "unknown", f"could not parse a build sha from: {version!r}"
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
