#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""Align canonical AY-claims docs with Cargo.lock.

test_vnncomp_ay_pin_coherence.py::test_internal_scored_audit_names_the_canonical_ay_revision
requires docs/SCORED_REPRO_AUDIT_2026-07-19.md and
docs/AY_BRANCH_HINT_CANARY.md to name the exact AY revision resolved in
Cargo.lock. AY bumps land frequently; run this after any bump (or let the bump
tooling call it) so the claims and lock never drift:

    python3 scripts/align_scored_audit_ay_pin.py [--check]

--check exits 1 without writing when the doc is out of date (CI-friendly).
Only the revision strings change; the docs' prose is untouched.
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCORED_AUDIT = ROOT / "docs" / "SCORED_REPRO_AUDIT_2026-07-19.md"
BRANCH_HINT_CANARY = ROOT / "docs" / "AY_BRANCH_HINT_CANARY.md"
DOCS = (SCORED_AUDIT, BRANCH_HINT_CANARY)
LOCK = ROOT / "Cargo.lock"


def resolved_ay_revision() -> str:
    lock = LOCK.read_text(encoding="utf-8")
    m = re.search(
        r'name = "ay-milp"\nversion = "[^"]+"\nsource = "git\+[^"]*?'
        r'rev=([0-9a-f]{40})',
        lock,
    )
    if not m:
        sys.exit("could not resolve ay-milp's git revision from Cargo.lock")
    return m.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the doc is stale; write nothing")
    args = parser.parse_args()

    rev = resolved_ay_revision()
    updates = []
    for doc in DOCS:
        original = doc.read_text(encoding="utf-8")
        updated = re.sub(
            r"(revision-pinned\s+to AY at\s+`)[0-9a-f]{40}(`)",
            lambda m: f"{m.group(1)}{rev}{m.group(2)}",
            original,
        )
        updated = re.sub(
            r"(`ay-milp` Git dependency at\s+`)[0-9a-f]{40}(`)",
            lambda m: f"{m.group(1)}{rev}{m.group(2)}",
            updated,
        )
        updated = re.sub(
            r"((?:revision-pinned to|Exact Git-pinned) AY `)[0-9a-f]{8}(`)",
            lambda m: f"{m.group(1)}{rev[:8]}{m.group(2)}",
            updated,
        )
        if updated != original:
            updates.append((doc, updated))

    if not updates:
        print(f"already aligned to AY {rev[:8]}")
        return 0
    if args.check:
        stale = ", ".join(str(doc.relative_to(ROOT)) for doc, _ in updates)
        print(f"STALE: {stale} does not name AY {rev[:8]}")
        return 1
    # pathlib.Path.write_text did not accept ``newline`` on the Python 3.9
    # baseline used by supported macOS hosts.  Open explicitly so the helper
    # stays portable while still normalizing generated documentation to LF.
    for doc, updated in updates:
        with doc.open("w", encoding="utf-8", newline="\n") as output:
            output.write(updated)
    print(f"aligned {len(updates)} docs to AY {rev[:8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
