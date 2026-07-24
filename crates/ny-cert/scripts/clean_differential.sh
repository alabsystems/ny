#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Scaled Proof-Carrying Verification round-trip: emit certificates for many
# deterministically-generated ReLU-1 networks and verify EVERY one with Clean's
# exact pinned external-certificate CLI.
#
# This is the cross-repo counterpart of tests/differential.rs: that test checks
# NY's in-tree mirror of Clean's algorithm; this checks the genuine Clean code.
#
# Usage:
#   CLEAN_DIR=/path/to/clean ./clean_differential.sh [SEED_START] [COUNT]
set -euo pipefail

SEED_START="${1:-0}"
COUNT="${2:-500}"

NY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
source "$(dirname "${BASH_SOURCE[0]}")/_clean_pinned.sh"
prepare_pinned_clean "$WORK/clean"

echo "Emitting NY certificates for seeds [$SEED_START, $((SEED_START+COUNT)))…"
OUT="$WORK/certs"; mkdir -p "$OUT"
( cd "$NY_DIR" && cargo run -q --release -p ny-cert --example emit_random -- "$SEED_START" "$COUNT" "$OUT" >/dev/null )

echo "=== Checking every emitted certificate with Clean's verifier ==="
BATCH="$WORK/certificates.json"
python3 - "$OUT" "$BATCH" <<'PY'
import json
import sys
from pathlib import Path

files = sorted(Path(sys.argv[1]).glob("*.json"))
if not files:
    raise SystemExit("no NY certificates were emitted")
certificates = [json.loads(path.read_text(encoding="utf-8")) for path in files]
Path(sys.argv[2]).write_text(json.dumps(certificates) + "\n", encoding="utf-8")
print(f"Prepared {len(certificates)} certificates for exact Clean batch verification")
PY
"$CLEAN_BIN" cert verify-external-batch "$BATCH" --threads 1
echo "All NY certificates accepted by Clean's real verifier."
