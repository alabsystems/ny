#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Pillar 2 cross-repo proof: NY emits SBAR attention support-bound certificates
# (simplex/water-filling LP duality) and Clean's REAL external-cert verifier
# accepts every one through Clean's exact pinned CLI.
#
# Usage: CLEAN_DIR=/path/to/clean ./clean_sbar.sh [SEED_START] [COUNT]
set -euo pipefail

SEED_START="${1:-0}"
COUNT="${2:-500}"
NY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
source "$(dirname "${BASH_SOURCE[0]}")/_clean_pinned.sh"
prepare_pinned_clean "$WORK/clean"

OUT="$WORK/certs"; mkdir -p "$OUT"
( cd "$NY_DIR" && cargo run -q --release -p ny-cert --example emit_sbar -- "$SEED_START" "$COUNT" "$OUT" )
BATCH="$WORK/sbar-certificates.json"
python3 - "$OUT" "$BATCH" <<'PY'
import json
import sys
from pathlib import Path

files = sorted(Path(sys.argv[1]).glob("*.json"))
if not files:
    raise SystemExit("no SBAR certificates were emitted")
certificates = [json.loads(path.read_text(encoding="utf-8")) for path in files]
Path(sys.argv[2]).write_text(json.dumps(certificates) + "\n", encoding="utf-8")
print(f"Prepared {len(certificates)} SBAR certificates for exact Clean verification")
PY
"$CLEAN_BIN" cert verify-external-batch "$BATCH" --threads 1
echo "All SBAR certificates accepted by Clean's real verifier."
