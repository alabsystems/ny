#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end Proof-Carrying Verification round-trip:
#   NY's ny-cert emits exact-rational CROWN certificates  ->  Clean's external
#   certificate verifier accepts them.
#
# The verifier is Clean's exact pinned CLI. No Clean source is copied into NY or
# into a generated harness.
#
# Usage:
#   CLEAN_DIR=/path/to/clean ./roundtrip_with_clean.sh
#   # or, if unset, the script clones github.com/alabsystems/clean (needs auth)
set -euo pipefail

NY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
source "$(dirname "${BASH_SOURCE[0]}")/_clean_pinned.sh"
prepare_pinned_clean "$WORK/clean"

# --- generate NY's certificates and check them with Clean's verifier ---
echo "Generating NY certificates…"
OUT="$WORK/certs"; mkdir -p "$OUT"
( cd "$NY_DIR" && NY_CERT_OUT_DIR="$OUT" cargo test -q -p ny-cert --test relu_entailment_roundtrip json_matches >/dev/null )

echo "=== Clean verifier verdicts on NY-emitted certificates ==="
"$CLEAN_BIN" cert verify-external "$OUT/ny_relu1_entailment.json"
"$CLEAN_BIN" cert verify-external "$OUT/ny_relu1_farkas.json"
echo "Round-trip OK."
