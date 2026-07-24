#!/usr/bin/env bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

# Shared first-party Clean dependency setup for NY's cross-repo tests. The
# Lake manifest is the single source of truth in both internal development and
# transformed public exports; never duplicate the URL or revision here.
# Source this file, then call prepare_pinned_clean.

prepare_pinned_clean() {
  local scripts_dir lean_root manifest_identity clean_url clean_pin
  scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  lean_root="$(cd "$scripts_dir/../proofs/lean" && pwd)"
  manifest_identity="$(python3 - "$lean_root/lake-manifest.json" <<'PY'
import json
import re
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
packages = [p for p in manifest.get("packages", []) if p.get("name") == "crownproof"]
if len(packages) != 1:
    raise SystemExit("expected exactly one crownproof package in Lake manifest")
package = packages[0]
url = package.get("url")
rev = package.get("rev")
if package.get("type") != "git" or package.get("inherited") is not False:
    raise SystemExit("crownproof must be a direct Git dependency")
if package.get("inputRev") != rev or package.get("subDir") != "crown-proofs/lean":
    raise SystemExit("crownproof Lake identity is not exact")
if not isinstance(url, str) or not isinstance(rev, str) or re.fullmatch(r"[0-9a-f]{40}", rev) is None:
    raise SystemExit("crownproof Lake URL/revision is malformed")
print(url, rev)
PY
)"
  read -r clean_url clean_pin <<<"$manifest_identity"
  local clean_root="${CLEAN_DIR:-}"

  if [[ -z "$clean_root" ]]; then
    clean_root="$lean_root/.lake/packages/crownproof"
    if [[ ! -d "$clean_root/.git" ]]; then
      echo "Clean Lake checkout missing; resolving exact dependency ${clean_pin:0:12}…"
      (cd "$lean_root" && lake update)
    fi
  fi

  [[ -d "$clean_root/.git" ]] || {
    echo "CLEAN_DIR is not a Git checkout: $clean_root" >&2
    return 1
  }
  local actual
  actual="$(git -C "$clean_root" rev-parse HEAD)"
  [[ "$actual" == "$clean_pin" ]] || {
    echo "Clean revision mismatch: expected $clean_pin from $clean_url, found $actual" >&2
    return 1
  }
  [[ -z "$(git -C "$clean_root" status --porcelain)" ]] || {
    echo "Clean checkout is dirty; cross-repo evidence requires the exact clean pin" >&2
    return 1
  }

  echo "Using exact Clean revision ${clean_pin:0:12} from $clean_url at $clean_root"
  cargo build --locked --release --manifest-path "$clean_root/Cargo.toml" -p clean
  CLEAN_ROOT="$clean_root"
  CLEAN_BIN="$clean_root/target/release/clean"
  [[ -x "$CLEAN_BIN" ]] || {
    echo "Clean build did not produce $CLEAN_BIN" >&2
    return 1
  }
}
