#!/usr/bin/env python3
# Copyright 2026 Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Atomically promote a JSON batch of sealed regular-track results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import promote_regular_bank as promote

REQUEST_SCHEMA = "ny_regular_bank_promotion_batch_request_v2"
PREVIOUS_REQUEST_SCHEMA = "ny_regular_bank_promotion_batch_request_v1"
PREVIOUS_REQUEST_KEYS = frozenset(
    {
        "artifact_root",
        "run_id",
        "category",
        "instance_index",
        "benchmark_root",
        "official_results",
        "measured_dir",
        "exact_commit",
        "evidence_index",
    }
)
REQUEST_KEYS = PREVIOUS_REQUEST_KEYS | {"migrate_legacy_decided_row"}


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise promote.PromotionError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _load_requests(path: Path) -> list[promote.PromotionRequest]:
    try:
        data = promote.evidence.stable_bytes(path, "batch request")
        value = json.loads(
            data,
            object_pairs_hook=_object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        OSError,
        ValueError,
    ) as error:
        raise promote.PromotionError(
            f"batch request is not strict JSON: {error}"
        ) from error
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "requests"}
        or value.get("schema") not in {REQUEST_SCHEMA, PREVIOUS_REQUEST_SCHEMA}
        or not isinstance(value.get("requests"), list)
        or not value["requests"]
    ):
        raise promote.PromotionError("batch request has an unsupported schema")
    request_schema = value["schema"]
    expected_keys = (
        REQUEST_KEYS if request_schema == REQUEST_SCHEMA else PREVIOUS_REQUEST_KEYS
    )
    requests: list[promote.PromotionRequest] = []
    for index, raw in enumerate(value["requests"], start=1):
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise promote.PromotionError(
                f"batch request {index} does not have exact canonical fields"
            )
        path_fields = (
            "artifact_root",
            "benchmark_root",
            "official_results",
            "measured_dir",
        )
        if (
            not all(isinstance(raw[field], str) and raw[field] for field in path_fields)
            or not isinstance(raw["run_id"], str)
            or not isinstance(raw["category"], str)
            or type(raw["instance_index"]) is not int
            or not isinstance(raw["exact_commit"], str)
            or (
                request_schema == REQUEST_SCHEMA
                and type(raw["migrate_legacy_decided_row"]) is not bool
            )
            or (
                raw["evidence_index"] is not None
                and not isinstance(raw["evidence_index"], str)
            )
        ):
            raise promote.PromotionError(
                f"batch request {index} has invalid field types"
            )
        requests.append(
            promote.PromotionRequest(
                artifact_root=Path(raw["artifact_root"]),
                run_id=raw["run_id"],
                category=raw["category"],
                instance_index=raw["instance_index"],
                benchmark_root=Path(raw["benchmark_root"]),
                official_results=Path(raw["official_results"]),
                measured_dir=Path(raw["measured_dir"]),
                exact_commit=raw["exact_commit"],
                evidence_index=(
                    Path(raw["evidence_index"])
                    if raw["evidence_index"] is not None
                    else None
                ),
                migrate_legacy_decided_row=(
                    raw["migrate_legacy_decided_row"]
                    if request_schema == REQUEST_SCHEMA
                    else False
                ),
            )
        )
    if promote.evidence.stable_bytes(path, "batch request") != data:
        raise promote.PromotionError("batch request changed while it was loaded")
    return requests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the index-first atomic transaction (default: dry run)",
    )
    args = parser.parse_args(argv)
    try:
        summary = promote.promote_batch(
            _load_requests(args.requests),
            apply=args.apply,
        )
    except (OSError, promote.PromotionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
