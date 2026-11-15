"""Official source rows are occurrences, not a set of model/property pairs."""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any

from .evidence import ExperimentError, pin, stable_read


def load_cohort(config: dict[str, Any], base: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kind = config.get("kind")
    if kind not in {"official", "synthetic"}:
        raise ExperimentError("cohort.kind must explicitly be official or synthetic")
    suite, category = config.get("suite"), config.get("category")
    if not isinstance(suite, str) or not suite or not isinstance(category, str) or not category:
        raise ExperimentError("cohort requires named suite and category")
    csv_path = (base / config["instances_csv"]).resolve(strict=True)
    root = (base / config.get("input_root", str(csv_path.parent))).resolve(strict=True)
    raw, csv_pin = stable_read(csv_path)
    requested = config.get("source_indices")
    if requested is not None:
        if (not isinstance(requested, list) or not requested
                or any(isinstance(i, bool) or not isinstance(i, int) or i <= 0 for i in requested)
                or len(set(requested)) != len(requested)):
            raise ExperimentError("source_indices must be distinct positive occurrence indices")
        requested = set(requested)
    seen: dict[tuple[str, str, str], int] = {}
    rows = []
    source_count = 0
    for index, fields in enumerate(csv.reader(io.StringIO(raw.decode("utf-8"))), 1):
        source_count = index
        if len(fields) != 3:
            raise ExperimentError(f"official source row {index} must have three fields")
        model, prop, timeout_text = fields
        try:
            timeout = float(timeout_text)
        except ValueError as error:
            raise ExperimentError(f"invalid timeout in source row {index}") from error
        if not math.isfinite(timeout) or timeout <= 0:
            raise ExperimentError(f"invalid timeout in source row {index}")
        key = tuple(fields)
        occurrence = seen.get(key, 0) + 1
        seen[key] = occurrence
        if requested is not None and index not in requested:
            continue
        record: dict[str, Any] = {
            "suite": suite, "category": category, "kind": kind,
            "source_index": index, "occurrence": occurrence,
            "declared_model": model, "declared_property": prop,
            "timeout_seconds": timeout, "source_csv_sha256": csv_pin["sha256"],
        }
        # Multi-network syntax needs its own adapter. Retain the occurrence and
        # record refusal instead of silently dropping a difficult benchmark row.
        if model.lstrip().startswith("["):
            record["unsupported"] = "multi_network_cohort_row"
        else:
            for name, declared in (("model", model), ("property", prop)):
                declared_path = root / declared
                path = declared_path.resolve(strict=True)
                if path != root and root not in path.parents:
                    raise ExperimentError(f"source row {index} input escapes input_root")
                record[name] = pin(declared_path)
        rows.append(record)
    if requested is not None and requested - {r["source_index"] for r in rows}:
        raise ExperimentError("source_indices include absent official occurrences")
    if not rows:
        raise ExperimentError("empty cohort")
    return rows, {"csv": csv_pin, "input_root": str(root), "total_source_rows": source_count,
                  "selected_rows": len(rows), "selection_order": "official_source_order"}
