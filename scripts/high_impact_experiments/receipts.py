"""Strict telemetry admission. None of these observations mints a proof."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .evidence import ExperimentError, count, pin, strict_json

PHASES = ("loading", "lowering", "preprocessing", "root_construction",
          "intermediate_refresh", "proposal_search", "certification", "branching",
          "witness_validation", "output")


def native_ny_observations(path: Path) -> dict[str, Any]:
    """Ingest current production frozen-dual receipts without inventing phases.

    Per-class receipts are not added to their tier summary again. Dense reads
    are an equivalent-work allowance from the backend, never observed physical
    reads by a second solver. Rounded class times cannot stand in for complete
    exclusive phase instrumentation.
    """
    if not path.exists():
        return {"status": "unavailable", "reason": "no_solver_stderr"}
    rows = []
    summaries = []
    integer_keys = ("class", "nodes", "certified_prunes", "breakpoint_calls",
                    "breakpoint_steps", "breakpoint_accepted", "breakpoint_prunes",
                    "breakpoint_refused", "breakpoint_skipped", "support_compiled",
                    "support_fallback", "support_untouched", "support_grouped",
                    "support_sign_folded", "support_build_reads", "support_peak_bytes",
                    "support_dense_bytes", "state_bytes", "build_estimate_bytes", "build_columns")
    with path.open("r", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.startswith("[frozen-dual] "):
                continue
            values = dict(re.findall(r"([a-zA-Z_][a-zA-Z_0-9]*)=([^\s]+)", line))
            if "class" not in values:
                summaries.append({"line": line_number, "fields": values})
                continue
            record: dict[str, Any] = {"line": line_number, "counts": {}, "diagnostic_times": {}}
            for key in integer_keys:
                if key in values:
                    if not values[key].isdigit():
                        raise ExperimentError(f"malformed native NY count {key}")
                    record["counts"][key] = int(values[key])
            for key in ("support_columns", "support_events", "support_reads"):
                if key in values:
                    left, sep, right = values[key].partition("/")
                    if not sep or not left.isdigit() or not right.isdigit():
                        raise ExperimentError(f"malformed native NY paired count {key}")
                    record["counts"][key] = {"actual": int(left), "comparison": int(right)}
            for key, divisor in (("build_ms", 1000), ("secs", 1)):
                if key in values:
                    record["diagnostic_times"][key] = nonnegative(float(values[key]), key) / divisor
            record["class_verdict_observation"] = values.get("verdict", "unavailable")
            record["phase_time_status"] = "unavailable; these are rounded aggregate class costs"
            rows.append(record)
    return {"status": "observed_untrusted_telemetry" if rows or summaries else "unavailable",
            "source": pin(path), "per_class": rows, "tier_summaries": summaries,
            "authority": "none; original certificate replay remains required",
            "dense_reads_meaning": "equivalent_uncompressed_work_allowance_not_observed_reads",
            "support_events_meaning": "actual_sorted_events/comparison_ungrouped_events",
            "missing_reason_details": "native log exposes refusal counts, not individual backend reasons"}


def nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ExperimentError(f"{label} must be finite and nonnegative")
    return float(value)


def telemetry(path: Path, wall_seconds: float) -> dict[str, Any]:
    missing = {name: {"status": "unavailable", "reason": "solver_did_not_emit_phase_receipt"}
               for name in PHASES}
    if not path.exists():
        return {"status": "unavailable", "phases": missing, "methods": []}
    raw = strict_json(path)
    if raw.get("schema") != "ny_experiment_phases_v1":
        raise ExperimentError("unsupported phase receipt schema")
    phases = raw.get("phases")
    if not isinstance(phases, dict) or set(phases) - set(PHASES):
        raise ExperimentError("invalid phase names")
    exclusive_total = 0.0
    for name, phase in phases.items():
        if not isinstance(phase, dict):
            raise ExperimentError("phase observation must be an object")
        if phase.get("status") == "measured":
            seconds = nonnegative(phase.get("seconds"), f"{name}.seconds")
            if phase.get("clock") != "exclusive_monotonic":
                raise ExperimentError("phase time must use exclusive_monotonic")
            exclusive_total += seconds
            if "peak_bytes" in phase:
                count(phase["peak_bytes"], f"{name}.peak_bytes")
        elif phase.get("status") != "unavailable" or not phase.get("reason"):
            raise ExperimentError("phase must be measured or explicitly unavailable")
        missing[name] = phase
    if exclusive_total > wall_seconds + max(0.05, wall_seconds * 0.01):
        raise ExperimentError("exclusive phase costs exceed observed whole query time")
    methods = raw.get("methods", [])
    if not isinstance(methods, list):
        raise ExperimentError("methods must be a list")
    names = set()
    for method in methods:
        if not isinstance(method, dict) or not isinstance(method.get("name"), str) or not method["name"]:
            raise ExperimentError("each method receipt requires a name")
        if method["name"] in names:
            raise ExperimentError("duplicate method receipt")
        names.add(method["name"])
        for key in ("admitted", "attempted", "completed", "refused", "skipped",
                    "certified_improvements", "actual_eliminations", "peak_bytes",
                    "lookup_hits", "checked_scope_hits", "coefficient_reads"):
            count(method.get(key), f"{method['name']}.{key}")
        if not (method["completed"] <= method["attempted"] <= method["admitted"]
                and method["attempted"] + method["skipped"] <= method["admitted"]
                and method["checked_scope_hits"] <= method["lookup_hits"]):
            raise ExperimentError("method counts do not describe actual admitted/checkable work")
        nonnegative(method.get("seconds"), "method.seconds")
        reasons = method.get("reasons")
        if not isinstance(reasons, dict) or any(not isinstance(k, str) for k in reasons):
            raise ExperimentError("method reasons require a count map")
        for reason, value in reasons.items():
            count(value, f"method.reasons.{reason}")
        if sum(reasons.values()) != method["refused"] + method["skipped"]:
            raise ExperimentError("refused/skipped method calls require counted reasons")
    return {"status": "observed_untrusted_telemetry", "artifact": pin(path),
            "phases": missing, "methods": methods,
            "authority": "none; certification/elimination counts remain solver observations"}


def proposed_verdict(path: Path) -> str:
    if not path.exists():
        return "missing"
    with path.open("rb") as stream:
        first = stream.readline(1024)
    if not first:
        return "missing"
    token = first.decode("utf-8", errors="replace").strip().lower()
    return token if token in {"sat", "unsat", "unknown", "timeout", "error"} else "malformed"
