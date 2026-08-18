#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Architecture layer guard for ny workspace crate dependencies.

Validates that workspace crate dependency edges conform to the layer policy
defined in configs/architecture_layers.toml.

Usage:
    python3 scripts/check_architecture_layers.py          # validate
    python3 scripts/check_architecture_layers.py --verbose # show all edges
    python3 scripts/check_architecture_layers.py --json    # machine-readable

Part of #2126, Part of #1696.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = REPO_ROOT / "configs" / "architecture_layers.toml"


def load_policy(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def get_workspace_graph() -> tuple[set[str], list[dict]]:
    """Return workspace package names and workspace-internal dependency edges."""
    result = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        log.error("cargo metadata failed:\n%s", result.stderr)
        sys.exit(2)

    meta = json.loads(result.stdout)
    workspace_ids = set(meta["workspace_members"])
    workspace_packages = [p for p in meta["packages"] if p["id"] in workspace_ids]
    workspace_names = {p["name"] for p in workspace_packages}

    edges = []
    for pkg in workspace_packages:
        for dep in pkg["dependencies"]:
            if dep["name"] in workspace_names:
                edges.append(
                    {
                        "from": pkg["name"],
                        "to": dep["name"],
                        "kind": dep.get("kind") or "normal",
                    }
                )
    return workspace_names, edges


def get_workspace_deps() -> list[dict]:
    """Return workspace-internal dependency edges."""
    return get_workspace_graph()[1]


def build_layer_map(policy: dict) -> dict[str, str]:
    """Map crate name -> layer name from the policy."""
    layer_map = {}
    for layer_name, crates in policy.get("layers", {}).items():
        for crate in crates:
            layer_map[crate] = layer_name
    return layer_map


def layer_order(layer: str) -> int:
    """Return numeric order for a layer. Lower = more foundational."""
    if layer == "sidecar":
        return -1
    if layer.startswith("L"):
        return int(layer[1:])
    return 999


def build_exception_set(policy: dict) -> set[tuple[str, str, str]]:
    """Build set of (from, to, kind) tuples from policy exceptions."""
    return {(exc["from"], exc["to"], exc["kind"]) for exc in policy.get("exceptions", [])}


def validate_policy_assignments(
    policy: dict, workspace_names: set[str]
) -> list[dict]:
    """Require every workspace crate to have exactly one valid assignment."""
    violations = []
    assignments: dict[str, str] = {}

    for layer_name, crates in policy.get("layers", {}).items():
        for crate in crates:
            if not isinstance(crate, str) or not crate.strip():
                violations.append(
                    {
                        "reason": f"{layer_name} contains an empty or invalid crate name",
                        "severity": "error",
                    }
                )
                continue
            if crate in assignments:
                violations.append(
                    {
                        "reason": (
                            f"{crate} is assigned to both {assignments[crate]} "
                            f"and {layer_name}"
                        ),
                        "severity": "error",
                    }
                )
                continue
            assignments[crate] = layer_name

    for crate in sorted(workspace_names - assignments.keys()):
        violations.append(
            {
                "reason": f"{crate} is not assigned to any layer",
                "severity": "error",
            }
        )
    for crate in sorted(assignments.keys() - workspace_names):
        violations.append(
            {
                "reason": f"{crate} is assigned in policy but is not a workspace crate",
                "severity": "error",
            }
        )

    return violations


def _check_edge(
    edge: dict, layer_map: dict[str, str], exceptions: set[tuple[str, str, str]]
) -> dict | None:
    """Check a single edge for layer violations. Return violation dict or None."""
    src, dst, kind = edge["from"], edge["to"], edge["kind"]
    src_layer, dst_layer = layer_map.get(src), layer_map.get(dst)

    if src_layer is None:
        return {**edge, "reason": f"{src} not in any layer", "severity": "error"}
    if dst_layer is None:
        return {**edge, "reason": f"{dst} not in any layer", "severity": "error"}
    if src_layer == "sidecar":
        return None
    if dst_layer == "sidecar":
        if kind == "normal" and (src, dst, kind) not in exceptions:
            return {**edge, "reason": f"{src} ({src_layer}) has normal dep on sidecar {dst}", "severity": "error"}
        return None
    if layer_order(src_layer) < layer_order(dst_layer) and (src, dst, kind) not in exceptions:
        return {
            **edge,
            "reason": f"{src} ({src_layer}) depends on {dst} ({dst_layer}): lower layer cannot depend on higher layer",
            "severity": "error",
        }
    return None


def validate_exceptions(
    policy: dict,
    edges: list[dict],
    layer_map: dict[str, str],
) -> list[dict]:
    """Reject stale exceptions and exceptions for edges allowed by the policy."""
    violations = []
    edge_keys = {(edge["from"], edge["to"], edge["kind"]) for edge in edges}
    seen: set[tuple[str, str, str]] = set()

    for exception in policy.get("exceptions", []):
        key = (exception["from"], exception["to"], exception["kind"])
        if key in seen:
            violations.append(
                {
                    "reason": (
                        f"duplicate exception for {key[0]} -> {key[1]} "
                        f"({key[2]})"
                    ),
                    "severity": "error",
                }
            )
            continue
        seen.add(key)

        if key not in edge_keys:
            violations.append(
                {
                    "reason": (
                        f"stale exception for nonexistent edge "
                        f"{key[0]} -> {key[1]} ({key[2]})"
                    ),
                    "severity": "error",
                }
            )
            continue

        edge = {"from": key[0], "to": key[1], "kind": key[2]}
        if _check_edge(edge, layer_map, set()) is None:
            violations.append(
                {
                    "reason": (
                        f"unnecessary exception for allowed edge "
                        f"{key[0]} -> {key[1]} ({key[2]})"
                    ),
                    "severity": "error",
                }
            )

    return violations


def validate(
    edges: list[dict], layer_map: dict[str, str], exceptions: set[tuple[str, str, str]]
) -> list[dict]:
    """Validate edges against layer policy. Return list of violations."""
    violations = []
    for edge in edges:
        v = _check_edge(edge, layer_map, exceptions)
        if v is not None:
            violations.append(v)
    return violations


def _find_sccs(adj: dict[str, set[str]], all_nodes: set[str]) -> list[list[str]]:
    """Find strongly connected components via Tarjan's algorithm."""
    index_counter = [0]
    stack: list[str] = []
    lowlink: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: set[str] = set()
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in adj.get(v, set()):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1:
                sccs.append(sorted(scc))

    for node in sorted(all_nodes):
        if node not in index:
            strongconnect(node)
    return sccs


def _scc_has_unexcepted_violation(
    scc: list[str],
    adj: dict[str, set[str]],
    edges: list[dict],
    layer_map: dict[str, str],
    exceptions: set[tuple[str, str, str]],
) -> tuple[bool, list[tuple[str, str, str]]]:
    """Check if an SCC contains an unexcepted layer violation. Return (has_violation, cycle_edges)."""
    cycle_edges = []
    for s in scc:
        for t in adj.get(s, set()):
            if t in scc:
                for edge in edges:
                    if edge["from"] == s and edge["to"] == t:
                        cycle_edges.append((s, t, edge["kind"]))

    for s, t, k in cycle_edges:
        if layer_map.get(s) == "sidecar":
            continue
        if layer_map.get(t) == "sidecar":
            if k == "normal" and (s, t, k) not in exceptions:
                return True, cycle_edges
            continue
        s_ord = layer_order(layer_map.get(s, "sidecar"))
        t_ord = layer_order(layer_map.get(t, "sidecar"))
        if s_ord < t_ord and (s, t, k) not in exceptions:
            return True, cycle_edges
    return False, cycle_edges


def check_cycles(
    edges: list[dict], layer_map: dict[str, str], exceptions: set[tuple[str, str, str]]
) -> list[dict]:
    """Detect cycles including dev-dependency edges. Return cycle descriptions."""
    adj: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        src, dst = edge["from"], edge["to"]
        if src in layer_map and dst in layer_map:
            adj[src].add(dst)

    sccs = _find_sccs(adj, set(layer_map.keys()))

    violations = []
    for scc in sccs:
        has_violation, cycle_edges = _scc_has_unexcepted_violation(
            scc, adj, edges, layer_map, exceptions
        )
        if has_violation:
            violations.append({
                "cycle": scc,
                "edges": [{"from": s, "to": t, "kind": k} for s, t, k in cycle_edges],
                "reason": f"Dependency cycle: {' <-> '.join(scc)}",
                "severity": "warning",
            })
    return violations


def _format_verbose(policy: dict, edges: list[dict], layer_map: dict[str, str], exceptions: set) -> str:
    """Format verbose output showing layer assignments and all edges."""
    lines = ["Layer assignments:"]
    for layer_name in sorted(policy.get("layers", {}).keys(), key=layer_order):
        crates = policy["layers"][layer_name]
        lines.append(f"  {layer_name}: {', '.join(crates)}")
    lines.append("")
    lines.append(f"Dependency edges ({len(edges)} total):")
    for edge in sorted(edges, key=lambda e: (e["from"], e["to"])):
        src_layer = layer_map.get(edge["from"], "?")
        dst_layer = layer_map.get(edge["to"], "?")
        marker = " [EXCEPTED]" if (edge["from"], edge["to"], edge["kind"]) in exceptions else ""
        lines.append(f"  {edge['from']} ({src_layer}) -> {edge['to']} ({dst_layer}) [{edge['kind']}]{marker}")
    lines.append("")
    return "\n".join(lines)


def _format_result(
    policy: dict, edges: list[dict], layer_map: dict[str, str],
    exceptions: set, all_violations: list[dict], verbose: bool,
) -> str:
    """Format human-readable output."""
    parts = []
    if verbose:
        parts.append(_format_verbose(policy, edges, layer_map, exceptions))
    if exceptions:
        lines = [f"Temporary exceptions ({len(exceptions)}):"]
        for exc in policy.get("exceptions", []):
            lines.append(f"  {exc['from']} -> {exc['to']} ({exc['kind']}) — {exc['owner']}: {exc['reason']}")
        lines.append("")
        parts.append("\n".join(lines))
    if all_violations:
        lines = [f"VIOLATIONS ({len(all_violations)}):"]
        for v in all_violations:
            lines.append(f"  [{v['severity'].upper()}] {v['reason']}")
            if "cycle" in v:
                for ce in v["edges"]:
                    lines.append(f"    {ce['from']} -> {ce['to']} ({ce['kind']})")
        lines.append("")
        lines.append("FAILED: architecture layer violations detected.")
        parts.append("\n".join(lines))
    else:
        parts.append(f"OK: {len(edges)} dependency edges checked, {len(exceptions)} exceptions, 0 violations.")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate workspace crate dependencies against architecture layer policy."
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all dependency edges")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--policy", type=Path, default=POLICY_PATH, help="Path to policy TOML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.policy.exists():
        log.error("Policy file not found: %s", args.policy)
        return 2

    policy = load_policy(args.policy)
    layer_map = build_layer_map(policy)
    exceptions = build_exception_set(policy)
    workspace_names, edges = get_workspace_graph()

    all_violations = (
        validate_policy_assignments(policy, workspace_names)
        + validate_exceptions(policy, edges, layer_map)
        + validate(edges, layer_map, exceptions)
        + check_cycles(edges, layer_map, exceptions)
    )

    if args.json:
        sys.stdout.write(json.dumps(
            {"violations": all_violations, "edge_count": len(edges), "exception_count": len(exceptions), "pass": len(all_violations) == 0},
            indent=2,
        ) + "\n")
    else:
        sys.stdout.write(_format_result(policy, edges, layer_map, exceptions, all_violations, args.verbose) + "\n")

    return 1 if all_violations else 0


if __name__ == "__main__":
    sys.exit(main())
