#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Differential bound oracle for NY and the VNN-COMP 2025 alpha-beta-CROWN.

This is deliberately a diagnostic, not a verifier.  It joins NY's existing
``NY_LPOPT_DUMP``/``NY_LPOPT_SPLIT_BOUNDS`` artifacts to a pinned export from
the official alpha-beta-CROWN checkout, then compares the exact same VNNLIB
input box, output rows, ReLU pre-activation boxes, and final margins.

The expensive ``abc-root`` command must be run with alpha-beta-CROWN's Python
environment and through the host GPU guard.  The other commands need only the
Python standard library.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "ny_abc_bound_parity_v1"
ABC_SHA = "e5c7e17bf0488843acb77b7519f59876717a49f4"
AUTOLIRPA_SHA = "5a098e8f9fb5786a428a024981d833d303921f2d"
CIFAR100_CONFIG_SHA256 = (
    "3ec72979f7a6748734802021c4da19f5b638d30c0682cf46d131d75506ad494b"
)
CIFAR100_MEDIUM_ONNX_SHA256 = (
    "aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4"
)
PROP1761_VNNLIB_SHA256 = (
    "f2a5e14de263f19a36d06a2200197e111f3cb7467eaf0f524edb09e3253b667b"
)
EXPECTED_EXPORT_PINS = {
    "alpha_beta_crown_git": ABC_SHA,
    "auto_lirpa_git": AUTOLIRPA_SHA,
    "config_sha256": CIFAR100_CONFIG_SHA256,
    "onnx_sha256": CIFAR100_MEDIUM_ONNX_SHA256,
}


class ParityError(ValueError):
    """An artifact cannot support an exact differential comparison."""


@dataclass(frozen=True)
class Bounds:
    shape: tuple[int, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    @property
    def size(self) -> int:
        return len(self.lower)


@dataclass(frozen=True)
class Premise:
    relu: str
    neuron: int
    active: bool


@dataclass(frozen=True)
class ChildMetadata:
    depth: int
    binding_objective: int
    binding_lower: float
    premises: tuple[Premise, ...]


@dataclass(frozen=True)
class NyDump:
    input_bounds: Bounds
    relu_map: tuple[tuple[str, str], ...]
    nodes: dict[str, Bounds]
    child: ChildMetadata | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checked_float(token: str, context: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise ParityError(f"{context}: invalid float {token!r}") from exc
    if not math.isfinite(value):
        raise ParityError(f"{context}: non-finite float {token!r}")
    return value


def _product(shape: Sequence[int]) -> int:
    value = 1
    for dim in shape:
        if dim <= 0:
            raise ParityError(f"non-positive tensor dimension {dim}")
        value *= dim
    return value


def parse_premises(raw: str) -> tuple[Premise, ...]:
    """Parse NY's ``Relu_N:index:A|I`` split-history encoding."""

    if raw == "":
        return ()
    premises: list[Premise] = []
    seen: set[tuple[str, int]] = set()
    for item in raw.split(","):
        parts = item.split(":")
        if len(parts) != 3 or parts[2] not in {"A", "I"}:
            raise ParityError(f"invalid split premise {item!r}")
        relu = parts[0]
        if not relu:
            raise ParityError(f"invalid split premise {item!r}: empty node")
        try:
            neuron = int(parts[1])
        except ValueError as exc:
            raise ParityError(f"invalid split neuron in {item!r}") from exc
        if neuron < 0:
            raise ParityError(f"negative split neuron in {item!r}")
        key = (relu, neuron)
        if key in seen:
            raise ParityError(f"duplicate split premise for {relu}:{neuron}")
        seen.add(key)
        premises.append(Premise(relu, neuron, parts[2] == "A"))
    return tuple(premises)


_CHILD_HEADER = re.compile(
    r"^# ny lpopt child-bounds dump v1 "
    r"depth=(\d+) bind_obj=(\d+) bind_lb=([^ ]+) premises=(.*)$"
)


def _parse_bounds_record(
    lines: Sequence[str], index: int, kind: str
) -> tuple[str | None, Bounds, int]:
    parts = lines[index].split()
    if kind == "INPUT":
        if len(parts) < 3:
            raise ParityError("INPUT record is missing shape dimensions")
        name: str | None = None
        size_index = 1
        shape_index = 2
    else:
        if len(parts) < 4:
            raise ParityError("NODE record is missing name or shape dimensions")
        name = parts[1]
        size_index = 2
        shape_index = 3
    try:
        size = int(parts[size_index])
        shape = tuple(int(item) for item in parts[shape_index:])
    except ValueError as exc:
        raise ParityError(f"malformed {kind} size/shape at line {index + 1}") from exc
    if size <= 0 or _product(shape) != size:
        raise ParityError(
            f"{kind} {name or ''}: size {size} does not match shape {shape}"
        )
    if index + 2 >= len(lines):
        raise ParityError(f"truncated {kind} record at line {index + 1}")
    lower_parts = lines[index + 1].split()
    upper_parts = lines[index + 2].split()
    if not lower_parts or lower_parts[0] != "L":
        raise ParityError(f"{kind} {name or ''}: expected L record")
    if not upper_parts or upper_parts[0] != "U":
        raise ParityError(f"{kind} {name or ''}: expected U record")
    lower = tuple(
        _checked_float(item, f"{kind} {name or ''} lower") for item in lower_parts[1:]
    )
    upper = tuple(
        _checked_float(item, f"{kind} {name or ''} upper") for item in upper_parts[1:]
    )
    if len(lower) != size or len(upper) != size:
        raise ParityError(
            f"{kind} {name or ''}: expected {size} values, got "
            f"{len(lower)} lower/{len(upper)} upper"
        )
    for offset, (lo, hi) in enumerate(zip(lower, upper)):
        if lo > hi:
            raise ParityError(
                f"{kind} {name or ''}[{offset}]: lower {lo} exceeds upper {hi}"
            )
    return name, Bounds(shape, lower, upper), index + 3


def parse_ny_dump(path: Path) -> NyDump:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ParityError(f"empty NY dump: {path}")
    child: ChildMetadata | None = None
    if lines[0] == "# ny lpopt dump v1":
        pass
    else:
        match = _CHILD_HEADER.fullmatch(lines[0])
        if match is None:
            raise ParityError(f"unsupported NY dump header {lines[0]!r}")
        child = ChildMetadata(
            depth=int(match.group(1)),
            binding_objective=int(match.group(2)),
            binding_lower=_checked_float(match.group(3), "child binding lower"),
            premises=parse_premises(match.group(4)),
        )
        if child.depth != len(child.premises):
            raise ParityError(
                f"child depth {child.depth} != {len(child.premises)} premises"
            )

    input_bounds: Bounds | None = None
    relu_map: list[tuple[str, str]] = []
    nodes: dict[str, Bounds] = {}
    index = 1
    while index < len(lines):
        line = lines[index]
        if not line:
            index += 1
            continue
        if line.startswith("INPUT "):
            if input_bounds is not None:
                raise ParityError("duplicate INPUT record")
            _, input_bounds, index = _parse_bounds_record(lines, index, "INPUT")
        elif line.startswith("RELUMAP "):
            if relu_map:
                raise ParityError("duplicate RELUMAP record")
            parts = line.split()
            if len(parts) != 2:
                raise ParityError("malformed RELUMAP record")
            try:
                count = int(parts[1])
            except ValueError as exc:
                raise ParityError("malformed RELUMAP count") from exc
            if count < 0 or index + count >= len(lines):
                raise ParityError("truncated RELUMAP record")
            seen_relu: set[str] = set()
            for offset in range(count):
                pair = lines[index + 1 + offset].split()
                if len(pair) != 2 or pair[0] in seen_relu:
                    raise ParityError(
                        f"malformed/duplicate RELUMAP entry at line {index + 2 + offset}"
                    )
                seen_relu.add(pair[0])
                relu_map.append((pair[0], pair[1]))
            index += count + 1
        elif line.startswith("NODE "):
            name, bounds, index = _parse_bounds_record(lines, index, "NODE")
            assert name is not None
            if name in nodes:
                raise ParityError(f"duplicate NODE record {name!r}")
            nodes[name] = bounds
        else:
            raise ParityError(f"unknown record at line {index + 1}: {line!r}")
    if input_bounds is None:
        raise ParityError("NY dump has no INPUT record")
    if not nodes:
        raise ParityError("NY dump has no NODE records")
    for _, preactivation in relu_map:
        if preactivation not in nodes:
            raise ParityError(f"RELUMAP target {preactivation!r} has no NODE record")
    return NyDump(input_bounds, tuple(relu_map), nodes, child)


def parse_ny_margins(path: Path) -> tuple[tuple[float, float, float], ...]:
    margins: dict[int, tuple[float, float, float]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ParityError(f"malformed margin line {line_number}")
        try:
            index = int(parts[0])
        except ValueError as exc:
            raise ParityError(f"invalid margin index at line {line_number}") from exc
        if index < 0 or index in margins:
            raise ParityError(f"duplicate/negative margin index {index}")
        margins[index] = tuple(
            _checked_float(token, f"margin {index}") for token in parts[1:]
        )  # type: ignore[assignment]
    if not margins or sorted(margins) != list(range(len(margins))):
        raise ParityError("margin indices must be contiguous from zero")
    return tuple(margins[index] for index in range(len(margins)))


def _tensor_list(tensor: Any) -> list[float]:
    return tensor.detach().to(device="cpu", dtype=None).reshape(-1).tolist()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def write_export(path: Path, value: Any) -> str:
    payload = gzip.compress(_canonical_json_bytes(value), compresslevel=9, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def read_export(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(gzip.decompress(path.read_bytes()))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParityError(f"invalid parity export {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ParityError(f"unsupported parity export schema in {path}")
    return value


def _validate_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ParityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_export_pins(
    abc: dict[str, Any],
    expected_vnnlib_sha256: str = PROP1761_VNNLIB_SHA256,
) -> None:
    if abc.get("kind") != "abc_root":
        raise ParityError("parity export is not an ABC root artifact")
    pins = abc.get("pins", {})
    if not isinstance(pins, dict):
        raise ParityError("ABC export pins must be an object")
    for key, expected in EXPECTED_EXPORT_PINS.items():
        if pins.get(key) != expected:
            raise ParityError(f"ABC export pin {key} is not the supported value")
    actual_vnnlib_sha256 = _validate_sha256(
        pins.get("vnnlib_sha256"), "ABC export pin vnnlib_sha256"
    )
    expected_vnnlib_sha256 = _validate_sha256(
        expected_vnnlib_sha256, "expected VNN-LIB SHA-256"
    )
    if actual_vnnlib_sha256 != expected_vnnlib_sha256:
        raise ParityError(
            "ABC export pin vnnlib_sha256 does not match the expected property: "
            f"expected {expected_vnnlib_sha256}, got {actual_vnnlib_sha256}"
        )


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ParityError(
            f"{label} hash mismatch: expected {expected}, got {actual} ({path})"
        )


def export_abc_root(args: argparse.Namespace) -> int:
    """Run the pinned official root and serialize all parity-relevant state."""

    abc_repo = args.abc_repo.resolve()
    config = args.config.resolve()
    onnx = args.onnx.resolve()
    vnnlib = args.vnnlib.resolve()
    if _git_head(abc_repo) != ABC_SHA:
        raise ParityError(f"alpha-beta-CROWN must be pinned to {ABC_SHA}")
    if _git_head(abc_repo / "auto_LiRPA") != AUTOLIRPA_SHA:
        raise ParityError(f"auto_LiRPA must be pinned to {AUTOLIRPA_SHA}")
    _require_hash(config, CIFAR100_CONFIG_SHA256, "cifar100 config")
    _require_hash(onnx, CIFAR100_MEDIUM_ONNX_SHA256, "CIFAR100 medium ONNX")
    _require_hash(
        vnnlib,
        _validate_sha256(args.expected_vnnlib_sha256, "expected VNN-LIB SHA-256"),
        "CIFAR100 VNN-LIB",
    )

    complete_verifier = abc_repo / "complete_verifier"
    sys.path.insert(0, str(complete_verifier))
    os.chdir(abc_repo)
    # Imports intentionally live here: compare/replay-plan remain stdlib-only.
    import arguments  # noqa: PLC0415
    from abcrown import ABCROWN  # noqa: PLC0415

    cli = [
        "--config",
        str(config),
        "--onnx_path",
        str(onnx),
        "--vnnlib_path",
        str(vnnlib),
        "--complete_verifier",
        "skip",
        "--return_optimized_model",
        "--save_output",
        "--pgd_order",
        "skip",
        "--deterministic",
        "--deterministic_opt",
        "--start",
        "0",
        "--end",
        "1",
    ]
    verifier = ABCROWN(args=cli)
    model = verifier.main()
    if model is None or not hasattr(model, "net"):
        raise ParityError("official root did not return an optimized LiRPANet")

    specs = verifier.vnnlib_handler.all_specs.get("cpu")
    bounded_x, c_tensor, rhs_tensor = specs[:3]
    x_lower = bounded_x.ptb.x_L[0]
    x_upper = bounded_x.ptb.x_U[0]
    c_rows = c_tensor.reshape(-1, c_tensor.shape[-1])
    rhs = rhs_tensor.reshape(-1)
    sparse_c = [
        [[index, float(value)] for index, value in enumerate(row.tolist()) if value]
        for row in c_rows
    ]

    output = arguments.Globals.get("out", {})
    initial_crown = output.get("init_crown_bounds")
    effective = output.get("init_alpha_crown")
    if initial_crown is None or effective is None:
        raise ParityError(
            "official save_output did not expose init_crown_bounds/init_alpha_crown"
        )
    initial_values = _tensor_list(initial_crown)
    raw_effective_values = _tensor_list(effective)
    if len(initial_values) != len(sparse_c):
        raise ParityError(
            f"initial official margins have {len(initial_values)} rows, "
            f"VNNLIB has {len(sparse_c)}"
        )
    if len(raw_effective_values) != len(sparse_c):
        raise ParityError(
            f"effective official margins have {len(raw_effective_values)} rows, "
            f"VNNLIB has {len(sparse_c)}"
        )
    if any(not math.isfinite(value) for value in initial_values):
        raise ParityError("official initial CROWN margins contain non-finite values")
    # PruneAfterCROWN uses +inf as the recovered sentinel for rows already
    # verified by the initial pass.  Such a sentinel is not a numeric bound to
    # compare; the retained sound bound for that row is its initial CROWN value.
    # The one optimized row is combined by MAX, since alpha optimization cannot
    # weaken the retained lower bound.
    effective_values = [
        max(initial, optimized) if math.isfinite(optimized) else initial
        for initial, optimized in zip(initial_values, raw_effective_values)
    ]

    layers = []
    for ordinal, node in enumerate(model.net.split_nodes):
        if node.lower is None or node.upper is None:
            raise ParityError(f"official split node {node.name} has no root bounds")
        lower = _tensor_list(node.lower)
        upper = _tensor_list(node.upper)
        if len(lower) != len(upper):
            raise ParityError(f"official split node {node.name} has mismatched bounds")
        layers.append(
            {
                "ordinal": ordinal,
                "preactivation": node.name,
                "shape": list(node.lower.shape[1:]),
                "lower": lower,
                "upper": upper,
            }
        )

    cfg = arguments.Config
    contract = {
        "deterministic_overlay": True,
        "onnx_optimization_flags": cfg["model"]["onnx_optimization_flags"],
        "alpha": {
            key: cfg["solver"]["alpha-crown"][key]
            for key in ("iteration", "lr_alpha", "full_conv_alpha")
        },
        "beta": {
            key: cfg["solver"]["beta-crown"][key]
            for key in (
                "iteration",
                "lr_alpha",
                "lr_beta",
                "enable_opt_interm_bounds",
                "all_node_split_LP",
            )
        },
        "branching": {
            key: cfg["bab"]["branching"][key]
            for key in ("method", "candidates", "reduceop")
        },
        "clip_interm_domain": dict(cfg["bab"]["clip_n_verify"]["clip_interm_domain"]),
        "cuts_enabled": cfg["bab"]["cut"]["enabled"],
        "biccos_enabled": cfg["bab"]["cut"]["biccos"]["enabled"],
        "complete_verifier_for_root_export": "skip",
    }
    artifact = {
        "schema": SCHEMA,
        "kind": "abc_root",
        "pins": {
            "alpha_beta_crown_git": ABC_SHA,
            "auto_lirpa_git": AUTOLIRPA_SHA,
            "config_sha256": _sha256(config),
            "onnx_sha256": _sha256(onnx),
            "vnnlib_sha256": _sha256(vnnlib),
        },
        "recipe": contract,
        "input": {
            "shape": list(x_lower.shape),
            "lower": _tensor_list(x_lower),
            "upper": _tensor_list(x_upper),
        },
        "specification": {
            "rows": sparse_c,
            "rhs": _tensor_list(rhs),
        },
        "root": {
            "initial_crown_margins": initial_values,
            "effective_margins": effective_values,
            "optimized_margin_finite_rows": sum(
                math.isfinite(value) for value in raw_effective_values
            ),
            "split_layers": layers,
        },
    }
    digest = write_export(args.output.resolve(), artifact)
    print(f"wrote {args.output.resolve()} sha256={digest}")
    return 0


def _bounds_profile(
    ny: Bounds, abc: dict[str, Any], tolerance: float
) -> dict[str, Any]:
    abc_lower = tuple(float(value) for value in abc["lower"])
    abc_upper = tuple(float(value) for value in abc["upper"])
    if len(abc_lower) != ny.size or len(abc_upper) != ny.size:
        raise ParityError(
            f"layer size mismatch: NY={ny.size}, ABC={len(abc_lower)}/{len(abc_upper)}"
        )
    ny_width = tuple(hi - lo for lo, hi in zip(ny.lower, ny.upper))
    abc_width = tuple(hi - lo for lo, hi in zip(abc_lower, abc_upper))
    lower_gain = tuple(a - n for n, a in zip(ny.lower, abc_lower))
    upper_gain = tuple(n - a for n, a in zip(ny.upper, abc_upper))
    nested = sum(
        a_lo >= n_lo - tolerance and a_hi <= n_hi + tolerance
        for n_lo, n_hi, a_lo, a_hi in zip(ny.lower, ny.upper, abc_lower, abc_upper)
    )
    ny_sum = sum(ny_width)
    abc_sum = sum(abc_width)
    return {
        "size": ny.size,
        "ny_width_sum": ny_sum,
        "abc_width_sum": abc_sum,
        "abc_over_ny_width": abc_sum / ny_sum if ny_sum else None,
        "mean_width_reduction": (ny_sum - abc_sum) / ny.size,
        "mean_lower_gain": sum(lower_gain) / ny.size,
        "mean_upper_gain": sum(upper_gain) / ny.size,
        "max_lower_gain": max(lower_gain),
        "max_upper_gain": max(upper_gain),
        "abc_nested_fraction": nested / ny.size,
        "ny_unstable": sum(lo < 0.0 < hi for lo, hi in zip(ny.lower, ny.upper)),
        "abc_unstable": sum(lo < 0.0 < hi for lo, hi in zip(abc_lower, abc_upper)),
    }


def compare_artifacts(
    ny_dump: NyDump,
    margins: Sequence[tuple[float, float, float]],
    abc: dict[str, Any],
    *,
    tolerance: float,
    expected_vnnlib_sha256: str = PROP1761_VNNLIB_SHA256,
) -> dict[str, Any]:
    _validate_export_pins(abc, expected_vnnlib_sha256)

    abc_input = abc["input"]
    if tuple(abc_input["shape"]) != ny_dump.input_bounds.shape:
        raise ParityError(
            f"input shape mismatch NY={ny_dump.input_bounds.shape} "
            f"ABC={tuple(abc_input['shape'])}"
        )
    abc_input_lower = tuple(float(value) for value in abc_input["lower"])
    abc_input_upper = tuple(float(value) for value in abc_input["upper"])
    if len(abc_input_lower) != ny_dump.input_bounds.size:
        raise ParityError("input element count mismatch")
    input_max_abs = max(
        [abs(a - b) for a, b in zip(ny_dump.input_bounds.lower, abc_input_lower)]
        + [abs(a - b) for a, b in zip(ny_dump.input_bounds.upper, abc_input_upper)]
    )
    if input_max_abs > tolerance:
        raise ParityError(
            f"input boxes differ by {input_max_abs}, tolerance is {tolerance}"
        )

    abc_layers = abc["root"]["split_layers"]
    if len(ny_dump.relu_map) != len(abc_layers):
        raise ParityError(
            f"ReLU layer count mismatch NY={len(ny_dump.relu_map)} "
            f"ABC={len(abc_layers)}"
        )
    layer_profiles = []
    for ordinal, ((relu, preactivation), abc_layer) in enumerate(
        zip(ny_dump.relu_map, abc_layers)
    ):
        if int(abc_layer["ordinal"]) != ordinal:
            raise ParityError("ABC split-layer ordinals are not contiguous")
        ny_bounds = ny_dump.nodes[preactivation]
        abc_shape = tuple(int(dim) for dim in abc_layer["shape"])
        if _product(abc_shape) != ny_bounds.size:
            raise ParityError(
                f"layer {ordinal} size mismatch: NY {preactivation}={ny_bounds.shape}, "
                f"ABC {abc_layer['preactivation']}={abc_shape}"
            )
        profile = _bounds_profile(ny_bounds, abc_layer, tolerance)
        profile.update(
            {
                "ordinal": ordinal,
                "ny_relu": relu,
                "ny_preactivation": preactivation,
                "ny_shape": list(ny_bounds.shape),
                "abc_preactivation": abc_layer["preactivation"],
                "abc_shape": list(abc_shape),
            }
        )
        layer_profiles.append(profile)

    initial_official = tuple(
        float(value) for value in abc["root"]["initial_crown_margins"]
    )
    official = tuple(float(value) for value in abc["root"]["effective_margins"])
    if len(official) != len(margins):
        raise ParityError(
            f"margin count mismatch NY={len(margins)} ABC={len(official)}"
        )
    rhs = abc["specification"]["rhs"]
    rows = abc["specification"]["rows"]
    if len(rows) != len(margins) or len(rhs) != len(margins):
        raise ParityError("official C/rhs row count is inconsistent")
    ny_lower = tuple(item[0] - item[2] for item in margins)
    initial_margin_gain = tuple(a - n for n, a in zip(ny_lower, initial_official))
    margin_gain = tuple(a - n for n, a in zip(ny_lower, official))
    ny_binding = min(range(len(ny_lower)), key=ny_lower.__getitem__)
    initial_binding = min(
        range(len(initial_official)), key=initial_official.__getitem__
    )
    abc_binding = min(range(len(official)), key=official.__getitem__)
    return {
        "schema": SCHEMA,
        "kind": "comparison",
        "pins": dict(abc["pins"]),
        "input_max_abs_difference": input_max_abs,
        "recipe": abc["recipe"],
        "layers": layer_profiles,
        "margins": {
            "count": len(margins),
            "ny_verified_count": sum(value > 0.0 for value in ny_lower),
            "abc_initial_crown_verified_count": sum(
                value > 0.0 for value in initial_official
            ),
            "abc_verified_count": sum(value > 0.0 for value in official),
            "ny_binding_objective": ny_binding,
            "ny_binding_lower": ny_lower[ny_binding],
            "abc_initial_crown_binding_objective": initial_binding,
            "abc_initial_crown_binding_lower": initial_official[initial_binding],
            "abc_binding_objective": abc_binding,
            "abc_binding_lower": official[abc_binding],
            "abc_minus_ny_binding_objective_gain": margin_gain[abc_binding],
            "max_abc_minus_ny_gain": max(margin_gain),
            "max_abc_initial_crown_minus_ny_gain": max(initial_margin_gain),
            "per_objective": [
                {
                    "objective": index,
                    "c": rows[index],
                    "rhs": rhs[index],
                    "ny_lower": ny_lower[index],
                    "abc_initial_crown_lower": initial_official[index],
                    "abc_initial_crown_minus_ny": initial_margin_gain[index],
                    "abc_lower": official[index],
                    "abc_minus_ny": margin_gain[index],
                }
                for index in range(len(margins))
            ],
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    margins = result["margins"]
    lines = [
        "# CIFAR100 NY / alpha-beta-CROWN bound parity",
        "",
        f"Input max absolute difference: `{result['input_max_abs_difference']:.9g}`.",
        "",
        (
            f"Root margins: NY verifies **{margins['ny_verified_count']}/{margins['count']}**; "
            f"official pre-alpha CROWN verifies **{margins['abc_initial_crown_verified_count']}/{margins['count']}**; "
            f"official effective alpha-CROWN verifies **{margins['abc_verified_count']}/{margins['count']}**. "
            f"NY binding row {margins['ny_binding_objective']} is "
            f"`{margins['ny_binding_lower']:+.7f}`; official binding row "
            f"{margins['abc_binding_objective']} is "
            f"`{margins['abc_binding_lower']:+.7f}`."
        ),
        "",
        "| # | NY preactivation | ABC preactivation | ABC/NY width | mean width reduction | ABC nested | unstable NY→ABC |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for layer in result["layers"]:
        ratio = layer["abc_over_ny_width"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.6f}"
        lines.append(
            f"| {layer['ordinal']} | `{layer['ny_preactivation']}` | "
            f"`{layer['abc_preactivation']}` | {ratio_text} | "
            f"{layer['mean_width_reduction']:+.7f} | "
            f"{layer['abc_nested_fraction']:.3%} | "
            f"{layer['ny_unstable']}→{layer['abc_unstable']} |"
        )
    lines.extend(
        [
            "",
            "The official recipe has multi-neuron, MILP, all-node LP, cuts, BICCOS, "
            "and optimized intermediate bounds disabled; `clip_interm_domain` is enabled. "
            "A positive official result therefore falsifies any claim that multi-neuron/LP "
            "is required for this instance.",
            "",
        ]
    )
    return "\n".join(lines)


def build_replay_plan(
    root: NyDump,
    child: NyDump,
    abc: dict[str, Any],
    *,
    expected_vnnlib_sha256: str = PROP1761_VNNLIB_SHA256,
) -> dict[str, Any]:
    _validate_export_pins(abc, expected_vnnlib_sha256)
    if not root.relu_map:
        raise ParityError("root dump needs RELUMAP for replay mapping")
    if child.child is None:
        raise ParityError("child dump header has no split premises")
    if child.input_bounds != root.input_bounds:
        raise ParityError("root/child input bounds differ")
    abc_layers = abc["root"]["split_layers"]
    if len(root.relu_map) != len(abc_layers):
        raise ParityError("root/ABC ReLU layer count mismatch")
    row_count = len(abc["specification"]["rows"])
    if child.child.binding_objective >= row_count:
        raise ParityError(
            f"child binding objective {child.child.binding_objective} exceeds "
            f"official row count {row_count}"
        )
    mapping = {
        relu: (ordinal, preactivation, abc_layers[ordinal]["preactivation"])
        for ordinal, (relu, preactivation) in enumerate(root.relu_map)
    }
    mapped = []
    for premise in child.child.premises:
        if premise.relu not in mapping:
            raise ParityError(f"split premise uses unknown NY ReLU {premise.relu!r}")
        ordinal, ny_pre, abc_pre = mapping[premise.relu]
        size = root.nodes[ny_pre].size
        if premise.neuron >= size:
            raise ParityError(
                f"split neuron {premise.relu}:{premise.neuron} exceeds size {size}"
            )
        if (
            ny_pre not in child.nodes
            or child.nodes[ny_pre].shape != root.nodes[ny_pre].shape
        ):
            raise ParityError(f"child has no shape-compatible node {ny_pre!r}")
        child_bounds = child.nodes[ny_pre]
        clamped_endpoint = (
            child_bounds.lower[premise.neuron]
            if premise.active
            else child_bounds.upper[premise.neuron]
        )
        if (premise.active and clamped_endpoint < 0.0) or (
            not premise.active and clamped_endpoint > 0.0
        ):
            state = "active lower" if premise.active else "inactive upper"
            raise ParityError(
                f"child {premise.relu}:{premise.neuron} {state} endpoint "
                f"is not clamped at zero: {clamped_endpoint}"
            )
        mapped.append(
            {
                "ny_relu": premise.relu,
                "ny_preactivation": ny_pre,
                "abc_layer_index": ordinal,
                "abc_preactivation": abc_pre,
                "neuron": premise.neuron,
                "state": "active" if premise.active else "inactive",
                "abc_history_sign": 1.0 if premise.active else -1.0,
                "abc_bound_clamp": "lower=0" if premise.active else "upper=0",
            }
        )
    return {
        "schema": SCHEMA,
        "kind": "abc_child_replay_plan",
        "pins": dict(abc["pins"]),
        "ny_child": {
            "depth": child.child.depth,
            "binding_objective": child.child.binding_objective,
            "binding_lower": child.child.binding_lower,
        },
        "mapped_premises": mapped,
        "official_protocol": [
            "Run the pinned abc root and retain SpecHandler.post_process reference_dict.",
            "Call SpecHandler.expand_intermediate(reference_dict) exactly as complete verification does.",
            "For each mapped premise, append (neuron, sign, bias=0) to history for abc_preactivation and clamp that node's lower or upper bound to zero.",
            "Initialize empty beta values for the injected history; retain root alpha values and the exact singleton binding C/rhs row.",
            "Call activation_split.update_bounds_pre/core/post with fix_interm_bounds=True, enable_clip_domains=True, enable_decision_precompute=False, and beta-crown iteration=10.",
            "Export the post-clip intermediate boxes and final lower margin, then compare them with the NY child dump using this tool's layer-order mapping.",
        ],
        "official_source_anchors": [
            "complete_verifier/domain_updater.py: DomainUpdater._set_history_and_bounds",
            "complete_verifier/activation_split/update_bounds_phases.py: update_bounds_pre/core/post",
            "complete_verifier/domain_clipper.py: DomainClipper.optimize_interm_bounds",
        ],
    }


def _path(value: str) -> Path:
    return Path(value).expanduser()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    root = subparsers.add_parser("abc-root", help="export the pinned official root")
    root.add_argument("--abc-repo", type=_path, required=True)
    root.add_argument("--config", type=_path, required=True)
    root.add_argument("--onnx", type=_path, required=True)
    root.add_argument("--vnnlib", type=_path, required=True)
    root.add_argument(
        "--expected-vnnlib-sha256",
        default=PROP1761_VNNLIB_SHA256,
        help=(
            "required content identity for the property; defaults to the pinned "
            "prop1761 artifact"
        ),
    )
    root.add_argument("--output", type=_path, required=True)

    compare = subparsers.add_parser("compare", help="compare NY root to ABC export")
    compare.add_argument("--ny-dump", type=_path, required=True)
    compare.add_argument("--ny-margins", type=_path, required=True)
    compare.add_argument("--abc-export", type=_path, required=True)
    compare.add_argument(
        "--expected-vnnlib-sha256",
        default=PROP1761_VNNLIB_SHA256,
        help=(
            "required content identity for the exported property; defaults to "
            "the pinned prop1761 artifact"
        ),
    )
    compare.add_argument("--json-output", type=_path)
    compare.add_argument("--markdown-output", type=_path)
    compare.add_argument("--tolerance", type=float, default=1e-6)

    replay = subparsers.add_parser(
        "replay-plan",
        help="translate an NY split history into official node coordinates",
    )
    replay.add_argument("--ny-root", type=_path, required=True)
    replay.add_argument("--ny-child", type=_path, required=True)
    replay.add_argument("--abc-export", type=_path, required=True)
    replay.add_argument(
        "--expected-vnnlib-sha256",
        default=PROP1761_VNNLIB_SHA256,
        help=(
            "required content identity for the exported property; defaults to "
            "the pinned prop1761 artifact"
        ),
    )
    replay.add_argument("--output", type=_path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "abc-root":
        return export_abc_root(args)
    if args.command == "compare":
        if not math.isfinite(args.tolerance) or args.tolerance < 0.0:
            raise ParityError("tolerance must be finite and non-negative")
        result = compare_artifacts(
            parse_ny_dump(args.ny_dump),
            parse_ny_margins(args.ny_margins),
            read_export(args.abc_export),
            tolerance=args.tolerance,
            expected_vnnlib_sha256=args.expected_vnnlib_sha256,
        )
        markdown = render_markdown(result)
        if args.json_output:
            args.json_output.write_bytes(_canonical_json_bytes(result))
        if args.markdown_output:
            args.markdown_output.write_text(markdown, encoding="utf-8")
        if not args.json_output and not args.markdown_output:
            print(markdown, end="")
        return 0
    if args.command == "replay-plan":
        plan = build_replay_plan(
            parse_ny_dump(args.ny_root),
            parse_ny_dump(args.ny_child),
            read_export(args.abc_export),
            expected_vnnlib_sha256=args.expected_vnnlib_sha256,
        )
        payload = json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ParityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
