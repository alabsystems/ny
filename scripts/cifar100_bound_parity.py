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
import copy
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
NY_CHILD_BOUND_SEMANTICS = "effective_node_bounds_plus_separate_split_history"
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


def _git_status_porcelain(path: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _gitlink_commit(path: Path, entry: str) -> str:
    output = subprocess.run(
        ["git", "ls-tree", "HEAD", "--", entry],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n")
    metadata, separator, actual_entry = output.partition("\t")
    fields = metadata.split()
    if (
        separator != "\t"
        or actual_entry != entry
        or len(fields) != 3
        or fields[:2] != ["160000", "commit"]
        or not re.fullmatch(r"[0-9a-f]{40}", fields[2])
    ):
        raise ParityError(f"{entry!r} is not an exact gitlink in {path}")
    return fields[2]


def _require_clean_git_tree(path: Path, label: str) -> None:
    dirty = _git_status_porcelain(path).splitlines()
    if not dirty:
        return
    preview = ", ".join(dirty[:5])
    if len(dirty) > 5:
        preview += f", ... ({len(dirty)} entries total)"
    raise ParityError(
        f"{label} checkout must be clean for a pinned replay ({path}): {preview}"
    )


def _require_pinned_abc_checkout(abc_repo: Path) -> None:
    auto_lirpa_repo = abc_repo / "auto_LiRPA"
    if _git_head(abc_repo) != ABC_SHA:
        raise ParityError(f"alpha-beta-CROWN must be pinned to {ABC_SHA}")
    if _gitlink_commit(abc_repo, "auto_LiRPA") != AUTOLIRPA_SHA:
        raise ParityError(
            f"alpha-beta-CROWN auto_LiRPA gitlink must be pinned to {AUTOLIRPA_SHA}"
        )
    if _git_head(auto_lirpa_repo) != AUTOLIRPA_SHA:
        raise ParityError(f"auto_LiRPA must be pinned to {AUTOLIRPA_SHA}")
    _require_clean_git_tree(abc_repo, "alpha-beta-CROWN")
    _require_clean_git_tree(auto_lirpa_repo, "auto_LiRPA")


def _require_module_under(module: Any, expected_root: Path, label: str) -> None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise ParityError(f"{label} import has no filesystem identity")
    resolved = Path(module_file).resolve()
    root = expected_root.resolve()
    if not resolved.is_relative_to(root):
        raise ParityError(
            f"{label} import resolved outside the pinned checkout: {resolved}"
        )


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


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ParityError(f"{label} must be a finite number")
    return result


def _plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ParityError(f"{label} must be an integer >= {minimum}")
    return value


def abc_iteration_for_optimizer_updates(updates: Any) -> int:
    """Translate step count to this pin's nonzero bound-evaluation count."""

    return _plain_int(updates, "beta optimizer updates") + 1


def _number_list(value: Any, label: str, *, size: int | None = None) -> list[float]:
    if not isinstance(value, list):
        raise ParityError(f"{label} must be an array")
    result = [
        _finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)
    ]
    if size is not None and len(result) != size:
        raise ParityError(f"{label} has {len(result)} values; expected {size}")
    return result


def _validate_abc_root_structure(
    abc: dict[str, Any],
    expected_vnnlib_sha256: str = PROP1761_VNNLIB_SHA256,
) -> None:
    """Validate every root field consumed by comparison or child replay."""

    _validate_export_pins(abc, expected_vnnlib_sha256)
    abc_input = abc.get("input")
    if not isinstance(abc_input, dict):
        raise ParityError("ABC export input must be an object")
    raw_shape = abc_input.get("shape")
    if not isinstance(raw_shape, list):
        raise ParityError("ABC export input shape must be an array")
    shape = tuple(
        _plain_int(dim, f"ABC export input shape[{index}]", minimum=1)
        for index, dim in enumerate(raw_shape)
    )
    if not shape:
        raise ParityError("ABC export input shape must not be empty")
    input_size = _product(shape)
    lower = _number_list(
        abc_input.get("lower"), "ABC export input lower", size=input_size
    )
    upper = _number_list(
        abc_input.get("upper"), "ABC export input upper", size=input_size
    )
    if any(lo > hi for lo, hi in zip(lower, upper)):
        raise ParityError("ABC export input contains reversed bounds")

    specification = abc.get("specification")
    if not isinstance(specification, dict):
        raise ParityError("ABC export specification must be an object")
    rows = specification.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ParityError("ABC export specification rows must be a non-empty array")
    _number_list(
        specification.get("rhs"), "ABC export specification rhs", size=len(rows)
    )
    for row_index, row in enumerate(rows):
        if not isinstance(row, list) or not row:
            raise ParityError(f"ABC export specification row {row_index} is empty")
        seen_outputs: set[int] = set()
        for term_index, term in enumerate(row):
            if not isinstance(term, list) or len(term) != 2:
                raise ParityError(
                    f"ABC export specification row {row_index} term {term_index} "
                    "must be [output, coefficient]"
                )
            output = _plain_int(
                term[0],
                f"ABC export specification row {row_index} output",
            )
            coefficient = _finite_number(
                term[1],
                f"ABC export specification row {row_index} coefficient",
            )
            if output in seen_outputs or coefficient == 0.0:
                raise ParityError(
                    f"ABC export specification row {row_index} has a duplicate "
                    "output or zero coefficient"
                )
            seen_outputs.add(output)

    root = abc.get("root")
    if not isinstance(root, dict):
        raise ParityError("ABC export root must be an object")
    _number_list(
        root.get("initial_crown_margins"),
        "ABC export initial CROWN margins",
        size=len(rows),
    )
    _number_list(
        root.get("effective_margins"),
        "ABC export effective margins",
        size=len(rows),
    )
    layers = root.get("split_layers")
    if not isinstance(layers, list) or not layers:
        raise ParityError("ABC export split_layers must be a non-empty array")
    seen_names: set[str] = set()
    for ordinal, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ParityError(f"ABC export split layer {ordinal} must be an object")
        if layer.get("ordinal") != ordinal:
            raise ParityError("ABC export split-layer ordinals are not contiguous")
        name = layer.get("preactivation")
        if not isinstance(name, str) or not name or name in seen_names:
            raise ParityError(
                f"ABC export split layer {ordinal} has an empty/duplicate name"
            )
        seen_names.add(name)
        raw_layer_shape = layer.get("shape")
        if not isinstance(raw_layer_shape, list):
            raise ParityError(
                f"ABC export split layer {ordinal} shape must be an array"
            )
        layer_shape = tuple(
            _plain_int(
                dim,
                f"ABC export split layer {ordinal} shape[{index}]",
                minimum=1,
            )
            for index, dim in enumerate(raw_layer_shape)
        )
        if not layer_shape:
            raise ParityError(f"ABC export split layer {ordinal} shape is empty")
        size = _product(layer_shape)
        layer_lower = _number_list(
            layer.get("lower"),
            f"ABC export split layer {ordinal} lower",
            size=size,
        )
        layer_upper = _number_list(
            layer.get("upper"),
            f"ABC export split layer {ordinal} upper",
            size=size,
        )
        if any(lo > hi for lo, hi in zip(layer_lower, layer_upper)):
            raise ParityError(f"ABC export split layer {ordinal} has reversed bounds")


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
    _require_pinned_abc_checkout(abc_repo)
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
    import auto_LiRPA  # noqa: N813, PLC0415
    from abcrown import ABCROWN  # noqa: PLC0415

    _require_module_under(
        auto_LiRPA, abc_repo / "auto_LiRPA" / "auto_LiRPA", "auto_LiRPA"
    )

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
    _validate_abc_root_structure(abc, expected_vnnlib_sha256)

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
    _validate_abc_root_structure(abc, expected_vnnlib_sha256)
    if not root.relu_map:
        raise ParityError("root dump needs RELUMAP for replay mapping")
    if child.child is None:
        raise ParityError("child dump header has no split premises")
    if child.input_bounds != root.input_bounds:
        raise ParityError("root/child input bounds differ")
    abc_layers = abc["root"]["split_layers"]
    if len(root.relu_map) != len(abc_layers):
        raise ParityError("root/ABC ReLU layer count mismatch")
    for ordinal, (_, preactivation) in enumerate(root.relu_map):
        ny_size = root.nodes[preactivation].size
        abc_size = _product(tuple(abc_layers[ordinal]["shape"]))
        if ny_size != abc_size:
            raise ParityError(
                f"root/ABC ReLU layer {ordinal} size mismatch: {ny_size} != {abc_size}"
            )
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
        root_bounds = root.nodes[ny_pre]
        size = root_bounds.size
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
        root_interval = [
            root_bounds.lower[premise.neuron],
            root_bounds.upper[premise.neuron],
        ]
        child_interval = [
            child_bounds.lower[premise.neuron],
            child_bounds.upper[premise.neuron],
        ]
        abc_interval = [
            abc_layers[ordinal]["lower"][premise.neuron],
            abc_layers[ordinal]["upper"][premise.neuron],
        ]
        if not root_interval[0] < 0.0 < root_interval[1]:
            raise ParityError(
                f"split premise {premise.relu}:{premise.neuron} is not unstable "
                f"in the NY root: [{root_interval[0]}, {root_interval[1]}]"
            )
        if not abc_interval[0] < 0.0 < abc_interval[1]:
            raise ParityError(
                f"split premise {premise.relu}:{premise.neuron} maps to an "
                f"official root coordinate that is not unstable: "
                f"[{abc_interval[0]}, {abc_interval[1]}]"
            )
        if (premise.active and child_interval[1] < 0.0) or (
            not premise.active and child_interval[0] > 0.0
        ):
            state = "active" if premise.active else "inactive"
            raise ParityError(
                f"child bounds for {premise.relu}:{premise.neuron} conflict "
                f"with its {state} split premise: "
                f"[{child_interval[0]}, {child_interval[1]}]"
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
                "ny_root_interval": root_interval,
                "ny_child_raw_interval": child_interval,
                "abc_root_interval": abc_interval,
            }
        )
    plan = {
        "schema": SCHEMA,
        "kind": "abc_child_replay_plan",
        "pins": dict(abc["pins"]),
        "ny_child": {
            "depth": child.child.depth,
            "binding_objective": child.child.binding_objective,
            "binding_lower": child.child.binding_lower,
            "selection_scope": "independent_child_at_recorded_depth",
            "lineage_relationship": "not_encoded",
            "node_bounds_semantics": NY_CHILD_BOUND_SEMANTICS,
        },
        "mapped_premises": mapped,
        "official_protocol": [
            "Run the pinned abc root and retain SpecHandler.post_process reference_dict.",
            "Call SpecHandler.expand_intermediate(reference_dict) exactly as complete verification does.",
            "Treat the NY child node bounds as effective raw boxes with split premises carried separately; never require premise endpoints to be pre-clamped in the dump.",
            "For each mapped premise, append (neuron, sign, bias=0) to history for abc_preactivation, reconstruct the official intermediate bounds, and explicitly apply the corresponding exact zero endpoint clamp.",
            "Validate every applied endpoint again after update_bounds_pre before entering the bound optimizer.",
            "Initialize empty beta values for the injected history; retain root alpha values and the exact singleton binding C/rhs row.",
            "Call activation_split.update_bounds_pre/core/post with fix_interm_bounds=True and enable_decision_precompute=False.",
            "Export the post-clip intermediate boxes and final lower margin, then compare them with the NY child dump using this tool's layer-order mapping.",
        ],
        "official_source_anchors": [
            "complete_verifier/domain_updater.py: DomainUpdater._set_history_and_bounds",
            "complete_verifier/state/intermediate_bounds.py: WorkingIntermBoundsInfo.from_histories",
            "complete_verifier/activation_split/update_bounds_phases.py: update_bounds_pre/core/post",
            "complete_verifier/domain_clipper.py: DomainClipper.optimize_interm_bounds",
        ],
    }
    validate_replay_plan(plan, abc, expected_vnnlib_sha256=expected_vnnlib_sha256)
    return plan


def validate_replay_plan(
    plan: dict[str, Any],
    abc: dict[str, Any],
    *,
    expected_vnnlib_sha256: str = PROP1761_VNNLIB_SHA256,
) -> None:
    """Fail closed before a malformed plan can reach the official runtime."""

    _validate_abc_root_structure(abc, expected_vnnlib_sha256)
    if plan.get("schema") != SCHEMA or plan.get("kind") != "abc_child_replay_plan":
        raise ParityError("unsupported ABC child replay plan")
    if plan.get("pins") != abc["pins"]:
        raise ParityError("replay plan pins do not exactly match the ABC root export")

    child = plan.get("ny_child")
    if not isinstance(child, dict):
        raise ParityError("replay plan ny_child must be an object")
    depth = _plain_int(child.get("depth"), "replay plan child depth", minimum=1)
    objective = _plain_int(
        child.get("binding_objective"), "replay plan binding objective"
    )
    _finite_number(child.get("binding_lower"), "replay plan binding lower")
    if objective >= len(abc["specification"]["rows"]):
        raise ParityError("replay plan binding objective exceeds official row count")
    if child.get("selection_scope") != "independent_child_at_recorded_depth":
        raise ParityError("replay plan must identify the NY dump selection scope")
    if child.get("lineage_relationship") != "not_encoded":
        raise ParityError("replay plan must not claim an unrecorded parent lineage")
    if child.get("node_bounds_semantics") != NY_CHILD_BOUND_SEMANTICS:
        raise ParityError(
            "replay plan must identify NY child bounds as separate from split history"
        )

    mapped = plan.get("mapped_premises")
    if not isinstance(mapped, list) or len(mapped) != depth:
        raise ParityError("replay plan premise count does not match child depth")
    layers = abc["root"]["split_layers"]
    seen: set[tuple[int, int]] = set()
    for premise_index, premise in enumerate(mapped):
        if not isinstance(premise, dict):
            raise ParityError(f"replay plan premise {premise_index} must be an object")
        layer_index = _plain_int(
            premise.get("abc_layer_index"),
            f"replay plan premise {premise_index} layer",
        )
        if layer_index >= len(layers):
            raise ParityError(
                f"replay plan premise {premise_index} layer is out of range"
            )
        layer = layers[layer_index]
        if premise.get("abc_preactivation") != layer["preactivation"]:
            raise ParityError(
                f"replay plan premise {premise_index} ABC layer name mismatch"
            )
        if not isinstance(premise.get("ny_relu"), str) or not premise["ny_relu"]:
            raise ParityError(
                f"replay plan premise {premise_index} has no NY ReLU name"
            )
        if (
            not isinstance(premise.get("ny_preactivation"), str)
            or not premise["ny_preactivation"]
        ):
            raise ParityError(
                f"replay plan premise {premise_index} has no NY preactivation"
            )
        neuron = _plain_int(
            premise.get("neuron"),
            f"replay plan premise {premise_index} neuron",
        )
        if neuron >= _product(tuple(layer["shape"])):
            raise ParityError(
                f"replay plan premise {premise_index} neuron is out of range"
            )
        coordinate = (layer_index, neuron)
        if coordinate in seen:
            raise ParityError(
                f"replay plan premise {premise_index} duplicates an ABC coordinate"
            )
        seen.add(coordinate)
        state = premise.get("state")
        expected = {
            "active": (1.0, "lower=0"),
            "inactive": (-1.0, "upper=0"),
        }.get(state)
        if expected is None:
            raise ParityError(f"replay plan premise {premise_index} has invalid state")
        sign = _finite_number(
            premise.get("abc_history_sign"),
            f"replay plan premise {premise_index} sign",
        )
        if sign != expected[0] or premise.get("abc_bound_clamp") != expected[1]:
            raise ParityError(
                f"replay plan premise {premise_index} has inconsistent state metadata"
            )
        ny_root_interval = _number_list(
            premise.get("ny_root_interval"),
            f"replay plan premise {premise_index} NY root interval",
            size=2,
        )
        child_interval = _number_list(
            premise.get("ny_child_raw_interval"),
            f"replay plan premise {premise_index} NY child interval",
            size=2,
        )
        abc_interval = _number_list(
            premise.get("abc_root_interval"),
            f"replay plan premise {premise_index} ABC root interval",
            size=2,
        )
        if not ny_root_interval[0] < 0.0 < ny_root_interval[1]:
            raise ParityError(
                f"replay plan premise {premise_index} is not NY-root unstable"
            )
        expected_abc_interval = [
            layer["lower"][neuron],
            layer["upper"][neuron],
        ]
        if abc_interval != expected_abc_interval:
            raise ParityError(
                f"replay plan premise {premise_index} ABC root interval mismatch"
            )
        if not abc_interval[0] < 0.0 < abc_interval[1]:
            raise ParityError(
                f"replay plan premise {premise_index} is not ABC-root unstable"
            )
        if child_interval[0] > child_interval[1]:
            raise ParityError(
                f"replay plan premise {premise_index} has reversed NY child bounds"
            )
        if (state == "active" and child_interval[1] < 0.0) or (
            state == "inactive" and child_interval[0] > 0.0
        ):
            raise ParityError(
                f"replay plan premise {premise_index} conflicts with NY child bounds"
            )


def build_abc_history(
    plan: dict[str, Any], split_node_names: Sequence[str]
) -> dict[str, tuple[list[int], list[float], list[float], list[float], list[float]]]:
    """Translate a validated replay plan into ABC's legacy history container."""

    if len(set(split_node_names)) != len(split_node_names):
        raise ParityError("official split-node names are not unique")
    history = {name: ([], [], [], [], []) for name in split_node_names}
    for premise in plan["mapped_premises"]:
        name = premise["abc_preactivation"]
        if name not in history:
            raise ParityError(f"replay history references unknown ABC node {name!r}")
        layer_history = history[name]
        layer_history[0].append(premise["neuron"])
        layer_history[1].append(premise["abc_history_sign"])
        layer_history[2].append(0.0)
    return history


def _validate_live_history_coordinates(
    plan: dict[str, Any],
    unstable_mask: dict[str, Any],
    clipper_mapping: dict[str, dict[int, int]],
) -> None:
    """Require each injected decision to be a genuine live-root split."""

    for premise_index, premise in enumerate(plan["mapped_premises"]):
        name = premise["abc_preactivation"]
        neuron = premise["neuron"]
        mask = unstable_mask.get(name)
        if mask is None or mask.shape[0] != 1:
            raise ParityError(
                f"replay premise {premise_index} node {name!r} is not "
                "live-root unstable"
            )
        flat_mask = mask.reshape(1, -1)
        if neuron >= flat_mask.shape[1] or not bool(flat_mask[0, neuron].item()):
            raise ParityError(
                f"replay premise {premise_index} coordinate {name}:{neuron} "
                "is not live-root unstable"
            )
        if name not in clipper_mapping or neuron not in clipper_mapping[name]:
            raise ParityError(
                f"replay premise {premise_index} coordinate {name}:{neuron} "
                "is absent from the official clipper mapping"
            )


def _replay_bound_coordinate(
    lower_bounds: dict[str, Any],
    upper_bounds: dict[str, Any],
    premise: dict[str, Any],
    torch: Any,
    *,
    context: str,
) -> tuple[Any, Any, int, float, float]:
    """Resolve one singleton split coordinate and validate its tensor identity."""

    name = premise["abc_preactivation"]
    if name not in lower_bounds or name not in upper_bounds:
        raise ParityError(f"{context} is missing replay split node {name!r}")
    lower = lower_bounds[name]
    upper = upper_bounds[name]
    if not torch.is_tensor(lower) or not torch.is_tensor(upper):
        raise ParityError(f"{context} split bounds for {name!r} are not tensors")
    if (
        lower is upper
        or lower.shape != upper.shape
        or len(lower.shape) < 2
        or lower.shape[0] != 1
        or lower.numel() != upper.numel()
    ):
        raise ParityError(
            f"{context} split bounds for {name!r} are not matching singleton tensors"
        )
    if (
        not lower.is_floating_point()
        or not upper.is_floating_point()
        or lower.dtype != upper.dtype
        or lower.device != upper.device
        or bool(getattr(lower, "requires_grad", False))
        or bool(getattr(upper, "requires_grad", False))
    ):
        raise ParityError(
            f"{context} split bounds for {name!r} have an unsupported tensor identity"
        )
    flat_lower = lower.reshape(1, -1)
    flat_upper = upper.reshape(1, -1)
    neuron = premise["neuron"]
    if neuron >= flat_lower.shape[1]:
        raise ParityError(
            f"{context} replay coordinate {name}:{neuron} is out of range"
        )
    lower_value = float(flat_lower[0, neuron].item())
    upper_value = float(flat_upper[0, neuron].item())
    if not math.isfinite(lower_value) or not math.isfinite(upper_value):
        raise ParityError(f"{context} replay coordinate {name}:{neuron} is non-finite")
    if lower_value > upper_value:
        raise ParityError(
            f"{context} replay coordinate {name}:{neuron} has reversed bounds"
        )
    return flat_lower, flat_upper, neuron, lower_value, upper_value


def _require_replay_history_clamps(
    lower_bounds: dict[str, Any],
    upper_bounds: dict[str, Any],
    plan: dict[str, Any],
    torch: Any,
    *,
    context: str,
) -> None:
    """Require exact official zero endpoints for every recorded ReLU premise."""

    for premise_index, premise in enumerate(plan["mapped_premises"]):
        _, _, _, lower, upper = _replay_bound_coordinate(
            lower_bounds, upper_bounds, premise, torch, context=context
        )
        if premise["state"] == "active":
            valid = lower == 0.0 and upper > 0.0
        else:
            valid = lower < 0.0 and upper == 0.0
        if not valid:
            raise ParityError(
                f"{context} did not preserve exact {premise['state']} clamp "
                f"for replay premise {premise_index}: [{lower}, {upper}]"
            )


def _apply_replay_history_clamps(
    lower_bounds: dict[str, Any],
    upper_bounds: dict[str, Any],
    plan: dict[str, Any],
    torch: Any,
) -> list[dict[str, Any]]:
    """Apply the exact endpoint update used by ABC's ReLU DomainUpdater.

    The NY child dump deliberately stores effective node boxes and its split
    history separately.  This function intersects the reconstructed official
    singleton with that validated history.  It refuses any update that would
    loosen a bound or apply a premise to an already stable/conflicting box.
    """

    applied = []
    for premise_index, premise in enumerate(plan["mapped_premises"]):
        flat_lower, flat_upper, neuron, lower, upper = _replay_bound_coordinate(
            lower_bounds,
            upper_bounds,
            premise,
            torch,
            context="official reconstructed domain",
        )
        state = premise["state"]
        if state == "active":
            if lower > 0.0 or upper <= 0.0:
                raise ParityError(
                    f"official reconstructed domain cannot apply active replay "
                    f"premise {premise_index}: [{lower}, {upper}]"
                )
            flat_lower[0, neuron] = 0.0
        else:
            if lower >= 0.0 or upper < 0.0:
                raise ParityError(
                    f"official reconstructed domain cannot apply inactive replay "
                    f"premise {premise_index}: [{lower}, {upper}]"
                )
            flat_upper[0, neuron] = 0.0
        after_lower = float(flat_lower[0, neuron].item())
        after_upper = float(flat_upper[0, neuron].item())
        if state == "active":
            unchanged = after_upper == upper
        else:
            unchanged = after_lower == lower
        if not unchanged:
            raise ParityError(
                f"official replay premise {premise_index} modified both endpoints"
            )
        applied.append(
            {
                "abc_preactivation": premise["abc_preactivation"],
                "neuron": neuron,
                "state": state,
                "before": [lower, upper],
                "after": [after_lower, after_upper],
            }
        )
    if len(applied) != plan["ny_child"]["depth"]:
        raise ParityError(
            "applied official replay clamp count does not match the NY child depth"
        )
    _require_replay_history_clamps(
        lower_bounds,
        upper_bounds,
        plan,
        torch,
        context="official reconstructed domain",
    )
    return applied


def _tensor_manifest(value: Any, torch: Any) -> dict[str, Any]:
    """Return deterministic metadata and content digests for a tensor tree."""

    tensors: list[dict[str, Any]] = []

    def walk(item: Any, path: str) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().to(device="cpu").contiguous()
            raw = tensor.view(torch.uint8).numpy().tobytes()
            record: dict[str, Any] = {
                "path": path,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            if tensor.numel() and (tensor.is_floating_point() or tensor.is_complex()):
                finite = torch.isfinite(tensor)
                record["finite"] = int(finite.sum().item())
                if bool(finite.any()):
                    finite_values = tensor[finite]
                    record["finite_min"] = float(finite_values.real.min().item())
                    record["finite_max"] = float(finite_values.real.max().item())
            tensors.append(record)
            return
        if isinstance(item, dict):
            for key in sorted(item, key=str):
                walk(item[key], f"{path}.{key}" if path else str(key))
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")
            return
        data = getattr(item, "_data", None)
        if isinstance(data, (dict, list, tuple)):
            walk(data, path)
            return
        sparse_beta_fields = {
            name: getattr(item, name)
            for name in ("val", "loc", "sign", "bias")
            if getattr(item, name, None) is not None
        }
        if sparse_beta_fields:
            walk(sparse_beta_fields, path)

    walk(value, "")
    return {
        "tensor_count": len(tensors),
        "total_numel": sum(item["numel"] for item in tensors),
        "tensors": tensors,
    }


def _require_absent_beta_warm_start(value: Any, context: str) -> None:
    if value is not None:
        raise ParityError(f"{context} unexpectedly contains a beta warm start")


def _require_empty_root_history(
    value: Any, expected_split_nodes: Sequence[str]
) -> None:
    if not isinstance(value, dict) or set(value) != set(expected_split_nodes):
        raise ParityError(
            "official singleton root history does not match its split nodes"
        )
    for name, entry in value.items():
        if not isinstance(entry, (list, tuple)) or len(entry) != 5:
            raise ParityError(
                f"official singleton root history for {name!r} is malformed"
            )
        try:
            nonempty = any(len(component) != 0 for component in entry)
        except TypeError as exc:
            raise ParityError(
                f"official singleton root history for {name!r} is malformed"
            ) from exc
        if nonempty:
            raise ParityError(
                f"official singleton root history for {name!r} contains split decisions"
            )


def _require_singleton_absent_domain_beta(value: Any) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 1 or value[0] is not None:
        raise ParityError(
            "official singleton domain unexpectedly contains a beta warm start"
        )


def _require_zero_sparse_beta_values(
    value: Any, expected_numel: int, torch: Any
) -> None:
    data = getattr(value, "_data", None)
    if not isinstance(data, dict):
        raise ParityError("official beta initialization is not BetaFullData")
    total_numel = 0
    for layer_name, sparse_betas in data.items():
        if not isinstance(sparse_betas, (list, tuple)):
            raise ParityError(
                f"official beta initialization for {layer_name!r} is malformed"
            )
        for sparse_beta in sparse_betas:
            beta = getattr(sparse_beta, "val", None)
            if not torch.is_tensor(beta):
                raise ParityError(
                    f"official beta initialization for {layer_name!r} "
                    "has no value tensor"
                )
            total_numel += beta.numel()
            if beta.numel() and bool((beta.detach() != 0).any().item()):
                raise ParityError(
                    f"official beta initialization for {layer_name!r} is nonzero"
                )
    if total_numel != expected_numel:
        raise ParityError(
            "official beta initialization size does not match the recorded "
            f"history: expected {expected_numel}, got {total_numel}"
        )


def _clone_bound_pairs(bounds: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    snapshot = {}
    for name, pair in bounds.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ParityError(
                f"official bounds for {name!r} are not a lower/upper pair"
            )
        lower, upper = pair
        if lower is None or upper is None:
            raise ParityError(f"official bounds for {name!r} contain None")
        snapshot[name] = (
            lower.detach().to(device="cpu").clone(),
            upper.detach().to(device="cpu").clone(),
        )
    return snapshot


def _export_bound_pairs(
    bounds: dict[str, tuple[Any, Any]],
    split_node_names: Sequence[str],
    torch: Any,
) -> dict[str, Any]:
    missing = [name for name in split_node_names if name not in bounds]
    if missing:
        raise ParityError(
            "official bound snapshot is missing split nodes: " + ", ".join(missing)
        )
    layers = []
    for ordinal, name in enumerate(split_node_names):
        if name not in bounds:
            continue
        lower, upper = bounds[name]
        lower = lower.detach().to(device="cpu")
        upper = upper.detach().to(device="cpu")
        if lower.shape != upper.shape or lower.shape[0] != 1:
            raise ParityError(f"official bounds for {name!r} are not singleton-shaped")
        if not bool(torch.isfinite(lower).all()) or not bool(
            torch.isfinite(upper).all()
        ):
            raise ParityError(
                f"official intermediate bounds for {name!r} are non-finite"
            )
        if bool((lower > upper).any()):
            raise ParityError(
                f"official intermediate bounds for {name!r} are infeasible"
            )
        layers.append(
            {
                "ordinal": ordinal,
                "preactivation": name,
                "shape": list(lower.shape[1:]),
                "lower": _tensor_list(lower),
                "upper": _tensor_list(upper),
            }
        )
    return {
        "layers": layers,
        "present": [item["preactivation"] for item in layers],
        "missing": [],
    }


def _working_bound_pairs(working: Any) -> dict[str, tuple[Any, Any]]:
    return {
        name: (item.lower_bound, item.upper_bound)
        for name, item in working.items()
        if item.lower_bound is not None and item.upper_bound is not None
    }


def _sparse_rows(c_tensor: Any) -> list[list[list[float | int]]]:
    rows = c_tensor.detach().to(device="cpu").reshape(-1, c_tensor.shape[-1])
    return [
        [
            [index, float(value)]
            for index, value in enumerate(row.tolist())
            if value != 0.0
        ]
        for row in rows
    ]


def _validate_live_root(
    abc: dict[str, Any],
    bounded_x: Any,
    c_tensor: Any,
    rhs_tensor: Any,
    model: Any,
    torch: Any,
) -> None:
    x_lower = bounded_x.ptb.x_L[0].detach().to(device="cpu")
    x_upper = bounded_x.ptb.x_U[0].detach().to(device="cpu")
    abc_input = abc["input"]
    if list(x_lower.shape) != abc_input["shape"]:
        raise ParityError("live official input shape does not match the root export")
    expected_lower = torch.tensor(abc_input["lower"], dtype=x_lower.dtype).reshape(
        x_lower.shape
    )
    expected_upper = torch.tensor(abc_input["upper"], dtype=x_upper.dtype).reshape(
        x_upper.shape
    )
    if not torch.equal(x_lower, expected_lower) or not torch.equal(
        x_upper, expected_upper
    ):
        raise ParityError("live official input box does not match the root export")
    if _sparse_rows(c_tensor) != abc["specification"]["rows"]:
        raise ParityError("live official C matrix does not match the root export")
    live_rhs = _tensor_list(rhs_tensor.reshape(-1))
    if live_rhs != abc["specification"]["rhs"]:
        raise ParityError("live official rhs does not match the root export")

    split_nodes = list(model.net.split_nodes)
    layers = abc["root"]["split_layers"]
    if len(split_nodes) != len(layers):
        raise ParityError(
            "live official split-layer count does not match the root export"
        )
    for ordinal, (node, layer) in enumerate(zip(split_nodes, layers)):
        if node.name != layer["preactivation"]:
            raise ParityError(f"live official split-layer {ordinal} name mismatch")
        if (
            node.lower is None
            or node.upper is None
            or list(node.lower.shape[1:]) != layer["shape"]
            or node.lower.shape != node.upper.shape
        ):
            raise ParityError(f"live official split-layer {ordinal} shape mismatch")
        live_lower = node.lower.detach().to(device="cpu")
        live_upper = node.upper.detach().to(device="cpu")
        expected_layer_lower = torch.tensor(
            layer["lower"], dtype=live_lower.dtype
        ).reshape(live_lower.shape)
        expected_layer_upper = torch.tensor(
            layer["upper"], dtype=live_upper.dtype
        ).reshape(live_upper.shape)
        for endpoint, live, expected in (
            ("lower", live_lower, expected_layer_lower),
            ("upper", live_upper, expected_layer_upper),
        ):
            if not torch.allclose(live, expected, rtol=0.0, atol=1e-6):
                maximum = float((live - expected).abs().max().item())
                raise ParityError(
                    f"live official split-layer {ordinal} {endpoint} bounds "
                    f"differ from the root export by {maximum}"
                )


def _require_sole_official_survivor(unverified: Any, binding_objective: int) -> None:
    """Bind replay to the exact singleton objective retained by ABC."""

    if unverified != [binding_objective]:
        raise ParityError(
            "NY binding objective is not the sole official root survivor: "
            f"NY={binding_objective}, official={unverified}"
        )


def build_abc_child_artifact(
    plan: dict[str, Any],
    arm: dict[str, Any],
    measurements: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the diagnostic artifact without granting it verifier authority."""

    required_arm = {
        "beta_enabled",
        "clip_enabled",
        "beta_crown_iteration",
        "beta_optimizer_step_budget",
        "bab_iteration",
    }
    if not required_arm <= set(arm):
        raise ParityError("ABC child arm metadata is incomplete")
    return {
        "schema": SCHEMA,
        "kind": "abc_child",
        "pins": dict(plan["pins"]),
        "diagnostic_only": True,
        "verifier_authority": False,
        "provenance": {
            "initialization": "official_singleton_root_alpha_zero_beta",
            "replay_semantics": "static_root_state_plus_exact_recorded_path",
            "parent_alpha_beta_warm_start": "not_available",
            "ny_dump_selection_scope": plan["ny_child"]["selection_scope"],
            "lineage_relationship": plan["ny_child"]["lineage_relationship"],
            "ny_child_node_bounds": plan["ny_child"]["node_bounds_semantics"],
            "split_history_application": (
                "explicit_validated_zero_clamp_after_official_reconstruction"
            ),
        },
        "arm": dict(arm),
        "ny_child": dict(plan["ny_child"]),
        "mapped_premises": copy.deepcopy(plan["mapped_premises"]),
        "measurements": measurements,
    }


def export_abc_child(args: argparse.Namespace) -> int:
    """Replay one exact NY path through the pinned official child-bound phases."""

    abc_repo = args.abc_repo.resolve()
    config = args.config.resolve()
    onnx = args.onnx.resolve()
    vnnlib = args.vnnlib.resolve()
    expected_vnnlib = _validate_sha256(
        args.expected_vnnlib_sha256, "expected VNN-LIB SHA-256"
    )
    _require_pinned_abc_checkout(abc_repo)
    _require_hash(config, CIFAR100_CONFIG_SHA256, "cifar100 config")
    _require_hash(onnx, CIFAR100_MEDIUM_ONNX_SHA256, "CIFAR100 medium ONNX")
    _require_hash(vnnlib, expected_vnnlib, "CIFAR100 VNN-LIB")
    beta_crown_iteration = abc_iteration_for_optimizer_updates(
        args.beta_optimizer_updates
    )
    if args.bab_iteration < 1:
        raise ParityError("BaB iteration must be >= 1")

    abc = read_export(args.abc_export.resolve())
    root_dump = parse_ny_dump(args.ny_root.resolve())
    child_dump = parse_ny_dump(args.ny_child.resolve())
    plan = build_replay_plan(
        root_dump,
        child_dump,
        abc,
        expected_vnnlib_sha256=expected_vnnlib,
    )
    validate_replay_plan(plan, abc, expected_vnnlib_sha256=expected_vnnlib)

    complete_verifier = abc_repo / "complete_verifier"
    sys.path.insert(0, str(complete_verifier))
    os.chdir(abc_repo)
    # These imports intentionally remain command-local. All other commands are
    # stdlib-only and can inspect artifacts without installing ABC or Torch.
    import arguments  # noqa: PLC0415
    import auto_LiRPA  # noqa: N813, PLC0415
    import torch  # noqa: PLC0415
    from abcrown import ABCROWN  # noqa: PLC0415
    from activation_split.update_bounds_phases import (  # noqa: PLC0415
        update_bounds_core,
        update_bounds_post,
        update_bounds_pre,
    )
    from auto_LiRPA.utils import (  # noqa: PLC0415
        multi_spec_keep_func_all,
        stop_criterion_batch_any,
    )
    from beta_CROWN_solver import LiRPANet  # noqa: PLC0415
    from branching_domains import BatchedDomainList  # noqa: PLC0415
    from prune import prune_alphas  # noqa: PLC0415
    from state import IntermBoundsFactory, WorkingIntermBoundsInfo  # noqa: PLC0415
    from utils import Timer, get_unstable_neurons  # noqa: PLC0415

    _require_module_under(
        auto_LiRPA, abc_repo / "auto_LiRPA" / "auto_LiRPA", "auto_LiRPA"
    )

    captured: dict[str, Any] = {}
    original_build = LiRPANet.build

    def capture_build(model: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        result = original_build(model, *call_args, **call_kwargs)
        if captured:
            raise ParityError("official root unexpectedly called LiRPANet.build twice")
        captured["model"] = model
        captured["result"] = result
        return result

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
    LiRPANet.build = capture_build
    try:
        verifier = ABCROWN(args=cli)
        model = verifier.main()
    finally:
        LiRPANet.build = original_build
    if model is None or captured.get("model") is not model or "result" not in captured:
        raise ParityError("official root state capture failed")

    full_specs = verifier.vnnlib_handler.all_specs.get("cpu")
    full_x, full_c, full_rhs = full_specs[:3]
    _validate_live_root(abc, full_x, full_c, full_rhs, model, torch)

    global_lb, reference = captured["result"]
    if not isinstance(reference, dict) or not reference:
        raise ParityError("official root did not retain reusable reference state")
    spec_handler = verifier.spec_handler_incomplete
    spec_handler.set_unverified_or_mask(global_lb)
    unverified = spec_handler.unverified_or_indices.detach().cpu().tolist()
    objective = plan["ny_child"]["binding_objective"]
    _require_sole_official_survivor(unverified, objective)
    reference = spec_handler.post_process(model, reference)
    spec_handler.expand_intermediate(reference)

    live_specs = verifier.vnnlib_handler.all_specs.get(device=model.device)
    bounded_x, c_tensor, rhs_tensor = live_specs[:3]
    if _sparse_rows(c_tensor) != [abc["specification"]["rows"][objective]]:
        raise ParityError("pruned official C row is not the NY binding objective")
    if _tensor_list(rhs_tensor.reshape(-1)) != [abc["specification"]["rhs"][objective]]:
        raise ParityError("pruned official rhs is not the NY binding objective")

    _require_absent_beta_warm_start(
        reference.get("refined_betas"), "official retained root reference"
    )
    root_state = model.build_with_refined_bounds(
        bounded_x,
        c_tensor,
        rhs_tensor,
        stop_criterion_batch_any,
        reference["lower_bounds"],
        reference["upper_bounds"],
        reference["lA"],
        reference["alphas"],
        None,
    )
    _require_absent_beta_warm_start(root_state.get("betas"), "official singleton root")
    _require_empty_root_history(
        root_state.get("history"), [node.name for node in model.net.split_nodes]
    )
    if root_state["global_lb"].shape[0] != 1:
        raise ParityError("official root bootstrap did not produce one domain")

    beta_args = arguments.Config["solver"]["beta-crown"]
    beta_args["beta"] = args.beta == "on"
    # At this pin, iteration=N evaluates N bounds and takes at most N-1
    # optimizer steps. Literal zero crashes after leaving best_ret unset.
    beta_args["iteration"] = beta_crown_iteration
    beta_args["lr_alpha"] = 0.0
    beta_args["enable_opt_interm_bounds"] = False
    beta_args["all_node_split_LP"] = False
    arguments.Config["bab"]["get_upper_bound"] = False
    arguments.Config["bab"]["cut"]["enabled"] = False
    arguments.Config["bab"]["cut"]["biccos"]["enabled"] = False
    arguments.Config["bab"]["branching"]["branching_input_and_activation"] = False
    arguments.Config["bab"]["branching"]["input_split"]["enable"] = False
    arguments.Config["general"]["deterministic_opt"] = True

    root_state["alphas"] = prune_alphas(root_state["alphas"], model.alpha_start_nodes)
    updated_mask, model.tot_ambi_nodes = get_unstable_neurons(root_state["mask"], model)
    factory = IntermBoundsFactory.from_interm(
        WorkingIntermBoundsInfo.from_two_dicts(
            root_state["lower_bounds"], root_state["upper_bounds"]
        ),
        final_name=model.final_name,
        device=model.device,
    )
    timer = Timer()
    domains = BatchedDomainList(
        ret=root_state,
        c=c_tensor,
        lAs=root_state["lA"],
        global_lbs=root_state["global_lb"],
        global_ubs=root_state["global_ub"],
        alphas=root_state["alphas"],
        history=copy.deepcopy(root_state["history"]),
        thresholds=rhs_tensor,
        net=model,
        x=bounded_x,
        branching_input_and_activation=False,
        timer=timer,
    )
    domains.update_unstable_mask(updated_mask)
    model.unstable_mask = domains.unstable_mask
    if model.domain_clipper is None:
        raise ParityError(
            "pinned configuration did not initialize intermediate-domain clipping"
        )
    model.domain_clipper.update_unstable_idx(updated_mask, model)
    _validate_live_history_coordinates(
        plan, domains.unstable_mask, model.domain_clipper.mapping
    )

    d = domains.pick_out(batch=1, device=model.device)
    _require_singleton_absent_domain_beta(d.get("betas"))
    split_node_names = [node.name for node in model.net.split_nodes]
    history = build_abc_history(plan, split_node_names)
    d["history"] = [history]
    d["depths"] = [plan["ny_child"]["depth"]]
    factory.construct_interm_bounds_in_d(d, domains.unstable_mask)
    applied_clamps = _apply_replay_history_clamps(
        d["lower_bounds"], d["upper_bounds"], plan, torch
    )
    for name, lower in d["lower_bounds"].items():
        upper = d["upper_bounds"][name]
        if bool((lower > upper).any()):
            raise ParityError(
                f"recorded history is infeasible under official root bounds at {name!r}"
            )

    history_export = {
        name: {
            "neuron": (
                layer_history[0].tolist()
                if hasattr(layer_history[0], "tolist")
                else list(layer_history[0])
            ),
            "sign": (
                layer_history[1].tolist()
                if hasattr(layer_history[1], "tolist")
                else list(layer_history[1])
            ),
            "bias": (
                layer_history[2].tolist()
                if hasattr(layer_history[2], "tolist")
                else list(layer_history[2])
            ),
        }
        for name, layer_history in history.items()
        if layer_history[0]
    }
    initial_alpha_manifest = _tensor_manifest(d["alphas"], torch)
    pre = update_bounds_pre(
        d=d,
        final_name=model.final_name,
        net_c=model.c,
        net_x=model.x,
        timer=timer,
        device=model.device,
        beta_bias=False,
    )
    _require_replay_history_clamps(
        {name: pair[0] for name, pair in pre.interm_bounds.items()},
        {name: pair[1] for name, pair in pre.interm_bounds.items()},
        plan,
        torch,
        context="official update_bounds_pre",
    )
    if args.beta == "on":
        _require_zero_sparse_beta_values(
            pre.betas_by_layer, plan["ny_child"]["depth"], torch
        )
    initial_beta_manifest = (
        _tensor_manifest(pre.betas_by_layer, torch)
        if args.beta == "on"
        else {"enabled": False, "tensor_count": 0, "total_numel": 0, "tensors": []}
    )

    pre_clip_pairs = {
        name: pair
        for name, pair in _clone_bound_pairs(pre.interm_bounds).items()
        if name not in model.alpha_start_nodes
    }
    clip_capture: dict[str, Any] = {"calls": 0}
    clipper = model.domain_clipper
    had_instance_override = "optimize_interm_bounds" in clipper.__dict__
    old_instance_override = clipper.__dict__.get("optimize_interm_bounds")
    original_clip = clipper.optimize_interm_bounds

    def capture_clip(*call_args: Any, **call_kwargs: Any) -> Any:
        if len(call_args) < 4:
            raise ParityError("unexpected DomainClipper call signature")
        clip_capture["calls"] += 1
        clip_capture["pre"] = _clone_bound_pairs(call_args[3])
        result = original_clip(*call_args, **call_kwargs)
        clip_capture["post"] = _clone_bound_pairs(result)
        return result

    clip_on = args.clip == "on"
    if clip_on:
        clipper.get_stop_criterion_and_iter(
            stop_criterion_batch_any, args.bab_iteration
        )
        try:
            clipper.get_constraints(d["history"])
        except Exception as exc:
            raise ParityError(
                f"official clipper rejected the recorded history: {exc}"
            ) from exc
        clipper.optimize_interm_bounds = capture_clip
    try:
        core = update_bounds_core(
            net=model,
            pre_result=pre,
            fix_interm_bounds=True,
            stop_criterion_func=stop_criterion_batch_any(d["thresholds"]),
            multi_spec_keep_func=multi_spec_keep_func_all,
            branching_heuristic=None,
            precompute_bfs_flag=False,
            batch_device_limit=1,
            is_multitree_bab=False,
            domain_clip_scorer=None,
            iter_idx=args.bab_iteration,
            enable_clip_domains=clip_on,
            enable_decision_precompute=False,
            visited_num=0,
        )
    finally:
        if clip_on:
            if had_instance_override:
                clipper.optimize_interm_bounds = old_instance_override
            else:
                del clipper.optimize_interm_bounds
    if clip_on and clip_capture["calls"] != 1:
        raise ParityError(
            "official intermediate clipper was expected exactly once, got "
            f"{clip_capture['calls']}"
        )
    if not clip_on and clip_capture["calls"] != 0:
        raise ParityError("official intermediate clipper ran in the disabled arm")

    final_working_pairs = _working_bound_pairs(core.working_interm_bounds)
    optimized_alpha_manifest = _tensor_manifest(core.working_alpha, torch)
    optimized_beta_manifest = (
        _tensor_manifest(core.working_beta, torch)
        if args.beta == "on"
        else {"enabled": False, "tensor_count": 0, "total_numel": 0, "tensors": []}
    )
    post = update_bounds_post(
        core_result=core,
        timer=timer,
        final_name=model.final_name,
        split_node_names=split_node_names,
        layers_requiring_bounds_names=[
            node.name for node in model.net.layers_requiring_bounds
        ],
        unstable_mask=model.unstable_mask,
        interm_transfer=True,
    )

    actual_pre_clip = clip_capture["pre"] if clip_on else pre_clip_pairs
    actual_post_clip = clip_capture["post"] if clip_on else pre_clip_pairs
    raw_lower = _tensor_list(core.lb)
    threshold = _tensor_list(d["thresholds"])
    if len(raw_lower) != len(threshold) or not raw_lower:
        raise ParityError("official child returned an inconsistent final bound shape")
    if any(not math.isfinite(value) for value in raw_lower + threshold):
        raise ParityError("official child returned a non-finite final lower bound")
    final_margin = [lower - rhs for lower, rhs in zip(raw_lower, threshold)]
    post_lower = _tensor_list(post.lower_bounds[model.final_name])
    if any(not math.isfinite(value) for value in post_lower):
        raise ParityError(
            "official child postprocess returned a non-finite lower bound"
        )

    arm = {
        "beta_enabled": args.beta == "on",
        "clip_enabled": clip_on,
        "beta_crown_iteration": beta_crown_iteration,
        "beta_optimizer_step_budget": args.beta_optimizer_updates,
        "bound_evaluation_budget": beta_crown_iteration,
        "alpha_learning_rate": 0.0,
        "bab_iteration": args.bab_iteration,
        "fix_intermediate_bounds": True,
        "decision_precompute": False,
    }
    measurements = {
        "source": {
            "abc_root_export_sha256": _sha256(args.abc_export.resolve()),
            "ny_root_sha256": _sha256(args.ny_root.resolve()),
            "ny_child_sha256": _sha256(args.ny_child.resolve()),
        },
        "history": {
            "depth": plan["ny_child"]["depth"],
            "by_layer": history_export,
            "applied_clamps": applied_clamps,
        },
        "bounds": {
            "pre_clip": _export_bound_pairs(actual_pre_clip, split_node_names, torch),
            "post_clip": _export_bound_pairs(actual_post_clip, split_node_names, torch),
            "final_working": _export_bound_pairs(
                final_working_pairs, split_node_names, torch
            ),
            "clipper_calls": clip_capture["calls"],
        },
        "final": {
            "raw_lower": raw_lower,
            "rhs": threshold,
            "lower_minus_rhs": final_margin,
            "post_lower": post_lower,
            "verified": all(lower > rhs for lower, rhs in zip(raw_lower, threshold)),
            "n_verified": core.n_verified,
            "n_splits": core.n_splits,
        },
        "state": {
            "initial_alpha": initial_alpha_manifest,
            "initial_beta": initial_beta_manifest,
            "optimized_alpha": optimized_alpha_manifest,
            "optimized_beta": optimized_beta_manifest,
            "post_alpha": _tensor_manifest(post.alphas, torch),
            "post_beta": _tensor_manifest(post.betas, torch),
        },
    }
    artifact = build_abc_child_artifact(plan, arm, measurements)
    digest = write_export(args.output.resolve(), artifact)
    print(f"wrote {args.output.resolve()} sha256={digest}")
    return 0


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

    child = subparsers.add_parser(
        "abc-child",
        help="replay one NY child path through pinned official update phases",
    )
    child.add_argument("--abc-repo", type=_path, required=True)
    child.add_argument("--config", type=_path, required=True)
    child.add_argument("--onnx", type=_path, required=True)
    child.add_argument("--vnnlib", type=_path, required=True)
    child.add_argument("--ny-root", type=_path, required=True)
    child.add_argument("--ny-child", type=_path, required=True)
    child.add_argument("--abc-export", type=_path, required=True)
    child.add_argument(
        "--expected-vnnlib-sha256",
        default=PROP1761_VNNLIB_SHA256,
        help=(
            "required content identity for the property; defaults to the pinned "
            "prop1761 artifact"
        ),
    )
    child.add_argument(
        "--beta",
        choices=("off", "on"),
        required=True,
        help="toggle split beta constraints while retaining frozen root alpha",
    )
    child.add_argument(
        "--clip",
        choices=("off", "on"),
        required=True,
        help="toggle the pinned intermediate-domain clipper",
    )
    child.add_argument(
        "--beta-optimizer-updates",
        type=int,
        default=10,
        help=(
            "requested optimizer-step budget; translated to N+1 auto_LiRPA "
            "iterations because iteration=1 means zero steps"
        ),
    )
    child.add_argument(
        "--bab-iteration",
        type=int,
        required=True,
        help="explicit official BaB round used by clipping; not inferred from depth",
    )
    child.add_argument("--output", type=_path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "abc-root":
        return export_abc_root(args)
    if args.command == "abc-child":
        return export_abc_child(args)
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
