#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Sampled per-node bound-containment check against an ONNX Runtime oracle.

WHAT THIS PROVES AND WHAT IT DOES NOT
-------------------------------------
NY claims, for every node of the network and every ``x`` in the declared input
box, that the node's true value lies inside NY's computed interval
``[lo, hi]``.  That claim is refutable by a single point: sample ``x`` from the
box, evaluate the network with a trusted oracle (ONNX Runtime), and check every
intermediate against NY's interval.

* A CONFIRMED containment violation **PROVES the bound wrong**.  It needs no
  ground truth, no second verifier and no property -- it is a defect in NY.
* Finding no violation **PROVES NOTHING**.  This tool samples; it does not
  enumerate.  A clean run is *evidence of no defect found*, never a soundness
  certificate.  Every report repeats that asymmetry.

Unlike the counterexample moat (which can only flag an NY ``unsat`` where the
field produced an accepted counterexample), this check works in categories with
no ``sat`` ground truth at all -- e.g. ``vit_2023``, where no tool has ever
falsified an instance.  It also localises: it names the FIRST node in execution
order whose interval is violated, instead of only condemning the verdict.

FLOAT HONESTY
-------------
NY's dumped bounds are the certified f32 intervals (its rounding terms are
already folded outward into ``[lo, hi]``), so the dumped interval is exactly the
claim under test.  The oracle side is not exact: ONNX Runtime accumulates its
own f32 rounding.  A raw excess smaller than that accumulated error is NOT a
defect.  This tool therefore refuses to call a violation unless it survives all
of:

1. two independent f32 evaluations of the same graph (ORT with graph
   optimisations disabled vs. enabled -- same mathematics, different operator
   fusion and summation order), both violating;
2. an excess exceeding ``noise-k`` times the measured disagreement between
   those two evaluations (an empirical, per-element estimate of the f32
   accumulation noise for this graph at this point);
3. an excess exceeding ``tol-abs + tol-rel * scale``;
4. optionally (``--f64-confirm``) an independent float64 re-evaluation in numpy
   via ``onnx.reference``, which shares no arithmetic with ORT.

Anything that fails 1-3 is reported as ``NOISE`` and never as a defect.

USAGE
-----
    python3 scripts/bound_containment.py run \\
        --binary target/release/ny --category vit_2023 \\
        --onnx  benchmarks/.../onnx/ibp_3_3_8.onnx \\
        --vnnlib benchmarks/.../vnnlib/ibp_3_3_8_8325.vnnlib \\
        --timeout 100 --samples 256 --json out.json

``dump`` only runs NY; ``check`` only consumes an existing dump.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "ny_bound_containment_v1"

# One-sided-proof banner. Printed on every report so no reader can mistake a
# clean run for a soundness certificate.
DISCLAIMER = (
    "ONE-SIDED PROOF: a CONFIRMED violation proves NY's bound wrong. "
    "Finding none proves NOTHING -- this samples the box, it does not "
    "enumerate it. NEVER read a clean run as 'verified sound'."
)


class ContainmentError(RuntimeError):
    """The check cannot be performed soundly; never a verdict about NY."""


# --------------------------------------------------------------------------
# NY_LPOPT_DUMP parsing (format: crates/.../multi_objective/root.rs run_lpopt_dump)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Bounds:
    shape: tuple[int, ...]
    lower: np.ndarray
    upper: np.ndarray

    @property
    def size(self) -> int:
        return int(self.lower.size)


@dataclass
class NyDump:
    input_bounds: Bounds
    relu_map: tuple[tuple[str, str], ...]
    nodes: dict[str, Bounds]


def _parse_bounds_record(
    lines: list[str], index: int, kind: str
) -> tuple[str | None, Bounds, int]:
    parts = lines[index].split()
    if kind == "INPUT":
        if len(parts) < 3:
            raise ContainmentError("INPUT record is missing shape dimensions")
        name: str | None = None
        size_index, shape_index = 1, 2
    else:
        if len(parts) < 4:
            raise ContainmentError("NODE record is missing name or shape dimensions")
        name = parts[1]
        size_index, shape_index = 2, 3
    size = int(parts[size_index])
    shape = tuple(int(item) for item in parts[shape_index:])
    if size <= 0 or math.prod(shape) != size:
        raise ContainmentError(f"{kind} {name or ''}: size {size} != shape {shape}")
    if index + 2 >= len(lines):
        raise ContainmentError(f"truncated {kind} record at line {index + 1}")
    lower_parts = lines[index + 1].split()
    upper_parts = lines[index + 2].split()
    if not lower_parts or lower_parts[0] != "L":
        raise ContainmentError(f"{kind} {name or ''}: expected L record")
    if not upper_parts or upper_parts[0] != "U":
        raise ContainmentError(f"{kind} {name or ''}: expected U record")
    lower = np.asarray(lower_parts[1:], dtype=np.float64)
    upper = np.asarray(upper_parts[1:], dtype=np.float64)
    if lower.size != size or upper.size != size:
        raise ContainmentError(
            f"{kind} {name or ''}: expected {size} values, got "
            f"{lower.size} lower / {upper.size} upper"
        )
    inverted = int(np.count_nonzero(lower > upper))
    if inverted:
        # An inverted interval is itself an NY defect, but it is a *different*
        # one; refuse rather than silently normalising it away.
        raise ContainmentError(
            f"{kind} {name or ''}: {inverted} elements have lower > upper"
        )
    return name, Bounds(shape, lower, upper), index + 3


def parse_ny_dump(path: Path) -> NyDump:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "# ny lpopt dump v1":
        raise ContainmentError(f"unsupported NY dump header in {path}")
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
            _, input_bounds, index = _parse_bounds_record(lines, index, "INPUT")
        elif line.startswith("RELUMAP "):
            count = int(line.split()[1])
            for offset in range(count):
                pair = lines[index + 1 + offset].split()
                if len(pair) != 2:
                    raise ContainmentError("malformed RELUMAP entry")
                relu_map.append((pair[0], pair[1]))
            index += count + 1
        elif line.startswith("NODE "):
            name, bounds, index = _parse_bounds_record(lines, index, "NODE")
            assert name is not None
            if name in nodes:
                raise ContainmentError(f"duplicate NODE record {name!r}")
            nodes[name] = bounds
        else:
            raise ContainmentError(f"unknown record at line {index + 1}: {line!r}")
    if input_bounds is None:
        raise ContainmentError("NY dump has no INPUT record")
    if not nodes:
        raise ContainmentError("NY dump has no NODE records")
    return NyDump(input_bounds, tuple(relu_map), nodes)


# --------------------------------------------------------------------------
# VNNLIB input box (cross-check that NY samples the box it was handed)
# --------------------------------------------------------------------------

_GE = re.compile(r"\(\s*assert\s*\(\s*>=\s*X_(\d+)\s+([-+0-9eE.]+)\s*\)\s*\)")
_LE = re.compile(r"\(\s*assert\s*\(\s*<=\s*X_(\d+)\s+([-+0-9eE.]+)\s*\)\s*\)")
_ASSIGN = re.compile(r"\(\s*X_(\d+)\s+([-+0-9eEn.]+)\s*\)")


def parse_witness_point(path: Path, count: int) -> np.ndarray | None:
    """Parse a VNN-COMP counterexample assignment ``(X_i v)`` into a point.

    This is what makes the control experiment meaningful: on a row where the
    field DID produce an accepted counterexample -- the only place the existing
    moat can see -- that exact point is fed to the containment check, so both
    detectors are looking at the same witness.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    point = np.full(count, np.nan, dtype=np.float64)
    for match in _ASSIGN.finditer(text):
        index = int(match.group(1))
        if index < count:
            try:
                point[index] = float(match.group(2))
            except ValueError:
                return None
    if np.isnan(point).any():
        return None
    return point


def parse_vnnlib_input_box(path: Path, count: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Best-effort declared input box, or None if the spec is not plain box form.

    Only the simple ``(assert (>= X_i c))`` / ``(assert (<= X_i c))`` form is
    recognised.  Anything else returns None and the cross-check is reported as
    ``skipped`` rather than silently passing.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lower = np.full(count, -np.inf, dtype=np.float64)
    upper = np.full(count, np.inf, dtype=np.float64)
    seen_l = np.zeros(count, dtype=bool)
    seen_u = np.zeros(count, dtype=bool)
    for match in _GE.finditer(text):
        idx = int(match.group(1))
        if idx < count:
            lower[idx] = max(lower[idx], float(match.group(2)))
            seen_l[idx] = True
    for match in _LE.finditer(text):
        idx = int(match.group(1))
        if idx < count:
            upper[idx] = min(upper[idx], float(match.group(2)))
            seen_u[idx] = True
    if not (seen_l.all() and seen_u.all()):
        return None
    return lower, upper


# --------------------------------------------------------------------------
# ONNX Runtime oracle with every intermediate exposed
# --------------------------------------------------------------------------


class Oracle:
    """Two independent f32 ORT evaluations of the same graph, plus optional f64."""

    def __init__(self, onnx_path: Path, wanted: set[str], batch: int = 1) -> None:
        import onnx  # noqa: PLC0415  (optional dependency, imported eagerly by caller)
        import onnxruntime as ort  # noqa: PLC0415

        self._onnx = onnx
        model = onnx.load(str(onnx_path))
        graph = model.graph
        produced: list[str] = []
        # ONNX graphs are required to be topologically sorted, so node position
        # IS execution order.  This is what lets the report name the FIRST node
        # whose interval fails -- NY's dump is a hash map and carries no order.
        self.topo_index: dict[str, int] = {}
        # NY names a graph node after the ONNX *node* (`spec.name`), not after
        # the tensor it produces, and it fuses activations into the producer.
        # Both facts are recorded here so the mapping can be resolved without
        # guessing -- see `candidate_tensors`.
        self.node_outputs: dict[str, list[str]] = {}
        self.consumers: dict[str, list[Any]] = {}
        self.node_by_output: dict[str, Any] = {}
        for position, node in enumerate(graph.node):
            if node.name:
                self.node_outputs.setdefault(node.name, [n for n in node.output if n])
            for name in node.input:
                if name:
                    self.consumers.setdefault(name, []).append(node)
            for name in node.output:
                if name:
                    produced.append(name)
                    self.topo_index.setdefault(name, position)
                    self.node_by_output[name] = node
                    if node.name:
                        self.topo_index.setdefault(node.name, position)
        # Old-style ONNX also lists initializers in `graph.input`; only the
        # entries with no initializer are real network inputs.
        initializer_names = {init.name for init in graph.initializer}
        self.graph_input_names = [
            entry.name for entry in graph.input if entry.name not in initializer_names
        ]
        # Resolve every dumped NY node name to the ONNX tensor(s) that could
        # carry its value, WITHOUT guessing which one NY meant.
        self.candidates: dict[str, list[str]] = {}
        self.unresolved: dict[str, str] = {}
        for ny_name in wanted:
            resolved, reason = self._resolve(ny_name, wanted)
            if resolved:
                self.candidates[ny_name] = resolved
            else:
                self.unresolved[ny_name] = reason
        existing = {out.name for out in graph.output}
        initializers = {init.name for init in graph.initializer}
        needed = {name for names in self.candidates.values() for name in names}
        extra = [
            name
            for name in dict.fromkeys(produced)
            if name not in existing and name not in initializers and name in needed
        ]
        for name in extra:
            graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
        self.exposed = set(existing) | set(extra)
        self.input_name = graph.input[0].name
        self.input_shape = [
            dim.dim_value if dim.HasField("dim_value") and dim.dim_value > 0 else batch
            for dim in graph.input[0].type.tensor_type.shape.dim
        ]
        self._model_bytes = model.SerializeToString()
        self._onnx_path = onnx_path

        def session(level: Any) -> Any:
            options = ort.SessionOptions()
            options.graph_optimization_level = level
            options.log_severity_level = 3
            return ort.InferenceSession(
                self._model_bytes, options, providers=["CPUExecutionProvider"]
            )

        self.session_a = session(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
        self.session_b = session(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        self.names_a = [out.name for out in self.session_a.get_outputs()]
        self.names_b = [out.name for out in self.session_b.get_outputs()]
        self._f64: Any = None
        self._f64_failed = False
        # Drop candidates ORT could not actually surface (e.g. an optimiser
        # folded them away in session B); a node with no live candidate is
        # unmapped, never silently passing.
        live = (set(self.names_a) & set(self.names_b)) | {self.input_name}
        for ny_name, names in list(self.candidates.items()):
            kept = [name for name in names if name in live]
            if kept:
                self.candidates[ny_name] = kept
            else:
                del self.candidates[ny_name]
                self.unresolved[ny_name] = "ORT did not surface any candidate tensor"

    # Ops a verifier front end routinely folds into the preceding node.  Only
    # elementwise / affine-in-place ops belong here: folding one of these does
    # not change the tensor's shape, so the fused node's bounds are still a
    # per-element claim about the SAME positions.
    _FOLDABLE = frozenset(
        {
            "Relu",
            "Clip",
            "Sigmoid",
            "Tanh",
            "LeakyRelu",
            "Elu",
            "Softplus",
            "HardSigmoid",
            "BatchNormalization",
            "Dropout",
            "Identity",
        }
    )

    def _resolve(self, ny_name: str, dumped: set[str]) -> tuple[list[str], str]:
        """Candidate ONNX tensors for one NY node name.

        NY keys graph nodes by the ONNX *node* name and folds following ops into
        the producer while keeping the producer's name -- CIFAR100_resnet_medium
        dumps 41 nodes for a 59-node ONNX graph because all 19
        ``BatchNormalization_*`` are folded into their Conv.  Comparing NY's
        ``Conv_0`` against the raw Conv output there is comparing two different
        tensors, and it produced a spurious "violation" on the first real run.

        The fold is inferred STRUCTURALLY, from the dump's own key set: walk the
        forced single-consumer chain while each consumer's name is ABSENT from
        the dump and its op is shape-preserving.  If every step is forced, the
        end of the chain is the tensor NY's node denotes and it is returned
        alone.  If any step is not forced the walk stops and both ends are
        returned, in which case a violation must hold for ALL candidates -- so
        an unresolved mapping costs coverage and can never invent a violation.
        """
        if ny_name in self.node_by_output:
            return [ny_name], "tensor-name match"
        if ny_name in self.graph_input_names:
            return [ny_name], "graph input"
        # `ny_build::graph::INPUT_NODE_NAME` -- NY's synthetic name for the
        # network input, which has no ONNX counterpart.
        if ny_name == "_input" and len(self.graph_input_names) == 1:
            return [self.graph_input_names[0]], "NY synthetic input node"
        outputs = self.node_outputs.get(ny_name)
        if not outputs:
            return [], "no ONNX node or tensor of that name"
        cursor = outputs[0]
        forced = True
        for _ in range(8):
            consumers = self.consumers.get(cursor, [])
            if len(consumers) != 1:
                break
            consumer = consumers[0]
            if not consumer.output or consumer.op_type not in self._FOLDABLE:
                break
            if consumer.name and consumer.name in dumped:
                break  # NY kept this op as its own node; the chain ends here.
            if not consumer.name:
                forced = False  # unnamed op: cannot tell whether NY kept it
                cursor = consumer.output[0]
                continue
            cursor = consumer.output[0]
        if cursor == outputs[0]:
            return [outputs[0]], "node-name match"
        if forced:
            return [cursor], f"node-name match, folded chain -> {cursor}"
        return [outputs[0], cursor], "node-name match, AMBIGUOUS fold"

    def run(self, x: np.ndarray, which: str = "a") -> dict[str, np.ndarray]:
        tensor = x.astype(np.float32, copy=False).reshape(self.input_shape)
        session = self.session_a if which == "a" else self.session_b
        names = self.names_a if which == "a" else self.names_b
        values = session.run(None, {self.input_name: tensor})
        out = {
            name: np.asarray(value).astype(np.float64, copy=False).ravel()
            for name, value in zip(names, values)
        }
        # ORT never returns a graph input as an output, but NY dumps a `_input`
        # node whose interval the sampled point must also satisfy.
        out.setdefault(self.input_name, tensor.astype(np.float64).ravel())
        return out

    def run_f64(self, x: np.ndarray) -> dict[str, np.ndarray] | None:
        """Independent float64 re-evaluation in numpy (shares no code with ORT)."""
        if self._f64_failed:
            return None
        if self._f64 is None:
            try:
                from onnx import reference  # noqa: PLC0415

                model = self._onnx.load(str(self._onnx_path))
                _cast_model_to_double(self._onnx, model)
                self._f64 = reference.ReferenceEvaluator(model)
            except Exception:  # noqa: BLE001 - best effort escalation only
                self._f64_failed = True
                return None
        try:
            tensor = x.astype(np.float64, copy=False).reshape(self.input_shape)
            outputs = self._f64.run(None, {self.input_name: tensor}, intermediate=True)
        except Exception:  # noqa: BLE001
            self._f64_failed = True
            return None
        if isinstance(outputs, dict):
            return {
                key: np.asarray(value, dtype=np.float64).ravel()
                for key, value in outputs.items()
                if isinstance(value, np.ndarray) and value.dtype.kind == "f"
            }
        return None


def _cast_model_to_double(onnx_mod: Any, model: Any) -> None:
    """Retype every float32 initializer / value-info in place to float64."""
    from onnx import numpy_helper  # noqa: PLC0415

    fp32 = onnx_mod.TensorProto.FLOAT
    fp64 = onnx_mod.TensorProto.DOUBLE
    for init in model.graph.initializer:
        if init.data_type == fp32:
            array = numpy_helper.to_array(init).astype(np.float64)
            init.CopyFrom(numpy_helper.from_array(array, init.name))
    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for entry in collection:
            if entry.type.tensor_type.elem_type == fp32:
                entry.type.tensor_type.elem_type = fp64
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx_mod.AttributeProto.TENSOR and attr.t.data_type == fp32:
                array = numpy_helper.to_array(attr.t).astype(np.float64)
                attr.t.CopyFrom(numpy_helper.from_array(array, attr.t.name))
            if node.op_type == "Cast" and attr.name == "to" and attr.i == fp32:
                attr.i = fp64


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def build_samples(
    lower: np.ndarray,
    upper: np.ndarray,
    count: int,
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    """Deliberate samples: extremes first, interior last.

    Containment violations live at the extremes -- a bound that is tighter than
    true is usually tight *somewhere on the boundary*, so uniform interior
    sampling alone is close to useless.  Order: centre, the two extreme corners,
    random vertices (every coordinate at ``lo`` or ``hi``), coordinate-block
    corners, then uniform interior points.
    """
    rng = np.random.default_rng(seed)
    width = upper - lower
    centre = 0.5 * (lower + upper)
    samples: list[tuple[str, np.ndarray]] = [
        ("centre", centre.copy()),
        ("corner-lo", lower.copy()),
        ("corner-hi", upper.copy()),
    ]
    n = lower.size
    while len(samples) < count:
        remaining = count - len(samples)
        kind = len(samples) % 4
        if kind in (0, 1):  # random vertex of the box
            mask = rng.random(n) < 0.5
            samples.append(("vertex", np.where(mask, lower, upper)))
        elif kind == 2:  # half-box corner: a contiguous block flipped
            start = int(rng.integers(0, max(n, 1)))
            length = int(rng.integers(1, max(n // 2, 2)))
            point = lower.copy()
            end = min(n, start + length)
            point[start:end] = upper[start:end]
            samples.append(("block-corner", point))
        else:  # uniform interior
            samples.append(("interior", lower + rng.random(n) * width))
        if remaining <= 0:
            break
    return samples[:count]


def spsa_ascent(
    oracle: Oracle,
    lower: np.ndarray,
    upper: np.ndarray,
    score: Any,
    start: np.ndarray,
    steps: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Gradient-free PGD substitute: SPSA ascent on the worst normalised excess.

    Two oracle evaluations per step give a stochastic gradient estimate; the
    iterate is projected back onto the box after every step, so every point it
    ever proposes is a legal witness.  ONNX Runtime exposes no gradients, so a
    true PGD would need a second framework; SPSA needs only forward passes and
    optimises exactly the quantity a violation is defined by.
    """
    rng = np.random.default_rng(seed)
    width = np.maximum(upper - lower, 0.0)
    x = np.clip(start.copy(), lower, upper)
    best_x, best = x.copy(), score(oracle.run(x, "a"))
    if not np.any(width > 0):
        return best_x, best
    for step in range(steps):
        # Perturbation and step size both shrink with the box width so the
        # search stays scale-free across categories.
        c = 0.25 * width * (0.7 ** (step / max(steps // 4, 1)))
        delta = rng.choice(np.array([-1.0, 1.0]), size=x.size)
        plus = np.clip(x + c * delta, lower, upper)
        minus = np.clip(x - c * delta, lower, upper)
        s_plus = score(oracle.run(plus, "a"))
        s_minus = score(oracle.run(minus, "a"))
        for candidate, value in ((plus, s_plus), (minus, s_minus)):
            if value > best:
                best, best_x = value, candidate.copy()
        direction = np.sign(s_plus - s_minus) * delta
        x = np.clip(x + 0.5 * width * direction * (0.8 ** (step / 4.0)), lower, upper)
    return best_x, best


# --------------------------------------------------------------------------
# Containment evaluation
# --------------------------------------------------------------------------


@dataclass
class NodeCheck:
    name: str
    size: int
    order: int
    tensors: list[str] = field(default_factory=list)


@dataclass
class Violation:
    node: str
    node_order: int
    index: int
    lower: float
    upper: float
    value_a: float
    value_b: float
    excess: float
    scale: float
    noise: float
    envelope: float
    sample_kind: str
    sample_id: int
    verdict: str
    tensor: str = ""
    value_f64: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Tolerance:
    abs: float = 1e-6
    rel: float = 1e-5
    noise_k: float = 8.0


def _envelope(
    lo: np.ndarray, hi: np.ndarray, va: np.ndarray, vb: np.ndarray, tol: Tolerance
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.maximum.reduce(
        [np.abs(lo), np.abs(hi), np.abs(va), np.ones_like(va)]
    )
    noise = np.abs(va - vb)
    envelope = np.maximum(tol.abs + tol.rel * scale, tol.noise_k * noise)
    return scale, noise, envelope


def _excess(lo: np.ndarray, hi: np.ndarray, value: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(lo - value, value - hi), 0.0)


class Checker:
    def __init__(
        self,
        dump: NyDump,
        oracle: Oracle,
        tol: Tolerance,
        node_order: list[str],
    ) -> None:
        self.dump = dump
        self.oracle = oracle
        self.tol = tol
        self.checked: list[NodeCheck] = []
        self.unmapped: dict[str, str] = dict(oracle.unresolved)
        # The single closest approach to any wall seen so far, with the node it
        # happened at -- this is what tells a reader whether a clean run was
        # comfortable or a hair's breadth from firing.
        self.tightest: dict[str, Any] = {}
        for order, name in enumerate(node_order):
            bounds = dump.nodes[name]
            tensors = oracle.candidates.get(name)
            if not tensors:
                self.unmapped.setdefault(name, "unresolved NY node name")
                continue
            self.checked.append(NodeCheck(name, bounds.size, order, list(tensors)))

    def _values(
        self, outputs: dict[str, np.ndarray], node: NodeCheck
    ) -> list[tuple[str, np.ndarray]]:
        found = []
        for tensor in node.tensors:
            value = outputs.get(tensor)
            if value is not None and value.size == node.size:
                found.append((tensor, value))
        return found

    def worst_normalised(self, outputs: dict[str, np.ndarray]) -> float:
        """Scalar the SPSA search maximises: worst scale-normalised excess.

        With several mapping candidates the MINIMUM over candidates is used, so
        the search chases only excursions every candidate agrees on.
        """
        worst = -np.inf
        for node in self.checked:
            per_candidate = []
            for _, value in self._values(outputs, node):
                bounds = self.dump.nodes[node.name]
                scale = np.maximum.reduce(
                    [np.abs(bounds.lower), np.abs(bounds.upper), np.ones_like(value)]
                )
                # Negative when contained; the distance to the nearer wall.
                slack = np.minimum(value - bounds.lower, bounds.upper - value) / scale
                per_candidate.append(float(-np.min(slack)))
            if per_candidate:
                worst = max(worst, min(per_candidate))
        return worst if math.isfinite(worst) else -1.0

    def evaluate(
        self, x: np.ndarray, kind: str, sample_id: int, f64_confirm: bool
    ) -> tuple[list[Violation], float]:
        out_a = self.oracle.run(x, "a")
        out_b = self.oracle.run(x, "b")
        violations: list[Violation] = []
        worst_norm = -np.inf
        for node in self.checked:
            values_a = self._values(out_a, node)
            values_b = dict(self._values(out_b, node))
            if not values_a:
                self.unmapped.setdefault(
                    node.name, f"no candidate tensor with NY size {node.size}"
                )
                continue
            bounds = self.dump.nodes[node.name]
            lo, hi = bounds.lower, bounds.upper
            # Every candidate must violate before anything is reported; the
            # weakest candidate supplies the evidence.  A mapping mistake can
            # therefore only cost coverage, never produce a false alarm.
            per_candidate: list[tuple[float, Violation]] = []
            node_worst = np.inf
            node_worst_index = -1
            for tensor, va in values_a:
                vb = values_b.get(tensor)
                if vb is None:
                    continue
                scale, noise, env = _envelope(lo, hi, va, vb, self.tol)
                slack = np.minimum(va - lo, hi - va) / scale
                candidate_worst = float(-np.min(slack))
                if candidate_worst < node_worst:
                    node_worst = candidate_worst
                    node_worst_index = int(np.argmin(slack))
                ex_a = _excess(lo, hi, va)
                ex_b = _excess(lo, hi, vb)
                hit = (ex_a > env) & (ex_b > env)
                if not np.any(hit):
                    per_candidate = []
                    break
                index = int(np.argmax(np.where(hit, ex_a / np.maximum(env, 1e-300), 0.0)))
                per_candidate.append(
                    (
                        float(ex_a[index]),
                        Violation(
                            node=node.name,
                            node_order=node.order,
                            index=index,
                            lower=float(lo[index]),
                            upper=float(hi[index]),
                            value_a=float(va[index]),
                            value_b=float(vb[index]),
                            excess=float(ex_a[index]),
                            scale=float(scale[index]),
                            noise=float(noise[index]),
                            envelope=float(env[index]),
                            sample_kind=kind,
                            sample_id=sample_id,
                            verdict="CONFIRMED-f32",
                            tensor=tensor,
                        ),
                    )
                )
            if math.isfinite(node_worst) and node_worst > worst_norm:
                worst_norm = node_worst
                self.tightest = {
                    "node": node.name,
                    "index": node_worst_index,
                    "normalised_excess": node_worst,
                    "sample_kind": kind,
                    "sample_id": sample_id,
                }
            if per_candidate and len(per_candidate) == len(node.tensors):
                violations.append(min(per_candidate, key=lambda item: item[0])[1])
        if violations and f64_confirm:
            out64 = self.oracle.run_f64(x)
            for violation in violations:
                value = out64.get(violation.tensor) if out64 else None
                if value is None or value.size <= violation.index:
                    violation.verdict = "CONFIRMED-f32 (f64 unavailable)"
                    continue
                v64 = float(value[violation.index])
                violation.value_f64 = v64
                ex64 = max(violation.lower - v64, v64 - violation.upper, 0.0)
                violation.verdict = (
                    "CONFIRMED-f32+f64"
                    if ex64 > violation.envelope
                    else "f32-only (f64 contained) -- treat as NOISE"
                )
        violations.sort(key=lambda v: v.node_order)
        return violations, worst_norm


# --------------------------------------------------------------------------
# NY invocation
# --------------------------------------------------------------------------


def run_ny_dump(
    binary: Path,
    category: str,
    onnx_path: Path,
    vnnlib_path: Path,
    dump_path: Path,
    timeout: int,
    result_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    env = dict(os.environ)
    env["NY_LPOPT_DUMP"] = str(dump_path)
    command = [
        str(binary),
        "vnncomp",
        "v1",
        category,
        str(onnx_path),
        str(vnnlib_path),
        str(result_path),
        str(timeout),
    ]
    started = time.time()
    with log_path.open("wb") as log:
        completed = subprocess.run(  # noqa: S603
            ["timeout", "-k", "30", str(timeout + 120), *command],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - started
    text = log_path.read_text(encoding="utf-8", errors="replace")
    preset = [line for line in text.splitlines() if "Loading preset:" in line]
    verdict = ""
    if result_path.exists():
        verdict = result_path.read_text(encoding="utf-8", errors="replace").strip()
    return {
        "command": command,
        "returncode": completed.returncode,
        "seconds": round(elapsed, 2),
        "preset_lines": preset,
        "verdict": verdict.splitlines()[0] if verdict else "",
        "log": str(log_path),
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def report(payload: dict[str, Any], stream: Any = sys.stdout) -> None:
    print(f"[{SCHEMA}] {DISCLAIMER}", file=stream)
    inst = payload["instance"]
    print(
        f"  category={inst['category']} onnx={Path(inst['onnx']).name} "
        f"vnnlib={Path(inst['vnnlib']).name} ny_verdict={inst.get('ny_verdict', '?')}",
        file=stream,
    )
    cov = payload["coverage"]
    print(
        f"  nodes: {cov['checked']} checked / {cov['dumped']} dumped "
        f"({cov['unmapped']} unmapped, {cov['ambiguous_mappings']} ambiguous), "
        f"elements checked={cov['elements']}, "
        f"samples={payload['sampling']['samples']} "
        f"(+{payload['sampling']['spsa_evaluations']} SPSA evaluations), "
        f"witness={payload['sampling']['witness']}",
        file=stream,
    )
    box = payload["input_box"]
    print(f"  input box vs VNNLIB: {box['status']} -- {box['detail']}", file=stream)
    if cov["checked"] == 0:
        print("  RESULT: VACUOUS -- zero nodes checked; the run proves nothing.", file=stream)
        return
    confirmed = payload["violations"]["confirmed"]
    if confirmed:
        first = confirmed[0]
        print(
            f"  RESULT: {len(confirmed)} CONFIRMED containment violation(s). "
            f"FIRST node in execution order: {first['node']}[{first['index']}] "
            f"value={first['value_a']!r} outside [{first['lower']!r}, {first['upper']!r}] "
            f"excess={first['excess']:.6g} envelope={first['envelope']:.6g} "
            f"({first['verdict']}, sample={first['sample_kind']}#{first['sample_id']})",
            file=stream,
        )
        print("  THIS PROVES NY's BOUND WRONG at that node.", file=stream)
    else:
        tight = payload["tightest"]
        print(
            "  RESULT: no containment violation found "
            f"({payload['violations']['noise']} sub-envelope excursions treated as float noise).",
            file=stream,
        )
        print(
            f"  Closest approach: {tight['normalised_excess']:.6g} normalised excess "
            f"at {tight.get('node', '?')}[{tight.get('index', '?')}] "
            f"(sample {tight.get('sample_kind', '?')}#{tight.get('sample_id', '?')}); "
            "negative means strictly inside.",
            file=stream,
        )
        print("  This is NOT a soundness certificate. See the banner above.", file=stream)


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def mutate_shrink(dump: NyDump, fraction: float, node: str | None) -> list[str]:
    """NEGATIVE CONTROL: shrink NY's intervals toward their midpoint.

    A detector that has never fired is not yet trustworthy.  Shrinking a bound
    makes it *provably* too tight wherever the true value lived in the removed
    sliver, so a working checker MUST report a confirmed violation here.  A
    clean run under mutation means the checker is blind, not that NY is sound.
    """
    touched: list[str] = []
    for name, bounds in dump.nodes.items():
        if node is not None and name != node:
            continue
        mid = 0.5 * (bounds.lower + bounds.upper)
        half = 0.5 * (bounds.upper - bounds.lower) * (1.0 - fraction)
        dump.nodes[name] = Bounds(bounds.shape, mid - half, mid + half)
        touched.append(name)
    return touched


def do_check(args: argparse.Namespace) -> dict[str, Any]:
    dump = parse_ny_dump(Path(args.dump))
    mutated: list[str] = []
    if args.mutate_shrink > 0.0:
        mutated = mutate_shrink(dump, args.mutate_shrink, args.mutate_node)
        if not mutated:
            raise ContainmentError(
                f"--mutate-node {args.mutate_node!r} matches no dumped node"
            )
    tol = Tolerance(abs=args.tol_abs, rel=args.tol_rel, noise_k=args.noise_k)
    dumped_names = list(dump.nodes.keys())
    oracle = Oracle(Path(args.onnx), set(dumped_names))
    # Execution order, so "first violating node" localises the defect to the
    # earliest point the bound went wrong rather than to a downstream symptom.
    node_order = sorted(
        dumped_names,
        key=lambda name: (oracle.topo_index.get(name, len(oracle.topo_index)), name),
    )
    checker = Checker(dump, oracle, tol, node_order)

    lower = dump.input_bounds.lower
    upper = dump.input_bounds.upper
    declared = parse_vnnlib_input_box(Path(args.vnnlib), lower.size)
    if declared is None:
        box_status, box_detail = "skipped", "spec is not plain box form"
    else:
        d_lo, d_hi = declared
        # NY's box must not be a strict subset of the declared box: verifying a
        # smaller region than the spec declares is itself unsound.
        narrow_lo = float(np.max(lower - d_lo))
        narrow_hi = float(np.max(d_hi - upper))
        narrow = max(narrow_lo, narrow_hi)
        box_status = "ok" if narrow <= 1e-6 else "NARROWER-THAN-SPEC"
        box_detail = (
            f"max inward shrink {narrow:.6g} "
            f"(lo {narrow_lo:.6g}, hi {narrow_hi:.6g})"
        )

    samples = build_samples(lower, upper, args.samples, args.seed)
    witness_status = "not requested"
    if args.witness:
        point = parse_witness_point(Path(args.witness), lower.size)
        if point is None:
            witness_status = f"unparseable/incomplete: {args.witness}"
        else:
            outside = int(np.count_nonzero((point < lower) | (point > upper)))
            witness_status = (
                f"loaded ({outside} coordinates outside NY's box)"
                if outside
                else "loaded (inside NY's box)"
            )
            samples.insert(0, ("field-witness", np.clip(point, lower, upper)))
    confirmed: list[Violation] = []
    noise_count = 0
    worst = -np.inf
    worst_sample: np.ndarray | None = None
    for sample_id, (kind, x) in enumerate(samples):
        violations, worst_norm = checker.evaluate(x, kind, sample_id, args.f64_confirm)
        for violation in violations:
            if violation.verdict.startswith("CONFIRMED"):
                confirmed.append(violation)
            else:
                noise_count += 1
        if worst_norm > worst:
            worst, worst_sample = worst_norm, x
        if confirmed and args.stop_on_first:
            break

    spsa_evals = 0
    if not confirmed and args.spsa_steps > 0 and worst_sample is not None:
        start = worst_sample
        best_x, _ = spsa_ascent(
            oracle,
            lower,
            upper,
            checker.worst_normalised,
            start,
            args.spsa_steps,
            args.seed + 1,
        )
        spsa_evals = 2 * args.spsa_steps + 1
        violations, worst_norm = checker.evaluate(
            best_x, "spsa", len(samples), args.f64_confirm
        )
        for violation in violations:
            if violation.verdict.startswith("CONFIRMED"):
                confirmed.append(violation)
            else:
                noise_count += 1
        worst = max(worst, worst_norm)

    confirmed.sort(key=lambda v: (v.node_order, -v.excess))
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "disclaimer": DISCLAIMER,
        "instance": {
            "category": args.category,
            "onnx": str(args.onnx),
            "vnnlib": str(args.vnnlib),
            "dump": str(args.dump),
            "ny_verdict": args.ny_verdict,
        },
        "tolerance": {"abs": tol.abs, "rel": tol.rel, "noise_k": tol.noise_k},
        "mutation": {
            "shrink": args.mutate_shrink,
            "node": args.mutate_node,
            "nodes_mutated": len(mutated),
        },
        "coverage": {
            "dumped": len(dump.nodes),
            "checked": len(checker.checked),
            "unmapped": len(checker.unmapped),
            "unmapped_detail": dict(list(checker.unmapped.items())[:20]),
            "elements": int(sum(node.size for node in checker.checked)),
            "ambiguous_mappings": sum(
                1 for node in checker.checked if len(node.tensors) > 1
            ),
        },
        "input_box": {
            "status": box_status,
            "detail": box_detail,
            "size": int(lower.size),
        },
        "sampling": {
            "samples": len(samples),
            "seed": args.seed,
            "spsa_steps": args.spsa_steps,
            "spsa_evaluations": spsa_evals,
            "witness": witness_status,
        },
        "violations": {
            "confirmed": [v.as_dict() for v in confirmed[: args.max_report]],
            "confirmed_total": len(confirmed),
            "noise": noise_count,
        },
        "tightest": {
            "normalised_excess": float(worst) if math.isfinite(worst) else None,
            **checker.tightest,
        },
    }
    return payload


def do_run(args: argparse.Namespace) -> dict[str, Any]:
    dump_path = Path(args.dump)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    ny = run_ny_dump(
        Path(args.binary),
        args.category,
        Path(args.onnx),
        Path(args.vnnlib),
        dump_path,
        args.timeout,
        Path(args.result or (str(dump_path) + ".result.txt")),
        Path(args.log or (str(dump_path) + ".log")),
    )
    if not ny["preset_lines"]:
        raise ContainmentError(
            "no 'Loading preset:' line in the NY log -- the run did not take the "
            f"competition path; refusing to report. See {ny['log']}"
        )
    if not dump_path.exists():
        raise ContainmentError(
            f"NY produced no dump at {dump_path} (verdict={ny['verdict']!r}, "
            f"rc={ny['returncode']}); the graph BaB root was never reached. "
            f"See {ny['log']}"
        )
    args.ny_verdict = ny["verdict"]
    payload = do_check(args)
    payload["ny_run"] = ny
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--category", required=True)
        p.add_argument("--onnx", required=True)
        p.add_argument("--vnnlib", required=True)
        p.add_argument("--dump", required=True)
        p.add_argument("--samples", type=int, default=64)
        p.add_argument("--seed", type=int, default=20260808)
        p.add_argument("--spsa-steps", type=int, default=24)
        p.add_argument("--tol-abs", type=float, default=1e-6)
        p.add_argument("--tol-rel", type=float, default=1e-5)
        p.add_argument("--noise-k", type=float, default=8.0)
        p.add_argument("--f64-confirm", action="store_true")
        p.add_argument(
            "--mutate-shrink",
            type=float,
            default=0.0,
            help="NEGATIVE CONTROL: shrink every interval toward its midpoint by "
            "this fraction before checking; a working checker must then FIRE.",
        )
        p.add_argument("--mutate-node", default=None)
        p.add_argument(
            "--witness",
            default=None,
            help="file containing a VNN-COMP counterexample assignment (X_i v); "
            "the point is checked FIRST, tying this check to the row the "
            "counterexample moat can already see.",
        )
        p.add_argument("--stop-on-first", action="store_true")
        p.add_argument("--max-report", type=int, default=10)
        p.add_argument("--json", default=None)
        p.add_argument("--ny-verdict", default="")

    p_check = sub.add_parser("check", help="check an existing NY_LPOPT_DUMP")
    common(p_check)

    p_run = sub.add_parser("run", help="run NY with NY_LPOPT_DUMP, then check")
    common(p_run)
    p_run.add_argument("--binary", required=True)
    p_run.add_argument("--timeout", type=int, default=100)
    p_run.add_argument("--result", default=None)
    p_run.add_argument("--log", default=None)

    args = parser.parse_args(argv)
    try:
        payload = do_run(args) if args.command == "run" else do_check(args)
    except ContainmentError as error:
        print(f"[{SCHEMA}] ABORT: {error}", file=sys.stderr)
        return 2
    report(payload)
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 1 if payload["violations"]["confirmed_total"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
