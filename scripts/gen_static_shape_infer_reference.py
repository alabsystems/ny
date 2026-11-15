#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the ORT reference shape tables for the static shape-inference
fallback's corpus test.

ny's loader asks ONNX Runtime for every tensor shape (see
``crates/ny-onnx/src/loader/shape_infer/mod.rs``: ``expose_intermediate_outputs``
+ ``collect_shapes_from_session``). When that is unavailable (the
``__shape-infer`` child cannot run), ``shape_infer::static_infer`` fills what it
can by exact rules. The corpus test
(``static_infer_corpus_tests.rs``) proves that on every tensor the fallback
fills, its shape EQUALS the one ORT reports. This script produces those ORT
tables so the test needs no ONNX Runtime at test time.

It reproduces ny's ORT extraction pipeline step for step, on the SAME bytes the
loader would hand ORT:

1. ``normalize_model_for_ort``: ir_version <= 0 -> 9; default-domain opset
   version <= 0 -> 13; add a default-domain opset 13 when absent.
2. ``batch_resolve::resolve_batch_dim(graph, 1)``: pin a non-positive /
   symbolic leading dim of every runtime (non-initializer) graph input to 1.
3. ``clear_intermediate_value_info_shapes``: drop every value_info shape.
4. ``clear_reconsumed_output_shapes``: drop the shape of every graph output
   that some node consumes.
5. ``normalize_transpose_perms_for_ort``: a no-op unless a Transpose perm's
   length disagrees with its input's known rank; the script refuses exactly
   those models rather than approximating the rewrite.
6. No runtime inputs -> empty table (ny returns before ORT).
7. Expose every node output as a graph output, typed only for Cast/CastLike
   and the shape-preserving op list (input 0's type), untyped otherwise.
8. ``InferenceSession`` on those bytes; the table is every TENSOR-typed
   session input and output (sequence / map / optional values are skipped, as
   ``ValueType::tensor_shape()`` skips them in Rust) whose shape is
   informative (rank >= 1 with any dim > 0), symbolic dims reported as -1
   exactly as the Rust ``ort`` crate reports them.
9. ``resolve_zero_dim_ranks``: a ZERO-dim report is ambiguous (ORT reports a
   tensor it could not type with zero dims, exactly like a rank-0 one), so
   each is settled by ONE more session on the same bytes plus a ``Shape`` node
   per candidate: the entry is ``[]`` only when that ``Shape`` output's
   inferred dims are exactly ``[0]`` (ORT's type has rank 0), and absent
   otherwise, including when the probe session fails.

Not mirrored: ny decodes into its own prost schema, whose ``TypeProto`` holds
only ``tensor_type``, so the Rust loader hands ORT a sequence / map / optional
/ sparse type DECLARATION as an empty type, while this script keeps it. The
two can then differ (a ``value_info`` declaring a sequence's element shape
``[]`` reaches ORT here but not from Rust). The reference corpora have no such
values (measured 2026-09-28: no non-tensor session value in any of the 959
distinct VNN-COMP models), so the tables are unaffected.

Usage (read-only use of the onnx/onnxruntime venv on the GB10)::

    <home>/x86emu/venv-arm/bin/python \
        scripts/gen_static_shape_infer_reference.py \
        --benchmarks benchmarks \
        --out crates/ny-onnx/src/loader/shape_infer/testdata/static_infer_ort_reference.json

The model list is every ``*.onnx`` / ``*.onnx.gz`` under the corpus
directories named in ``CORPORA`` for both vnncomp2025 and vnncomp2026,
deduplicated by the SHA-256 of the decompressed model bytes.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import onnx
import onnxruntime as ort

YEARS = ("vnncomp2025", "vnncomp2026")
CORPORA = (
    "linearizenn_2024",
    "lsnc_relu",
    "acasxu_2023",
    "tllverifybench_2023",
    "cersyve",
    "dist_shift_2023",
    "collins_rul_cnn_2022",
    "safenlp_2024",
)

# Mirrors `shape_preserving_ops` in `expose_intermediate_outputs`.
SHAPE_PRESERVING_OPS = {
    "Relu", "Sigmoid", "Tanh", "LeakyRelu", "Elu", "Selu", "Softplus",
    "Softsign", "HardSigmoid", "HardSwish", "Mish", "Erf", "Gelu", "Abs",
    "Neg", "Ceil", "Floor", "Round", "Sign", "Sqrt", "Exp", "Log",
    "Reciprocal", "Not", "Clip", "Dropout", "BatchNormalization",
    "InstanceNormalization", "GroupNormalization", "Identity",
}

DEFAULT_IR_VERSION = 9
DEFAULT_OPSET_VERSION = 13
VERIFICATION_BATCH_SIZE = 1


def is_standard_domain(domain: str) -> bool:
    return domain in ("", "ai.onnx")


def normalize_model_for_ort(model: onnx.ModelProto) -> None:
    if model.ir_version <= 0:
        model.ir_version = DEFAULT_IR_VERSION
    has_default = False
    for opset in model.opset_import:
        if opset.domain == "":
            has_default = True
            if opset.version <= 0:
                opset.version = DEFAULT_OPSET_VERSION
    if not has_default:
        entry = model.opset_import.add()
        entry.domain = ""
        entry.version = DEFAULT_OPSET_VERSION


def resolve_batch_dim(graph: onnx.GraphProto, batch: int) -> None:
    inits = {i.name for i in graph.initializer}
    for inp in graph.input:
        if inp.name in inits:
            continue
        if not inp.HasField("type") or not inp.type.HasField("tensor_type"):
            continue
        tt = inp.type.tensor_type
        if not tt.HasField("shape") or len(tt.shape.dim) == 0:
            continue
        lead = tt.shape.dim[0]
        if lead.HasField("dim_value") and lead.dim_value > 0:
            continue
        lead.dim_value = batch


def clear_shape(info: onnx.ValueInfoProto) -> None:
    if info.HasField("type") and info.type.HasField("tensor_type"):
        info.type.tensor_type.ClearField("shape")


def expose(model_bytes: bytes):
    model = onnx.load_from_string(model_bytes)
    normalize_model_for_ort(model)
    graph = model.graph
    resolve_batch_dim(graph, VERIFICATION_BATCH_SIZE)
    for info in graph.value_info:
        clear_shape(info)
    consumed = {name for node in graph.node for name in node.input if name}
    for info in graph.output:
        if info.name in consumed:
            clear_shape(info)
    # `normalize_transpose_perms_for_ort` only rewrites a perm whose length
    # disagrees with the input's KNOWN rank (initializer dims or a graph-input
    # declaration) and whose entries are all non-negative; everything else is
    # left as authored. That rewrite is not mirrored, so refuse exactly the
    # models that would need it.
    known_ranks = {init.name: len(init.dims) for init in graph.initializer}
    for inp in graph.input:
        if inp.HasField("type") and inp.type.HasField("tensor_type"):
            if inp.type.tensor_type.HasField("shape"):
                known_ranks[inp.name] = len(inp.type.tensor_type.shape.dim)
    for node in graph.node:
        if node.op_type != "Transpose" or not is_standard_domain(node.domain):
            continue
        if not node.input or node.input[0] not in known_ranks:
            continue
        perms = [a.ints for a in node.attribute if a.name == "perm"]
        if not perms or any(v < 0 for v in perms[0]):
            continue
        if len(perms[0]) != known_ranks[node.input[0]]:
            raise NotImplementedError(
                "rank-mismatched Transpose perm (ny rewrites it for ORT); not mirrored"
            )
    inits = {i.name for i in graph.initializer}
    if not any(i.name and i.name not in inits for i in graph.input):
        return None
    existing_outputs = {o.name for o in graph.output}
    input_names = {i.name for i in graph.input}
    type_map = {}
    for info in list(graph.input) + list(graph.output) + list(graph.value_info):
        if info.HasField("type"):
            type_map[info.name] = copy.deepcopy(info.type)
    for init in graph.initializer:
        if init.data_type > 0:
            ty = onnx.TypeProto()
            ty.tensor_type.elem_type = init.data_type
            ty.tensor_type.shape.SetInParent()
            for d in init.dims:
                ty.tensor_type.shape.dim.add().dim_value = d
            type_map[init.name] = ty
    new_outputs = []
    new_names = set()
    for node in graph.node:
        inferred = None
        if is_standard_domain(node.domain):
            first = node.input[0] if node.input and node.input[0] else None
            if node.op_type == "Cast":
                if first in type_map:
                    targets = [a for a in node.attribute if a.name == "to"]
                    if len(targets) == 1 and targets[0].type == onnx.AttributeProto.INT:
                        inferred = copy.deepcopy(type_map[first])
                        if inferred.HasField("tensor_type"):
                            inferred.tensor_type.elem_type = targets[0].i
                        else:
                            inferred = None
            elif node.op_type == "CastLike":
                second = node.input[1] if len(node.input) > 1 and node.input[1] else None
                if first in type_map and second in type_map and type_map[second].HasField("tensor_type"):
                    inferred = copy.deepcopy(type_map[first])
                    if inferred.HasField("tensor_type"):
                        inferred.tensor_type.elem_type = type_map[second].tensor_type.elem_type
                    else:
                        inferred = None
            elif node.op_type in SHAPE_PRESERVING_OPS:
                if first in type_map:
                    inferred = copy.deepcopy(type_map[first])
        for out in node.output:
            if (
                not out
                or out in existing_outputs
                or out in input_names
                or out in inits
                or out in new_names
            ):
                continue
            new_names.add(out)
            vi = onnx.ValueInfoProto()
            vi.name = out
            if inferred is not None:
                vi.type.CopyFrom(inferred)
            new_outputs.append(vi)
            if inferred is not None:
                type_map[out] = inferred
    graph.output.extend(new_outputs)
    return model.SerializeToString()


def informative(shape) -> bool:
    # Mirrors `insert_shape_if_informative`: a NON-EMPTY shape with a positive
    # dim. A zero-dim report is never informative by itself; only the rank
    # probe below may record `[]`.
    return any(d > 0 for d in shape)


def to_dims(shape):
    return [d if isinstance(d, int) else -1 for d in shape]


# Mirrors `RANK_PROBE_PREFIX`, `collect_graph_names` and `fresh_name`.
RANK_PROBE_PREFIX = "__ny_rank_probe_"


def graph_names(graph: onnx.GraphProto, names: set) -> None:
    for info in list(graph.input) + list(graph.output) + list(graph.value_info):
        names.add(info.name)
    for init in graph.initializer:
        names.add(init.name)
    for sparse in graph.sparse_initializer:
        names.add(sparse.values.name)
    for node in graph.node:
        names.add(node.name)
        names.update(node.input)
        names.update(node.output)
        for attr in node.attribute:
            if attr.HasField("g"):
                graph_names(attr.g, names)
            for sub in attr.graphs:
                graph_names(sub, names)


def fresh_name(taken: set, base: str) -> str:
    name, suffix = base, 0
    while name in taken:
        suffix += 1
        name = f"{base}_{suffix}"
    taken.add(name)
    return name


def proven_rank_zero(exposed: bytes, candidates, opts) -> set:
    """Mirror of `resolve_zero_dim_ranks` / `build_rank_probe_model`: the
    candidates whose `Shape` probe ORT types as exactly `[0]`."""
    model = onnx.load_from_string(exposed)
    graph = model.graph
    taken: set = set()
    graph_names(graph, taken)
    probes = {}
    # Rust iterates a BTreeSet<String>; byte order == code-point order.
    for index, name in enumerate(sorted(candidates)):
        output = fresh_name(taken, f"{RANK_PROBE_PREFIX}{index}")
        node_name = fresh_name(taken, f"{RANK_PROBE_PREFIX}node_{index}")
        graph.node.append(onnx.helper.make_node("Shape", [name], [output], name=node_name))
        graph.output.add().name = output
        probes[output] = name
    try:
        sess = ort.InferenceSession(
            model.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
    except Exception as err:  # the Rust path records none of them either
        print(f"rank probe failed ({type(err).__name__}: {err}); "
              f"{len(probes)} zero-dim entries left out", file=sys.stderr)
        return set()
    return {
        probes[arg.name]
        for arg in sess.get_outputs()
        if arg.name in probes and arg.shape is not None and to_dims(arg.shape) == [0]
    }


def ort_table(model_bytes: bytes):
    exposed = expose(model_bytes)
    if exposed is None:
        return {}
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(exposed, opts, providers=["CPUExecutionProvider"])
    table = {}
    zero_dims = set()
    for arg in list(sess.get_inputs()) + list(sess.get_outputs()):
        if not arg.name or arg.shape is None:
            continue
        # Mirrors `ValueType::tensor_shape()` in the Rust `ort` crate, which
        # `collect_shapes_from_session` reads: only tensor (and sparse-tensor)
        # values carry a shape; a sequence / map / optional is skipped there.
        # Python reports such a value's shape as [], so without this filter it
        # would join the rank probe, whose `Shape` node on it is an invalid
        # graph that fails the WHOLE probe — unlike the Rust path.
        if not arg.type.startswith(("tensor(", "sparse_tensor(")):
            continue
        dims = to_dims(arg.shape)
        if not dims:
            zero_dims.add(arg.name)
        elif informative(dims):
            table[arg.name] = dims
    # A name reported both with and without dims is dropped from both.
    for name in zero_dims & set(table):
        del table[name]
        zero_dims.discard(name)
    if zero_dims:
        for name in proven_rank_zero(exposed, zero_dims, opts):
            table[name] = []
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmarks", default="benchmarks", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    groups: dict[str, dict] = {}
    for year in YEARS:
        for corpus in CORPORA:
            base = args.benchmarks / year / "benchmarks" / corpus
            if not base.is_dir():
                print(f"corpus missing: {base}", file=sys.stderr)
                return 2
            found = []
            for dirpath, _dirs, files in os.walk(base, followlinks=True):
                for name in files:
                    if name.endswith(".onnx") or name.endswith(".onnx.gz"):
                        found.append(Path(dirpath) / name)
            for path in sorted(found):
                raw = path.read_bytes()
                data = gzip.decompress(raw) if path.suffix == ".gz" else raw
                sha = hashlib.sha256(data).hexdigest()
                rel = path.relative_to(args.benchmarks).as_posix()
                entry = groups.setdefault(
                    sha, {"sha256": sha, "corpus": corpus, "paths": [], "_bytes": data}
                )
                entry["paths"].append(rel)

    models = []
    for sha in sorted(groups, key=lambda s: (groups[s]["corpus"], groups[s]["paths"][0])):
        entry = groups.pop(sha)
        data = entry.pop("_bytes")
        try:
            entry["ort_shapes"] = dict(sorted(ort_table(data).items()))
            entry["ort_error"] = None
        except Exception as err:  # recorded, never silently dropped
            entry["ort_shapes"] = {}
            entry["ort_error"] = f"{type(err).__name__}: {err}"
        entry["paths"] = sorted(entry["paths"])
        models.append(entry)

    doc = {
        "schema": 1,
        "generator": "scripts/gen_static_shape_infer_reference.py",
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "pipeline": "ny expose_intermediate_outputs mirror; batch pinned to 1; "
        "table = informative session input/output shapes, symbolic dims -1; "
        "[] only where a Shape rank probe proves rank 0",
        "corpora": list(CORPORA),
        "years": list(YEARS),
        "models": models,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        handle.write("{\n")
        head = {k: v for k, v in doc.items() if k != "models"}
        for key, value in head.items():
            handle.write(f"  {json.dumps(key)}: {json.dumps(value)},\n")
        handle.write('  "models": [\n')
        for index, model in enumerate(models):
            sep = "," if index + 1 < len(models) else ""
            handle.write(f"    {json.dumps(model, separators=(',', ':'))}{sep}\n")
        handle.write("  ]\n}\n")
    errors = sum(1 for m in models if m["ort_error"])
    print(f"wrote {len(models)} distinct models ({errors} ORT errors) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
