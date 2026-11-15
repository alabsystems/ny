#!/usr/bin/env python3
"""Export exact ONNX affine/ReLU chains for the paired-difference experiment.

Every source operator is retained as a separate stage. FLOAT initializers are
widened exactly to binary64; affine layers are never composed. Bounds are
rounded outward to binary32 and the property threshold inward. This exporter
is intentionally restricted to the isomorphic ACAS-Xu benchmark grammar.
"""
from __future__ import annotations

import argparse
import ast
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import onnx
from onnx import numpy_helper


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sexprs(text):
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text))
    stack, roots = [], []
    for token in tokens:
        if token == "(":
            stack.append([])
        elif token == ")":
            if not stack:
                raise ValueError("unbalanced property parentheses")
            expr = stack.pop()
            (stack[-1] if stack else roots).append(expr)
        elif stack:
            stack[-1].append(token)
        else:
            raise ValueError("property token outside expression")
    if stack:
        raise ValueError("unclosed property expression")
    return roots


def flatten(expr, op):
    if isinstance(expr, list) and expr and expr[0] == op:
        return [x for child in expr[1:] for x in flatten(child, op)]
    return [expr]


def directed_f32(value, upward):
    exact = Decimal(value)
    rounded = np.float32(float(exact))
    represented = Decimal.from_float(float(rounded))
    if (upward and represented < exact) or (not upward and represented > exact):
        rounded = np.nextafter(rounded, np.float32(np.inf if upward else -np.inf))
    if not np.isfinite(rounded):
        raise ValueError("nonfinite property constant")
    return float(rounded)


def parse_property(path):
    roots = sexprs(Path(path).read_text())
    if any(not root or root[0] not in ("assert", "vnnlib-version", "declare-network") for root in roots):
        raise ValueError("unsupported property top-level form")
    version = [root for root in roots if root[0] == "vnnlib-version"]
    networks = [root for root in roots if root[0] == "declare-network"]
    if version != [["vnnlib-version", "<2.0>"]] or len(networks) != 2:
        raise ValueError("expected VNN-LIB 2.0 and exactly two networks")
    declarations = {root[1]: root[2:] for root in networks}
    for tag in ("f", "g"):
        if tag not in declarations:
            raise ValueError("unexpected network names")
        declaration = declarations[tag]
        expected = [["declare-input", f"X_{tag}", "real", "[1,", "1,", "1,", "5]"],
                    ["declare-output", f"Y_{tag}", "real", "[1,", "5]"]]
        if tag == "g":
            expected.insert(0, ["isomorphic-to", "f"])
        if declaration != expected:
            raise ValueError("unsupported network declaration")
    assertion_forms = [root for root in roots if root[0] == "assert"]
    if any(len(root) != 2 for root in assertion_forms):
        raise ValueError("malformed assertion")
    assertions = [root[1] for root in assertion_forms]
    if len(assertions) != 11:
        raise ValueError(f"unexpected assertion count: {len(assertions)}")
    lo, hi, shared, band = {}, {}, set(), []
    for assertion in assertions:
        for atom in flatten(assertion, "and"):
            if atom[0] in ("<=", ">="):
                m = re.fullmatch(r"X_f\[0,0,0,([0-4])\]", atom[1])
                if not m or len(atom) != 3 or isinstance(atom[2], list):
                    raise ValueError(f"unexpected domain atom: {atom}")
                target = hi if atom[0] == "<=" else lo
                axis = int(m[1])
                if axis in target:
                    raise ValueError("duplicate domain bound")
                target[axis] = atom[2]
            elif atom[0] == "==":
                m = re.fullmatch(r"X_f\[0,0,0,([0-4])\]", atom[1])
                if len(atom) != 3 or not m or atom[2] != f"X_g[0,0,0,{m[1]}]":
                    raise ValueError(f"unexpected shared-input atom: {atom}")
                shared.add(int(m[1]))
            elif atom[0] == "or":
                band.extend(flatten(atom, "or"))
            else:
                raise ValueError(f"unsupported property assertion: {atom}")
    if set(lo) != set(range(5)) or set(hi) != set(range(5)) or shared != set(range(5)):
        raise ValueError("property is not a complete shared five-dimensional box")
    signs, epsilons = set(), set()
    for atom in band:
        if len(atom) != 3 or atom[0] not in (">", "<"):
            raise ValueError(f"unexpected output atom: {atom}")
        m = re.fullmatch(r"Y_g\[0,([0-4])\]", atom[1])
        right = atom[2]
        expected_op = "+" if atom[0] == ">" else "-"
        if (not m or not isinstance(right, list) or len(right) != 3
                or right[0] != expected_op or right[1] != f"Y_f[0,{m[1]}]"):
            raise ValueError(f"unexpected output difference: {atom}")
        signs.add((int(m[1]), atom[0]))
        epsilons.add(Decimal(right[2]))
    if len(band) != 10 or signs != {(i, op) for i in range(5) for op in (">", "<")} or len(epsilons) != 1:
        raise ValueError("property is not one symmetric five-output difference band")
    epsilon = epsilons.pop()
    if epsilon <= 0:
        raise ValueError("nonpositive band threshold")
    lower = [directed_f32(lo[i], False) for i in range(5)]
    upper = [directed_f32(hi[i], True) for i in range(5)]
    if any(a > b for a, b in zip(lower, upper)):
        raise ValueError("empty input domain")
    return lower, upper, directed_f32(str(epsilon), False), str(epsilon), lo, hi


def source_stages(path):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    constants = {}
    for init in model.graph.initializer:
        if init.data_type != onnx.TensorProto.FLOAT:
            raise ValueError("initializer must be authored FLOAT")
        a = numpy_helper.to_array(init)
        if not np.all(np.isfinite(a)):
            raise ValueError("nonfinite initializer")
        constants[init.name] = a.astype(np.float64)
    inputs = [value for value in model.graph.input if value.name not in constants]
    if len(inputs) != 1 or len(model.graph.output) != 1:
        raise ValueError("model must have exactly one dynamic input and output")
    dims = tuple(d.dim_value for d in inputs[0].type.tensor_type.shape.dim)
    if dims != (1, 1, 1, 5):
        raise ValueError(f"unexpected input shape: {dims}")
    current, width, stages = inputs[0].name, 5, []
    for node in model.graph.node:
        if len(node.output) != 1 or not node.input or node.input[0] != current:
            raise ValueError(f"source graph is not a strict dynamic chain at {node.name}")
        if node.op_type == "Flatten":
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if len(node.input) != 1 or attrs.get("axis", 1) != 1:
                raise ValueError("unsupported Flatten")
        elif node.op_type == "Relu":
            if len(node.input) != 1:
                raise ValueError("unsupported ReLU arity")
            stages.append({"kind": "R", "width": width, "source_node": node.name})
        elif node.op_type in ("Sub", "Add", "MatMul"):
            if len(node.input) != 2 or node.input[1] not in constants or node.attribute:
                raise ValueError(f"unsupported affine node: {node.name}")
            constant = constants[node.input[1]]
            if node.op_type == "MatMul":
                if constant.ndim != 2 or constant.shape[0] != width:
                    raise ValueError("unsupported MatMul shape")
                weights = constant.T.copy()
                bias = np.zeros(weights.shape[0], dtype=np.float64)
            else:
                if constant.size != width or constant.shape[-1] != width:
                    raise ValueError("unsupported affine broadcast")
                weights = np.eye(width, dtype=np.float64)
                bias = constant.reshape(-1).copy()
                if node.op_type == "Sub":
                    bias = -bias
            stages.append({"kind": "A", "rows": len(bias), "cols": width,
                           "weights": weights.tolist(), "bias": bias.tolist(),
                           "source_node": node.name, "source_op": node.op_type})
            width = len(bias)
        else:
            raise ValueError(f"unsupported operator: {node.op_type}")
        current = node.output[0]
    if current != model.graph.output[0].name or width != 5:
        raise ValueError("unexpected output")
    return stages


def source_forward(path, points):
    """Independent source-node numpy float64 evaluator; no exported-stage input.

    This is a differential/checking oracle, not an outward-rounded proof or
    an implementation of the official binary32 ONNX Runtime semantics.
    """
    model = onnx.load(path)
    constants = {i.name: numpy_helper.to_array(i).astype(np.float64) for i in model.graph.initializer}
    dynamic = next(i.name for i in model.graph.input if i.name not in constants)
    values = {dynamic: np.asarray(points, dtype=np.float64).reshape(-1, 1, 1, 5)}
    for node in model.graph.node:
        args = [values[n] if n in values else constants[n] for n in node.input]
        if node.op_type == "Sub": result = args[0] - args[1]
        elif node.op_type == "Add": result = args[0] + args[1]
        elif node.op_type == "MatMul": result = args[0] @ args[1]
        elif node.op_type == "Flatten": result = args[0].reshape(len(points), -1)
        elif node.op_type == "Relu": result = np.maximum(args[0], 0.0)
        else: raise ValueError(f"unsupported independent forward op {node.op_type}")
        values[node.output[0]] = result
    return values[model.graph.output[0].name].reshape(len(points), 5)


def stages_forward(stages, points):
    out = np.asarray(points, dtype=np.float64)
    for stage in stages:
        if stage["kind"] == "R": out = np.maximum(out, 0.0)
        else: out = out @ np.asarray(stage["weights"]).T + np.asarray(stage["bias"])
    return out


def write_tokens(path, sf, sg, lower, upper, epsilon):
    def nums(values):
        return " ".join(repr(float(v)) for v in np.asarray(values).reshape(-1))
    lines = [f"PAIRED_V1 5 5 {len(sf)} {epsilon!r}", nums(lower), nums(upper)]
    if len(sf) != len(sg):
        raise ValueError("models have different stage counts")
    for a, b in zip(sf, sg):
        if a["kind"] != b["kind"]:
            raise ValueError("models have different operator chains")
        if a["kind"] == "R":
            if a["width"] != b["width"]: raise ValueError("ReLU width mismatch")
            lines.append(f"R {a['width']}")
        else:
            if (a["rows"], a["cols"]) != (b["rows"], b["cols"]): raise ValueError("affine shape mismatch")
            lines.extend([f"A {a['rows']} {a['cols']}", nums(a["weights"]), nums(b["weights"]), nums(a["bias"]), nums(b["bias"])])
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, nargs="*")
    ap.add_argument("--samples", type=int, default=32)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    rows = list(csv.reader((args.benchmark / "instances.csv").open()))
    for index, row in enumerate(rows):
        if args.rows is not None and index not in args.rows: continue
        models = ast.literal_eval(row[0])
        if len(models) != 2 or [m[0] for m in models] != ["f", "g"]: raise ValueError("unexpected network tuple")
        fpath, gpath = [args.benchmark / m[1] for m in models]
        prop = args.benchmark / row[1]
        lower, upper, epsilon, exact_eps, exact_lo, exact_hi = parse_property(prop)
        sf, sg = source_stages(fpath), source_stages(gpath)
        output = args.out / f"instance_{index}.paired"
        write_tokens(output, sf, sg, lower, upper, epsilon)
        rng = np.random.default_rng(index)
        points = rng.uniform(lower, upper, size=(args.samples, 5))
        points = np.vstack([points, lower, upper])
        oracle_f, oracle_g = source_forward(fpath, points), source_forward(gpath, points)
        err_f = float(np.max(np.abs(oracle_f - stages_forward(sf, points))))
        err_g = float(np.max(np.abs(oracle_g - stages_forward(sg, points))))
        # BLAS may choose a different summation order for a transposed copied
        # initializer. This check detects export errors; it is not proof of
        # floating-point equivalence and its measured discrepancy is retained.
        if err_f > 1e-12 or err_g > 1e-12: raise ValueError(f"stage export differs from source forward: {err_f}, {err_g}")
        metadata = {"index": index, "models": row[0], "property": row[1], "budget_secs": float(row[2]),
                    "f_sha256": sha256(fpath), "g_sha256": sha256(gpath), "property_sha256": sha256(prop),
                    "export_sha256": sha256(output), "epsilon_exact_decimal": exact_eps,
                    "epsilon_inward_f32": epsilon, "lo_outward_f32": lower, "hi_outward_f32": upper,
                    "lo_exact_decimal": exact_lo, "hi_exact_decimal": exact_hi,
                    "stages": len(sf), "source_forward_max_export_error": [err_f, err_g],
                    "sample_count": len(points), "sampled_max_abs_difference": float(np.max(np.abs(oracle_f-oracle_g)))}
        (args.out / f"instance_{index}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        manifest.append(metadata)
        print(f"exported row={index} stages={len(sf)} samples={len(points)} source_error={err_f},{err_g}", flush=True)
    (args.out / "manifest.json").write_text(json.dumps({"instances_sha256": sha256(args.benchmark / 'instances.csv'),
                                                      "exporter_sha256": sha256(__file__), "rows": manifest}, indent=2) + "\n")


if __name__ == "__main__":
    main()
