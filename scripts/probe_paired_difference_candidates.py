#!/usr/bin/env python3
"""Diagnostic concrete sampling of paired exports; never publishes SAT/UNSAT.

Runs under the NY box measurement lock. Samples representable binary32 inputs
inside the original decimal box, evaluates exact exported coefficients with
NumPy float64, and independently replays the strongest candidates through the
original ONNX nodes and CPU ONNX Runtime at two optimization levels. Sampling
cannot establish an output enclosure or rule out an undiscovered violation.
"""
from __future__ import annotations

import argparse
import ast
import csv
from decimal import Decimal
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import time

# Fix BLAS threading before importing NumPy, so this diagnostic does not create
# a second unbounded worker pool while the box lock is held.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import onnxruntime as ort
import export_paired_difference_instances as exporter


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_stages(path):
    tokens = iter(path.read_text().split())
    def take():
        try:
            return next(tokens)
        except StopIteration as exc:
            raise ValueError("truncated paired input") from exc
    if take() != "PAIRED_V1":
        raise ValueError("bad paired input version")
    inputs, outputs, count, epsilon = int(take()), int(take()), int(take()), float(take())
    if inputs != 5 or outputs != 5 or not 0 < count <= 128:
        raise ValueError("unsupported paired input dimensions")
    def vector(size):
        values = np.asarray([float(take()) for _ in range(size)], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("nonfinite exported coefficient")
        return values
    lower, upper = vector(inputs), vector(inputs)
    sf, sg, width = [], [], inputs
    for _ in range(count):
        kind = take()
        if kind == "A":
            rows, cols = int(take()), int(take())
            if cols != width or not 0 < rows <= 512 or cols > 512:
                raise ValueError("exported affine shape mismatch")
            wf, wg = vector(rows * cols).reshape(rows, cols), vector(rows * cols).reshape(rows, cols)
            bf, bg = vector(rows), vector(rows)
            sf.append(("A", wf, bf)); sg.append(("A", wg, bg)); width = rows
        elif kind == "R":
            if int(take()) != width:
                raise ValueError("exported ReLU shape mismatch")
            sf.append(("R", None, None)); sg.append(("R", None, None))
        else:
            raise ValueError("unknown exported stage")
    if width != outputs or next(tokens, None) is not None:
        raise ValueError("bad paired output/trailing tokens")
    return sf, sg, lower, upper, epsilon


def evaluate(stages, points):
    out = np.asarray(points, dtype=np.float64)
    for kind, weight, bias in stages:
        out = np.maximum(out, 0.0) if kind == "R" else out @ weight.T + bias
    if not np.all(np.isfinite(out)):
        raise ValueError("nonfinite concrete forward result")
    return out


def sample_points(metadata, count, seed):
    # Inward faces ensure even a clipped binary32 corner is inside the exact
    # ORIGINAL decimal box, unlike the outward box used for certification.
    lower = np.asarray([exporter.directed_f32(metadata["lo_exact_decimal"][str(i)], True) for i in range(5)])
    upper = np.asarray([exporter.directed_f32(metadata["hi_exact_decimal"][str(i)], False) for i in range(5)])
    if np.any(lower > upper):
        raise ValueError("original box contains no representable binary32 point")
    corners = np.asarray(list(itertools.product(*zip(lower, upper))), dtype=np.float64)
    center = (lower + upper)[None, :] * 0.5
    if count < len(corners) + 1:
        raise ValueError("sample count must cover all 32 corners and the center")
    generator = np.random.Generator(np.random.PCG64(seed))
    random = generator.uniform(lower, upper, size=(count - len(corners) - 1, 5))
    points = np.vstack([corners, center, random]).astype(np.float32)
    points = np.clip(points, lower.astype(np.float32), upper.astype(np.float32))
    return points.astype(np.float64)


def exact_in_box(point, metadata):
    return all(Decimal(metadata["lo_exact_decimal"][str(i)]) <= Decimal.from_float(float(x))
               <= Decimal(metadata["hi_exact_decimal"][str(i)]) for i, x in enumerate(point))


def ort_sessions(fpath, gpath):
    sessions = {}
    for label, level in (("disable_all", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
                         ("enable_all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL)):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = level
        options.log_severity_level = 3
        sessions[label] = [ort.InferenceSession(str(path), sess_options=options,
                                               providers=["CPUExecutionProvider"]) for path in (fpath, gpath)]
    return sessions


def ort_forward(session, point):
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].type != "tensor(float)":
        raise ValueError("unexpected dynamic ONNX Runtime input")
    shape = inputs[0].shape
    if shape != [1, 1, 1, 5]:
        raise ValueError(f"unexpected ONNX Runtime input shape {shape}")
    return np.asarray(session.run(None, {inputs[0].name: np.asarray(point, dtype=np.float32).reshape(shape)})[0],
                      dtype=np.float64).reshape(5)


def output_record(f, g, exact_epsilon):
    signed = np.asarray(f) - np.asarray(g)
    maximum = float(np.max(np.abs(signed)))
    margin = Decimal.from_float(maximum) - Decimal(exact_epsilon)
    return {"f": np.asarray(f).tolist(), "g": np.asarray(g).tolist(),
            "signed_difference": signed.tolist(), "max_abs_difference": maximum,
            "margin_vs_exact_decimal_epsilon": float(margin), "observed_strict_violation": margin > 0}


def probe(index, inputs, benchmark, count, seed, top, batch_size):
    started = time.monotonic()
    paired = inputs / f"instance_{index}.paired"
    receipt_path = inputs / f"instance_{index}.json"
    metadata = json.loads(receipt_path.read_text())
    if metadata["index"] != index or metadata["export_sha256"] != sha256(paired):
        raise ValueError(f"export identity/hash mismatch for row {index}")
    models = ast.literal_eval(metadata["models"])
    if len(models) != 2 or [m[0] for m in models] != ["f", "g"]:
        raise ValueError("unexpected paired model tuple")
    fpath, gpath = [(benchmark / m[1]).resolve() for m in models]
    prop = (benchmark / metadata["property"]).resolve()
    for path, field in ((fpath, "f_sha256"), (gpath, "g_sha256"), (prop, "property_sha256")):
        if not path.is_relative_to(benchmark.resolve()) or sha256(path) != metadata[field]:
            raise ValueError(f"original source hash/path mismatch: {path}")
    _, _, _, exact_epsilon, lo_decimal, hi_decimal = exporter.parse_property(prop)
    lo_decimal = {str(axis): value for axis, value in lo_decimal.items()}
    hi_decimal = {str(axis): value for axis, value in hi_decimal.items()}
    if exact_epsilon != metadata["epsilon_exact_decimal"] or lo_decimal != metadata["lo_exact_decimal"] or hi_decimal != metadata["hi_exact_decimal"]:
        raise ValueError("source property and exporter receipt disagree")
    sf, sg, lower, upper, epsilon = paired_stages(paired)
    if lower.tolist() != metadata["lo_outward_f32"] or upper.tolist() != metadata["hi_outward_f32"] or epsilon != metadata["epsilon_inward_f32"]:
        raise ValueError("paired program domain differs from exporter receipt")
    points = sample_points(metadata, count, seed + index)
    values_f, values_g = [], []
    for start in range(0, count, batch_size):
        batch = points[start:start + batch_size]
        values_f.append(evaluate(sf, batch)); values_g.append(evaluate(sg, batch))
    values_f, values_g = np.vstack(values_f), np.vstack(values_g)
    scores = np.max(np.abs(values_f - values_g), axis=1)
    order = np.lexsort((np.arange(count), -scores))[:top]
    candidate_points = points[order]
    # This source-node evaluator consumes original ONNX constants and operators,
    # not exported-stage coefficients or certificate bounds.
    source_f = exporter.source_forward(fpath, candidate_points)
    source_g = exporter.source_forward(gpath, candidate_points)
    discrepancy = max(float(np.max(np.abs(source_f - values_f[order]))),
                      float(np.max(np.abs(source_g - values_g[order]))))
    if discrepancy > 1e-10:
        raise ValueError(f"exported/source concrete forward disagreement {discrepancy}")
    sessions = ort_sessions(fpath, gpath)
    candidates = []
    for position, sample_index in enumerate(order):
        x = points[sample_index]
        in_box = exact_in_box(x, metadata)
        if not in_box:
            raise ValueError("generated candidate is outside the exact original box")
        original_f64 = output_record(source_f[position], source_g[position], exact_epsilon)
        replay = {level: output_record(ort_forward(pair[0], x), ort_forward(pair[1], x), exact_epsilon)
                  for level, pair in sessions.items()}
        candidates.append({"sample_index": int(sample_index), "x_shared": x.tolist(),
                           "exact_original_box_contains_x": in_box,
                           "source_node_numpy_float64": original_f64, "original_onnxruntime_float32": replay,
                           "observed_violation_in_both_ort_levels": all(r["observed_strict_violation"] for r in replay.values()),
                           "observed_violation_in_source_f64_and_both_ort_levels": original_f64["observed_strict_violation"] and all(r["observed_strict_violation"] for r in replay.values())})
    return {"row": index, "samples": count, "seed": seed + index,
            "sampler": "32 corners + center + deterministic PCG64 uniform; binary32 in exact original box",
            "sampled_max_abs_difference_export_numpy_f64": float(scores[order[0]]),
            "sampled_margin_vs_exact_decimal_epsilon": float(Decimal.from_float(float(scores[order[0]])) - Decimal(exact_epsilon)),
            "export_source_max_concrete_discrepancy_on_candidates": discrepancy,
            "candidates": candidates, "elapsed_seconds": time.monotonic() - started,
            "input_sha256": sha256(paired), "metadata_sha256": sha256(receipt_path),
            "original_source_hashes": {field: metadata[field] for field in ("f_sha256", "g_sha256", "property_sha256")},
            "protocol_verdict": None, "formal_enclosure": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--benchmark-root", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rows", default="", help="comma separated IDs; default selects every undecided historical-bank row")
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()
    if not 33 <= args.samples <= 1000000 or not 1 <= args.top <= 64 or args.top > args.samples or not 1 <= args.batch_size <= 8192 or args.seed < 0:
        ap.error("invalid sampling/resource parameters")
    bank = {}
    with args.bank.open(newline="") as stream:
        for row in csv.reader(stream):
            if len(row) != 6:
                ap.error("bank must contain six-column instance rows")
            match = re.fullmatch(r"(?:\./)?vnnlib/instance_(\d+)\.vnnlib", row[2])
            if not match or int(match[1]) in bank:
                ap.error("bank does not have unique expected six-column instance identities")
            bank[int(match[1])] = row[4]
    selected = sorted(i for i, verdict in bank.items() if verdict not in ("sat", "unsat")) if not args.rows else [int(i) for i in args.rows.split(",")]
    if not selected or any(i not in bank for i in selected) or len(set(selected)) != len(selected):
        ap.error("nonempty unique bank-backed row selection required")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"scope": "diagnostic concrete sampling; no SAT/UNSAT publication, no enclosure or score claim",
              "script_sha256": sha256(__file__), "exporter_sha256": sha256(exporter.__file__),
              "bank_path": str(args.bank.resolve()), "bank_sha256": sha256(args.bank),
              "benchmark_root": str(args.benchmark_root.resolve()), "selected_rows": selected,
              "numpy_version": np.__version__, "onnxruntime_version": ort.__version__,
              "threads": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
              "rows": []}
    for index in selected:
        row = probe(index, args.inputs, args.benchmark_root, args.samples, args.seed, args.top, args.batch_size)
        row["historical_bank_verdict"] = bank[index]
        result["rows"].append(row)
        (args.output / f"instance_{index}.json").write_text(json.dumps(row, indent=2) + "\n")
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        observed = any(c["observed_violation_in_source_f64_and_both_ort_levels"] for c in row["candidates"])
        print(f"row={index} sampled_max={row['sampled_max_abs_difference_export_numpy_f64']:.9g} candidate_replayed_violation={observed} diagnostic_only=True", flush=True)
    result["summary"] = {"completed_rows": len(selected), "samples_per_row": args.samples,
                         "rows_with_replayed_violation_candidates": [r["row"] for r in result["rows"] if any(c["observed_violation_in_source_f64_and_both_ort_levels"] for c in r["candidates"])],
                         "protocol_verdicts_published": 0, "formal_enclosures_proved": 0,
                         "no_candidate_does_not_imply_unsat": True}
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
