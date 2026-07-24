#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Independent sat-witness auditor (#sb-rebank): margins on TWO engines.

The sat-bank standard (sibling audit): a banked `sat` must reproduce on
current main AND its witness must violate the property on an INDEPENDENT
onnxruntime 1.19.2 + an exact-f64 numpy forward, at zero tolerance, WITH
MARGIN. This script is the second half of that bar:

  input : witness (raw SMT-LIB `((X_i v)...)` or a VNN-COMP results file
          whose first line is `sat`/`violated`), the ONNX model, the VNN-LIB
          property;
  output: per-clause margins on BOTH engines + the property margin
          (disjunction: max over clauses of min over conjuncts), the
          zero-tolerance input-box check, and an accept verdict;
  accept: property margin >= --bar (default 1e-5) on BOTH engines AND the
          witness X is inside the declared box at zero tolerance (f64 parse,
          exactly the organizer's asserts).

Engines:
  * onnxruntime (version-pinned; refuses to run unless __version__ ==
    --ort-version, default 1.19.2) on the f32 cast of the witness — the
    organizer's view;
  * an exact-f64 numpy re-implementation of the graph (Gemm / Relu /
    Reshape / Conv / Flatten / Constant / Add / Sub / MatMul / Sigmoid,
    f32 weights widened exactly to f64), evaluated on the SAME f32-cast
    input widened to f64 — the build-independent referee. Unsupported ops
    fail loudly (no silent skip).

Generic over the standard VNN-LIB robustness shape: `(assert (<=|>= X_i c))`
input bounds plus one `(assert (or (and <atoms>...) ...))` — or a plain
conjunction — with atoms `(<=|>= Y_i c)` and `(<=|>= Y_i Y_j)`. This covers
soundnessbench and the currently banked sats; anything else raises.

Usage:
  python3 scripts/sb_witness_check.py \
    --onnx benchmarks/.../model.onnx --vnnlib .../model_1.vnnlib \
    --witness res_model_1.txt [--bar 1e-5] [--json-out out.json]

Exit code 0 = ACCEPT, 2 = REJECT (below bar / out of box / not violated),
1 = error (bad inputs, unsupported op, wrong ORT version).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


# ----------------------------- VNN-LIB parsing -----------------------------

_ATOM = re.compile(
    r"\(\s*(<=|>=)\s+([XY])_(\d+)\s+(?:([XY])_(\d+)|([-+0-9.eE]+))\s*\)"
)


def _tokenize_sexpr(text: str):
    """Yield top-level s-expressions of the file (comments stripped)."""
    src = re.sub(r";[^\n]*", "", text)
    depth = 0
    start = None
    for i, ch in enumerate(src):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                yield src[start : i + 1]
                start = None
    if depth != 0:
        raise ValueError("unbalanced parentheses in vnnlib")


class Atom:
    """One comparison: margin > 0 <=> strictly satisfied (violation sense)."""

    def __init__(self, op: str, lhs, rhs):
        self.op = op  # "<=" or ">="
        self.lhs = lhs  # ("Y", i)
        self.rhs = rhs  # ("Y", j) or ("const", c)

    def margin(self, y: np.ndarray) -> float:
        a = float(y[self.lhs[1]])
        b = float(self.rhs[1]) if self.rhs[0] == "const" else float(y[self.rhs[1]])
        return (b - a) if self.op == "<=" else (a - b)

    def __repr__(self):
        rhs = f"{self.rhs[1]}" if self.rhs[0] == "const" else f"Y_{self.rhs[1]}"
        return f"({self.op} Y_{self.lhs[1]} {rhs})"


def parse_vnnlib(path: Path):
    """Return (input_bounds: [(lo, hi)], clauses: [[Atom, ...], ...]).

    The property is the DISJUNCTION of the clauses (a plain conjunction
    parses as one clause). Per-clause input atoms are not supported here —
    soundnessbench and the banked-sat set are global-box properties.
    """
    text = path.read_text()
    n_inputs = len(re.findall(r"declare-const X_\d+", text))
    lo = [float("-inf")] * n_inputs
    hi = [float("inf")] * n_inputs
    clauses: list[list[Atom]] = []

    for sexpr in _tokenize_sexpr(text):
        if not sexpr.startswith("(assert"):
            continue
        body = sexpr[len("(assert") : -1].strip()
        if body.startswith("(or") or body.startswith("(and"):
            if body.startswith("(and"):
                groups = [body]
            else:
                groups = re.findall(r"\(and[^()]*(?:\([^()]*\)[^()]*)*\)", body)
                if not groups:
                    raise ValueError(f"cannot split disjunction: {body[:120]}")
            for g in groups:
                atoms: list[Atom] = []
                for m in _ATOM.finditer(g):
                    op, v1, i1, v2, i2, const = m.groups()
                    if v1 != "Y":
                        raise ValueError(
                            "per-clause input atoms unsupported by this auditor"
                        )
                    rhs = ("Y", int(i2)) if v2 == "Y" else ("const", float(const))
                    atoms.append(Atom(op, ("Y", int(i1)), rhs))
                if not atoms:
                    raise ValueError(f"empty clause in: {g[:120]}")
                clauses.append(atoms)
        else:
            m = _ATOM.match(body)
            if not m:
                raise ValueError(f"unsupported assert: {body[:120]}")
            op, v1, i1, v2, i2, const = m.groups()
            if v1 == "X" and v2 is None:
                idx, c = int(i1), float(const)
                if op == "<=":
                    hi[idx] = min(hi[idx], c)
                else:
                    lo[idx] = max(lo[idx], c)
            elif v1 == "Y":
                rhs = ("Y", int(i2)) if v2 == "Y" else ("const", float(const))
                clauses.append([Atom(op, ("Y", int(i1)), rhs)])
            else:
                raise ValueError(f"unsupported assert: {body[:120]}")
    if not clauses:
        raise ValueError("no output property found")
    return list(zip(lo, hi)), clauses


def property_margin(clauses, y: np.ndarray) -> tuple[float, list[float]]:
    """(max over clauses of min over atoms, per-clause margins)."""
    per_clause = [min(a.margin(y) for a in clause) for clause in clauses]
    return max(per_clause), per_clause


# ----------------------------- witness parsing -----------------------------

def parse_witness(path: Path, n_inputs: int) -> np.ndarray:
    """Parse `(X_i v)` lines out of a witness or VNN-COMP results file."""
    vals: dict[int, float] = {}
    for m in re.finditer(r"\(\s*X_(\d+)\s+([-+0-9.eE]+)\s*\)", path.read_text()):
        vals[int(m.group(1))] = float(m.group(2))
    if len(vals) != n_inputs or set(vals) != set(range(n_inputs)):
        raise ValueError(
            f"witness has {len(vals)} X values, property declares {n_inputs}"
        )
    return np.array([vals[i] for i in range(n_inputs)], dtype=np.float64)


# --------------------------- exact-f64 forward -----------------------------

def _to_f64(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


class F64Forward:
    """Exact-f64 numpy executor for the supported op set (fails loudly)."""

    SUPPORTED = {
        "Gemm", "Relu", "Reshape", "Conv", "Flatten", "Constant",
        "Add", "Sub", "MatMul", "Sigmoid", "Identity",
    }

    def __init__(self, onnx_path: Path):
        import onnx
        from onnx import numpy_helper

        self.model = onnx.load(str(onnx_path))
        g = self.model.graph
        unsupported = {n.op_type for n in g.node} - self.SUPPORTED
        if unsupported:
            raise ValueError(f"exact-f64 forward: unsupported ops {sorted(unsupported)}")
        self.inits = {i.name: _to_f64(numpy_helper.to_array(i)) for i in g.initializer}
        self.raw_inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
        self.input_name = g.input[0].name
        self.output_name = g.output[0].name
        dims = [
            d.dim_value if d.HasField("dim_value") else 1
            for d in g.input[0].type.tensor_type.shape.dim
        ]
        self.input_shape = [max(d, 1) for d in dims]

    @staticmethod
    def _attr(node, name, default=None):
        for a in node.attribute:
            if a.name == name:
                if a.ints:
                    return list(a.ints)
                if a.HasField("i"):
                    return a.i
                if a.HasField("f"):
                    return a.f
                if a.HasField("t"):
                    from onnx import numpy_helper

                    return numpy_helper.to_array(a.t)
        return default

    def run(self, x_flat: np.ndarray) -> np.ndarray:
        """x_flat: float64 flat input (already the f32-cast view widened)."""
        # macOS Accelerate raises spurious FP-state flags on some strided f64
        # matmuls; results are verified finite below, so silence the flags.
        with np.errstate(all="ignore"):
            out = self._run_inner(x_flat)
        if not np.all(np.isfinite(out)):
            raise ValueError("exact-f64 forward produced non-finite outputs")
        return out

    def _run_inner(self, x_flat: np.ndarray) -> np.ndarray:
        vals: dict[str, np.ndarray] = dict(self.inits)
        vals[self.input_name] = _to_f64(x_flat).reshape(self.input_shape)
        for node in self.model.graph.node:
            t = node.op_type
            inp = [vals[i] for i in node.input if i]
            if t == "Constant":
                out = _to_f64(self._attr(node, "value"))
            elif t == "Identity":
                out = inp[0]
            elif t == "Relu":
                out = np.maximum(inp[0], 0.0)
            elif t == "Sigmoid":
                out = 1.0 / (1.0 + np.exp(-inp[0]))
            elif t == "Add":
                out = inp[0] + inp[1]
            elif t == "Sub":
                out = inp[0] - inp[1]
            elif t == "MatMul":
                out = inp[0] @ inp[1]
            elif t == "Gemm":
                a, b = inp[0], inp[1]
                if self._attr(node, "transA", 0):
                    a = a.T
                if self._attr(node, "transB", 0):
                    b = b.T
                alpha = float(self._attr(node, "alpha", 1.0))
                beta = float(self._attr(node, "beta", 1.0))
                out = alpha * (a @ b)
                if len(inp) > 2:
                    out = out + beta * inp[2]
            elif t == "Reshape":
                shape = [int(v) for v in inp[1].ravel()]
                # ONNX semantics: 0 copies the input dim; numpy resolves -1.
                resolved = [
                    inp[0].shape[i] if s == 0 else s for i, s in enumerate(shape)
                ]
                out = inp[0].reshape(resolved)
            elif t == "Flatten":
                axis = int(self._attr(node, "axis", 1))
                shp = inp[0].shape
                axis = axis if axis >= 0 else len(shp) + axis
                before = int(np.prod(shp[:axis])) if axis > 0 else 1
                out = inp[0].reshape(before, -1)
            elif t == "Conv":
                out = self._conv(node, inp)
            else:  # pragma: no cover — filtered in __init__
                raise ValueError(f"unsupported op {t}")
            vals[node.output[0]] = out
        return vals[self.output_name].ravel()

    def _conv(self, node, inp):
        x, w = inp[0], inp[1]
        b = inp[2] if len(inp) > 2 else None
        strides = self._attr(node, "strides", [1, 1])
        pads = self._attr(node, "pads", [0, 0, 0, 0])
        dil = self._attr(node, "dilations", [1, 1])
        group = int(self._attr(node, "group", 1))
        if group != 1 or dil != [1, 1]:
            raise ValueError("exact-f64 conv: group/dilation unsupported")
        if x.ndim == 3:
            x = x[None, ...]
        n, ic, ih, iw = x.shape
        oc, _, kh, kw = w.shape
        sh, sw = strides
        ph0, pw0, ph1, pw1 = pads
        xp = np.pad(x, ((0, 0), (0, 0), (ph0, ph1), (pw0, pw1)))
        oh = (ih + ph0 + ph1 - kh) // sh + 1
        ow = (iw + pw0 + pw1 - kw) // sw + 1
        # im2col in f64 (exact data movement), one f64 matmul per conv.
        cols = np.empty((n, ic * kh * kw, oh * ow), dtype=np.float64)
        idx = 0
        for c in range(ic):
            for ki in range(kh):
                for kj in range(kw):
                    patch = xp[:, c, ki : ki + sh * oh : sh, kj : kj + sw * ow : sw]
                    cols[:, idx, :] = patch.reshape(n, -1)
                    idx += 1
        wmat = w.reshape(oc, ic * kh * kw)
        out = np.einsum("ok,nkp->nop", wmat, cols, optimize=True)
        if b is not None:
            out = out + b.reshape(1, oc, 1)
        return out.reshape(n, oc, oh, ow)


# --------------------------------- driver ----------------------------------

def check_one(onnx_path: Path, vnnlib_path: Path, witness_path: Path,
              bar: float, ort_version: str):
    import onnxruntime as ort

    if ort.__version__ != ort_version:
        raise RuntimeError(
            f"independent-ORT pin violated: have {ort.__version__}, "
            f"need {ort_version} (override with --ort-version)"
        )

    bounds, clauses = parse_vnnlib(vnnlib_path)
    x64 = parse_witness(witness_path, len(bounds))

    # Zero-tolerance box membership on the f64 parse (organizer's asserts).
    in_box = all(lo <= v <= hi for v, (lo, hi) in zip(x64, bounds))

    # Both engines see the f32 cast (the organizer feeds ORT f32 tensors).
    x32 = x64.astype(np.float32)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in in_meta.shape]
    y_ort = sess.run(None, {in_meta.name: x32.reshape(shape)})[0].ravel()
    ort_margin, ort_clauses = property_margin(clauses, _to_f64(y_ort))

    fwd = F64Forward(onnx_path)
    y_f64 = fwd.run(_to_f64(x32))
    f64_margin, f64_clauses = property_margin(clauses, y_f64)

    accept = bool(in_box and ort_margin >= bar and f64_margin >= bar)
    return {
        "onnx": str(onnx_path),
        "vnnlib": str(vnnlib_path),
        "witness": str(witness_path),
        "in_box_zero_tol": in_box,
        "ort_version": ort.__version__,
        "ort_property_margin": ort_margin,
        "f64_property_margin": f64_margin,
        "joint_margin": min(ort_margin, f64_margin),
        "bar": bar,
        "accept": accept,
        "ort_clause_margins": ort_clauses,
        "f64_clause_margins": f64_clauses,
        "max_engine_disagreement": float(
            np.max(np.abs(_to_f64(y_ort) - y_f64))
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--vnnlib", required=True, type=Path)
    ap.add_argument("--witness", required=True, type=Path,
                    help="witness or VNN-COMP results file containing (X_i v) lines")
    ap.add_argument("--bar", type=float, default=1e-5,
                    help="accept margin on BOTH engines (default 1e-5)")
    ap.add_argument("--ort-version", default="1.19.2")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    try:
        report = check_one(args.onnx, args.vnnlib, args.witness, args.bar,
                           args.ort_version)
    except Exception as e:  # noqa: BLE001 — audit tool: surface everything
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    line = (
        f"{'ACCEPT' if report['accept'] else 'REJECT'} "
        f"in_box={report['in_box_zero_tol']} "
        f"ort_margin={report['ort_property_margin']:+.6e} "
        f"f64_margin={report['f64_property_margin']:+.6e} "
        f"bar={report['bar']:.1e} "
        f"engine_disagreement={report['max_engine_disagreement']:.3e}"
    )
    print(line)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2))
    return 0 if report["accept"] else 2


if __name__ == "__main__":
    sys.exit(main())
