#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the ONNX Runtime oracle fixtures of `tests/lowering_ort.rs`.

The checker's independent lowering (`ny-proofcheck`, B2) is cross-checked
against ONNX Runtime on small models that exercise every routing form it
admits (Slice, Gather, Concat, Reshape, Flatten, MatMul with a constant on
either side, rank-1 and non-batch graph inputs). The models are built with
the official `onnx` helper -- a different writer from the Rust test support's
-- and evaluated by onnxruntime at fixed points. Every weight and input is a
small dyadic rational, so ORT's float32 arithmetic is EXACT here and the
committed outputs are the exact real values: the Rust test asserts strict
containment in the checker's certified point enclosure, with no tolerance.

Run (any python with onnx + onnxruntime + numpy; on the GB10 host the venv
`<home>/x86emu/venv-arm` has onnx 1.21 / onnxruntime 1.26):

    python3 crates/ny-proofcheck/tests/data/ort_forms/gen_ort_forms.py

It rewrites the `*.onnx` files and `manifest.json` next to itself and fails
(non-zero) if any output is not exactly representable or not finite.
"""

import hashlib
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
I64_MAX = 9223372036854775807
I64_MIN = -9223372036854775808
OPSET = 13


def dyadic(rng, shape, choices=(-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)):
    return rng.choice(np.array(choices, dtype=np.float32), size=shape).astype(np.float32)


class G:
    """A tiny graph builder: constants are initializers, names are unique."""

    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.inits = []
        self.k = 0

    def fresh(self, base):
        self.k += 1
        return f"{base}_{self.k}"

    def i64(self, vals, scalar=False):
        n = self.fresh("i")
        arr = np.array(vals, dtype=np.int64)
        if scalar:
            arr = arr.reshape(())
        self.inits.append(numpy_helper.from_array(arr, n))
        return n

    def f32(self, arr):
        n = self.fresh("c")
        self.inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), n))
        return n

    def op(self, op_type, inputs, **attrs):
        out = self.fresh(op_type.lower())
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def slice(self, x, starts, ends, axes=None, steps=None):
        ins = [x, self.i64(starts), self.i64(ends)]
        if axes is not None or steps is not None:
            ins.append(self.i64(axes) if axes is not None else "")
        if steps is not None:
            ins.append(self.i64(steps))
        return self.op("Slice", ins)

    def flat(self, x):
        return self.op("Reshape", [x, self.i64([1, -1])])

    def model(self, inp, in_shape, out_parts):
        out = self.op("Concat", out_parts, axis=1) if len(out_parts) > 1 else out_parts[0]
        final = "y"
        self.nodes.append(helper.make_node("Identity", [out], [final]))
        graph = helper.make_graph(
            self.nodes,
            self.name,
            [helper.make_tensor_value_info(inp, TensorProto.FLOAT, in_shape)],
            [helper.make_tensor_value_info(final, TensorProto.FLOAT, [1, "n"])],
            self.inits,
        )
        m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
        m.ir_version = 8
        onnx.checker.check_model(m)
        return m


def fx_slice(rng):
    g = G("slice")
    x = "x"
    parts = []
    # Negative steps, INT64_MIN/MAX ends, negative axes, omitted axes/steps.
    parts.append(g.flat(g.slice(x, [2, -1], [I64_MIN, 0], [1, 3], [-1, -2])))
    parts.append(g.flat(g.slice(x, [1], [I64_MAX], [-2], [2])))
    parts.append(g.flat(g.slice(x, [0, 1, 3], [1, 3, 99])))
    parts.append(g.flat(g.slice(x, [-2, 4], [-100, -6], [2, 3], [-3, -1])))
    parts.append(g.flat(g.slice(x, [5, -4], [0, 9], [3, 1], [-2, 2])))
    return g.model(x, [1, 3, 4, 5], parts), [1, 3, 4, 5], True


def fx_gather(rng):
    g = G("gather")
    x = "x"
    parts = []
    parts.append(g.flat(g.op("Gather", [x, g.i64([[3, -1], [0, 0]])], axis=2)))
    parts.append(g.flat(g.op("Gather", [x, g.i64([-2], scalar=True)], axis=1)))
    parts.append(g.flat(g.op("Gather", [x, g.i64([0, 0])], axis=0)))
    parts.append(g.flat(g.op("Gather", [x, g.i64([2, 1, 2])], axis=-2)))
    return g.model(x, [1, 3, 4], parts), [1, 3, 4], True


def fx_concat(rng):
    g = G("concat")
    x = "x"
    a = g.slice(x, [0], [2], [2])
    b = g.slice(x, [1], [3], [2])
    parts = [
        g.flat(g.op("Concat", [a, b, a], axis=2)),
        g.flat(g.op("Concat", [a, b], axis=-2)),
        g.flat(g.op("Concat", [b, x, a], axis=2)),
        g.flat(g.op("Concat", [x, x], axis=0)),
    ]
    return g.model(x, [1, 2, 3], parts), [1, 2, 3], True


def fx_reshape(rng):
    g = G("reshape")
    x = "x"
    parts = [
        g.flat(g.op("Reshape", [x, g.i64([0, -1, 2])])),
        g.flat(g.op("Reshape", [x, g.i64([4, 0, 3])])),
        g.flat(g.op("Flatten", [x], axis=0)),
        g.flat(g.op("Flatten", [x], axis=-1)),
        g.flat(g.op("Flatten", [x], axis=4)),
        g.flat(g.slice(g.op("Reshape", [x, g.i64([3, -1])]), [2], [0], [0], [-1])),
    ]
    return g.model(x, [1, 2, 3, 4], parts), [1, 2, 3, 4], True


def fx_matmul(rng):
    g = G("matmul")
    x = "x"
    xa = g.op("Reshape", [x, g.i64([4, 6])])
    xb = g.op("Reshape", [x, g.i64([24])])
    xc = g.op("Reshape", [x, g.i64([2, 3, 4])])
    xd = g.op("Reshape", [x, g.i64([1, 6, 4])])
    xe = g.op("Reshape", [x, g.i64([2, 1, 3, 4])])
    parts = [
        g.flat(g.op("MatMul", [g.f32(dyadic(rng, (3, 4))), xa])),
        g.flat(g.op("MatMul", [xb, g.f32(dyadic(rng, (24, 2)))])),
        g.flat(g.op("MatMul", [xc, g.f32(dyadic(rng, (4, 2)))])),
        g.flat(g.op("MatMul", [g.f32(dyadic(rng, (2, 3, 6))), xd])),
        g.flat(g.op("MatMul", [g.f32(dyadic(rng, (4,))), xa])),
        g.flat(g.op("MatMul", [xa, g.f32(dyadic(rng, (6,)))])),
        g.flat(g.op("MatMul", [xe, g.f32(dyadic(rng, (5, 4, 2)))])),
        g.flat(g.op("MatMul", [x, g.f32(dyadic(rng, (24, 3)))])),
    ]
    return g.model(x, [1, 24], parts), [1, 24], False


def fx_rank1(rng):
    g = G("rank1")
    x = "x"
    s = g.slice(x, [1], [5], [0])
    h = g.op(
        "Gemm",
        [g.op("Reshape", [s, g.i64([1, 4])]), g.f32(dyadic(rng, (3, 4))), g.f32(dyadic(rng, (3,)))],
        transB=1,
    )
    r = g.op("Relu", [h])
    m = g.op("MatMul", [x, g.f32(dyadic(rng, (6, 2)))])
    parts = [r, g.flat(m), g.flat(g.op("Gather", [x, g.i64([5, 0, 5])], axis=0))]
    return g.model(x, [6], parts), [6], False


def fx_nonbatch(rng):
    g = G("nonbatch")
    x = "x"
    parts = [
        g.flat(g.op("Gather", [x, g.i64([2, 0])], axis=0)),
        g.op("Flatten", [x], axis=0),
        g.flat(g.op("MatMul", [g.f32(dyadic(rng, (2, 3))), x])),
    ]
    return g.model(x, [3, 2], parts), [3, 2], False


def fx_routing_mlp(rng):
    g = G("routing_mlp")
    x = "x"
    lo = g.slice(x, [0], [3], [1])
    hi = g.slice(x, [3], [I64_MAX], [1])
    col = g.op("Reshape", [lo, g.i64([3, 1])])
    z1 = g.op("MatMul", [g.f32(dyadic(rng, (4, 3))), col])
    z2 = g.op("MatMul", [hi, g.f32(dyadic(rng, (3, 4)))])
    h1 = g.op("Relu", [g.op("Reshape", [z1, g.i64([1, 4])])])
    h2 = g.op("Relu", [z2])
    cat = g.op("Concat", [h2, h1], axis=1)
    perm = g.op("Gather", [cat, g.i64([7, 0, 3, 3, 5, 1, 6, 2, 4])], axis=1)
    y = g.op("Gemm", [perm, g.f32(dyadic(rng, (2, 9))), g.f32(dyadic(rng, (2,)))], transB=1)
    return g.model(x, [1, 6], [y]), [1, 6], False


FIXTURES = [
    fx_slice,
    fx_gather,
    fx_concat,
    fx_reshape,
    fx_matmul,
    fx_rank1,
    fx_nonbatch,
    fx_routing_mlp,
]


def exact_f32(v):
    return np.isfinite(v) and float(np.float32(v)) == float(v)


def main():
    rng = np.random.default_rng(20261008)
    entries = []
    for fx in FIXTURES:
        m, in_shape, routing_only = fx(rng)
        name = m.graph.name
        path = os.path.join(HERE, f"{name}.onnx")
        data = m.SerializeToString()
        with open(path, "wb") as f:
            f.write(data)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        points, outputs = [], []
        for _ in range(4):
            p = (rng.integers(-16, 17, size=in_shape) / 8.0).astype(np.float32)
            y = sess.run(None, {"x": p})[0]
            if y.ndim != 2 or y.shape[0] != 1:
                raise SystemExit(f"{name}: output shape {y.shape} is not [1, N]")
            ys = [float(v) for v in y.reshape(-1)]
            if not all(exact_f32(v) for v in ys):
                raise SystemExit(f"{name}: a non-finite or inexact output")
            points.append([repr(float(v)) for v in p.reshape(-1)])
            outputs.append([repr(v) for v in ys])
        entries.append(
            {
                "file": f"{name}.onnx",
                "sha256": hashlib.sha256(data).hexdigest(),
                "routing_only": routing_only,
                "points": points,
                "outputs": outputs,
            }
        )
    manifest = {
        "generator": "gen_ort_forms.py",
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "opset": OPSET,
        "fixtures": entries,
    }
    with open(os.path.join(HERE, "manifest.json"), "w") as f:
        json.dump(manifest, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {len(entries)} fixtures with onnxruntime {ort.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
