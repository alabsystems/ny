#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the install-time self-test network `abs_relu.onnx`.

    f(x) = relu(x) + relu(-x) = |x|,   x in R^1 (shape [1, 1]).

`install_tool.sh` runs the packaged binary through `ny vnncomp` on this network
with two properties over the box x in [-1, 1], whose answers follow from the
arithmetic alone (max |x| on the box is exactly 1, attained at x = +-1):

    abs_relu_unsat.vnnlib   violation Y_0 >= 2     unreachable  -> unsat
    abs_relu_sat.vnnlib     violation Y_0 >= 0.5   reachable    -> sat

The ONNX file is committed; this script only documents and reproduces it
(`python3 make_selftest.py` with the `onnx` package installed). It needs no
ONNX Runtime; `--check` additionally evaluates the model with onnxruntime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent


def build() -> onnx.ModelProto:
    w1 = numpy_helper.from_array(np.array([[1.0, -1.0]], dtype=np.float32), "W1")
    w2 = numpy_helper.from_array(np.array([[1.0], [1.0]], dtype=np.float32), "W2")
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["X", "W1"], ["H"], name="hidden"),
            helper.make_node("Relu", ["H"], ["R"], name="relu"),
            helper.make_node("MatMul", ["R", "W2"], ["Y"], name="out"),
        ],
        "abs_relu",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])],
        initializer=[w1, w2],
    )
    model = helper.make_model(
        graph,
        producer_name="ny-install-selftest",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main() -> int:
    model = build()
    out = HERE / "abs_relu.onnx"
    out.write_bytes(model.SerializeToString(deterministic=True))
    print(f"wrote {out}")
    if "--check" in sys.argv[1:]:
        import onnxruntime as ort

        session = ort.InferenceSession(out.read_bytes(), providers=["CPUExecutionProvider"])
        for x in (-1.0, -0.5, 0.0, 0.25, 1.0):
            (y,) = session.run(None, {"X": np.array([[x]], dtype=np.float32)})
            assert float(y[0, 0]) == abs(x), (x, y)
            print(f"f({x}) = {float(y[0, 0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
