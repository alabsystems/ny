#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Create ONNX test models for Floor, Ceil, Round, Sign, Reciprocal ops using onnx directly."""

import os

import onnx
from onnx import TensorProto, helper

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "models")


def create_unary_op_model(op_name: str, filename: str):
    """Create a simple ONNX model with a single unary op."""
    # Input
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])

    # Output
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

    # Node
    node = helper.make_node(
        op_name,
        inputs=["input"],
        outputs=["output"],
    )

    # Graph
    graph = helper.make_graph(
        [node],
        f"{op_name}_test",
        [input_tensor],
        [output_tensor],
    )

    # Model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    # Validate and save
    onnx.checker.check_model(model)
    output_path = os.path.join(MODELS_DIR, filename)
    onnx.save(model, output_path)
    print(f"Created: {output_path}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    ops = [
        ("Floor", "floor_test.onnx"),
        ("Ceil", "ceil_test.onnx"),
        ("Round", "round_test.onnx"),
        ("Sign", "sign_test.onnx"),
        ("Reciprocal", "reciprocal_test.onnx"),
    ]

    for op_name, filename in ops:
        create_unary_op_model(op_name, filename)

    print("\nAll models created successfully.")


if __name__ == "__main__":
    main()
