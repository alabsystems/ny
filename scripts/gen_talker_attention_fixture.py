#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Generate a minimal synthetic talker_attention_layer0.onnx fixture.

Creates a simplified single-head attention ONNX model with the same input/output
names as the real Qwen3-TTS talker attention layer:
  - hidden_states [B, T, 2048]
  - cos [1, T, 64]       (3D — no Squeeze needed for broadcast with [B, T, 64])
  - sin [1, T, 64]       (3D — same)
  - mask [1, T, T]       (3D — same)
  - attn_output [B, T, 2048]

Contains MatMul and Softmax nodes to satisfy the parse-structure gate.
Uses tiny random weights (~KBs instead of ~50MB).

Usage:
    python3 scripts/gen_talker_attention_fixture.py
"""

import logging
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

log = logging.getLogger(__name__)

D_MODEL = 2048
D_HEAD = 64  # Matches RoPE dim and keeps model small
OPSET = 17


def make_initializer(name: str, shape: list, scale: float = 0.02) -> onnx.TensorProto:
    """Create a small random weight initializer."""
    rng = np.random.default_rng(seed=hash(name) % (2**31))
    data = (rng.standard_normal(shape) * scale).astype(np.float32)
    return numpy_helper.from_array(data, name=name)


def build_graph() -> onnx.GraphProto:
    # --- Inputs (3D cos/sin/mask — no Squeeze needed) ---
    hidden_states = helper.make_tensor_value_info(
        "hidden_states", TensorProto.FLOAT, ["B", "T", D_MODEL]
    )
    cos_input = helper.make_tensor_value_info(
        "cos", TensorProto.FLOAT, [1, "T", D_HEAD]
    )
    sin_input = helper.make_tensor_value_info(
        "sin", TensorProto.FLOAT, [1, "T", D_HEAD]
    )
    mask_input = helper.make_tensor_value_info(
        "mask", TensorProto.FLOAT, [1, "T", "T"]
    )

    # --- Output ---
    attn_output = helper.make_tensor_value_info(
        "attn_output", TensorProto.FLOAT, ["B", "T", D_MODEL]
    )

    # --- Weight initializers ---
    # Fold 1/sqrt(d) scaling into Wq so no separate Mul(scale_factor) is needed.
    scale = 1.0 / np.sqrt(D_HEAD)
    initializers = [
        make_initializer("Wq", [D_MODEL, D_HEAD], scale=0.02 * scale),
        make_initializer("Wk", [D_MODEL, D_HEAD]),
        make_initializer("Wv", [D_MODEL, D_HEAD]),
        make_initializer("Wo", [D_HEAD, D_MODEL]),
    ]

    nodes = []

    # Step 1: Q = hidden_states @ Wq  -> [B, T, D_HEAD]
    nodes.append(helper.make_node("MatMul", ["hidden_states", "Wq"], ["q_raw"]))

    # Step 2: K = hidden_states @ Wk  -> [B, T, D_HEAD]
    nodes.append(helper.make_node("MatMul", ["hidden_states", "Wk"], ["k_raw"]))

    # Step 3: V = hidden_states @ Wv  -> [B, T, D_HEAD]
    nodes.append(helper.make_node("MatMul", ["hidden_states", "Wv"], ["v"]))

    # Step 4: Simplified RoPE using 3D cos/sin directly (no Squeeze needed)
    # q_rope = q_raw * cos  (simplified, cos broadcasts [1,T,64] with [B,T,64])
    nodes.append(helper.make_node("Mul", ["q_raw", "cos"], ["q_cos"]))
    # k with sin: k_sin = k_raw * sin
    nodes.append(helper.make_node("Mul", ["k_raw", "sin"], ["k_sin"]))
    # k = k_raw + k_sin (combine)
    nodes.append(helper.make_node("Add", ["k_raw", "k_sin"], ["k"]))

    # Step 5: Scores = q_cos @ Transpose(k) -> [B, T, T]
    nodes.append(
        helper.make_node("Transpose", ["k"], ["k_t"], perm=[0, 2, 1])
    )
    nodes.append(helper.make_node("MatMul", ["q_cos", "k_t"], ["scores"]))

    # Step 6: Add mask (3D [1, T, T] broadcasts with [B, T, T])
    # (scaling already folded into Wq)
    nodes.append(helper.make_node("Add", ["scores", "mask"], ["scores_masked"]))

    # Step 8: Softmax -> attention probabilities
    nodes.append(
        helper.make_node("Softmax", ["scores_masked"], ["attn_probs"], axis=-1)
    )

    # Step 9: Context = attn_probs @ V  -> [B, T, D_HEAD]
    nodes.append(helper.make_node("MatMul", ["attn_probs", "v"], ["context"]))

    # Step 10: Output projection: context @ Wo -> [B, T, D_MODEL]
    nodes.append(helper.make_node("MatMul", ["context", "Wo"], ["attn_output"]))

    graph = helper.make_graph(
        nodes,
        "talker_attention_layer0",
        [hidden_states, cos_input, sin_input, mask_input],
        [attn_output],
        initializer=initializers,
    )
    return graph


def main():
    logging.basicConfig(level=logging.INFO)
    graph = build_graph()
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9

    onnx.checker.check_model(model)

    output_path = "tests/models/talker_attention_layer0.onnx"
    onnx.save(model, output_path)
    size_kb = len(model.SerializeToString()) / 1024
    log.info("Saved %s (%.1f KB)", output_path, size_kb)
    log.info("  Inputs: %s", [inp.name for inp in model.graph.input])
    log.info("  Outputs: %s", [out.name for out in model.graph.output])
    log.info("  Nodes: %d", len(model.graph.node))
    log.info("  Node types: %s", [n.op_type for n in model.graph.node])


if __name__ == "__main__":
    main()
