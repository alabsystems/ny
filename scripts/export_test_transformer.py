#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Export minimal transformer components to ONNX for ny testing."""

import os

import torch
import torch.nn as nn

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "models")
CAUSAL_MASK_VALUE = -1.0e4


class SimpleLayerNorm(nn.Module):
    """LayerNorm only."""
    def __init__(self, dim=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class SimpleSoftmax(nn.Module):
    """Softmax only."""
    def __init__(self, dim=-1):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.softmax(x)


class SimpleGELU(nn.Module):
    """GELU only."""
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x)


class SimpleMLP(nn.Module):
    """MLP with GELU activation (transformer FFN pattern)."""
    def __init__(self, dim=4, hidden_mult=4):
        super().__init__()
        hidden_dim = dim * hidden_mult
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class SimpleAttention(nn.Module):
    """Minimal single-head attention."""
    def __init__(self, dim=4):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        # x: (batch, seq, dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out)


class MinimalTransformerBlock(nn.Module):
    """One transformer block: Attention + FFN with residuals and LayerNorm."""
    def __init__(self, dim=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimpleAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SimpleMLP(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CausalAttention(nn.Module):
    """Causal (masked) single-head attention for decoders."""
    def __init__(self, dim=4):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        # x: (batch, seq, dim)
        batch, seq, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Attention: softmax(Q @ K^T / sqrt(d) + mask) @ V
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask: a large finite negative value for future positions.
        # ONNX constant folding can turn 0 * -inf into NaN, and ny rejects
        # non-finite constants when converting fixtures to propagation layers.
        mask = torch.triu(
            torch.full((seq, seq), CAUSAL_MASK_VALUE, dtype=attn.dtype, device=attn.device),
            diagonal=1,
        )
        attn = attn + mask.to(attn.device)

        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out)


class DecoderBlock(nn.Module):
    """Decoder transformer block with causal self-attention."""
    def __init__(self, dim=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CausalAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SimpleMLP(dim)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttention(nn.Module):
    """Cross-attention for encoder-decoder models."""
    def __init__(self, dim=4):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x, encoder_output):
        # x: (batch, seq_q, dim) - decoder input
        # encoder_output: (batch, seq_k, dim) - encoder output
        q = self.q_proj(x)
        k = self.k_proj(encoder_output)
        v = self.v_proj(encoder_output)

        # Cross-attention: no causal mask
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out)


class EncoderDecoderBlock(nn.Module):
    """Full encoder-decoder decoder block (like Whisper decoder)."""
    def __init__(self, dim=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CausalAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = SimpleMLP(dim)

    def forward(self, x, encoder_output):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), encoder_output)
        x = x + self.mlp(self.norm3(x))
        return x


def export_model(model, name, input_shape, opset=14):
    """Export a model to ONNX."""
    model.eval()
    dummy_input = torch.randn(*input_shape)
    output_path = os.path.join(OUTPUT_DIR, f"{name}.onnx")

    # Use dynamo=False to use legacy TorchScript-based export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,  # Use legacy exporter
    )
    print(f"Exported {name} to {output_path}")

    # Print ONNX ops used
    import onnx
    onnx_model = onnx.load(output_path)
    ops = set(node.op_type for node in onnx_model.graph.node)
    print(f"  ONNX ops: {sorted(ops)}")
    print(f"  Size: {os.path.getsize(output_path)} bytes")
    return output_path


def export_dual_input_model(model, name, input_shapes, input_names, opset=14):
    """Export a model with two inputs to ONNX."""
    model.eval()
    dummy_inputs = tuple(torch.randn(*shape) for shape in input_shapes)
    output_path = os.path.join(OUTPUT_DIR, f"{name}.onnx")

    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported {name} to {output_path}")

    import onnx
    onnx_model = onnx.load(output_path)
    ops = set(node.op_type for node in onnx_model.graph.node)
    print(f"  ONNX ops: {sorted(ops)}")
    print(f"  Size: {os.path.getsize(output_path)} bytes")
    return output_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Exporting transformer test models...\n")

    # Single operations
    export_model(SimpleSoftmax(dim=-1), "softmax", (1, 4))
    export_model(SimpleGELU(), "gelu", (1, 4))
    export_model(SimpleLayerNorm(4), "layer_norm", (1, 4))

    # Transformer components
    export_model(SimpleMLP(4), "transformer_mlp", (1, 4))
    export_model(SimpleAttention(4), "simple_attention", (1, 2, 4))  # batch=1, seq=2, dim=4

    # Full transformer block (encoder style - bidirectional attention)
    export_model(MinimalTransformerBlock(4), "transformer_block", (1, 2, 4))

    # Causal attention (decoder style)
    export_model(CausalAttention(4), "causal_attention", (1, 4, 4))  # batch=1, seq=4, dim=4

    # Decoder block (decoder-only transformer)
    export_model(DecoderBlock(4), "decoder_block", (1, 4, 4))

    # Cross attention (encoder-decoder)
    export_dual_input_model(
        CrossAttention(4),
        "cross_attention",
        [(1, 2, 4), (1, 4, 4)],  # decoder: seq=2, encoder: seq=4
        ["decoder_input", "encoder_output"],
    )

    # Full encoder-decoder decoder block
    export_dual_input_model(
        EncoderDecoderBlock(4),
        "encoder_decoder_block",
        [(1, 2, 4), (1, 4, 4)],
        ["decoder_input", "encoder_output"],
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
