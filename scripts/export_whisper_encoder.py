#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Export Whisper-tiny encoder from HuggingFace to ONNX for verification benchmarking.

This script creates the whisper_tiny_encoder.onnx test model needed for
VISION.md Phase 4: Practical Whisper-scale verification.

Requirements:
    pip install transformers torch onnx

Usage:
    python scripts/export_whisper_encoder.py

Output:
    tests/models/whisper_tiny_encoder.onnx (~33MB)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def export_whisper_encoder(output_path: str, opset_version: int = 18) -> None:
    """Export Whisper-tiny encoder to ONNX format.

    Args:
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version (default: 14)
    """
    try:
        import torch
        from transformers import WhisperModel
    except ImportError as e:
        print(f"Error: Missing dependencies - {e}")
        print("Install with: pip install transformers torch")
        sys.exit(1)

    print("Loading Whisper-tiny model from HuggingFace...")
    # Load just the encoder from whisper-tiny
    model = WhisperModel.from_pretrained("openai/whisper-tiny")
    encoder = model.get_encoder()
    encoder.eval()

    # Whisper-tiny encoder specs:
    # - hidden_dim: 384
    # - num_layers: 4
    # - num_heads: 6
    # - head_dim: 64 (384 / 6)
    # - ffn_dim: 1536 (384 * 4)
    # - input: mel spectrogram [batch, n_mels=80, seq_len]
    # After conv stem: [batch, seq_len', hidden_dim=384]

    print(f"Encoder config: {encoder.config}")
    print(f"  hidden_size: {encoder.config.d_model}")
    print(f"  num_layers: {encoder.config.encoder_layers}")
    print(f"  num_heads: {encoder.config.encoder_attention_heads}")

    # Create dummy input matching Whisper's expected input shape
    # Whisper encoder expects: [batch, feature_dim, seq_len]
    # For whisper-tiny: feature_dim=80 (mel bins), seq_len=3000 (30s audio)
    # Whisper expects full-length mel input; use 3000 to match model constraints.
    batch_size = 1
    n_mels = 80
    seq_len = 3000

    dummy_input = torch.randn(batch_size, n_mels, seq_len)

    print(f"Input shape: {dummy_input.shape}")
    print("Exporting to ONNX...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export to ONNX
    torch.onnx.export(
        encoder,
        dummy_input,
        output_path,
        input_names=["input_features"],
        output_names=["last_hidden_state"],
        opset_version=opset_version,
        external_data=False,
        do_constant_folding=True,
        dynamic_axes={
            "input_features": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"},
        },
    )

    # Verify the exported model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Print stats
    file_size = os.path.getsize(output_path)
    num_nodes = len(onnx_model.graph.node)
    op_types = sorted(set(n.op_type for n in onnx_model.graph.node))

    print(f"\nExported successfully to: {output_path}")
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
    print(f"  ONNX nodes: {num_nodes}")
    print(f"  Op types: {op_types}")

    repo_root = Path(__file__).resolve().parent.parent
    try:
        git_result = subprocess.run(
            ["git", "check-ignore", "-q", str(Path(output_path).resolve())],
            cwd=repo_root,
            check=False,
        )
        if git_result.returncode == 1:
            print(
                "\nNote: this model is large and should not be checked into git. "
                "Add it to .gitignore or store it with LFS."
            )
    except FileNotFoundError:
        pass

    # Count parameters
    param_count = sum(
        len(init.raw_data) // 4  # float32 = 4 bytes
        for init in onnx_model.graph.initializer
    )
    print(f"  Parameters: {param_count:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Whisper-tiny encoder to ONNX"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(Path(__file__).parent.parent / "tests" / "models" / "whisper_tiny_encoder.onnx"),
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)",
    )
    args = parser.parse_args()

    export_whisper_encoder(args.output, args.opset)


if __name__ == "__main__":
    main()
