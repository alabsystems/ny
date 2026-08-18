#!/bin/bash
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
#
# Benchmark ny diff command performance.
# Created for #79: Diff timing benchmarks needed to validate VISION/README claims.
#
# VISION.md claims: "Model diffing: detect where implementations diverge (seconds vs hours)"
# Target: Sub-10s for Whisper-scale models (~39M params)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_MODELS="$REPO_ROOT/tests/models"
TEMP_DIR="$(mktemp -d)"
CUSTOM_MODEL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            CUSTOM_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model <path>]"
            echo ""
            echo "Options:"
            echo "  --model <path>  Benchmark a specific model instead of test models"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

echo "=== Ny Diff Benchmark ==="
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Target: <10s for Whisper-scale (~39M params)"
echo ""

# Build ny-cli in release mode for accurate benchmarks
echo "Building ny-cli in release mode..."
BUILD_LOG="$TEMP_DIR/cargo-build.log"
if ! cargo build -p ny-cli --release >"$BUILD_LOG" 2>&1; then
    cat "$BUILD_LOG" >&2
    exit 1
fi
sed '/Compiling\|Downloaded/d' "$BUILD_LOG"
NY="$REPO_ROOT/target/release/ny"

if [ ! -f "$NY" ]; then
    echo "ERROR: ny binary not found at $NY"
    exit 1
fi

if [ ! -d "$TEST_MODELS" ]; then
    echo "ERROR: Test models directory not found: $TEST_MODELS"
    echo "Build models first: cargo test --no-run"
    exit 1
fi

echo ""
echo "=== Available Test Models ==="
ls -lh "$TEST_MODELS"/*.onnx 2>/dev/null | awk '{print $5, $9}' | while read size path; do
    name=$(basename "$path")
    echo "  $size  $name"
done

echo ""
echo "=== Benchmark Results ==="
echo ""

# Helper function to create a perturbed model copy
create_perturbed() {
    local model="$1"
    local output="$2"
    # For benchmarking, we use the same model as both inputs
    # The diff command will compare layer-by-layer even for identical models
    # This measures the full graph traversal time
    cp "$model" "$output"
}

benchmark_model() {
    local name="$1"
    local model="$2"
    local perturbed
    perturbed="$TEMP_DIR/perturbed_$(basename "$model")"

    create_perturbed "$model" "$perturbed"

    # Warm-up run
    "$NY" diff "$model" "$perturbed" --tolerance 1e-5 > /dev/null 2>&1

    # Timed run
    local start
    local end
    start=$(python3 -c "import time; print(time.time())")
    "$NY" diff "$model" "$perturbed" --tolerance 1e-5 > /dev/null 2>&1
    end=$(python3 -c "import time; print(time.time())")

    local elapsed
    elapsed=$(python3 -c "print(f'{$end - $start:.3f}')")

    # Get model size
    local size
    size=$(ls -lh "$model" | awk '{print $5}')

    printf "%-30s %8s %8ss\n" "$name" "$size" "$elapsed"
}

# Print header
printf "%-30s %8s %8s\n" "Model" "Size" "Time"
printf "%-30s %8s %8s\n" "-----" "----" "----"

if [ -n "$CUSTOM_MODEL" ]; then
    # Benchmark custom model only
    if [ ! -f "$CUSTOM_MODEL" ]; then
        echo "ERROR: Model not found: $CUSTOM_MODEL"
        exit 1
    fi
    benchmark_model "$(basename "$CUSTOM_MODEL")" "$CUSTOM_MODEL"
else
    # Benchmark existing test models (smallest to largest)
    benchmark_model "simple_mlp.onnx" "$TEST_MODELS/simple_mlp.onnx"
    benchmark_model "transformer_block.onnx" "$TEST_MODELS/transformer_block.onnx"
    benchmark_model "encoder_decoder_block.onnx" "$TEST_MODELS/encoder_decoder_block.onnx"
    benchmark_model "mnist_mlp_2x50_trained.onnx" "$TEST_MODELS/mnist_mlp_2x50_trained.onnx"
    benchmark_model "mnist_mlp_2x50.onnx" "$TEST_MODELS/mnist_mlp_2x50.onnx"
    benchmark_model "cifar10_mlp_2x100_trained.onnx" "$TEST_MODELS/cifar10_mlp_2x100_trained.onnx"
    benchmark_model "cifar10_mlp_2x100.onnx" "$TEST_MODELS/cifar10_mlp_2x100.onnx"

    echo ""
    echo "=== Analysis ==="
    echo ""
    echo "Largest tested model: cifar10_mlp_2x100.onnx (~2.5MB)"
    echo ""
    echo "Extrapolation to Whisper-tiny (~39M params, ~150MB):"
    echo "  - If linear in model size: expect ~60x of cifar10 time"
    echo "  - If sub-linear (graph structure): likely faster"
    echo ""
    echo "Note: To benchmark actual Whisper-scale models, run:"
    echo "  python3 scripts/export_docling_to_onnx.py --model granite-docling-258M --trust-remote-code"
    echo "  scripts/benchmark_diff.sh --model models/docling/granite-docling-258M/vision_encoder.onnx"
fi

echo ""
echo "=== Summary ==="
echo "All benchmarks complete. Compare times to 10s target."
