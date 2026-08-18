#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
Performance benchmarking script for ny.

Measures:
1. CLI command latency on test models
2. WGPU vs CPU speedup for IBP operations
3. Scaling with model size and sequence length

Run from repo root:
    python scripts/benchmark_performance.py
"""

import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# Paths
NY_BIN = Path("target/release/ny")
MODELS_DIR = Path("tests/models")

@dataclass
class BenchmarkResult:
    model: str
    command: str
    elapsed_ms: float
    success: bool
    returncode: int | None
    stdout: str
    stderr: str

def run_benchmark(cmd: list[str], timeout: float = 60.0) -> BenchmarkResult:
    """Run a command and measure time/memory."""
    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return BenchmarkResult(
            model=cmd[2] if len(cmd) > 2 else "N/A",
            command=cmd[1] if len(cmd) > 1 else "N/A",
            elapsed_ms=elapsed_ms,
            success=(
                result.returncode == 0
                and math.isfinite(elapsed_ms)
                and elapsed_ms > 0
            ),
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            model=cmd[2] if len(cmd) > 2 else "N/A",
            command=cmd[1] if len(cmd) > 1 else "N/A",
            elapsed_ms=timeout * 1000,
            success=False,
            returncode=None,
            stdout="",
            stderr="TIMEOUT",
        )


def parse_json_result(result: BenchmarkResult, label: str) -> dict:
    """Require a successful process and a JSON object."""
    if not result.success:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"{label} failed with code {result.returncode}: {detail[:300]}"
        )
    json_start = result.stdout.find("{")
    if json_start < 0:
        raise RuntimeError(f"{label} returned no JSON")
    try:
        payload = json.loads(result.stdout[json_start:])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} JSON was not an object")
    return payload


def finite_number(value, *, nonnegative: bool = False, positive: bool = False) -> bool:
    """Check a JSON scalar without accepting booleans as numbers."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    numeric = float(value)
    if not math.isfinite(numeric):
        return False
    if positive:
        return numeric > 0
    if nonnegative:
        return numeric >= 0
    return True


def validate_verify_payload(
    payload: dict,
    expected_outputs: int,
    expected_backend: str,
    label: str,
) -> list[tuple[float, float]]:
    """Validate status, backend, and the complete expected output shape."""
    aliases = {"safe": "verified", "violated": "falsified"}
    raw_status = str(payload.get("property_status", payload.get("status", ""))).lower()
    status = aliases.get(raw_status, raw_status)
    if status != "verified":
        raise RuntimeError(f"{label} did not complete verification: {raw_status!r}")
    soundness = payload.get("soundness")
    if not isinstance(soundness, dict) or soundness.get("mode") != "sound":
        raise RuntimeError(f"{label} did not report sound verification")
    actual_method = (
        str(payload.get("actual_method", ""))
        .lower()
        .replace("-", "")
        .replace("_", "")
    )
    if payload.get("method") != "ibp" or actual_method != "ibp":
        raise RuntimeError(f"{label} did not execute the requested IBP method")
    if payload.get("backend") != expected_backend:
        raise RuntimeError(
            f"{label} backend mismatch: {payload.get('backend')!r} "
            f"!= {expected_backend!r}"
        )
    bounds = payload.get("output_bounds")
    if not isinstance(bounds, list) or len(bounds) != expected_outputs:
        raise RuntimeError(
            f"{label} output shape mismatch: "
            f"{len(bounds) if isinstance(bounds, list) else 'invalid'} "
            f"!= {expected_outputs}"
        )
    parsed = []
    try:
        for bound in bounds:
            lower = float(bound["lower"])
            upper = float(bound["upper"])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValueError
            parsed.append((lower, upper))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} returned malformed or non-finite bounds") from exc
    return parsed


def validate_analysis_payload(
    payload: dict,
    command: str,
    expected_layers: int,
    label: str,
) -> None:
    """Validate the shape and finite metrics of an analysis command."""
    layers = payload.get("layers")
    if not isinstance(layers, list) or len(layers) != expected_layers:
        raise RuntimeError(
            f"{label} layer shape mismatch: "
            f"{len(layers) if isinstance(layers, list) else 'invalid'} "
            f"!= {expected_layers}"
        )

    if command == "sensitivity":
        top_fields = [
            "input_epsilon", "final_width", "max_sensitivity", "total_sensitivity"
        ]
        layer_fields = ["input_width", "output_width", "sensitivity"]
    elif command == "profile-bounds":
        top_fields = [
            "input_epsilon", "initial_width", "final_width", "difficulty_score",
            "max_growth_ratio", "total_expansion",
        ]
        layer_fields = [
            "input_width", "output_width", "mean_output_width",
            "median_output_width", "growth_ratio", "cumulative_expansion",
        ]
    elif command == "quantize-check":
        top_fields = ["input_epsilon"]
        layer_fields = ["min_bound", "max_bound", "max_abs", "int8_scale"]
    else:
        raise RuntimeError(f"no validator for command {command}")

    if not all(finite_number(payload.get(field), nonnegative=True) for field in top_fields):
        raise RuntimeError(f"{label} returned invalid top-level metrics")
    for layer in layers:
        if not isinstance(layer, dict):
            raise RuntimeError(f"{label} returned a malformed layer entry")
        if not all(finite_number(layer.get(field)) for field in layer_fields):
            raise RuntimeError(f"{label} returned non-finite layer metrics")
        if command == "quantize-check":
            if float(layer["min_bound"]) > float(layer["max_bound"]):
                raise RuntimeError(f"{label} returned inverted quantization bounds")
            if float(layer["max_abs"]) < 0 or float(layer["int8_scale"]) < 0:
                raise RuntimeError(f"{label} returned negative scale metrics")
        elif any(float(layer[field]) < 0 for field in layer_fields):
            raise RuntimeError(f"{label} returned negative width metrics")


def close_bounds(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> bool:
    """Compare complete bound vectors across backends."""
    return (
        len(left) == len(right)
        and all(
            math.isclose(l_lower, r_lower, rel_tol=1e-5, abs_tol=1e-6)
            and math.isclose(l_upper, r_upper, rel_tol=1e-5, abs_tol=1e-6)
            for (l_lower, l_upper), (r_lower, r_upper) in zip(left, right)
        )
    )

def benchmark_cli_commands():
    """Benchmark all CLI commands on test models."""
    print("=" * 60)
    print("CLI Command Benchmarks")
    print("=" * 60)

    models = [
        "single_linear", "linear_relu", "simple_mlp",
        "softmax", "layer_norm", "transformer_mlp"
    ]
    output_dims = {
        "single_linear": 3,
        "linear_relu": 3,
        "simple_mlp": 2,
        "softmax": 4,
        "layer_norm": 4,
        "transformer_mlp": 4,
    }
    layer_counts = {
        "single_linear": 1,
        "linear_relu": 2,
        "simple_mlp": 3,
        "softmax": 1,
        "layer_norm": 1,
        "transformer_mlp": 3,
    }

    commands = [
        (
            "verify",
            [
                "--method", "ibp", "--backend", "cpu", "--json",
                "--strict", "--require-sound",
            ],
        ),
        ("sensitivity", ["--json"]),
        ("quantize-check", ["--json"]),
        ("profile-bounds", ["--json"]),
    ]

    results = []
    for model in models:
        model_path = MODELS_DIR / f"{model}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"required benchmark fixture missing: {model_path}")

        for cmd_name, extra_args in commands:
            cmd = [str(NY_BIN), cmd_name, str(model_path), "--epsilon", "0.01"] + extra_args

            # Warmup
            warmup = run_benchmark(cmd)
            warmup_payload = parse_json_result(
                warmup, f"{model}/{cmd_name} warm-up"
            )
            if cmd_name == "verify":
                validate_verify_payload(
                    warmup_payload, output_dims[model], "cpu",
                    f"{model}/{cmd_name} warm-up",
                )
            else:
                validate_analysis_payload(
                    warmup_payload, cmd_name, layer_counts[model],
                    f"{model}/{cmd_name} warm-up",
                )

            # Measure (3 runs)
            times = []
            for _ in range(3):
                result = run_benchmark(cmd)
                payload = parse_json_result(result, f"{model}/{cmd_name}")
                if cmd_name == "verify":
                    validate_verify_payload(
                        payload, output_dims[model], "cpu",
                        f"{model}/{cmd_name}",
                    )
                else:
                    validate_analysis_payload(
                        payload, cmd_name, layer_counts[model],
                        f"{model}/{cmd_name}",
                    )
                times.append(result.elapsed_ms)

            if len(times) != 3:
                raise RuntimeError(f"{model}/{cmd_name} returned an incomplete run set")
            avg_time = sum(times) / len(times)
            if not finite_number(avg_time, positive=True):
                raise RuntimeError(f"{model}/{cmd_name} returned invalid timing")
            results.append((model, cmd_name, avg_time))
            print(f"  {model:20s} | {cmd_name:15s} | {avg_time:8.2f} ms")

    return results

def benchmark_gpu_vs_cpu():
    """Benchmark GPU vs CPU performance."""
    print("\n" + "=" * 60)
    print("GPU vs CPU Comparison")
    print("=" * 60)

    models = {
        "single_linear": 3,
        "simple_mlp": 2,
        "transformer_mlp": 4,
    }

    results = []
    for model, output_dim in models.items():
        model_path = MODELS_DIR / f"{model}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"required benchmark fixture missing: {model_path}")

        # CPU benchmark
        cmd_cpu = [
            str(NY_BIN), "verify", str(model_path), "--epsilon", "0.01",
            "--method", "ibp", "--backend", "cpu", "--json",
            "--strict", "--require-sound",
        ]
        cpu_warmup = run_benchmark(cmd_cpu)
        cpu_warmup_payload = parse_json_result(cpu_warmup, f"{model}/CPU warm-up")
        cpu_reference_bounds = validate_verify_payload(
            cpu_warmup_payload, output_dim, "cpu", f"{model}/CPU warm-up"
        )
        cpu_results = [run_benchmark(cmd_cpu) for _ in range(5)]
        for index, result in enumerate(cpu_results):
            payload = parse_json_result(result, f"{model}/CPU run {index + 1}")
            bounds = validate_verify_payload(
                payload, output_dim, "cpu", f"{model}/CPU run {index + 1}"
            )
            if not close_bounds(cpu_reference_bounds, bounds):
                raise RuntimeError(f"{model}/CPU bounds changed between runs")
        cpu_times = [result.elapsed_ms for result in cpu_results]
        cpu_avg = sum(cpu_times) / len(cpu_times)

        # WGPU benchmark
        cmd_gpu = [
            str(NY_BIN), "verify", str(model_path), "--epsilon", "0.01",
            "--method", "ibp", "--backend", "wgpu", "--json",
            "--strict", "--require-sound",
        ]
        gpu_warmup = run_benchmark(cmd_gpu)
        gpu_warmup_payload = parse_json_result(gpu_warmup, f"{model}/WGPU warm-up")
        gpu_reference_bounds = validate_verify_payload(
            gpu_warmup_payload, output_dim, "wgpu", f"{model}/WGPU warm-up"
        )
        if not close_bounds(cpu_reference_bounds, gpu_reference_bounds):
            raise RuntimeError(f"{model} CPU/WGPU bounds do not match")
        gpu_results = [run_benchmark(cmd_gpu) for _ in range(5)]
        for index, result in enumerate(gpu_results):
            payload = parse_json_result(result, f"{model}/WGPU run {index + 1}")
            bounds = validate_verify_payload(
                payload, output_dim, "wgpu", f"{model}/WGPU run {index + 1}"
            )
            if not close_bounds(gpu_reference_bounds, bounds):
                raise RuntimeError(f"{model}/WGPU bounds changed between runs")
        gpu_times = [result.elapsed_ms for result in gpu_results]
        gpu_avg = sum(gpu_times) / len(gpu_times)

        if not finite_number(cpu_avg, positive=True):
            raise RuntimeError(f"{model} CPU average timing is invalid")
        if not finite_number(gpu_avg, positive=True):
            raise RuntimeError(f"{model} WGPU average timing is invalid")
        speedup = cpu_avg / gpu_avg
        if not finite_number(speedup, positive=True):
            raise RuntimeError(f"{model} CPU/WGPU speedup is invalid")
        results.append((model, cpu_avg, gpu_avg, speedup))
        print(
            f"  {model:20s} | CPU: {cpu_avg:8.2f} ms | "
            f"WGPU: {gpu_avg:8.2f} ms | Speedup: {speedup:.2f}x"
        )

    return results

def benchmark_scaling():
    """Analyze how performance scales with model size."""
    print("\n" + "=" * 60)
    print("Built-in Benchmark Scaling Analysis")
    print("=" * 60)

    # Run built-in benchmarks
    completed = 0
    for bench_type in ["layer", "attention", "full"]:
        print(f"\n--- {bench_type.upper()} benchmarks ---")
        result = run_benchmark(
            [str(NY_BIN), "bench", "-b", bench_type, "--json"],
            timeout=300.0,
        )
        payload = parse_json_result(result, f"{bench_type} scaling benchmark")
        if payload.get("benchmark_type") != bench_type or payload.get("valid_type") is not True:
            raise RuntimeError(f"{bench_type} benchmark returned the wrong result type")
        rows = payload.get("results")
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"{bench_type} benchmark returned no measurement rows")
        names = set()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("name"), str):
                raise RuntimeError(f"{bench_type} benchmark returned a malformed row")
            iterations = row.get("iterations")
            if (
                not isinstance(iterations, int)
                or isinstance(iterations, bool)
                or iterations <= 0
            ):
                raise RuntimeError(f"{bench_type} benchmark returned invalid iterations")
            for field in [
                "per_iter_ms", "per_iter_ns", "per_iter_us", "total_ms", "total_ns"
            ]:
                if not finite_number(row.get(field), positive=True):
                    raise RuntimeError(
                        f"{bench_type} benchmark returned invalid {field}"
                    )
            if row["name"] in names:
                raise RuntimeError(f"{bench_type} benchmark returned duplicate rows")
            names.add(row["name"])
        print(result.stdout)
        completed += 1
    return completed

def load_criterion_data():
    """Load and display criterion benchmark data."""
    print("\n" + "=" * 60)
    print("Criterion Benchmark Results (from previous runs)")
    print("=" * 60)

    criterion_dir = Path("target/criterion")
    if not criterion_dir.exists():
        print("SKIP: no prior Criterion data found (run: cargo bench)")
        return {}

    results = {}
    for estimates_file in criterion_dir.rglob("estimates.json"):
        with open(estimates_file) as f:
            data = json.load(f)

        # Extract benchmark name from path
        parts = estimates_file.parts
        idx = parts.index("criterion")
        name = "/".join(parts[idx+1:-2])

        try:
            mean_ns = data["mean"]["point_estimate"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"malformed Criterion data: {estimates_file}") from exc
        if not finite_number(mean_ns, positive=True):
            raise RuntimeError(f"invalid Criterion mean: {estimates_file}")
        mean_ms = mean_ns / 1e6

        results[name] = mean_ms

    if not results:
        print("SKIP: Criterion directory contained no estimates")
        return {}

    # Group and display
    linear_results = {k: v for k, v in results.items() if "Linear" in k and "new" in k}
    matmul_results = {k: v for k, v in results.items() if "MatMul" in k and "new" in k}

    print("\n--- Linear IBP Scaling ---")
    for name, ms in sorted(linear_results.items()):
        print(f"  {name:50s} | {ms:8.2f} ms")

    print("\n--- MatMul Scaling (Attention) ---")
    for name, ms in sorted(matmul_results.items()):
        print(f"  {name:50s} | {ms:8.2f} ms")

    # Calculate GPU speedups for MatMul
    print("\n--- GPU Speedup for MatMul ---")
    matmul_cpu = {k.replace("/new", "").replace("cpu/", ""): v for k, v in results.items() if "MatMul" in k and "/cpu/" in k and "/new/" in k}
    matmul_gpu = {k.replace("/new", "").replace("accel/", ""): v for k, v in results.items() if "MatMul" in k and "/accel/" in k and "/new/" in k}

    for config in matmul_cpu:
        if config in matmul_gpu:
            cpu_ms = matmul_cpu[config]
            gpu_ms = matmul_gpu[config]
            if not finite_number(cpu_ms, positive=True):
                raise RuntimeError(f"invalid Criterion CPU timing for {config}")
            if not finite_number(gpu_ms, positive=True):
                raise RuntimeError(f"invalid Criterion GPU timing for {config}")
            speedup = cpu_ms / gpu_ms
            print(f"  {config:20s} | CPU: {cpu_ms:8.2f} ms | GPU: {gpu_ms:8.2f} ms | Speedup: {speedup:.1f}x")

    return results

def main() -> int:
    # Check binary exists
    if not NY_BIN.exists():
        print("Building release binary...")
        subprocess.run(["cargo", "build", "--release", "-p", "ny-cli"], check=True)

    print("=" * 60)
    print("ny Performance Benchmark Suite")
    print("=" * 60)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Binary: {NY_BIN}")
    print()

    # Run benchmarks
    cli_results = benchmark_cli_commands()
    gpu_results = benchmark_gpu_vs_cpu()
    scaling_results = benchmark_scaling()
    criterion_data = load_criterion_data()
    if len(cli_results) != 24:
        raise RuntimeError(f"incomplete CLI benchmark matrix: {len(cli_results)}/24")
    if len(gpu_results) != 3:
        raise RuntimeError(f"incomplete CPU/WGPU benchmark matrix: {len(gpu_results)}/3")
    if scaling_results != 3:
        raise RuntimeError(f"incomplete scaling benchmark matrix: {scaling_results}/3")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nMeasured CLI rows: {len(cli_results)}")
    print(f"Measured CPU/WGPU rows: {len(gpu_results)}")
    print(f"Measured scaling suites: {scaling_results}")

    if criterion_data:
        # Find best GPU speedup
        matmul_speedups = []
        for config in ["h6s64d64", "h8s128d64"]:
            cpu_key = f"Comparison_MatMul/cpu/{config}/new"
            gpu_key = f"Comparison_MatMul/accel/{config}/new"
            if cpu_key in criterion_data and gpu_key in criterion_data:
                speedup = criterion_data[cpu_key] / criterion_data[gpu_key]
                if not finite_number(speedup, positive=True):
                    raise RuntimeError(
                        f"invalid Criterion GPU speedup for {config}"
                    )
                matmul_speedups.append(speedup)

        if matmul_speedups:
            print(f"\nPeak GPU speedup for MatMul: {max(matmul_speedups):.0f}x")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
