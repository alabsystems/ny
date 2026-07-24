#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Benchmark CPU vs wgpu backends on Whisper sequential verification.

This uses `ny whisper-seq` and parses:
- `total_time_ms`
- `final_output_width`
- per-block widths and `gpu` column

It runs multiple repeats per configuration and reports median timings.
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
from dataclasses import dataclass

DEFAULT_MODEL = "tests/models/whisper_tiny_encoder.onnx"
DEFAULT_NY = "./target/release/ny"


@dataclass(frozen=True)
class WhisperSeqResult:
    time_ms: int
    final_output_width: float
    gpu_enabled: bool
    gpu_used_any: bool


_RE_TIME_MS = re.compile(r"total_time_ms=(\d+)")
_RE_FINAL_WIDTH = re.compile(r"final_output_width=([0-9eE+.\-]+)")
_RE_BACKEND = re.compile(r"^Backend:\s+(\S+)\s+\(GPU:\s+(enabled|disabled)\)", re.MULTILINE)
_RE_PER_BLOCK_ROW = re.compile(
    r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(yes|no)\s+(\d+)\s*$",
    re.MULTILINE,
)


def _run_whisper_seq(
    ny: str,
    model: str,
    backend: str,
    seq_len: int,
    blocks: int,
    epsilon: float,
    timeout_s: int,
) -> WhisperSeqResult:
    cmd = [
        ny,
        "whisper-seq",
        model,
        "--start-block",
        "0",
        "--end-block",
        str(blocks),
        "--epsilon",
        str(epsilon),
        "--seq-len",
        str(seq_len),
        "--backend",
        backend,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    output = completed.stdout + completed.stderr

    if completed.returncode != 0:
        raise RuntimeError(f"ny whisper-seq failed (backend={backend}):\n{output}")

    match_time = _RE_TIME_MS.search(output)
    if match_time is None:
        raise RuntimeError(f"Failed to parse total_time_ms (backend={backend}):\n{output}")
    time_ms = int(match_time.group(1))

    match_width = _RE_FINAL_WIDTH.search(output)
    if match_width is None:
        raise RuntimeError(f"Failed to parse final_output_width (backend={backend}):\n{output}")
    final_output_width = float(match_width.group(1))

    match_backend = _RE_BACKEND.search(output)
    gpu_enabled = False
    if match_backend is not None:
        gpu_enabled = match_backend.group(2) == "enabled"

    gpu_used_any = any(m.group(6) == "yes" for m in _RE_PER_BLOCK_ROW.finditer(output))

    return WhisperSeqResult(
        time_ms=time_ms,
        final_output_width=final_output_width,
        gpu_enabled=gpu_enabled,
        gpu_used_any=gpu_used_any,
    )


def _median_ms(samples: list[int]) -> int:
    # Use the lower-median to avoid averaging in the even-N case.
    # For small-N (1/2/3) this is typically more robust to a single outlier.
    return int(statistics.median_low(samples))


def _rel_diff(a: float, b: float) -> float:
    denom = max(1.0, abs(a), abs(b))
    return abs(a - b) / denom


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ny", default=DEFAULT_NY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller config set (faster, less coverage).",
    )
    args = parser.parse_args()

    if args.quick:
        configs: list[tuple[int, int]] = [
            (64, 2),
            (128, 2),
            (256, 2),
            (128, 4),
        ]
    else:
        configs = [
            (64, 2),
            (64, 4),
            (128, 2),
            (128, 4),
            (256, 2),
            (256, 4),
        ]

    print("=" * 60)
    print("CPU vs wgpu Backend Benchmark (whisper-seq)")
    print("=" * 60)
    print()
    print(f"Binary: {args.ny}")
    print(f"Model:  {args.model}")
    print(f"Config: repeats={args.repeats}, epsilon={args.epsilon}")
    print()
    print("| seq_len | blocks | CPU med (ms) | wgpu med (ms) | wgpu speedup |")
    print("|---------|--------|--------------|---------------|--------------|")

    correctness_failures: list[str] = []
    speedups_wgpu: list[float] = []

    for seq_len, blocks in configs:
        per_backend_times: dict[str, list[int]] = {"cpu": [], "wgpu": []}
        per_backend_widths: dict[str, list[float]] = {"cpu": [], "wgpu": []}
        per_backend_gpu_used_any: dict[str, bool] = {"cpu": False, "wgpu": False}

        for backend in ("cpu", "wgpu"):
            _run_whisper_seq(
                args.ny, args.model, backend, seq_len, blocks, args.epsilon, args.timeout_s
            )

            for _ in range(args.repeats):
                r = _run_whisper_seq(
                    args.ny,
                    args.model,
                    backend,
                    seq_len,
                    blocks,
                    args.epsilon,
                    args.timeout_s,
                )
                per_backend_times[backend].append(r.time_ms)
                per_backend_widths[backend].append(r.final_output_width)
                per_backend_gpu_used_any[backend] = per_backend_gpu_used_any[backend] or r.gpu_used_any

        cpu_ms = _median_ms(per_backend_times["cpu"])
        wgpu_ms = _median_ms(per_backend_times["wgpu"])

        wgpu_speedup = (cpu_ms / wgpu_ms) if wgpu_ms > 0 else 0.0

        speedups_wgpu.append(wgpu_speedup)

        print(
            f"| {seq_len:7d} | {blocks:6d} | {cpu_ms:12d} | {wgpu_ms:13d} | {wgpu_speedup:11.2f}x |"
        )

        cpu_width = statistics.median_low(per_backend_widths["cpu"])
        backend_width = statistics.median_low(per_backend_widths["wgpu"])
        diff = _rel_diff(cpu_width, backend_width)
        if diff > args.tolerance:
            correctness_failures.append(
                f"seq_len={seq_len} blocks={blocks} backend=wgpu rel_diff={diff:.3e} cpu={cpu_width:.6e} wgpu={backend_width:.6e}"
            )

        if seq_len >= 64 and not per_backend_gpu_used_any["wgpu"]:
            correctness_failures.append(
                f"seq_len={seq_len} blocks={blocks} backend=wgpu did not report any GPU usage"
            )

    print()
    print("Summary:")
    print(f"  Average wgpu speedup (median-of-medians): {statistics.mean(speedups_wgpu):.2f}x")
    print()

    if correctness_failures:
        print("Correctness check: FAILED")
        for line in correctness_failures:
            print(f"  - {line}")
        return 1

    print(f"Correctness check: OK (tolerance={args.tolerance:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
