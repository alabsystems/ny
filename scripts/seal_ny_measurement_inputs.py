#!/usr/bin/env python3
"""Seal one VNN-COMP benchmark pair before NY executes it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from archive_vnncomp_sat_result import seal_inputs, validate_input_preflight


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--instance-index", type=int, required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--vnnlib", required=True)
    parser.add_argument("--onnx-file", type=Path, required=True)
    parser.add_argument("--vnnlib-file", type=Path, required=True)
    parser.add_argument("--start-manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        preflight = seal_inputs(
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            category=args.category,
            instance_index=args.instance_index,
            onnx=args.onnx,
            vnnlib=args.vnnlib,
            onnx_file=args.onnx_file,
            vnnlib_file=args.vnnlib_file,
            start_manifest=args.start_manifest,
        )
        payload, _ = validate_input_preflight(
            preflight_manifest=preflight,
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            category=args.category,
            instance_index=args.instance_index,
            onnx=args.onnx,
            vnnlib=args.vnnlib,
            onnx_file=args.onnx_file,
            vnnlib_file=args.vnnlib_file,
            start_manifest=args.start_manifest,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    inputs = payload["inputs"]
    assert isinstance(inputs, dict)
    print(
        json.dumps(
            {
                "preflight_manifest": str(preflight.resolve()),
                "onnx_file": inputs["onnx"]["sealed"]["resolved_path"],
                "vnnlib_file": inputs["vnnlib"]["sealed"]["resolved_path"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
