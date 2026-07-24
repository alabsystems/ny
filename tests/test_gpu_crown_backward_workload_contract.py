# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import onnx
import pytest
from onnx import shape_inference


REPO_ROOT = Path(__file__).resolve().parent.parent
WORKLOADS_RS = (
    REPO_ROOT
    / "crates"
    / "ny-gpu"
    / "src"
    / "benchmark_support"
    / "crown_backward_workloads.rs"
)
POLICY_PATH = (
    REPO_ROOT / "configs" / "benchmark_regressions" / "gpu_crown_backward.json"
)


ConvSpec = tuple[int, int, int, tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class RustWorkloadContract:
    case_name: str
    parameter_count: int
    estimated_cpu_peak_bytes: int
    conv_specs: tuple[ConvSpec, ...]
    input_shape: tuple[int, ...] | None = None
    input_dim: int | None = None
    output_dim: int | None = None
    hidden_dim: int | None = None
    reshape_shape: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class OnnxWorkloadContract:
    parameter_count: int
    estimated_cpu_peak_bytes: int
    conv_specs: tuple[ConvSpec, ...]
    input_shape: tuple[int, ...] | None = None
    input_dim: int | None = None
    output_dim: int | None = None
    hidden_dim: int | None = None
    reshape_shape: tuple[int, int, int] | None = None


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"benchmark fixture missing: {path}")


def _source_text() -> str:
    return WORKLOADS_RS.read_text(encoding="utf-8")


def _const_value(source: str, name: str):
    match = re.search(
        rf"pub const {re.escape(name)}:[^=]+= (?P<value>.+?);",
        source,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing Rust constant `{name}` in {WORKLOADS_RS}"
    return ast.literal_eval(match.group("value").strip())


def _shape_product(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def _conv_output_dim(spec: ConvSpec) -> int:
    out_channels, _, kernel, (stride_h, stride_w), (pad_h, pad_w), (input_h, input_w) = spec
    out_h = (input_h + 2 * pad_h - kernel) // stride_h + 1
    out_w = (input_w + 2 * pad_w - kernel) // stride_w + 1
    return out_channels * out_h * out_w


def _estimate_dense_peak_bytes(conv_specs: tuple[ConvSpec, ...]) -> int:
    max_dim = max(_conv_output_dim(spec) for spec in conv_specs)
    return 4 * max_dim * max_dim * 4


def _count_conv_params(conv_specs: tuple[ConvSpec, ...]) -> int:
    total = 0
    for out_channels, in_channels, kernel, _, _, _ in conv_specs:
        total += out_channels * in_channels * kernel * kernel + out_channels
    return total


def _rust_metaroom_contract() -> RustWorkloadContract:
    source = _source_text()
    case_name = _const_value(source, "METAROOM_CASE_NAME")
    input_shape = tuple(_const_value(source, "METAROOM_INPUT_SHAPE"))
    output_dim = int(_const_value(source, "METAROOM_OUTPUT_DIM"))
    hidden_dim = int(_const_value(source, "METAROOM_HIDDEN_DIM"))
    conv_specs = tuple(_const_value(source, "METAROOM_CONV_SPECS"))
    flattened = _conv_output_dim(conv_specs[-1])
    parameter_count = (
        _count_conv_params(conv_specs)
        + hidden_dim * flattened
        + hidden_dim
        + output_dim * hidden_dim
        + output_dim
    )
    return RustWorkloadContract(
        case_name=case_name,
        input_shape=input_shape,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        conv_specs=conv_specs,
        parameter_count=parameter_count,
        estimated_cpu_peak_bytes=_estimate_dense_peak_bytes(conv_specs),
    )


def _rust_soundnessbench_contract() -> RustWorkloadContract:
    source = _source_text()
    case_name = _const_value(source, "SOUNDNESSBENCH_CASE_NAME")
    input_dim = int(_const_value(source, "SOUNDNESSBENCH_INPUT_DIM"))
    output_dim = int(_const_value(source, "SOUNDNESSBENCH_OUTPUT_DIM"))
    reshape_shape = tuple(_const_value(source, "SOUNDNESSBENCH_RESHAPE_SHAPE"))
    conv_specs = tuple(_const_value(source, "SOUNDNESSBENCH_CONV_SPECS"))
    reshape_dim = _shape_product(reshape_shape)
    parameter_count = (
        reshape_dim * input_dim
        + reshape_dim
        + _count_conv_params(conv_specs)
        + output_dim * output_dim
        + output_dim
    )
    return RustWorkloadContract(
        case_name=case_name,
        input_dim=input_dim,
        output_dim=output_dim,
        reshape_shape=reshape_shape,
        conv_specs=conv_specs,
        parameter_count=parameter_count,
        estimated_cpu_peak_bytes=_estimate_dense_peak_bytes(conv_specs),
    )


def _dim_values(value_info: onnx.ValueInfoProto) -> list[int | str]:
    dims: list[int | str] = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def _infer_model(path: Path) -> onnx.ModelProto:
    _skip_if_missing(path)
    return shape_inference.infer_shapes(onnx.load(path))


def _shape_map(model: onnx.ModelProto) -> dict[str, list[int | str]]:
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    return {value.name: _dim_values(value) for value in values}


def _initializer_map(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    return {initializer.name: initializer for initializer in model.graph.initializer}


def _initializer_param_count(model: onnx.ModelProto) -> int:
    total = 0
    for initializer in model.graph.initializer:
        param_count = 1
        for dim in initializer.dims:
            param_count *= int(dim)
        total += param_count
    return total


def _conv_specs_from_onnx(model: onnx.ModelProto) -> tuple[ConvSpec, ...]:
    shapes = _shape_map(model)
    initializers = _initializer_map(model)
    specs: list[ConvSpec] = []
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        weights = initializers[node.input[1]]
        out_channels, in_channels, kernel_h, kernel_w = (int(dim) for dim in weights.dims)
        assert kernel_h == kernel_w, (
            f"representative benchmark Conv kernel must stay square: {node.name or node.output[0]}"
        )
        attrs: dict[str, tuple[int, ...] | int] = {
            attribute.name: (
                tuple(int(value) for value in attribute.ints)
                if attribute.ints
                else int(attribute.i)
            )
            for attribute in node.attribute
        }
        pads = attrs.get("pads", (0, 0, 0, 0))
        strides = attrs.get("strides", (1, 1))
        input_shape = shapes[node.input[0]]
        assert len(input_shape) == 4, (
            f"Conv input shape should be NCHW, got {input_shape!r} for {node.name or node.output[0]}"
        )
        input_h = int(input_shape[-2])
        input_w = int(input_shape[-1])
        pad_h, pad_w, pad_h_tail, pad_w_tail = (int(value) for value in pads)
        assert (pad_h, pad_w) == (pad_h_tail, pad_w_tail), (
            f"benchmark Conv pads must stay symmetric: {pads!r}"
        )
        specs.append(
            (
                out_channels,
                in_channels,
                kernel_h,
                (int(strides[0]), int(strides[1])),
                (pad_h, pad_w),
                (input_h, input_w),
            )
        )
    return tuple(specs)


def _gemm_weight_shapes(model: onnx.ModelProto) -> tuple[tuple[int, int], ...]:
    initializers = _initializer_map(model)
    shapes: list[tuple[int, int]] = []
    for node in model.graph.node:
        if node.op_type != "Gemm":
            continue
        weights = initializers[node.input[1]]
        shapes.append(tuple(int(dim) for dim in weights.dims))
    return tuple(shapes)


def _onnx_metaroom_contract() -> OnnxWorkloadContract:
    model = _infer_model(
        REPO_ROOT
        / "benchmarks"
        / "vnncomp2025"
        / "benchmarks"
        / "metaroom_2023"
        / "onnx"
        / "6cnn_ry_0_0_no_custom_OP.onnx"
    )
    input_shape = tuple(int(dim) for dim in _dim_values(model.graph.input[0])[-3:])
    output_dim = int(_dim_values(model.graph.output[0])[-1])
    gemm_shapes = _gemm_weight_shapes(model)
    assert len(gemm_shapes) == 2, (
        f"expected 2 Gemm layers in representative metaroom model, got {len(gemm_shapes)}"
    )
    hidden_dim = gemm_shapes[0][0]
    flattened_dim = gemm_shapes[0][1]
    conv_specs = _conv_specs_from_onnx(model)
    assert flattened_dim == _conv_output_dim(conv_specs[-1]), (
        f"metaroom flatten width drifted: Gemm input {flattened_dim} vs conv output {_conv_output_dim(conv_specs[-1])}"
    )
    return OnnxWorkloadContract(
        input_shape=input_shape,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        conv_specs=conv_specs,
        parameter_count=_initializer_param_count(model),
        estimated_cpu_peak_bytes=_estimate_dense_peak_bytes(conv_specs),
    )


def _onnx_soundnessbench_contract() -> OnnxWorkloadContract:
    model = _infer_model(
        REPO_ROOT
        / "benchmarks"
        / "vnncomp2025"
        / "benchmarks"
        / "soundnessbench"
        / "onnx"
        / "model.onnx"
    )
    input_dim = int(_dim_values(model.graph.input[0])[-1])
    output_dim = int(_dim_values(model.graph.output[0])[-1])
    shapes = _shape_map(model)
    conv_specs = _conv_specs_from_onnx(model)
    gemm_shapes = _gemm_weight_shapes(model)
    assert len(gemm_shapes) == 2, (
        f"expected 2 Gemm layers in soundnessbench model, got {len(gemm_shapes)}"
    )
    first_conv_input = shapes[
        next(node.input[0] for node in model.graph.node if node.op_type == "Conv")
    ]
    reshape_shape = tuple(int(dim) for dim in first_conv_input[-3:])
    reshape_dim = _shape_product(reshape_shape)
    assert gemm_shapes[0][0] == reshape_dim, (
        f"soundnessbench front Gemm output {gemm_shapes[0][0]} != reshape volume {reshape_dim}"
    )
    return OnnxWorkloadContract(
        input_dim=input_dim,
        output_dim=output_dim,
        reshape_shape=reshape_shape,
        conv_specs=conv_specs,
        parameter_count=_initializer_param_count(model),
        estimated_cpu_peak_bytes=_estimate_dense_peak_bytes(conv_specs),
    )


def _policy_case_metadata(case_name: str) -> tuple[int, int]:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    checks = [check for check in payload["checks"] if check["case"] == case_name]
    assert checks, f"missing regression policy rows for `{case_name}`"
    parameter_counts = {int(check["expected_parameter_count"]) for check in checks}
    dense_peaks = {int(check["expected_estimated_cpu_peak_bytes"]) for check in checks}
    assert len(parameter_counts) == 1, (
        f"policy metadata drifted for `{case_name}` parameter counts: {parameter_counts!r}"
    )
    assert len(dense_peaks) == 1, (
        f"policy metadata drifted for `{case_name}` CPU dense peaks: {dense_peaks!r}"
    )
    return next(iter(parameter_counts)), next(iter(dense_peaks))


def test_metaroom_benchmark_workload_matches_representative_vnncomp_model() -> None:
    rust_contract = _rust_metaroom_contract()
    onnx_contract = _onnx_metaroom_contract()

    assert rust_contract.case_name == "metaroom_6cnn_ry_like", (
        f"metaroom case name mismatch: {rust_contract.case_name!r}"
    )
    assert rust_contract.input_shape == onnx_contract.input_shape, (
        f"metaroom input_shape: rust={rust_contract.input_shape} onnx={onnx_contract.input_shape}"
    )
    assert rust_contract.output_dim == onnx_contract.output_dim, (
        f"metaroom output_dim: rust={rust_contract.output_dim} onnx={onnx_contract.output_dim}"
    )
    assert rust_contract.hidden_dim == onnx_contract.hidden_dim, (
        f"metaroom hidden_dim: rust={rust_contract.hidden_dim} onnx={onnx_contract.hidden_dim}"
    )
    assert rust_contract.conv_specs == onnx_contract.conv_specs, (
        f"metaroom conv_specs diverged between Rust and ONNX"
    )
    assert rust_contract.parameter_count == onnx_contract.parameter_count, (
        f"metaroom parameter_count: rust={rust_contract.parameter_count} onnx={onnx_contract.parameter_count}"
    )
    assert rust_contract.estimated_cpu_peak_bytes == onnx_contract.estimated_cpu_peak_bytes, (
        f"metaroom estimated_cpu_peak_bytes: rust={rust_contract.estimated_cpu_peak_bytes} onnx={onnx_contract.estimated_cpu_peak_bytes}"
    )


def test_soundnessbench_benchmark_workload_matches_representative_vnncomp_model() -> None:
    rust_contract = _rust_soundnessbench_contract()
    onnx_contract = _onnx_soundnessbench_contract()

    assert rust_contract.case_name == "soundnessbench_exact_like", (
        f"soundnessbench case name mismatch: {rust_contract.case_name!r}"
    )
    assert rust_contract.input_dim == onnx_contract.input_dim, (
        f"soundnessbench input_dim: rust={rust_contract.input_dim} onnx={onnx_contract.input_dim}"
    )
    assert rust_contract.output_dim == onnx_contract.output_dim, (
        f"soundnessbench output_dim: rust={rust_contract.output_dim} onnx={onnx_contract.output_dim}"
    )
    assert rust_contract.reshape_shape == onnx_contract.reshape_shape, (
        f"soundnessbench reshape_shape: rust={rust_contract.reshape_shape} onnx={onnx_contract.reshape_shape}"
    )
    assert rust_contract.conv_specs == onnx_contract.conv_specs, (
        f"soundnessbench conv_specs diverged between Rust and ONNX"
    )
    assert rust_contract.parameter_count == onnx_contract.parameter_count, (
        f"soundnessbench parameter_count: rust={rust_contract.parameter_count} onnx={onnx_contract.parameter_count}"
    )
    assert rust_contract.estimated_cpu_peak_bytes == onnx_contract.estimated_cpu_peak_bytes, (
        f"soundnessbench estimated_cpu_peak_bytes: rust={rust_contract.estimated_cpu_peak_bytes} onnx={onnx_contract.estimated_cpu_peak_bytes}"
    )


def test_gpu_crown_backward_policy_metadata_matches_representative_models() -> None:
    metaroom_rust = _rust_metaroom_contract()
    soundnessbench_rust = _rust_soundnessbench_contract()

    metaroom_policy = _policy_case_metadata(metaroom_rust.case_name)
    soundnessbench_policy = _policy_case_metadata(soundnessbench_rust.case_name)

    assert metaroom_policy == (
        metaroom_rust.parameter_count,
        metaroom_rust.estimated_cpu_peak_bytes,
    ), (
        f"metaroom policy metadata mismatch: policy={metaroom_policy} rust=({metaroom_rust.parameter_count}, {metaroom_rust.estimated_cpu_peak_bytes})"
    )
    assert soundnessbench_policy == (
        soundnessbench_rust.parameter_count,
        soundnessbench_rust.estimated_cpu_peak_bytes,
    ), (
        f"soundnessbench policy metadata mismatch: policy={soundnessbench_policy} rust=({soundnessbench_rust.parameter_count}, {soundnessbench_rust.estimated_cpu_peak_bytes})"
    )
