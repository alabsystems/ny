# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "test_python_api.py"
BASE_EXPECTED_EXPORTS = [
    "diff",
    "sensitivity_analysis",
    "quantize_check",
    "profile_bounds",
    "load_model_info",
    "load_npy",
    "verify",
    "compare",
    "weights_info",
    "weights_diff",
    "bench",
    "DiffResult",
    "DiffStatus",
    "LayerComparison",
    "SensitivityResult",
    "QuantizationResult",
    "ProfileResult",
    "VerifyResult",
    "VerifyStatus",
    "OutputBound",
    "CompareResult",
    "BoundViolation",
    "WeightsInfo",
    "TensorInfo",
    "WeightsDiffResult",
    "TensorComparison",
    "BenchResult",
    "BenchResultItem",
    "BenchDimensions",
]


@dataclass
class FakeTensorSpec:
    name: str
    shape: list[int]
    dtype: str


@dataclass
class FakeModelInfo:
    inputs: list[FakeTensorSpec] = field(
        default_factory=lambda: [FakeTensorSpec("input", [1, 2], "float32")]
    )
    outputs: list[FakeTensorSpec] = field(
        default_factory=lambda: [FakeTensorSpec("output", [1, 2], "float32")]
    )
    layer_count: int = 3
    layer_names: list[str] = field(default_factory=lambda: ["fc1", "relu", "output"])


class FakeTensorComparisonStatus:
    pass


@dataclass
class FakeTensorComparison:
    status: FakeTensorComparisonStatus = field(default_factory=FakeTensorComparisonStatus)


@dataclass
class FakeWeightsDiffResult:
    is_match: bool = True
    max_diff: float = 0.0
    tolerance: float = 1e-6
    differing_count: int = 0
    total_tensors_a: int = 2
    total_tensors_b: int = 2
    comparisons: list[FakeTensorComparison] = field(
        default_factory=lambda: [FakeTensorComparison()]
    )


def load_python_api_script():
    spec = importlib.util.spec_from_file_location("test_python_api_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None, f"Expected a loader for {SCRIPT_PATH}"
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_fake_ny(*, include_typed_exports: bool):
    ny = types.SimpleNamespace(
        load_model_info=lambda _path: FakeModelInfo(),
        weights_diff=lambda _a, _b: FakeWeightsDiffResult(),
    )

    for name in BASE_EXPECTED_EXPORTS:
        setattr(ny, name, getattr(ny, name, type(name, (), {})()))

    if include_typed_exports:
        ny.TensorSpec = FakeTensorSpec
        ny.ModelInfo = FakeModelInfo
        ny.TensorComparisonStatus = FakeTensorComparisonStatus

    return ny


def test_python_api_script_accepts_typed_consumer_surface(monkeypatch):
    module = load_python_api_script()
    monkeypatch.setattr(
        module,
        "ny",
        make_fake_ny(include_typed_exports=True),
        raising=False,
    )
    monkeypatch.setattr(module, "HAS_NY", True)

    tests = module.PythonAPITests()
    tests.test_load_model_info_basic()
    tests.test_load_model_info_shapes()
    tests.test_weights_diff_result_attrs()
    tests.test_module_exports()

    results = {result.name: result for result in tests.results}
    assert results["load_model_info_basic"].passed, results["load_model_info_basic"].message
    assert results["load_model_info_shapes"].passed, results["load_model_info_shapes"].message
    assert (
        results["weights_diff_result_attrs"].passed
    ), results["weights_diff_result_attrs"].message
    assert results["module_exports"].passed, results["module_exports"].message


def test_python_api_script_requires_typed_exports(monkeypatch):
    module = load_python_api_script()
    monkeypatch.setattr(
        module,
        "ny",
        make_fake_ny(include_typed_exports=False),
        raising=False,
    )
    monkeypatch.setattr(module, "HAS_NY", True)

    tests = module.PythonAPITests()
    tests.test_module_exports()

    result = tests.results[-1]
    assert not result.passed, "Expected module exports check to fail without typed exports"
    assert "TensorSpec" in result.message, result.message
    assert "ModelInfo" in result.message, result.message
    assert "TensorComparisonStatus" in result.message, result.message
