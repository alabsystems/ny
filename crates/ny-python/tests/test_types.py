# Copyright 2026 Andrew Yates
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""Type checking + runtime smoke tests for ny module."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt

_NY_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import ny as _ny_mod
except ModuleNotFoundError as e:
    _ny_mod = None
    _NY_IMPORT_ERROR = e


class _MissingNy(ModuleType):
    def __getattr__(self, name: str):  # pragma: no cover
        raise AssertionError(
            "Python bindings not installed: cannot import `ny`.\n"
            "Build/install with: `python -m pip install -e crates/ny-python` "
            "(requires `maturin`)."
        ) from _NY_IMPORT_ERROR


ny = _ny_mod if _ny_mod is not None else _MissingNy("ny")

MODELS_DIR = Path(__file__).resolve().parents[3] / "tests" / "models"
SIMPLE_MLP = str(MODELS_DIR / "simple_mlp.onnx")


def test_diff_basic() -> None:
    """Test diff function type hints."""
    result: ny.DiffResult = ny.diff(SIMPLE_MLP, SIMPLE_MLP)
    _: bool = result.is_equivalent
    _2: float = result.max_divergence
    _3: list[ny.LayerComparison] = result.layers


def test_diff_with_options() -> None:
    """Test diff with optional parameters."""
    input_data: npt.NDArray[np.float32] = np.zeros((1, 2), dtype=np.float32)
    mapping: dict[str, str] = {"output": "output"}

    result: ny.DiffResult = ny.diff(
        SIMPLE_MLP,
        SIMPLE_MLP,
        tolerance=1e-4,
        input=input_data,
        continue_after_divergence=False,
        layer_mapping=mapping,
    )

    _: str = result.summary()
    _2: list[ny.DiffStatus] = result.statuses()


def test_layer_comparison() -> None:
    """Test LayerComparison type hints."""
    result = ny.diff(SIMPLE_MLP, SIMPLE_MLP)
    if result.layers:
        layer: ny.LayerComparison = result.layers[0]
        _: str = layer.name
        _2: float = layer.max_diff
        _3: bool = layer.exceeds_tolerance
        _4: list[int] = layer.shape_a


def test_run_with_intermediates() -> None:
    """Test run_with_intermediates type hints."""
    input_data: npt.NDArray[np.float32] = np.zeros((1, 2), dtype=np.float32)
    outputs: dict[str, npt.NDArray[np.float32]] = ny.run_with_intermediates(
        SIMPLE_MLP, input_data
    )
    for name, arr in outputs.items():
        _: str = name
        _2: npt.NDArray[np.float32] = arr


def test_load_model_info() -> None:
    """Test load_model_info type hints."""
    info: ny.ModelInfo = ny.load_model_info(SIMPLE_MLP)
    _: int = info.layer_count
    _2: list[str] = info.layer_names
    if info.inputs:
        spec: ny.TensorSpec = info.inputs[0]
        _3: str = spec.name
        _4: list[int] = spec.shape
        _5: str = spec.dtype


def test_load_npy(tmp_path: Path) -> None:
    """Test load_npy type hints."""
    npy_path = tmp_path / "data.npy"
    np.save(str(npy_path), np.zeros((2, 3), dtype=np.float32))
    data: npt.NDArray[np.float32] = ny.load_npy(str(npy_path))
    _ = data.shape


def test_diff_status_enum() -> None:
    """Test DiffStatus enum type hints."""
    _: ny.DiffStatus = ny.DiffStatus.Ok
    _2: ny.DiffStatus = ny.DiffStatus.DriftStarts
    _3: ny.DiffStatus = ny.DiffStatus.ExceedsTolerance
    _4: ny.DiffStatus = ny.DiffStatus.ShapeMismatch


def test_tensor_comparison_status_enum() -> None:
    """Test TensorComparisonStatus enum type hints."""
    _: ny.TensorComparisonStatus = ny.TensorComparisonStatus.Match
    _2: ny.TensorComparisonStatus = ny.TensorComparisonStatus.Differs
    _3: ny.TensorComparisonStatus = ny.TensorComparisonStatus.ShapeMismatch
    _4: ny.TensorComparisonStatus = ny.TensorComparisonStatus.MissingInA
    _5: ny.TensorComparisonStatus = ny.TensorComparisonStatus.MissingInB


def test_version() -> None:
    """Test __version__ type hint."""
    version: str = ny.__version__
    _ = version
