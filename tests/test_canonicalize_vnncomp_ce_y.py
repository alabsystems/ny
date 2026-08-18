# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "canonicalize_vnncomp_ce_y.py"
SCORING_DIR: Path
SPEC = importlib.util.spec_from_file_location("canonicalize_vnncomp_ce_y", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
canonicalizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = canonicalizer
SPEC.loader.exec_module(canonicalizer)


@pytest.fixture(autouse=True)
def hermetic_scoring_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a tiny official-checker-shaped tree for every test.

    The suite must not depend on the ignored external_tools checkout. The fake
    checker still replays X through ONNX Runtime and applies the generated
    VNN-LIB threshold, so the positive and low-margin negative controls retain
    their intended end-to-end meaning.
    """

    # Mirror the official checkout layout. Both the 2025 and 2026 settings
    # modules walk three parents from settings.py and require a sibling
    # vnncompYYYY_benchmarks checkout during import.
    layout = tmp_path / "official-layout"
    benchmark_repo = layout / "vnncomp2025_benchmarks"
    benchmark_repo.mkdir(parents=True)
    (benchmark_repo / "snapshot-layout-marker").write_text(
        "authenticated fixture benchmark\n", encoding="utf-8"
    )
    scoring = layout / "vnncomp2025_results" / "SCORING-ZERO-TOL"
    scoring.mkdir(parents=True)
    (scoring / "settings.py").write_text(
        r'''import os
from pathlib import Path

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
BENCHMARK_REPO = Path(base_dir) / "vnncomp2025_benchmarks"
assert BENCHMARK_REPO.is_dir(), f"missing sibling benchmark repo: {BENCHMARK_REPO}"
assert (BENCHMARK_REPO / "snapshot-layout-marker").read_text(encoding="utf-8") == "authenticated fixture benchmark\n"
MARKER = "hermetic-safe"
''',
        encoding="utf-8",
    )
    (scoring / "vnnlib.py").write_text(
        "# Hermetic sibling sealed into checker evidence.\n", encoding="utf-8"
    )
    (scoring / "counterexamples.py").write_text(
        r'''import enum
import re

import numpy as np
import onnxruntime as ort
import settings

if settings.MARKER != "hermetic-safe":
    raise RuntimeError("poisoned settings sibling was imported")

_Y_LIMIT = re.compile(r"\(assert\s+\(<=\s+Y_(\d+)\s+([^\s()]+)\)\s*\)")
_ENTRY = re.compile(r"\(\s*([XY])_(\d+)\s+([^\s()]+)\s*\)")


class Result(enum.Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"


def _limits(vnnlib_path):
    text = open(vnnlib_path, encoding="utf-8").read()
    limits = {int(index): float(value) for index, value in _Y_LIMIT.findall(text)}
    return [limits[index] for index in range(len(limits))]


def is_specification_vio(
    model_path, vnnlib_path, x_values, y_values, *, input_tol, output_tol
):
    del model_path, x_values, input_tol
    limits = _limits(vnnlib_path)
    violated = len(y_values) == len(limits) and all(
        value <= limit + output_tol for value, limit in zip(y_values, limits)
    )
    return violated, "hermetic written-witness threshold check"


def get_ce_diff(model_path, vnnlib_path, ce_path, abs_tol, rel_tol):
    del abs_tol, rel_tol
    entries = _ENTRY.findall(open(ce_path, encoding="utf-8").read())
    x_map = {int(index): float(value) for kind, index, value in entries if kind == "X"}
    x_values = [x_map[index] for index in range(len(x_map))]
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    shape = [int(dimension) for dimension in model_input.shape]
    dtype = np.float64 if model_input.type == "tensor(double)" else np.float32
    model_outputs = session.run(None, {model_input.name: np.asarray(x_values, dtype=dtype).reshape(shape)})
    if len(model_outputs) != 1:
        return Result.INCORRECT, "hermetic checker requires one output"
    actual = tuple(float(value) for value in np.asarray(model_outputs[0]).reshape(-1))
    violated, message = is_specification_vio(
        model_path,
        vnnlib_path,
        tuple(x_values),
        actual,
        input_tol=0.0,
        output_tol=0.0,
    )
    return (Result.CORRECT if violated else Result.INCORRECT), message
''',
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "SCORING_DIR", scoring)


def test_checker_snapshot_preserves_official_relative_benchmark_layout() -> None:
    snapshot, staged, evidence = canonicalizer._snapshot_official_checker(
        SCORING_DIR
    )
    try:
        assert staged.parent.name == "vnncomp2025_results"
        assert staged.name == "SCORING-ZERO-TOL"
        staged_benchmark = staged.parent.parent / "vnncomp2025_benchmarks"
        source_benchmark = SCORING_DIR.parent.parent / "vnncomp2025_benchmarks"
        assert staged_benchmark.is_dir()
        assert staged_benchmark.resolve() == source_benchmark.resolve()
        assert evidence["benchmark_repositories"] == {
            "vnncomp2025_benchmarks": str(source_benchmark.resolve())
        }
    finally:
        snapshot.cleanup()


def write_vnnlib(
    path: Path,
    *,
    n_inputs: int = 1,
    n_outputs: int = 1,
    y_upper: float = 100.0,
) -> Path:
    """Write a box property whose violation region is every Y_i <= y_upper."""

    lines = [f"(declare-const X_{i} Real)" for i in range(n_inputs)]
    lines.extend(f"(declare-const Y_{i} Real)" for i in range(n_outputs))
    for i in range(n_inputs):
        lines.append(f"(assert (<= X_{i} 100.0))")
        lines.append(f"(assert (>= X_{i} -100.0))")
    for i in range(n_outputs):
        lines.append(f"(assert (<= Y_{i} {y_upper}))")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def result_bytes(x_tokens: list[str], y_tokens: list[str]) -> bytes:
    entries = [f"(X_{i} {value})" for i, value in enumerate(x_tokens)]
    entries.extend(f"(Y_{i} {value})" for i, value in enumerate(y_tokens))
    return ("sat\n(" + "\n".join(entries) + ")\n").encode()


def identity_model_bytes(onnx, *, input_type=None, outputs: int = 1) -> bytes:
    input_type = input_type or onnx.TensorProto.FLOAT
    x_info = onnx.helper.make_tensor_value_info("input", input_type, [1])
    output_infos = [
        onnx.helper.make_tensor_value_info(f"output_{index}", input_type, [1])
        for index in range(outputs)
    ]
    nodes = [
        onnx.helper.make_node("Identity", ["input"], [f"output_{index}"])
        for index in range(outputs)
    ]
    graph = onnx.helper.make_graph(
        nodes, "canonicalizer-identity-test", [x_info], output_infos
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    model.ir_version = min(model.ir_version, 8)
    return model.SerializeToString()


def add_model_bytes(onnx, bias_value: float) -> bytes:
    x_info = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1]
    )
    y_info = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1]
    )
    bias = onnx.helper.make_tensor(
        "bias", onnx.TensorProto.FLOAT, [1], [bias_value]
    )
    node = onnx.helper.make_node("Add", ["input", "bias"], ["output"])
    graph = onnx.helper.make_graph(
        [node], "canonicalizer-swap-test", [x_info], [y_info], [bias]
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    model.ir_version = min(model.ir_version, 8)
    return model.SerializeToString()


def test_parse_and_render_preserve_exact_x_tokens() -> None:
    source = result_bytes(["-0.0", "1.0000000000000002"], ["9", "-4"])
    assignment = canonicalizer.parse_sat_result(source)
    rendered = canonicalizer.render_sat_result(assignment, [1.5, -2.25])

    assert assignment.x_tokens == ("-0.0", "1.0000000000000002")
    assert b"(X_0 -0.0)" in rendered
    assert b"(X_1 1.0000000000000002)" in rendered
    assert b"(Y_0 1.5)" in rendered
    assert b"(Y_1 -2.25)" in rendered


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (b"unsat\n", "SAT verdict"),
        (b"sat\n((X_1 0)(Y_0 0))\n", "contiguous"),
        (b"sat\n((X_0 nan)(Y_0 0))\n", "finite"),
        (b"sat\n((X_0 0)(Y_1 0))\n", "contiguous"),
        (b"sat\n((X_0 0)(Y_0 0)(X_1 1))\n", "precede"),
        (b"sat\n((X_0 0)(Y_0 0)garbage)\n", "unsupported"),
    ],
)
def test_parser_fails_closed(source: bytes, message: str) -> None:
    with pytest.raises(canonicalizer.CanonicalizationError, match=message):
        canonicalizer.parse_sat_result(source)


def test_render_rejects_output_arity_drift() -> None:
    assignment = canonicalizer.parse_sat_result(result_bytes(["0"], ["0", "1"]))
    with pytest.raises(canonicalizer.CanonicalizationError, match="arity"):
        canonicalizer.render_sat_result(assignment, [2.0])


def test_integration_recomputes_only_y_and_emits_receipt(tmp_path: Path) -> None:
    x_info = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, 2]
    )
    y_info = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 2]
    )
    bias = onnx.helper.make_tensor(
        "bias", onnx.TensorProto.FLOAT, [2], [0.25, -0.5]
    )
    node = onnx.helper.make_node("Add", ["input", "bias"], ["output"])
    graph = onnx.helper.make_graph([node], "canonicalizer-test", [x_info], [y_info], [bias])
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    model.ir_version = min(model.ir_version, 8)

    model_path = tmp_path / "model.onnx"
    source_path = tmp_path / "source.results"
    output_path = tmp_path / "canonical.results"
    receipt_path = tmp_path / "canonical.receipt.json"
    vnnlib_path = write_vnnlib(
        tmp_path / "property.vnnlib", n_inputs=2, n_outputs=2
    )
    onnx.save(model, model_path)
    source_path.write_bytes(result_bytes(["1.0", "-2.0"], ["99", "98"]))

    receipt = canonicalizer.canonicalize(
        model_path,
        source_path,
        output_path,
        vnnlib_path=vnnlib_path,
        scoring_dir=SCORING_DIR,
        receipt_path=receipt_path,
        required_ort_version=ort.__version__,
        required_provider="CPUExecutionProvider",
    )
    canonical = canonicalizer.parse_sat_result(output_path.read_bytes())

    assert canonical.x_tokens == ("1.0", "-2.0")
    assert [float(value) for value in canonical.y_tokens] == pytest.approx(
        [1.25, -2.5], abs=0.0, rel=0.0
    )
    assert source_path.read_bytes() == result_bytes(
        ["1.0", "-2.0"], ["99", "98"]
    )
    assert receipt["policy"]["x_tokens_preserved"] is True
    assert receipt["policy"]["source_overwritten"] is False
    assert receipt["runtime"]["onnxruntime"] == ort.__version__
    assert receipt["runtime"]["session_providers"] == ["CPUExecutionProvider"]
    assert receipt["runtime"]["selected_provider"] == "CPUExecutionProvider"
    assert receipt["runtime"]["output_dtype"] == "float32"
    assert receipt["runtime"]["output_shape"] == [1, 2]
    assert json.loads(receipt_path.read_text())["output_result"]["sha256"] == (
        receipt["output_result"]["sha256"]
    )

    assert receipt["replay_gate"]["official_replay_result"] == "correct"
    assert receipt["replay_gate"]["written_witness_violates_property"] is True
    assert receipt["replay_gate"]["abs_tolerance"] == 0.0
    assert receipt["replay_gate"]["rel_tolerance"] == 0.0
    assert receipt["policy"]["tol0_replay_gate"] == "required"
    assert receipt["vnnlib"]["sha256"] == hashlib.sha256(
        vnnlib_path.read_bytes()
    ).hexdigest()

    with pytest.raises(canonicalizer.CanonicalizationError, match="overwrite"):
        canonicalizer.canonicalize(
            model_path,
            source_path,
            output_path,
            vnnlib_path=vnnlib_path,
            scoring_dir=SCORING_DIR,
            receipt_path=None,
            required_ort_version=ort.__version__,
            required_provider="CPUExecutionProvider",
        )


def test_inference_uses_hashed_model_bytes_during_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_a = add_model_bytes(onnx, 0.0)
    model_b = add_model_bytes(onnx, 10.0)
    model_path = tmp_path / "model.onnx"
    source_path = tmp_path / "source.results"
    output_path = tmp_path / "canonical.results"
    receipt_path = tmp_path / "canonical.receipt.json"
    vnnlib_path = write_vnnlib(tmp_path / "property.vnnlib")
    model_path.write_bytes(model_a)
    source_path.write_bytes(result_bytes(["1.0"], ["99"]))

    original = canonicalizer.canonical_outputs

    def swap_during_inference(model_bytes, assignment, **kwargs):
        assert model_bytes == model_a
        model_path.write_bytes(model_b)
        try:
            return original(model_bytes, assignment, **kwargs)
        finally:
            model_path.write_bytes(model_a)

    monkeypatch.setattr(canonicalizer, "canonical_outputs", swap_during_inference)
    receipt = canonicalizer.canonicalize(
        model_path,
        source_path,
        output_path,
        vnnlib_path=vnnlib_path,
        scoring_dir=SCORING_DIR,
        receipt_path=receipt_path,
        required_ort_version=ort.__version__,
        required_provider="CPUExecutionProvider",
    )

    canonical = canonicalizer.parse_sat_result(output_path.read_bytes())
    assert canonical.y_tokens == ("1.0",)
    assert receipt["onnx"]["sha256"] == hashlib.sha256(model_a).hexdigest()


def test_rejects_multiple_graph_outputs(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    source_path = tmp_path / "source.results"
    output_path = tmp_path / "canonical.results"
    vnnlib_path = write_vnnlib(tmp_path / "property.vnnlib")
    model_path.write_bytes(identity_model_bytes(onnx, outputs=2))
    source_path.write_bytes(result_bytes(["1.0"], ["0"]))

    with pytest.raises(canonicalizer.CanonicalizationError, match="one model output"):
        canonicalizer.canonicalize(
            model_path,
            source_path,
            output_path,
            vnnlib_path=vnnlib_path,
            scoring_dir=SCORING_DIR,
            receipt_path=None,
            required_ort_version=ort.__version__,
            required_provider="CPUExecutionProvider",
        )
    assert not output_path.exists()


def test_rejects_multiple_noninitializer_inputs(tmp_path: Path) -> None:
    left = onnx.helper.make_tensor_value_info(
        "left", onnx.TensorProto.FLOAT, [1]
    )
    right = onnx.helper.make_tensor_value_info(
        "right", onnx.TensorProto.FLOAT, [1]
    )
    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1]
    )
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["left", "right"], ["output"])],
        "canonicalizer-two-input-test",
        [left, right],
        [output],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    model.ir_version = min(model.ir_version, 8)
    model_path = tmp_path / "model.onnx"
    source_path = tmp_path / "source.results"
    output_path = tmp_path / "canonical.results"
    vnnlib_path = write_vnnlib(tmp_path / "property.vnnlib")
    model_path.write_bytes(model.SerializeToString())
    source_path.write_bytes(result_bytes(["1.0"], ["0"]))

    with pytest.raises(canonicalizer.CanonicalizationError, match="one model input"):
        canonicalizer.canonicalize(
            model_path,
            source_path,
            output_path,
            vnnlib_path=vnnlib_path,
            scoring_dir=SCORING_DIR,
            receipt_path=None,
            required_ort_version=ort.__version__,
            required_provider="CPUExecutionProvider",
        )
    assert not output_path.exists()


def test_rejects_nonfloating_input_dtype(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    source_path = tmp_path / "source.results"
    output_path = tmp_path / "canonical.results"
    vnnlib_path = write_vnnlib(tmp_path / "property.vnnlib")
    model_path.write_bytes(
        identity_model_bytes(onnx, input_type=onnx.TensorProto.INT64)
    )
    source_path.write_bytes(result_bytes(["1.0"], ["0"]))

    with pytest.raises(
        canonicalizer.CanonicalizationError, match="only FLOAT and DOUBLE"
    ):
        canonicalizer.canonicalize(
            model_path,
            source_path,
            output_path,
            vnnlib_path=vnnlib_path,
            scoring_dir=SCORING_DIR,
            receipt_path=None,
            required_ort_version=ort.__version__,
            required_provider="CPUExecutionProvider",
        )
    assert not output_path.exists()


def test_provider_must_be_exclusively_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSession:
        def get_providers(self):
            return ["OtherExecutionProvider", "CPUExecutionProvider"]

    class FakeOrt:
        __version__ = ort.__version__
        SessionOptions = ort.SessionOptions

        def __init__(self):
            self.requested_providers = None

        @staticmethod
        def get_available_providers():
            return ["OtherExecutionProvider", "CPUExecutionProvider"]

        def InferenceSession(self, model, options, *, providers):  # noqa: N802
            self.requested_providers = providers
            return FakeSession()

    fake_ort = FakeOrt()
    monkeypatch.setattr(
        canonicalizer, "_load_runtime", lambda: (np, onnx, fake_ort)
    )
    assignment = canonicalizer.parse_sat_result(result_bytes(["1.0"], ["0"]))

    with pytest.raises(
        canonicalizer.CanonicalizationError, match="selected exclusively"
    ):
        canonicalizer.canonical_outputs(
            identity_model_bytes(onnx),
            assignment,
            required_ort_version=ort.__version__,
            required_provider="CPUExecutionProvider",
        )
    assert fake_ort.requested_providers == ["CPUExecutionProvider"]


def test_rejects_runtime_output_dtype_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSession:
        @staticmethod
        def get_providers():
            return ["CPUExecutionProvider"]

        @staticmethod
        def get_inputs():
            return [SimpleNamespace(name="input")]

        @staticmethod
        def get_outputs():
            return [SimpleNamespace(name="output_0")]

        @staticmethod
        def run(outputs, inputs):
            return [np.asarray([1.0], dtype=np.float64)]

    class FakeOrt:
        __version__ = ort.__version__
        SessionOptions = ort.SessionOptions

        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(model, options, *, providers):  # noqa: N802
            assert providers == ["CPUExecutionProvider"]
            return FakeSession()

    monkeypatch.setattr(canonicalizer, "_load_runtime", lambda: (np, onnx, FakeOrt))
    assignment = canonicalizer.parse_sat_result(result_bytes(["1.0"], ["0"]))

    with pytest.raises(canonicalizer.CanonicalizationError, match="output dtype"):
        canonicalizer.canonical_outputs(
            identity_model_bytes(onnx),
            assignment,
            required_ort_version=ort.__version__,
            required_provider="CPUExecutionProvider",
        )


@pytest.mark.parametrize(
    "symlinked_path",
    [
        "model_ancestor",
        "result_ancestor",
        "output_ancestor",
        "receipt_ancestor",
        "model_leaf",
        "result_leaf",
        "output_leaf",
        "receipt_leaf",
    ],
)
def test_rejects_symlink_in_every_path_chain(
    tmp_path: Path, symlinked_path: str
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    nested = real / "nested"
    nested.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    model_path = real / "model.onnx"
    result_path = real / "source.results"
    output_path = tmp_path / "canonical.results"
    receipt_path = tmp_path / "canonical.receipt.json"
    vnnlib_path = write_vnnlib(real / "property.vnnlib")
    model_path.write_bytes(identity_model_bytes(onnx))
    result_path.write_bytes(result_bytes(["1.0"], ["0"]))

    if symlinked_path == "model_ancestor":
        model_path = linked / "model.onnx"
    elif symlinked_path == "result_ancestor":
        result_path = linked / "source.results"
    elif symlinked_path == "output_ancestor":
        output_path = linked / "nested" / "canonical.results"
    elif symlinked_path == "receipt_ancestor":
        receipt_path = linked / "nested" / "canonical.receipt.json"
    elif symlinked_path == "model_leaf":
        model_link = tmp_path / "model-link.onnx"
        model_link.symlink_to(model_path)
        model_path = model_link
    elif symlinked_path == "result_leaf":
        result_link = tmp_path / "result-link.results"
        result_link.symlink_to(result_path)
        result_path = result_link
    elif symlinked_path == "output_leaf":
        output_path.symlink_to(real / "unused-output")
    else:
        receipt_path.symlink_to(real / "unused-receipt")

    with pytest.raises(canonicalizer.CanonicalizationError, match="symlink"):
        canonicalizer.canonicalize(
            model_path,
            result_path,
            output_path,
            vnnlib_path=vnnlib_path,
            scoring_dir=SCORING_DIR,
            receipt_path=receipt_path,
            required_ort_version=ort.__version__,
            required_provider="CPUExecutionProvider",
        )
    if symlinked_path.startswith("receipt"):
        assert not output_path.exists()


def test_output_and_receipt_must_be_distinct(tmp_path: Path) -> None:
    same_path = tmp_path / "same.results"
    with pytest.raises(canonicalizer.CanonicalizationError, match="distinct"):
        canonicalizer.canonicalize(
            tmp_path / "unused.onnx",
            tmp_path / "unused.results",
            same_path,
            vnnlib_path=tmp_path / "unused.vnnlib",
            scoring_dir=SCORING_DIR,
            receipt_path=same_path,
            required_ort_version="unused",
            required_provider="CPUExecutionProvider",
        )
    assert not same_path.exists()


def test_existing_receipt_prevents_output_publication(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    source_path = tmp_path / "source.results"
    output_path = tmp_path / "canonical.results"
    receipt_path = tmp_path / "canonical.receipt.json"
    vnnlib_path = write_vnnlib(tmp_path / "property.vnnlib")
    model_path.write_bytes(identity_model_bytes(onnx))
    source_path.write_bytes(result_bytes(["1.0"], ["0"]))
    receipt_path.write_bytes(b"preexisting receipt")

    with pytest.raises(
        canonicalizer.CanonicalizationError, match="existing receipt"
    ):
        canonicalizer.canonicalize(
            model_path,
            source_path,
            output_path,
            vnnlib_path=vnnlib_path,
            scoring_dir=SCORING_DIR,
            receipt_path=receipt_path,
            required_ort_version=ort.__version__,
            required_provider="CPUExecutionProvider",
        )
    assert not output_path.exists()
    assert receipt_path.read_bytes() == b"preexisting receipt"


def test_existing_output_prevents_receipt_publication(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    source_path = tmp_path / "source.results"
    output_path = tmp_path / "canonical.results"
    receipt_path = tmp_path / "canonical.receipt.json"
    vnnlib_path = write_vnnlib(tmp_path / "property.vnnlib")
    model_path.write_bytes(identity_model_bytes(onnx))
    source_path.write_bytes(result_bytes(["1.0"], ["0"]))
    output_path.write_bytes(b"preexisting output")

    with pytest.raises(
        canonicalizer.CanonicalizationError, match="existing output"
    ):
        canonicalizer.canonicalize(
            model_path,
            source_path,
            output_path,
            vnnlib_path=vnnlib_path,
            scoring_dir=SCORING_DIR,
            receipt_path=receipt_path,
            required_ort_version=ort.__version__,
            required_provider="CPUExecutionProvider",
        )
    assert output_path.read_bytes() == b"preexisting output"
    assert not receipt_path.exists()


def gate_case_paths(tmp_path: Path, *, y_upper: float) -> tuple[Path, Path, Path, Path, Path]:
    model_path = tmp_path / "model.onnx"
    source_path = tmp_path / "source.results"
    output_path = tmp_path / "canonical.results"
    receipt_path = tmp_path / "canonical.receipt.json"
    vnnlib_path = write_vnnlib(tmp_path / "property.vnnlib", y_upper=y_upper)
    source_path.write_bytes(result_bytes(["1.0"], ["0.0"]))
    return model_path, source_path, output_path, receipt_path, vnnlib_path


def test_replay_gate_passes_valid_witness_end_to_end(tmp_path: Path) -> None:
    # POSITIVE CONTROL: identity network, X_0=1.0 -> Y_0=1.0, violation
    # region Y_0 <= 100.  The canonical witness still violates at tol=0, so
    # the gate must admit it and record the strict verdict in the receipt.
    model_path, source_path, output_path, receipt_path, vnnlib_path = (
        gate_case_paths(tmp_path, y_upper=100.0)
    )
    model_path.write_bytes(identity_model_bytes(onnx))

    receipt = canonicalizer.canonicalize(
        model_path,
        source_path,
        output_path,
        vnnlib_path=vnnlib_path,
        scoring_dir=SCORING_DIR,
        receipt_path=receipt_path,
        required_ort_version=ort.__version__,
        required_provider="CPUExecutionProvider",
    )

    assert receipt["replay_gate"]["official_replay_result"] == "correct"
    assert receipt["replay_gate"]["written_witness_violates_property"] is True
    assert receipt["replay_gate"]["abs_tolerance"] == 0.0
    assert receipt["replay_gate"]["rel_tolerance"] == 0.0
    assert receipt["official_checker"]["files"]["counterexamples.py"]["sha256"]
    assert output_path.exists()


def test_replay_gate_ignores_poisoned_parent_checker_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path, source_path, output_path, receipt_path, vnnlib_path = (
        gate_case_paths(tmp_path, y_upper=100.0)
    )
    model_path.write_bytes(identity_model_bytes(onnx))
    monkeypatch.setitem(
        sys.modules,
        "settings",
        SimpleNamespace(MARKER="poisoned-parent-module"),
    )

    receipt = canonicalizer.canonicalize(
        model_path,
        source_path,
        output_path,
        vnnlib_path=vnnlib_path,
        scoring_dir=SCORING_DIR,
        receipt_path=receipt_path,
        required_ort_version=ort.__version__,
        required_provider="CPUExecutionProvider",
    )

    assert receipt["replay_gate"]["official_replay_result"] == "correct"
    assert output_path.exists()


def test_replay_gate_does_not_reintroduce_pythonpath_site_packages(
    tmp_path: Path,
) -> None:
    """The isolated checker must not bless a parent PYTHONPATH by its name."""

    model_path, source_path, output_path, receipt_path, vnnlib_path = (
        gate_case_paths(tmp_path, y_upper=100.0)
    )
    model_path.write_bytes(identity_model_bytes(onnx))

    # The checker genuinely imports cachier from the interpreter's declared
    # installation roots. Under the old `sys.path` filter, a parent path merely
    # named `site-packages` was forwarded ahead of that real installation and
    # this poison module executed inside the supposedly isolated worker.
    checker_path = SCORING_DIR / "counterexamples.py"
    checker_path.write_text(
        "import cachier\n" + checker_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    poison = tmp_path / "site-packages"
    poison.mkdir()
    (poison / "cachier.py").write_text(
        "raise RuntimeError('inherited PYTHONPATH reached checker worker')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(poison)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--onnx",
            str(model_path),
            "--result",
            str(source_path),
            "--output",
            str(output_path),
            "--vnnlib",
            str(vnnlib_path),
            "--scoring-dir",
            str(SCORING_DIR),
            "--receipt",
            str(receipt_path),
            "--require-onnxruntime",
            ort.__version__,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()
    assert receipt_path.exists()


def test_replay_gate_rejects_worker_runtime_version_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child that cannot reproduce the parent runtime may not certify."""

    model_path, source_path, output_path, receipt_path, vnnlib_path = (
        gate_case_paths(tmp_path, y_upper=100.0)
    )
    model_path.write_bytes(identity_model_bytes(onnx))
    fake_version = "0.invalid-worker-mismatch"
    original = canonicalizer.canonical_outputs

    def forged_parent_runtime(model_bytes, assignment, **kwargs):
        outputs, runtime = original(
            model_bytes,
            assignment,
            required_ort_version=ort.__version__,
            required_provider=kwargs["required_provider"],
        )
        runtime["onnxruntime"] = fake_version
        return outputs, runtime

    monkeypatch.setattr(
        canonicalizer,
        "canonical_outputs",
        forged_parent_runtime,
    )

    with pytest.raises(
        canonicalizer.CanonicalizationError,
        match="runtime binding.*onnxruntime version mismatch",
    ):
        canonicalizer.canonicalize(
            model_path,
            source_path,
            output_path,
            vnnlib_path=vnnlib_path,
            scoring_dir=SCORING_DIR,
            receipt_path=receipt_path,
            required_ort_version=fake_version,
            required_provider="CPUExecutionProvider",
        )
    assert not output_path.exists()
    assert not receipt_path.exists()


def test_replay_gate_negative_control_broken_y_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # NEGATIVE CONTROL 1 (check_offline_scorecard.sh discipline): a
    # deliberately-broken canonicalization writes Y values on the SAFE side of
    # the property (999 > 100).  The gate must refuse, else it certifies
    # nothing.
    model_path, source_path, output_path, receipt_path, vnnlib_path = (
        gate_case_paths(tmp_path, y_upper=100.0)
    )
    model_path.write_bytes(identity_model_bytes(onnx))
    original = canonicalizer.canonical_outputs

    def broken_outputs(*args, **kwargs):
        _outputs, runtime = original(*args, **kwargs)
        return [999.0], runtime

    monkeypatch.setattr(
        canonicalizer,
        "canonical_outputs",
        broken_outputs,
    )

    with pytest.raises(
        canonicalizer.CanonicalizationError, match="no longer violate"
    ):
        canonicalizer.canonicalize(
            model_path,
            source_path,
            output_path,
            vnnlib_path=vnnlib_path,
            scoring_dir=SCORING_DIR,
            receipt_path=receipt_path,
            required_ort_version=ort.__version__,
            required_provider="CPUExecutionProvider",
        )
    assert not output_path.exists()
    assert not receipt_path.exists()


def test_replay_gate_negative_control_low_margin_flip_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # NEGATIVE CONTROL 2 (the low-margin soundnessbench hazard): the written
    # Y claims a violation (0.4 <= 0.5) but the true tol=0 replay of X gives
    # Y=1.0, which does NOT violate.  The official strict replay must refuse.
    model_path, source_path, output_path, receipt_path, vnnlib_path = (
        gate_case_paths(tmp_path, y_upper=0.5)
    )
    model_path.write_bytes(identity_model_bytes(onnx))
    original = canonicalizer.canonical_outputs

    def forged_low_margin_outputs(*args, **kwargs):
        _outputs, runtime = original(*args, **kwargs)
        return [0.4], runtime

    monkeypatch.setattr(
        canonicalizer,
        "canonical_outputs",
        forged_low_margin_outputs,
    )

    with pytest.raises(
        canonicalizer.CanonicalizationError, match="official checker verdict"
    ):
        canonicalizer.canonicalize(
            model_path,
            source_path,
            output_path,
            vnnlib_path=vnnlib_path,
            scoring_dir=SCORING_DIR,
            receipt_path=receipt_path,
            required_ort_version=ort.__version__,
            required_provider="CPUExecutionProvider",
        )
    assert not output_path.exists()
    assert not receipt_path.exists()


def test_replay_gate_missing_scoring_dir_fails_closed(tmp_path: Path) -> None:
    model_path, source_path, output_path, receipt_path, vnnlib_path = (
        gate_case_paths(tmp_path, y_upper=100.0)
    )
    model_path.write_bytes(identity_model_bytes(onnx))

    with pytest.raises(
        canonicalizer.CanonicalizationError, match="SCORING directory is unavailable"
    ):
        canonicalizer.canonicalize(
            model_path,
            source_path,
            output_path,
            vnnlib_path=vnnlib_path,
            scoring_dir=tmp_path / "no-such-scoring-dir",
            receipt_path=receipt_path,
            required_ort_version=ort.__version__,
            required_provider="CPUExecutionProvider",
        )
    assert not output_path.exists()
    assert not receipt_path.exists()


def test_cli_requires_vnnlib_and_scoring_dir(tmp_path: Path) -> None:
    # The gate must not be skippable from the command line.
    with pytest.raises(SystemExit) as excinfo:
        canonicalizer.main(
            [
                "--onnx",
                str(tmp_path / "m.onnx"),
                "--result",
                str(tmp_path / "s.results"),
                "--output",
                str(tmp_path / "o.results"),
                "--require-onnxruntime",
                ort.__version__,
            ]
        )
    assert excinfo.value.code == 2
