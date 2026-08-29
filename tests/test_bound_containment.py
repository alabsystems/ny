#!/usr/bin/env python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Hermetic tests for the sampled bound-containment checker.

The point of these tests is the NEGATIVE CONTROL: a detector that has never
been shown to fire on a known-bad bound is not trustworthy.  Each test builds a
tiny ONNX network, computes an interval that is *provably* correct or
*provably* too tight, and asserts the checker's verdict.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bound_containment", REPO / "scripts" / "bound_containment.py"
)
assert SPEC is not None and SPEC.loader is not None
bc = importlib.util.module_from_spec(SPEC)
sys.modules["bound_containment"] = bc
SPEC.loader.exec_module(bc)

N_IN, N_HIDDEN, N_OUT = 4, 5, 3
INTERMEDIATES = ("mm1", "pre1", "act1", "mm2")


def _build_model(path: Path) -> None:
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(7)
    weights = {
        "W1": rng.normal(size=(N_IN, N_HIDDEN)).astype(np.float32),
        "b1": rng.normal(size=(N_HIDDEN,)).astype(np.float32),
        "W2": rng.normal(size=(N_HIDDEN, N_OUT)).astype(np.float32),
        "b2": rng.normal(size=(N_OUT,)).astype(np.float32),
    }
    nodes = [
        helper.make_node("MatMul", ["input", "W1"], ["mm1"], name="mm1"),
        helper.make_node("Add", ["mm1", "b1"], ["pre1"], name="pre1"),
        helper.make_node("Relu", ["pre1"], ["act1"], name="act1"),
        helper.make_node("MatMul", ["act1", "W2"], ["mm2"], name="mm2"),
        helper.make_node("Add", ["mm2", "b2"], ["output"], name="out"),
    ]
    graph = helper.make_graph(
        nodes,
        "containment_fixture",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, N_IN])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, N_OUT])],
        [numpy_helper.from_array(value, name) for name, value in weights.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _vertex_hull(path: Path, lower: np.ndarray, upper: np.ndarray) -> dict:
    from onnx import helper

    model = onnx.load(str(path))
    for name in INTERMEDIATES:
        model.graph.output.append(helper.make_empty_tensor_value_info(name))
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = [out.name for out in session.get_outputs()]
    hull: dict[str, list[np.ndarray]] = {}
    for bits in itertools.product([0, 1], repeat=N_IN):
        point = np.where(np.array(bits, dtype=bool), upper, lower)
        values = session.run(
            None, {"input": point.astype(np.float32).reshape(1, N_IN)}
        )
        for name, value in zip(names, values):
            flat = np.asarray(value, dtype=np.float64).ravel()
            if name not in hull:
                hull[name] = [flat.copy(), flat.copy()]
            else:
                hull[name][0] = np.minimum(hull[name][0], flat)
                hull[name][1] = np.maximum(hull[name][1], flat)
    return hull


def _write_dump(
    path: Path, lower: np.ndarray, upper: np.ndarray, hull: dict, pad: float
) -> None:
    def fmt(values: np.ndarray) -> str:
        return " ".join(repr(float(np.float32(v))) for v in values)

    lines = [
        "# ny lpopt dump v1",
        f"INPUT {N_IN} 1 {N_IN}",
        "L " + fmt(lower),
        "U " + fmt(upper),
        "RELUMAP 1",
        "act1 pre1",
    ]
    for name, (lo, hi) in hull.items():
        width = np.maximum(hi - lo, 1e-6)
        lines.append(f"NODE {name} {lo.size} 1 {lo.size}")
        lines.append("L " + fmt(lo - pad * width))
        lines.append("U " + fmt(hi + pad * width))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> dict:
    directory = tmp_path_factory.mktemp("containment")
    model = directory / "net.onnx"
    _build_model(model)
    lower = np.full(N_IN, -1.0)
    upper = np.full(N_IN, 1.0)
    hull = _vertex_hull(model, lower, upper)
    dumps = {}
    for label, pad in (("sound", 0.5), ("hull", 0.0), ("too_tight", -0.25)):
        dump = directory / f"{label}.dump"
        _write_dump(dump, lower, upper, hull, pad)
        dumps[label] = dump
    return {"model": model, "dumps": dumps, "dir": directory}


def _check(fixture: dict, label: str, **overrides) -> dict:
    args = bc.argparse.Namespace(
        command="check",
        category="unit",
        onnx=str(fixture["model"]),
        vnnlib=str(fixture["dir"] / "missing.vnnlib"),
        dump=str(fixture["dumps"][label]),
        samples=overrides.pop("samples", 64),
        seed=1234,
        spsa_steps=overrides.pop("spsa_steps", 8),
        tol_abs=1e-6,
        tol_rel=1e-5,
        noise_k=8.0,
        f64_confirm=overrides.pop("f64_confirm", False),
        mutate_shrink=overrides.pop("mutate_shrink", 0.0),
        mutate_node=overrides.pop("mutate_node", None),
        witness=overrides.pop("witness", None),
        stop_on_first=False,
        max_report=10,
        json=None,
        ny_verdict="",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return bc.do_check(args)


def test_sound_bounds_report_no_violation(fixture: dict) -> None:
    payload = _check(fixture, "sound")
    assert payload["coverage"]["checked"] == len(INTERMEDIATES) + 1
    assert payload["violations"]["confirmed_total"] == 0
    # A clean run must never claim soundness.
    assert "proves NOTHING" in payload["disclaimer"]


def test_too_tight_bounds_are_caught(fixture: dict) -> None:
    """NEGATIVE CONTROL: a 25%-shrunk interval must be refuted."""
    payload = _check(fixture, "too_tight", f64_confirm=True)
    assert payload["violations"]["confirmed_total"] > 0
    first = payload["violations"]["confirmed"][0]
    # Earliest node in execution order, not a downstream symptom.
    assert first["node"] == "mm1"
    assert first["excess"] > first["envelope"]
    assert first["verdict"].startswith("CONFIRMED")


def test_vertex_hull_is_unsound_past_the_relu(fixture: dict) -> None:
    """The affine-vertex hull is NOT an enclosure once a ReLU intervenes.

    This is why interior points are sampled too: ``mm2``'s extreme is not at a
    vertex of the input box, so a vertex-only checker would miss it.  256
    samples detect it on every seed tried (1, 2, 3, 1234, 20260808); 64 detect
    it on one of five -- the budget is load-bearing, which is exactly why a
    clean run is never a certificate.
    """
    payload = _check(fixture, "hull", samples=256, spsa_steps=24)
    nodes = {item["node"] for item in payload["violations"]["confirmed"]}
    assert payload["violations"]["confirmed_total"] > 0
    # Every affine-prefix node IS exactly hulled by the vertices; only the
    # post-ReLU compositions can be violated.
    assert nodes <= {"mm2", "output"}, nodes


def test_mutation_control_fires_on_sound_bounds(fixture: dict) -> None:
    """Shrinking a sound bound must make the checker fire; otherwise it is blind."""
    clean = _check(fixture, "sound")
    assert clean["violations"]["confirmed_total"] == 0
    mutated = _check(fixture, "sound", mutate_shrink=0.9)
    assert mutated["mutation"]["nodes_mutated"] == len(INTERMEDIATES) + 1
    assert mutated["violations"]["confirmed_total"] > 0


def test_node_name_resolves_to_its_output_tensor(fixture: dict) -> None:
    """NY keys nodes by the ONNX *node* name, which is often not a tensor name.

    The fixture's last node is named ``out`` and produces ``output``; resolving
    it by name is what keeps the check from silently skipping the root.
    """
    oracle = bc.Oracle(fixture["model"], {"out", "mm1"})
    assert oracle.candidates["out"] == ["output"]
    assert oracle.candidates["mm1"] == ["mm1"]


def _fusion_model(path: Path) -> None:
    """MatMul -> BatchNormalization -> Relu, with node names != tensor names.

    This is the CIFAR100_resnet shape: NY folds the BatchNormalization into its
    producer and keeps the producer's name, so NY's ``affine`` node denotes the
    POST-BN tensor.
    """
    from onnx import TensorProto, helper, numpy_helper

    initializers = [
        numpy_helper.from_array(np.eye(2, dtype=np.float32), "W"),
        numpy_helper.from_array(np.ones(2, dtype=np.float32), "scale"),
        numpy_helper.from_array(np.zeros(2, dtype=np.float32), "bias"),
        numpy_helper.from_array(np.zeros(2, dtype=np.float32), "mean"),
        numpy_helper.from_array(np.ones(2, dtype=np.float32), "var"),
    ]
    nodes = [
        helper.make_node("MatMul", ["input", "W"], ["pre"], name="affine"),
        helper.make_node(
            "BatchNormalization",
            ["pre", "scale", "bias", "mean", "var"],
            ["normed"],
            name="bn",
        ),
        helper.make_node("Relu", ["normed"], ["output"], name="relu"),
    ]
    graph = helper.make_graph(
        nodes,
        "fusion",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        initializers,
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


def test_folded_batchnorm_resolves_to_the_post_fold_tensor(tmp_path: Path) -> None:
    """The dump's own key set says which ops NY folded.

    ``bn`` is absent from the dump, so NY's ``affine`` node spans MatMul+BN and
    denotes ``normed``.  Comparing against the raw ``pre`` tensor instead is what
    produced a spurious violation on CIFAR100_resnet_medium.
    """
    path = tmp_path / "fusion.onnx"
    _fusion_model(path)
    oracle = bc.Oracle(path, {"affine", "relu"})
    assert oracle.candidates["affine"] == ["normed"]
    assert oracle.candidates["relu"] == ["output"]


def test_kept_op_stops_the_fold_walk(tmp_path: Path) -> None:
    """When NY dumped the intermediate op too, the chain must NOT be walked."""
    path = tmp_path / "fusion.onnx"
    _fusion_model(path)
    oracle = bc.Oracle(path, {"affine", "bn", "relu"})
    assert oracle.candidates["affine"] == ["pre"]
    assert oracle.candidates["bn"] == ["normed"]


def test_inverted_interval_is_refused(tmp_path: Path) -> None:
    dump = tmp_path / "bad.dump"
    dump.write_text(
        "# ny lpopt dump v1\nINPUT 1 1 1\nL 0.0\nU 1.0\nNODE a 1 1 1\nL 1.0\nU 0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(bc.ContainmentError, match="lower > upper"):
        bc.parse_ny_dump(dump)
