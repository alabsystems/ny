# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0 OR MIT

"""VNN-LIB 2.0 tensor-indexed spec support in ``scripts/extended_bank/vnnlib_ce.py``.

The module decides whether a counterexample is GENUINE, so the tests here are
built around the danger of ACCEPTING A BAD WITNESS, not merely of refusing a
good one:

(a) the three banked, independently ORT-confirmed traffic_signs witnesses must
    validate GENUINE against the 4-D NHWC 2.0 specs;
(b) negative controls must FAIL -- box centres, a witness nudged one ulp outside
    a bound (checked on both the float64 candidate and the float32 vector that
    ORT actually receives), and a witness permuted into the wrong tensor order,
    which is the specific way a mis-derived flat mapping would show up;
(c) legacy VNN-LIB 1.x ``X_0`` specs behave exactly as before;
(d) an unsupported or mixed-syntax spec raises the DISTINCT unsupported-syntax
    error rather than degrading into a structural "not searchable" verdict.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = REPO_ROOT / "scripts" / "extended_bank"


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vnnlib_ce = _load("vnnlib_ce_for_tensor_test", "scripts/extended_bank/vnnlib_ce.py")

BENCHMARK_DIR = (
    REPO_ROOT
    / "benchmarks/vnncomp2026/benchmarks/traffic_signs_recognition_2023/2.0"
)
TRAFFIC_ONNX = (
    BENCHMARK_DIR / "onnx/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx"
)
WITNESS_DIR = REPO_ROOT / "reports/measured-2026/traffic_signs_eps1_counterexamples"
BANKED_INDICES = (178, 1703, 6371)

def _require_corpus() -> None:
    """Hard-fail, never skip, when the corpus this lane needs is absent.

    These contracts previously carried ``pytest.mark.skipif`` on
    ``TRAFFIC_ONNX.is_file()``, which the source policy prohibits: a
    conditionally-vanishing test reports green on a machine where it never ran.
    They are now ``external_*`` functions, so ordinary ``pytest tests/`` does
    not collect them at all, and selecting the lane with
    ``--override-ini 'python_functions=external_*'`` opts into the complete
    contract — including this failure when the asset is missing.
    """
    assert TRAFFIC_ONNX.is_file(), (
        f"traffic_signs_recognition_2023/2.0 corpus is not materialized at "
        f"{TRAFFIC_ONNX}; this external lane requires it. Materialize the "
        f"corpus or do not select the lane."
    )


@pytest.fixture(autouse=True)
def _corpus_guard(request: pytest.FixtureRequest) -> None:
    """Apply the corpus requirement to the external lane only.

    A fixture rather than a call in each body: it cannot be forgotten when a
    contract is added to this lane later, and it leaves the hermetic ``test_*``
    functions in this module collectable with no corpus at all.
    """
    if request.node.name.startswith("external_"):
        _require_corpus()


def _spec(index: int) -> Path:
    return BENCHMARK_DIR / f"vnnlib/model_30_idx_{index}_eps_1.00000.vnnlib"


def _witness(index: int):
    import numpy as np

    path = WITNESS_DIR / f"CEX_model_30_idx_{index}_1.00000.npy"
    return np.load(path, allow_pickle=False).astype(np.float64).ravel()


def _values(vector) -> dict[int, float]:
    return {position: float(value) for position, value in enumerate(vector)}


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


# --------------------------------------------------------------------------
# Row-major mapping derived from the declared shape
# --------------------------------------------------------------------------


def test_flat_index_is_row_major_over_the_declared_shape() -> None:
    nhwc = vnnlib_ce.TensorLayout.from_shape("X", [1, 30, 30, 3])
    assert nhwc.size == 2700
    assert nhwc.strides == (2700, 90, 3, 1)
    assert nhwc.flat_index([0, 0, 0, 0]) == 0
    assert nhwc.flat_index([0, 0, 0, 2]) == 2
    assert nhwc.flat_index([0, 1, 2, 1]) == 1 * 90 + 2 * 3 + 1
    assert nhwc.flat_index([0, 29, 29, 2]) == 2699

    # The SAME derivation, not a second NHWC special case, handles NCHW.
    nchw = vnnlib_ce.TensorLayout.from_shape("X", [1, 3, 32, 32])
    assert nchw.strides == (3072, 1024, 32, 1)
    assert nchw.flat_index([0, 1, 2, 3]) == 1024 + 64 + 3

    flat = vnnlib_ce.TensorLayout.from_shape("X", [1, 3072])
    assert flat.flat_index([0, 17]) == 17


def test_every_axis_index_is_bounds_checked() -> None:
    layout = vnnlib_ce.TensorLayout.from_shape("X", [1, 30, 30, 3])
    for indices in ([0, 30, 0, 0], [0, 0, 30, 0], [0, 0, 0, 3], [1, 0, 0, 0]):
        with pytest.raises(vnnlib_ce.ValidationError, match="out of range"):
            layout.flat_index(indices)
    # A rank mismatch is unreadable, not merely out of range.
    with pytest.raises(vnnlib_ce.UnsupportedSyntaxError, match="rank"):
        layout.flat_index([0, 1, 2])


def test_a_non_unit_leading_dimension_is_refused_not_flattened() -> None:
    with pytest.raises(vnnlib_ce.UnsupportedSyntaxError, match="leading dimension"):
        vnnlib_ce.TensorLayout.from_shape("X", [4, 30, 30, 3])


def external_declared_shape_is_read_from_the_spec_header() -> None:
    summary = vnnlib_ce._scan_property(_spec(178))
    assert summary.layout is not None
    assert summary.layout.inputs.shape == (1, 30, 30, 3)
    assert summary.layout.outputs.shape == (1, 43)
    assert summary.input_count == 2700
    assert summary.output_count == 43
    assert summary.input_assertion_count == 5400


# --------------------------------------------------------------------------
# (a) the banked witnesses validate GENUINE -- 4-D NHWC mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize("index", BANKED_INDICES)
def external_banked_traffic_signs_witnesses_are_genuine(index: int) -> None:
    in_box, genuine, detail = vnnlib_ce.validate(
        TRAFFIC_ONNX, _spec(index), _values(_witness(index))
    )
    assert in_box is True, detail
    assert genuine is True, detail
    assert "complete_inputs=2700" in detail


@pytest.mark.parametrize("index", BANKED_INDICES)
def external_flat_mapping_agrees_with_the_independent_replay_oracle(index: int) -> None:
    """Check the 4-D mapping against ground truth, not against itself.

    ``validate_independent.py`` is the banked, ORT-confirmed replay script; it
    reads the same 2.0 spec with its own regex, keys the box by the tensor
    position ``(r, c, ch)``, and indexes the witness as ``witness[0, r, c, ch]``.
    Every one of the 2700 coordinates must land on the same value under this
    module's row-major flattening of the DECLARED shape.
    """
    replay = _load(
        "traffic_signs_replay_oracle",
        "reports/measured-2026/traffic_signs_eps1_counterexamples/"
        "validate_independent.py",
    )
    lower, upper, pairs = replay.load_vnnlib(_spec(index).read_bytes())
    assert len(lower) == 2700 and len(upper) == 2700

    low, high = _box(index)
    layout = vnnlib_ce._scan_property(_spec(index)).layout
    for position, bound in lower.items():
        flat = layout.inputs.flat_index([0, *position])
        assert low[flat] == bound, (position, flat)
    for position, bound in upper.items():
        flat = layout.inputs.flat_index([0, *position])
        assert high[flat] == bound, (position, flat)

    # The output pairs the oracle extracts must resolve to the same Y indices.
    for attack, target in pairs:
        assert vnnlib_ce.resolve_variable(f"Y[0,{attack}]", layout) == ("Y", attack)
        assert vnnlib_ce.resolve_variable(f"Y[0,{target}]", layout) == ("Y", target)


# --------------------------------------------------------------------------
# (b) negative controls
# --------------------------------------------------------------------------


def _box(index: int):
    """Per-coordinate (low, high) read straight out of the 2.0 spec."""
    import numpy as np

    summary = vnnlib_ce._scan_property(_spec(index))
    layout = summary.layout
    low = np.full(summary.input_count, -np.inf)
    high = np.full(summary.input_count, np.inf)
    for expression in vnnlib_ce._file_expressions(_spec(index)):
        if not isinstance(expression, list) or expression[0] != "assert":
            continue
        node = expression[1]
        if not isinstance(node, list) or node[0] not in {"<=", ">="}:
            continue
        resolved = vnnlib_ce.resolve_variable(str(node[1]), layout)
        if resolved is None or resolved[0] != "X":
            continue
        position, constant = resolved[1], float(node[2])
        if node[0] == "<=":
            high[position] = min(high[position], constant)
        else:
            low[position] = max(low[position], constant)
    return low, high


@pytest.mark.parametrize("index", BANKED_INDICES)
def external_box_centre_is_not_a_counterexample(index: int) -> None:
    low, high = _box(index)
    in_box, genuine, detail = vnnlib_ce.validate(
        TRAFFIC_ONNX, _spec(index), _values((low + high) / 2.0)
    )
    assert in_box is True, detail
    assert genuine is False, detail


@pytest.mark.parametrize("index", BANKED_INDICES)
def external_one_ulp_outside_the_box_is_refused_after_the_float32_cast(index: int) -> None:
    """A nudge must survive BOTH representations to count as out-of-box.

    float32 rounding can pull a float64 point that is barely outside a bound
    back inside, so a test that only perturbed the float64 candidate could pass
    while the vector ORT actually receives is still contained -- and a validator
    that gated on the wrong one would accept a bad witness.  The perturbation
    below is therefore chosen so the FLOAT32 image is strictly outside, and the
    assertion checks the float32 image explicitly.
    """
    import numpy as np

    low, high = _box(index)
    witness = _witness(index)
    position = int(np.argmax(np.isfinite(high)))
    bound = float(high[position])

    nudged = witness.copy()
    step = np.nextafter(np.float32(bound), np.float32(np.inf))
    nudged[position] = float(step)

    # The value fed to ORT really is outside the declared bound.
    assert _float32(nudged[position]) > bound
    assert float(nudged[position]) > bound

    in_box, genuine, detail = vnnlib_ce.validate(
        TRAFFIC_ONNX, _spec(index), _values(nudged)
    )
    assert in_box is False, detail
    assert genuine is False, detail
    assert detail.startswith("in_box=False"), detail


def external_a_witness_permuted_into_the_wrong_tensor_order_is_refused() -> None:
    """The declared mapping is load-bearing, not decorative.

    If ``X[0,r,c,ch]`` were flattened under any other axis order, this NCHW
    transpose of a genuine NHWC witness would still land in the box.  It must
    not.
    """
    import numpy as np

    witness = _witness(178).reshape(30, 30, 3)
    permuted = np.transpose(witness, (2, 0, 1)).ravel()
    assert not np.array_equal(permuted, witness.ravel())
    in_box, genuine, detail = vnnlib_ce.validate(
        TRAFFIC_ONNX, _spec(178), _values(permuted)
    )
    assert in_box is False, detail
    assert genuine is False, detail


def external_a_short_witness_is_refused_as_incomplete() -> None:
    values = _values(_witness(178))
    values.pop(2699)
    in_box, genuine, detail = vnnlib_ce.validate(TRAFFIC_ONNX, _spec(178), values)
    assert (in_box, genuine) == (False, False)
    assert detail.startswith("incomplete witness:"), detail


# --------------------------------------------------------------------------
# (c) legacy VNN-LIB 1.x behaves exactly as before
# --------------------------------------------------------------------------


LEGACY_PROPERTY = (
    "(declare-const X_0 Real)\n"
    "(declare-const X_1 Real)\n"
    "(declare-const Y_0 Real)\n"
    "(assert (>= X_0 0))\n"
    "(assert (<= X_0 1))\n"
    "(assert (>= X_1 0))\n"
    "(assert (<= X_1 1))\n"
    "(assert (>= Y_0 0))\n"
)


def _write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "property.vnnlib"
    path.write_text(source, encoding="utf-8")
    return path


def test_legacy_property_scan_is_unchanged(tmp_path: Path) -> None:
    summary = vnnlib_ce._scan_property(_write(tmp_path, LEGACY_PROPERTY))
    assert summary.layout is None
    assert summary.input_count == 2
    assert summary.output_count == 1
    assert summary.input_assertion_count == 4
    assert summary.output_assertion_count == 1

    requirements = vnnlib_ce.property_requirements(_write(tmp_path, LEGACY_PROPERTY))
    assert (requirements.input_count, requirements.input_assertion_count) == (2, 4)
    assert list(requirements.input_indices) == [0, 1]


def test_legacy_names_resolve_and_evaluate_unchanged() -> None:
    assert vnnlib_ce.resolve_variable("X_0") == ("X", 0)
    assert vnnlib_ce.resolve_variable("Y_12") == ("Y", 12)
    assert vnnlib_ce.resolve_variable("0.5") is None
    assert vnnlib_ce.resolve_variable("true") is None

    environment = vnnlib_ce._VariableEnvironment({0: 0.25, 1: -1.0}, [3.0])
    assert environment.layout is None
    assert vnnlib_ce.evaluate(">= X_0 0".split()[1], environment) == 0.25
    assert vnnlib_ce.evaluate([">=", "X_0", "0"], environment) is True
    assert vnnlib_ce.evaluate([">=", "X_1", "0"], environment) is False
    assert vnnlib_ce.evaluate(["<=", "Y_0", "3.0"], environment) is True

    # The plain-Mapping environment stays VNN-LIB 1.x only.
    assert vnnlib_ce.evaluate([">=", "Y_0", "0"], {"Y_0": 1.0}) is True
    with pytest.raises(vnnlib_ce.ValidationError, match="unknown"):
        vnnlib_ce.evaluate("1_0", {})


def test_legacy_cli_assignment_parser_is_unchanged() -> None:
    assert vnnlib_ce._extract_cli_assignment("((X_0 0.5) (X_1 -1))") == {
        0: 0.5,
        1: -1.0,
    }
    for source in ("((X_00 0))", "((X_0 1_0))", "((X_0 0) (X_0 1))"):
        with pytest.raises(vnnlib_ce.ValidationError):
            vnnlib_ce._extract_cli_assignment(source)


def test_a_tensor_named_witness_is_named_as_such_not_as_an_empty_one() -> None:
    """The WITNESS format stays flat whatever dialect the SPEC uses."""
    with pytest.raises(vnnlib_ce.UnsupportedSyntaxError, match="tensor-indexed"):
        vnnlib_ce._extract_cli_assignment("((X[0,0] 0.5))")
    with pytest.raises(vnnlib_ce.ValidationError, match="no X_i assignments"):
        vnnlib_ce._extract_cli_assignment("(())")


# --------------------------------------------------------------------------
# (d) an unreadable spec raises the DISTINCT error, never a structural verdict
# --------------------------------------------------------------------------


TENSOR_HEADER = (
    "(vnnlib-version <2.0>)\n"
    "(declare-network N\n"
    "    (declare-input  X float32 [1, 2])\n"
    "    (declare-output Y float32 [1, 1])\n"
    ")\n"
)
TENSOR_BODY = (
    "(assert (>= X[0,0] 0))\n"
    "(assert (<= X[0,0] 1))\n"
    "(assert (>= X[0,1] 0))\n"
    "(assert (<= X[0,1] 1))\n"
    "(assert (>= Y[0,0] 0))\n"
)


def test_a_minimal_tensor_spec_is_read(tmp_path: Path) -> None:
    summary = vnnlib_ce._scan_property(_write(tmp_path, TENSOR_HEADER + TENSOR_BODY))
    assert summary.layout is not None
    assert summary.input_count == 2
    assert summary.output_count == 1


def test_unsupported_syntax_error_is_a_distinct_type() -> None:
    assert issubclass(vnnlib_ce.UnsupportedSyntaxError, vnnlib_ce.ValidationError)
    assert vnnlib_ce.UnsupportedSyntaxError is not vnnlib_ce.ValidationError


@pytest.mark.parametrize(
    ("label", "source"),
    [
        (
            "tensor names without a declare-network header",
            "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n"
            "(assert (>= X[0,0] 0))\n(assert (>= Y_0 0))\n",
        ),
        (
            "legacy names inside a 2.0 property",
            TENSOR_HEADER + "(assert (>= X_0 0))\n(assert (>= Y[0,0] 0))\n",
        ),
        (
            "declare-const mixed with declare-network",
            TENSOR_HEADER + "(declare-const X_0 Real)\n" + TENSOR_BODY,
        ),
        (
            "declare-network mixed with declare-const",
            "(declare-const X_0 Real)\n" + TENSOR_HEADER + TENSOR_BODY,
        ),
        (
            "multi-input network",
            "(vnnlib-version <2.0>)\n(declare-network f\n"
            "    (declare-input X1 real [1, 2])\n"
            "    (declare-input X2 real [1, 2])\n"
            "    (declare-output Y real [1, 1])\n)\n"
            "(assert (>= X1[0,0] 0))\n(assert (>= Y[0,0] 0))\n",
        ),
        (
            "renamed tensors",
            "(vnnlib-version <2.0>)\n(declare-network f\n"
            "    (declare-input X_f real [1, 1, 1, 5])\n"
            "    (declare-output Y_f real [1, 5])\n)\n"
            "(assert (>= X_f[0,0,0,0] 0))\n(assert (>= Y_f[0,0] 0))\n",
        ),
        (
            "batch larger than one",
            "(vnnlib-version <2.0>)\n(declare-network N\n"
            "    (declare-input  X float32 [4, 2])\n"
            "    (declare-output Y float32 [1, 1])\n)\n"
            "(assert (>= X[0,0] 0))\n(assert (>= Y[0,0] 0))\n",
        ),
        (
            "unreadable version",
            "(vnnlib-version <3.0>)\n" + TENSOR_BODY,
        ),
        (
            "unreadable shape",
            "(vnnlib-version <2.0>)\n(declare-network N\n"
            "    (declare-input  X float32 [1, n])\n"
            "    (declare-output Y float32 [1, 1])\n)\n"
            "(assert (>= X[0,0] 0))\n(assert (>= Y[0,0] 0))\n",
        ),
    ],
)
def test_unreadable_specs_raise_the_distinct_error(
    tmp_path: Path, label: str, source: str
) -> None:
    with pytest.raises(vnnlib_ce.UnsupportedSyntaxError) as caught:
        vnnlib_ce._scan_property(_write(tmp_path, source))
    message = str(caught.value)
    assert "not searchable" not in message, label
    assert "do not reference" not in message, label


def test_an_out_of_range_tensor_index_is_structural_not_unreadable(
    tmp_path: Path,
) -> None:
    """A spec that IS readable but inconsistent stays a structural refusal."""
    source = TENSOR_HEADER + "(assert (>= X[0,7] 0))\n(assert (>= Y[0,0] 0))\n"
    with pytest.raises(vnnlib_ce.ValidationError) as caught:
        vnnlib_ce._scan_property(_write(tmp_path, source))
    assert not isinstance(caught.value, vnnlib_ce.UnsupportedSyntaxError)
    assert "out of range" in str(caught.value)


def test_validate_reports_unreadable_specs_with_a_distinct_prefix(
    tmp_path: Path,
) -> None:
    source = "(vnnlib-version <3.0>)\n" + TENSOR_BODY
    in_box, genuine, detail = vnnlib_ce.validate(
        tmp_path / "missing.onnx", _write(tmp_path, source), {0: 0.0}
    )
    assert (in_box, genuine) == (False, False)
    assert detail.startswith(vnnlib_ce.UNSUPPORTED_SYNTAX_PREFIX), detail
    assert "invalid property structure" not in detail
    assert "not searchable" not in detail


def _identity_model(tmp_path: Path, shape: list[int]) -> Path:
    """A one-node ONNX identity whose input tensor has exactly ``shape``."""
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["X"], ["Y"])],
        "identity",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = tmp_path / "identity.onnx"
    path.write_bytes(model.SerializeToString())
    return path


def test_onnx_shape_must_match_the_declared_input_shape(tmp_path: Path) -> None:
    """Equal element counts are NOT enough to license a flat mapping.

    ``[1,2,3]`` and ``[1,3,2]`` both hold six elements and both accept the
    indices used below, but they flatten ``X[0,1,1]`` to 4 and to 3
    respectively.  Executing a witness under one while checking the box under
    the other could accept a point that is outside the real box, so the
    disagreement must be refused rather than reconciled by element count.
    """
    source = (
        "(vnnlib-version <2.0>)\n"
        "(declare-network N\n"
        "    (declare-input  X float32 [1, 3, 2])\n"
        "    (declare-output Y float32 [1, 6])\n"
        ")\n"
    )
    for first in range(3):
        for second in range(2):
            source += f"(assert (>= X[0,{first},{second}] 0))\n"
            source += f"(assert (<= X[0,{first},{second}] 1))\n"
    source += "(assert (>= Y[0,0] 0))\n"
    spec = _write(tmp_path, source)

    summary = vnnlib_ce._scan_property(spec)
    assert summary.input_count == 6
    assert summary.layout.inputs.flat_index([0, 1, 1]) == 3

    model = _identity_model(tmp_path, [1, 2, 3])
    with pytest.raises(vnnlib_ce.ValidationError, match="ambiguous"):
        vnnlib_ce.validate(model, spec, {index: 0.5 for index in range(6)})
