# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""The ground-truth-free falsification audit of banked ``unsat`` rows.

This audit is ONE-SIDED: a hit proves a banked ``unsat`` wrong, a miss proves
nothing.  That asymmetry makes a silent failure catastrophic in one direction
only -- a broken searcher reports "0 refutations" and looks exactly like a clean
bank.  The tests below therefore concentrate on the failure modes that would
make the audit VACUOUS rather than merely wrong:

* the bank reader must dispatch on CSV width; reading the ``prepared`` column as
  the verdict yields ZERO banked ``unsat`` rows for the whole regular track;
* the vectorised search evaluator must agree with ``vnnlib_ce.evaluate``, or the
  search steers by a different property than the oracle checks;
* a genuinely violable instance must actually be REFUTED end to end, with the
  oracle's own ``GENUINE-IN-BOX-CE`` record attached;
* an instance that truly holds must come back ``NO-CE-FOUND`` and must not be
  described as verified;
* the blind-row inventory must REFUSE to answer without field ground truth,
  because "0 blind rows" is the most dangerous wrong answer it could give.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime  # noqa: F401  (required by the end-to-end audit contract)
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "extended_bank"))
audit = _load("audit_unsat_by_falsification", "scripts/audit_unsat_by_falsification.py")
vnnlib_ce = _load("vnnlib_ce_for_audit_test", "scripts/extended_bank/vnnlib_ce.py")


# ---------------------------------------------------------------- bank reader


def test_bank_reader_dispatches_on_csv_width(tmp_path: Path) -> None:
    """A 7-column row keeps its verdict in column 4, a 5-column row in column 3.

    The two banks genuinely disagree on layout.  Taking "the field after the
    vnnlib path" reads ``prepared`` on the 7-column schema, which silently
    reports zero ``unsat`` rows for acasxu_2023 and would make this whole audit
    a no-op that still prints a green summary.
    """
    directory = tmp_path / "bank"
    directory.mkdir()
    (directory / "wide.csv").write_text(
        "acasxu_2023,onnx/a.onnx,vnnlib/p.vnnlib,prepared,unsat,0.55,run-1\n"
        "acasxu_2023,onnx/b.onnx,vnnlib/p.vnnlib,prepared,sat,0.9\n",
        encoding="utf-8",
    )
    (directory / "narrow.csv").write_text(
        "cat,onnx,vnnlib,verdict,secs\n"
        "vit_2023,onnx/v.onnx,vnnlib/v.vnnlib,unsat,9.8\n",
        encoding="utf-8",
    )
    rows = audit.read_bank([directory])
    verdicts = {(r.category, r.onnx): (r.verdict, r.seconds) for r in rows}
    assert verdicts[("acasxu_2023", "onnx/a.onnx")] == ("unsat", "0.55")
    assert verdicts[("acasxu_2023", "onnx/b.onnx")] == ("sat", "0.9")
    assert verdicts[("vit_2023", "onnx/v.onnx")] == ("unsat", "9.8")
    assert sum(1 for r in rows if r.verdict == "unsat") == 2


def test_bank_reader_drops_harness_instances(tmp_path: Path) -> None:
    directory = tmp_path / "bank"
    directory.mkdir()
    (directory / "b.csv").write_text(
        "nn4sys,onnx/test_nano.onnx,vnnlib/test_nano.vnnlib,0,unsat,0\n"
        "nn4sys,onnx/real.onnx,vnnlib/real.vnnlib,prepared,unsat,1.0\n",
        encoding="utf-8",
    )
    assert [r.onnx for r in audit.read_bank([directory])] == ["onnx/real.onnx"]


def test_bank_reader_refuses_unsupported_width_instead_of_returning_partial_rows(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "bank"
    directory.mkdir()
    (directory / "mixed.csv").write_text(
        "vit_2023,onnx/good.onnx,vnnlib/good.vnnlib,unsat,1.0\n"
        "vit_2023,onnx/bad.onnx,vnnlib/bad.vnnlib,prepared,unsat,1.0,run,extra\n",
        encoding="utf-8",
    )

    with pytest.raises(audit.BankFormatError, match="unsupported 8-column row"):
        audit.read_bank([directory])


# ------------------------------------------------------------- bound extraction


def test_simple_input_bounds_handles_both_operand_orders() -> None:
    node = ["and", [">=", "X_0", "1.5"], ["<=", "3.5", "X_1"]]
    assert audit.simple_input_bounds(node) == [(0, ">=", 1.5), (1, ">=", 3.5)]


def test_simple_input_bounds_refuses_output_terms() -> None:
    assert audit.simple_input_bounds(["and", [">=", "X_0", "1"], ["<=", "Y_0", "2"]]) is None


def test_input_bounds_within_recovers_a_mixed_conjunction() -> None:
    """nn4sys states its whole input interval only inside a mixed (X, Y) arm.

    Ignoring those conjuncts leaves an unbounded box, and the row is dropped as
    unsearchable -- which is how 161 field-blind rows would go unaudited.
    """
    arm = ["and", [">=", "X_0", "0.1"], ["<=", "X_0", "0.2"], ["<=", "Y_0", "0.5"]]
    assert audit.simple_input_bounds(arm) is None
    assert audit._input_bounds_within(arm) == [(0, ">=", 0.1), (0, "<=", 0.2)]


# ------------------------------------------------------- evaluator equivalence


@pytest.mark.parametrize(
    "expression",
    [
        ["or", ["and", [">=", "Y_0", "Y_1"]], ["and", [">=", "Y_1", "Y_0"]]],
        ["and", ["<=", ["+", "Y_0", "Y_1"], "1.0"], [">", "X_0", "0.25"]],
        ["not", ["<=", ["*", "Y_0", "2.0"], ["-", "Y_1", "X_1"]]],
        ["=", "Y_0", "Y_1"],
    ],
)
def test_vectorised_evaluator_matches_the_trusted_scalar_evaluator(expression) -> None:
    """Steering and the gate must be the SAME property the oracle checks."""
    compiled = audit.compile_boolean(expression)
    generator = np.random.default_rng(7)
    x = generator.random((64, 2))
    y = generator.random((64, 2))
    _margin, holds = compiled(x, y)
    for row in range(x.shape[0]):
        environment = vnnlib_ce._VariableEnvironment(
            {0: float(x[row, 0]), 1: float(x[row, 1])},
            [float(y[row, 0]), float(y[row, 1])],
            executed_inputs=False,
        )
        assert bool(holds[row]) is bool(vnnlib_ce.evaluate(expression, environment))


def test_margin_sign_agrees_with_satisfaction() -> None:
    compiled = audit.compile_boolean(["or", [">=", "Y_0", "1.0"], [">=", "Y_1", "1.0"]])
    y = np.array([[0.4, 0.9], [1.2, 0.0]])
    margin, holds = compiled(np.zeros((2, 1)), y)
    assert list(holds) == [False, True]
    assert margin[0] < 0 <= margin[1]


# ------------------------------------------------------------------ end to end


def _write_linear_model(path: Path, weight: np.ndarray, bias: np.ndarray) -> None:
    """y = x @ weight + bias, batch axis symbolic so the runner can batch."""
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["input", "W"], ["mm"]),
            onnx.helper.make_node("Add", ["mm", "B"], ["output"]),
        ],
        "linear",
        [
            onnx.helper.make_tensor_value_info(
                "input", onnx.TensorProto.FLOAT, ["batch", weight.shape[0]]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                "output", onnx.TensorProto.FLOAT, ["batch", weight.shape[1]]
            )
        ],
        [
            onnx.numpy_helper.from_array(weight.astype(np.float32), "W"),
            onnx.numpy_helper.from_array(bias.astype(np.float32), "B"),
        ],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    onnx.save(model, str(path))


def _instance(root: Path, category: str, name: str, spec: str, weight, bias) -> None:
    (root / category / "onnx").mkdir(parents=True, exist_ok=True)
    (root / category / "vnnlib").mkdir(parents=True, exist_ok=True)
    _write_linear_model(root / category / "onnx" / f"{name}.onnx", weight, bias)
    (root / category / "vnnlib" / f"{name}.vnnlib").write_text(spec, encoding="utf-8")


VIOLABLE = """
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= Y_0 0.75))
"""

HOLDS = """
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= Y_0 5.0))
"""


def _audit(root: Path, name: str, verdict: str, tmp_path: Path) -> dict:
    row = audit.BankRow(
        category="synthetic",
        onnx=f"onnx/{name}.onnx",
        vnnlib=f"vnnlib/{name}.vnnlib",
        verdict=verdict,
        seconds="1.0",
        source="synthetic.csv:1",
    )
    return audit.audit_row(
        row=row,
        benchmark_root=root,
        budget_seconds=8.0,
        evidence_dir=tmp_path / "evidence",
        max_spec_mb=64.0,
        seed=1,
        threads=1,
        batch=64,
    )


def test_a_violable_row_is_refuted_with_the_oracles_own_record(tmp_path: Path) -> None:
    """y = x on [0, 1] reaches 1.0, so ``Y_0 >= 0.75`` is satisfiable in box."""
    root = tmp_path / "benchmarks"
    _instance(root, "synthetic", "violable", VIOLABLE, np.eye(1), np.zeros(1))
    record = _audit(root, "violable", "unsat", tmp_path)

    assert record["status"] == audit.ST_REFUTED
    assert record["oracle"]["verdict"] == "GENUINE-IN-BOX-CE"
    assert record["oracle"]["is_counterexample"] is True
    assert record["ort_output"][0] >= 0.75
    assert "REFUTED" in record["detail"]

    # The witness must be reproducible in one command by someone who does not
    # trust this script at all.
    witness = Path(record["counterexample_file"])
    assert witness.is_file()
    values = vnnlib_ce._extract_cli_assignment(witness.read_text(encoding="utf-8"))
    in_box, is_counterexample, _detail = vnnlib_ce.validate(
        record["onnx_path"], record["vnnlib_path"], values
    )
    assert (in_box, is_counterexample) == (True, True)


def test_a_row_that_holds_reports_a_miss_and_never_claims_soundness(
    tmp_path: Path,
) -> None:
    root = tmp_path / "benchmarks"
    _instance(root, "synthetic", "holds", HOLDS, np.eye(1), np.zeros(1))
    record = _audit(root, "holds", "unsat", tmp_path)

    assert record["status"] == audit.ST_NO_CE
    assert "PROVES NOTHING" in record["detail"]
    assert "never" in record["one_sided_note"]
    assert record["effort"]["points_tried"] > 1000
    assert record["effort"]["strategies_run"]


def test_effort_is_reported_so_a_miss_is_interpretable(tmp_path: Path) -> None:
    root = tmp_path / "benchmarks"
    _instance(root, "synthetic", "holds", HOLDS, np.eye(1), np.zeros(1))
    effort = _audit(root, "holds", "unsat", tmp_path)["effort"]
    for field in (
        "points_tried",
        "ort_forward_calls",
        "wall_seconds",
        "strategies_run",
        "best_full_property_margin",
        "free_inputs",
        "targets",
    ):
        assert field in effort, field
    assert effort["free_inputs"] == 1
    # A one-dimensional free input must get the dense sweep, not just gradients:
    # a piecewise-constant network defeats every gradient estimate.
    assert "grid" in effort["strategies_run"]


MIXED_ONLY = """
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (or
  (and (>= X_0 0.0) (<= X_0 1.0) (>= Y_0 0.75))
))
"""


def test_a_spec_the_full_oracle_refuses_is_still_searched_and_reported_apart(
    tmp_path: Path,
) -> None:
    """nn4sys's shape: the input box exists ONLY inside a mixed (X, Y) disjunct.

    ``_scan_property`` demands an input-only assertion per declared X_i, so the
    full validator refuses the file outright -- which means (a) the box must be
    recovered from the mixed arm or the row goes unsearched, and (b) a hit can
    never be graded GENUINE-IN-BOX-CE, so it must land in its own status rather
    than be quietly promoted or quietly dropped.
    """
    root = tmp_path / "benchmarks"
    _instance(root, "synthetic", "mixed", MIXED_ONLY, np.eye(1), np.zeros(1))
    record = _audit(root, "mixed", "unsat", tmp_path)

    assert record["full_oracle_can_validate_this_spec"] is False
    assert "input constraints do not reference X_0" in record["full_oracle_refusal"]
    assert record["status"] == audit.ST_REFUTED_REDUCED
    assert record["status"] != audit.ST_REFUTED
    assert record["oracle"]["is_counterexample"] is False
    assert record["reduced_oracle"]["all_assertions_hold"] is True
    assert "only _scan_property" in record["reduced_oracle"]["caveat"]


def test_missing_instance_files_are_skipped_not_silently_passed(tmp_path: Path) -> None:
    record = _audit(tmp_path / "empty", "absent", "unsat", tmp_path)
    assert record["status"] == audit.ST_SKIP
    assert "not found" in record["detail"]


# --------------------------------------------------------------- blind inventory


def test_blind_inventory_refuses_to_run_without_field_ground_truth(
    tmp_path: Path, capsys
) -> None:
    """Without ground truth the honest answer is "cannot tell", not "zero".

    Reporting zero moat-blind rows because the official tree was unreadable is
    precisely the vacuous pass this campaign has already paid for once.
    """
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "b.csv").write_text(
        "vit_2023,onnx/v.onnx,vnnlib/v.vnnlib,unsat,9.8\n", encoding="utf-8"
    )
    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    code = audit.main(
        [
            "--bank",
            str(bank),
            "--benchmark-root",
            str(benchmarks),
            "--official",
            str(tmp_path / "no-such-official-tree"),
            "--list-blind-rows",
        ]
    )
    assert code == 3
    assert "ENVIRONMENT ERROR" in capsys.readouterr().err


def test_unannotated_rows_default_to_blind_not_to_clean() -> None:
    """An unrecognised row must fail toward "the moat cannot see this"."""
    assert audit.UNANNOTATED["field_blind"] is True
    assert audit.UNANNOTATED["category_blind"] is True
    assert audit.UNANNOTATED["row_blind"] is True
