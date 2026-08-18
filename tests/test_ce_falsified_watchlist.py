# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0
"""The regular-track FIELD-FALSIFIED gate.

``longtable.tex`` — and ``True Result`` itself — print ``unsat`` on rows the
VNN-COMP field FALSIFIED, because ``process_results.py`` only promotes
``true_result`` to ``sat`` for an exactly-CORRECT witness.  A row whose every
ACCEPTED counterexample graded ``CORRECT_UP_TO_TOLERANCE`` keeps the ``unsat``
label while the tools that produced those witnesses were each paid +10 falsifier
credit.  Reading the label alone hands an ny ``unsat`` a silent +10 on an
instance the field PROVED violable — the regular-track twin of the extended
track's ``(v)`` blind spot.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scorecard = _load("ny_retroactive_scorecard")
watchlist_mod = _load("emit_ce_falsified_watchlist")
sweep = _load("ny_measured_sweep")


def _regular_fixture(
    tmp_path: Path,
    *,
    category: str,
    latex_name: str,
    rows: list[tuple[str, str, str, str, bool]],
) -> tuple[Path, Path, Path]:
    """Synthetic reference results.csv + longtable.tex + results.txt.

    ``rows`` entries are ``(onnx, vnnlib, published_result, marker, is_fals)``
    where ``marker`` is the ``(v)``/``(h)`` letter in the official Row cell and
    ``is_fals`` says whether that tool earned ACCEPTED-counterexample credit.
    """
    reference_csv = tmp_path / "results.csv"
    reference_csv.write_text(
        "".join(
            f"{category},onnx/{onnx}.onnx,vnnlib/{vnnlib}.vnnlib,0,unsat,1\n"
            for onnx, vnnlib, _res, _marker, _fals in rows
        ),
        encoding="utf-8",
    )
    longtable = tmp_path / "longtable.tex"
    longtable.write_text(
        "".join(
            f"{latex_name} & {index} & ~\\textsc{{{res}}} & - \\\\\n"
            for index, (_o, _v, res, _m, _f) in enumerate(rows)
        ),
        encoding="utf-8",
    )
    lines = [f"Category 2025_{category}:", "1 participating tools: ['alpha_beta_crown']"]
    for index, (onnx, vnnlib, res, marker, is_fals) in enumerate(rows):
        lines.append(f"Row: ['{onnx}-{vnnlib}', '1.0 ({marker})']")
        lines.append(f"True Result: {res}")
        lines.append(
            f"{index}: alpha_beta_crown score: 10, is_ver: False, "
            f"is_fals: {is_fals}, is_fastest: False"
        )
    results_txt = tmp_path / "results.txt"
    results_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return reference_csv, longtable, results_txt


def _official_dir(tmp_path: Path, reference_csv: Path) -> Path:
    official = tmp_path / "official"
    (official / "alpha_beta_crown").mkdir(parents=True, exist_ok=True)
    (official / "alpha_beta_crown" / "results.csv").write_text(
        reference_csv.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return official


# --- ground truth -----------------------------------------------------------


def test_published_ground_truth_refuses_holds_on_a_field_falsified_row(
    tmp_path: Path,
) -> None:
    """THE DEFECT: the published cell reads `unsat`, so the old reader emitted
    `holds` and the scorer paid an ny unsat +10 with nothing raised."""
    reference_csv, longtable, results_txt = _regular_fixture(
        tmp_path,
        category="cifar100_2024",
        latex_name="2025 Cifar100 2024",
        rows=[
            ("net", "prop_0", "unsat", "v", True),  # falsified, mislabelled
            ("net", "prop_1", "unsat", "h", False),  # a genuine hold
        ],
    )
    reference = scorecard.load_reference_instance_order(reference_csv)
    falsified = scorecard.load_accepted_ce_falsifications(results_txt, reference)
    ground_truth = scorecard.load_published_ground_truth(longtable, reference, falsified)
    first, second = reference["cifar100_2024"]

    assert ground_truth["cifar100_2024"][first] == scorecard.FIELD_FALSIFIED
    assert ground_truth["cifar100_2024"][second] == "holds"


def test_ground_truth_without_evidence_still_reads_the_bare_label(
    tmp_path: Path,
) -> None:
    """Documents what the unfixed reader did, so the fix cannot silently lapse."""
    reference_csv, longtable, _results_txt = _regular_fixture(
        tmp_path,
        category="cifar100_2024",
        latex_name="2025 Cifar100 2024",
        rows=[("net", "prop_0", "unsat", "v", True)],
    )
    reference = scorecard.load_reference_instance_order(reference_csv)

    ground_truth = scorecard.load_published_ground_truth(longtable, reference)

    (instance,) = reference["cifar100_2024"]
    assert ground_truth["cifar100_2024"][instance] == "holds"


# --- the falsification signal ----------------------------------------------


def test_accepted_ce_falsification_ignores_a_rejected_witness(tmp_path: Path) -> None:
    """A `(v)` whose counterexample was REJECTED falsified nothing: that tool was
    penalised -150, and ny may legitimately prove the row unsat.  Using the bare
    `(v)` marker here would manufacture false blockers."""
    reference_csv, _longtable, results_txt = _regular_fixture(
        tmp_path,
        category="cifar100_2024",
        latex_name="2025 Cifar100 2024",
        rows=[("net", "prop_0", "unsat", "v", False)],
    )
    reference = scorecard.load_reference_instance_order(reference_csv)

    falsified = scorecard.load_accepted_ce_falsifications(results_txt, reference)

    assert falsified.get("cifar100_2024", {}) == {}


def test_accepted_ce_falsification_keys_rows_by_index_not_row_id(
    tmp_path: Path,
) -> None:
    """The official row id is only <onnx-stem>-<vnnlib-stem> and COLLIDES:
    safenlp_2024 repeats identical basenames under medical/ and ruarobot/, so a
    rid-keyed watchlist silently keeps only the last of each colliding pair."""
    category = "safenlp_2024"
    reference_csv = tmp_path / "results.csv"
    reference_csv.write_text(
        f"{category},onnx/medical/perturbations_0.onnx,"
        f"vnnlib/medical/hyperrectangle_1.vnnlib,0,sat,1\n"
        f"{category},onnx/ruarobot/perturbations_0.onnx,"
        f"vnnlib/ruarobot/hyperrectangle_1.vnnlib,0,sat,1\n",
        encoding="utf-8",
    )
    lines = [f"Category 2025_{category}:", "1 participating tools: ['alpha_beta_crown']"]
    for index in range(2):
        # IDENTICAL row id for both rows — exactly what the organizers emit.
        lines.append("Row: ['perturbations_0-hyperrectangle_1', '1.0 (v)']")
        lines.append("True Result: unsat")
        lines.append(
            f"{index}: alpha_beta_crown score: 10, is_ver: False, "
            "is_fals: True, is_fastest: False"
        )
    results_txt = tmp_path / "results.txt"
    results_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reference = scorecard.load_reference_instance_order(reference_csv)

    falsified = scorecard.load_accepted_ce_falsifications(results_txt, reference)

    assert len(falsified[category]) == 2
    assert {info["idx"] for info in falsified[category].values()} == {0, 1}


def test_accepted_ce_falsification_fails_closed_on_broken_row_alignment(
    tmp_path: Path,
) -> None:
    """Positional binding between results.txt and results.csv is VERIFIED, so a
    future artifact reshuffle cannot yield a silently wrong watchlist."""
    reference_csv, _longtable, results_txt = _regular_fixture(
        tmp_path,
        category="cifar100_2024",
        latex_name="2025 Cifar100 2024",
        rows=[("net", "prop_0", "unsat", "v", True)],
    )
    reference_csv.write_text(
        "cifar100_2024,onnx/other.onnx,vnnlib/elsewhere.vnnlib,0,unsat,1\n",
        encoding="utf-8",
    )
    reference = scorecard.load_reference_instance_order(reference_csv)

    with pytest.raises(scorecard.FalsifiedRowAlignmentError, match="no longer holds"):
        scorecard.load_accepted_ce_falsifications(results_txt, reference)


# --- the moat ---------------------------------------------------------------


def test_regular_moat_blocks_ny_unsat_on_a_field_falsified_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference_csv, _longtable, results_txt = _regular_fixture(
        tmp_path,
        category="cifar100_2024",
        latex_name="2025 Cifar100 2024",
        rows=[("net", "prop_0", "unsat", "v", True)],
    )
    reference = scorecard.load_reference_instance_order(reference_csv)
    (instance,) = reference["cifar100_2024"]

    blocking = scorecard.regular_moat_check(
        results_txt,
        _official_dir(tmp_path, reference_csv),
        {"cifar100_2024": {instance: "holds"}},
    )
    out = capsys.readouterr().out

    assert blocking == 1
    assert "ny-unsat on a field-falsified regular row" in out
    assert "net-prop_0" in out
    assert "MOAT: zero contradictions" not in out


def test_regular_moat_leaves_ny_sat_on_a_field_falsified_row_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The symmetric partner: agreeing with the field's counterexample is fine."""
    reference_csv, _longtable, results_txt = _regular_fixture(
        tmp_path,
        category="cifar100_2024",
        latex_name="2025 Cifar100 2024",
        rows=[("net", "prop_0", "unsat", "v", True)],
    )
    reference = scorecard.load_reference_instance_order(reference_csv)
    (instance,) = reference["cifar100_2024"]

    blocking = scorecard.regular_moat_check(
        results_txt,
        _official_dir(tmp_path, reference_csv),
        {"cifar100_2024": {instance: "violated"}},
    )
    out = capsys.readouterr().out

    assert blocking == 0
    assert "MOAT: zero contradictions" in out


def test_published_artifact_projection_counts_field_falsified_holds(
    tmp_path: Path,
) -> None:
    """Points stay official-faithful at +10 — the organizer scorer really pays
    them — but the row is counted separately so the board can block on it."""
    reference_csv, longtable, results_txt = _regular_fixture(
        tmp_path,
        category="cifar100_2024",
        latex_name="2025 Cifar100 2024",
        rows=[("net", "prop_0", "unsat", "v", True)],
    )
    reference = scorecard.load_reference_instance_order(reference_csv)
    falsified = scorecard.load_accepted_ce_falsifications(results_txt, reference)
    ground_truth = scorecard.load_published_ground_truth(longtable, reference, falsified)
    (instance,) = reference["cifar100_2024"]

    _total, breakdown = scorecard.published_artifact_projection(
        {"cifar100_2024": {instance: "holds"}},
        reference,
        ground_truth,
        dict.fromkeys(scorecard.REGULAR, 100),
        ny_sat_status="correct-up-to-tolerance",
    )

    projection = breakdown["cifar100_2024"]
    assert (
        projection.raw,
        projection.credited,
        projection.contradictions,
    ) == (10, 1, 0)
    assert projection.field_falsified_holds == 1


# --- the emitted watchlist --------------------------------------------------


def test_watchlist_round_trips_to_the_scorecard_instance_key(tmp_path: Path) -> None:
    reference_csv, _longtable, results_txt = _regular_fixture(
        tmp_path,
        category="cifar100_2024",
        latex_name="2025 Cifar100 2024",
        rows=[
            ("net", "prop_0", "unsat", "v", True),
            ("net", "prop_1", "unsat", "h", False),
        ],
    )
    official = _official_dir(tmp_path, reference_csv)
    zero_tol = official / "SCORING-ZERO-TOL"
    zero_tol.mkdir(parents=True)
    (zero_tol / "results.txt").write_text(
        results_txt.read_text(encoding="utf-8"), encoding="utf-8"
    )

    payload = watchlist_mod.build(official)
    out = tmp_path / "watchlist.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = watchlist_mod.load(out)

    assert payload["counts"]["falsified_rows"] == 1
    assert payload["counts"]["labelled_unsat_anyway"] == 1
    assert loaded["cifar100_2024"] == {("onnx/net.onnx", "vnnlib/prop_0.vnnlib", 0)}
    # The non-falsified row is absent, not merely flagged.
    assert len(loaded["cifar100_2024"]) == 1


def test_watchlist_load_rejects_an_unknown_schema_version(tmp_path: Path) -> None:
    stale = tmp_path / "watchlist.json"
    stale.write_text(json.dumps({"version": 999, "categories": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="is not the expected"):
        watchlist_mod.load(stale)


def external_committed_watchlist_matches_the_official_artifacts() -> None:
    """Explicit evidence probe; the hermetic suite tests the schema with fixtures."""
    official = REPO_ROOT / "external_tools" / "vnncomp2025_results"
    results = official / "SCORING-ZERO-TOL" / "results.txt"
    assert results.is_file(), (
        "official vnncomp2025_results artifacts are not checked out: "
        f"{results}"
    )

    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "emit_ce_falsified_watchlist.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


# --- the banking gate -------------------------------------------------------


def test_sweep_refuses_to_bank_an_unsat_on_a_field_falsified_row() -> None:
    rows = [("onnx/net.onnx", "vnnlib/prop_0.vnnlib", 60)]
    watchlist = {"cifar100_2024": {("onnx/net.onnx", "vnnlib/prop_0.vnnlib", 0)}}

    refused = sweep.refused_falsified_unsats(
        "cifar100_2024", rows, {0: ("unsat", 1.0)}, watchlist
    )

    assert refused == [("onnx/net.onnx", "vnnlib/prop_0.vnnlib", "unsat")]


def test_sweep_banks_a_sat_or_timeout_on_the_same_row() -> None:
    """Only `unsat` contradicts an accepted counterexample."""
    rows = [("onnx/net.onnx", "vnnlib/prop_0.vnnlib", 60)]
    watchlist = {"cifar100_2024": {("onnx/net.onnx", "vnnlib/prop_0.vnnlib", 0)}}

    for token in ("sat", "timeout", "unknown"):
        assert (
            sweep.refused_falsified_unsats(
                "cifar100_2024", rows, {0: (token, 1.0)}, watchlist
            )
            == []
        )


def test_sweep_distinguishes_repeated_pair_occurrences() -> None:
    """sat_relu repeats an ONNX/VNN-LIB pair; only the listed occurrence is
    refused, so the gate cannot over-block a legitimate second measurement."""
    rows = [("onnx/n.onnx", "vnnlib/p.vnnlib", 60), ("onnx/n.onnx", "vnnlib/p.vnnlib", 60)]
    watchlist = {"sat_relu": {("onnx/n.onnx", "vnnlib/p.vnnlib", 1)}}

    refused = sweep.refused_falsified_unsats(
        "sat_relu", rows, {0: ("unsat", 1.0), 1: ("unsat", 1.0)}, watchlist
    )

    assert len(refused) == 1


def test_sweep_refuses_to_run_at_all_without_a_watchlist(tmp_path: Path) -> None:
    """Fail CLOSED: a missing watchlist must not read as 'nothing is listed'."""
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "ny_measured_sweep.py"),
            "cifar100_2024",
            "--watchlist",
            str(tmp_path / "absent.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 3
    assert "REFUSING TO SWEEP" in completed.stderr


# --- the merge seams: dynamic rescoring x the field-falsified moat -----------
#
# These pin the three places where main's dynamic organizer rescoring and this
# moat interact. Each one passes on ONE side alone and would silently regress if
# a later edit reverted the resolution.


def test_field_falsified_still_counts_as_published_holds_for_strict_rescoring() -> None:
    """THE SEAM. ``FIELD_FALSIFIED`` REFINES the published ``holds`` cell.

    The dynamic-rescoring gate asked ``truth == "holds"``. A relabelled row
    fails that string compare, so a strictly correct NY witness on a
    field-falsified row would skip the organizer rescore its truth flip demands
    — and that rescore is what keeps the incumbent denominator honest.
    """
    assert scorecard.published_truth_is_holds("holds")
    assert scorecard.published_truth_is_holds(scorecard.FIELD_FALSIFIED)
    assert not scorecard.published_truth_is_holds("violated")
    assert not scorecard.published_truth_is_holds("unknown")

    instance = ("onnx/a.onnx", "vnnlib/a.vnnlib", 0)
    reference = {category: [] for category in scorecard.REGULAR}
    reference["acasxu_2023"] = [instance]
    ground_truth = {"acasxu_2023": {instance: scorecard.FIELD_FALSIFIED}}

    # A strictly correct NY sat on a FIELD_FALSIFIED row with NO rescore must
    # still be refused, exactly as it is on a plain `holds` row.
    with pytest.raises(scorecard.MeasurementBudgetError, match="strict truth"):
        scorecard.published_artifact_projection(
            {"acasxu_2023": {instance: "violated"}},
            reference,
            ground_truth,
            dict.fromkeys(scorecard.REGULAR, 100),
            ny_sat_status="correct-up-to-tolerance",
            ny_sat_results={"acasxu_2023": {instance: "correct"}},
            organizer_rescores={},
        )


def test_apply_field_falsified_labels_relabels_a_preparsed_ground_truth() -> None:
    """The evidence-qualified path gets its truth PRE-PARSED from the pinned
    promoter context, which reads the bare longtable.tex cell. Relabelling only
    inside ``load_published_ground_truth`` would leave that path blind."""
    listed = ("onnx/a.onnx", "vnnlib/a.vnnlib", 0)
    clean = ("onnx/b.onnx", "vnnlib/b.vnnlib", 0)
    violated = ("onnx/c.onnx", "vnnlib/c.vnnlib", 0)
    pinned = {
        "cifar100_2024": {listed: "holds", clean: "holds", violated: "violated"}
    }
    falsified = {"cifar100_2024": {listed: {"rid": "a-a", "true": "unsat"}}}

    relabelled = scorecard.apply_field_falsified_labels(pinned, falsified)

    assert relabelled["cifar100_2024"][listed] == scorecard.FIELD_FALSIFIED
    assert relabelled["cifar100_2024"][clean] == "holds"
    # A `violated` cell is never upgraded, and the caller's dict is untouched.
    assert relabelled["cifar100_2024"][violated] == "violated"
    assert pinned["cifar100_2024"][listed] == "holds"


def test_evidence_qualified_report_blocks_on_a_pinned_field_falsified_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """THE PINNED-PATH HOLE. ``--require-evidence`` takes its ground truth from
    the pinned promoter context, NOT from ``load_published_ground_truth``.

    Without ``apply_field_falsified_labels`` on that branch the moat is blind on
    exactly the path that credits the validated promotion index, and the board
    prints green with an ny ``unsat`` sitting on a field-falsified row. Reverting
    that one call must turn this test red.
    """
    import regular_bank_evidence as bank_evidence  # noqa: PLC0415

    instance = ("onnx/a.onnx", "vnnlib/a.vnnlib", 0)
    reference = {category: [] for category in scorecard.REGULAR}
    reference["acasxu_2023"] = [instance]
    truth = {category: {} for category in scorecard.REGULAR}
    # The pinned context reads the BARE longtable.tex cell, so it says "holds".
    truth["acasxu_2023"] = {instance: "holds"}
    pinned = SimpleNamespace(
        context=SimpleNamespace(
            reference_order=reference,
            ground_truth=truth,
            winner_points=dict.fromkeys(scorecard.REGULAR, 100),
        )
    )
    monkeypatch.setattr(bank_evidence, "revalidate_official_results", lambda _: None)

    status = scorecard.report_published_artifact_projection(
        tmp_path,
        {"acasxu_2023": {instance: "holds"}},
        ny_sat_status="correct-up-to-tolerance",
        target=1566.9,
        field_falsified={
            "acasxu_2023": {instance: {"rid": "a-a", "true": "unsat"}}
        },
        evidence_qualified=True,
        pinned_official=pinned,
        ny_sat_results={},
    )

    output = capsys.readouterr().out
    assert status == 1, output
    assert "MOAT: 1 CONTRADICTION" in output
    # Official-faithful: the organizer really does pay this row, so the points
    # stay — the board is blocked, not silently rescored.
    assert "points stay official-faithful at +10" in output


def test_evidence_qualified_report_stays_green_without_a_falsified_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The matched control for the test above: same pinned path, same ny
    ``holds``, but the row is not on the watchlist."""
    import regular_bank_evidence as bank_evidence  # noqa: PLC0415

    instance = ("onnx/a.onnx", "vnnlib/a.vnnlib", 0)
    reference = {category: [] for category in scorecard.REGULAR}
    reference["acasxu_2023"] = [instance]
    truth = {category: {} for category in scorecard.REGULAR}
    truth["acasxu_2023"] = {instance: "holds"}
    pinned = SimpleNamespace(
        context=SimpleNamespace(
            reference_order=reference,
            ground_truth=truth,
            winner_points=dict.fromkeys(scorecard.REGULAR, 100),
        )
    )
    monkeypatch.setattr(bank_evidence, "revalidate_official_results", lambda _: None)

    status = scorecard.report_published_artifact_projection(
        tmp_path,
        {"acasxu_2023": {instance: "holds"}},
        ny_sat_status="correct-up-to-tolerance",
        target=1566.9,
        field_falsified={},
        evidence_qualified=True,
        pinned_official=pinned,
        ny_sat_results={},
    )

    output = capsys.readouterr().out
    assert status == 0, output
    assert "MOAT: zero ny-unsat verdicts on field-falsified regular rows." in output


def test_report_projection_cannot_be_called_without_the_moat_input() -> None:
    """``field_falsified`` is REQUIRED on purpose: a default would let a caller
    print a green board while silently skipping the moat."""
    with pytest.raises(TypeError, match="field_falsified"):
        scorecard.report_published_artifact_projection(
            Path("/nonexistent"),
            {},
            ny_sat_status="correct-up-to-tolerance",
            target=1566.9,
        )
