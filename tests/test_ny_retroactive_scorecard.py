# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "ny_retroactive_scorecard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ny_retroactive_scorecard", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scorecard = _load_module()


def test_load_tool_csv_excludes_harness_test_rows(tmp_path: Path) -> None:
    results = tmp_path / "results.csv"
    results.write_text(
        "cat,onnx/model.onnx,vnnlib/property.vnnlib,0,unsat,1\n"
        "cat,onnx/test_nano.onnx,vnnlib/test_nano.vnnlib,0,unsat,1\n"
        "cat,onnx/test_tiny.onnx,vnnlib/test_tiny.vnnlib,0,unsat,1\n",
        encoding="utf-8",
    )

    loaded = scorecard.load_tool_csv(results)

    assert len(loaded["cat"]) == 1


def test_load_tool_csv_preserves_duplicate_instance_occurrences(tmp_path: Path) -> None:
    results = tmp_path / "results.csv"
    row = "sat_relu,onnx/sat_v6_c27.onnx,vnnlib/sat_v6_c27.vnnlib,0,sat,1\n"
    results.write_text(row + row, encoding="utf-8")

    loaded = scorecard.load_tool_csv(results)["sat_relu"]

    assert len(loaded) == 2
    assert {key[-1] for key in loaded} == {0, 1}
    assert set(loaded.values()) == {"violated"}


def test_load_tool_csv_tolerates_trailing_measurement_run_id(tmp_path: Path) -> None:
    results = tmp_path / "results.csv"
    results.write_text(
        "acasxu_2023,onnx/a.onnx,vnnlib/a.vnnlib,0,unsat,1,20260718T120000Z-123\n",
        encoding="utf-8",
    )

    loaded = scorecard.load_tool_csv(results)

    assert list(loaded["acasxu_2023"].values()) == ["holds"]


def test_published_artifacts_map_ids_and_preserve_official_denominator(
    tmp_path: Path,
) -> None:
    reference_csv = tmp_path / "results.csv"
    reference_csv.write_text(
        "acasxu_2023,onnx/a.onnx,vnnlib/a.vnnlib,0,unsat,1\n"
        "acasxu_2023,onnx/test_nano.onnx,vnnlib/test_nano.vnnlib,0,unsat,1\n"
        "acasxu_2023,onnx/b.onnx,vnnlib/b.vnnlib,0,sat,1\n",
        encoding="utf-8",
    )
    longtable = tmp_path / "longtable.tex"
    longtable.write_text(
        "2025 Acasxu 2023 & 0 & ~\\textsc{unsat} & - \\\\\n"
        "2025 Acasxu 2023 & 1 & ~\\textsc{sat} & - \\\\\n",
        encoding="utf-8",
    )
    reference = scorecard.load_reference_instance_order(reference_csv)
    ground_truth = scorecard.load_published_ground_truth(longtable, reference)
    first, second = reference["acasxu_2023"]

    assert len(reference["acasxu_2023"]) == 2
    assert ground_truth["acasxu_2023"] == {
        first: "holds",
        second: "violated",
    }

    ny = {"acasxu_2023": {first: "holds", second: "violated"}}
    winner_points = dict.fromkeys(scorecard.REGULAR, 100)
    total, breakdown = scorecard.published_artifact_projection(
        ny,
        reference,
        ground_truth,
        winner_points,
        ny_sat_status="correct-up-to-tolerance",
    )

    # NY gets 20 raw but remains normalized by the published 100-point winner,
    # not a raw-verdict-recomputed denominator of 20.
    assert breakdown["acasxu_2023"][:3] == (20, 20.0, 100)
    assert total == 20.0


def test_official_artifact_mode_requires_explicit_ny_sat_assumption() -> None:
    # The requirement is enforced in main because argparse cannot express the
    # conditional cleanly; guard the accepted status vocabulary at unit level.
    with pytest.raises(ValueError, match="unsupported NY SAT status"):
        scorecard.published_artifact_projection(
            {},
            {},
            {},
            dict.fromkeys(scorecard.REGULAR, 1),
            ny_sat_status="strict-but-unchecked",
        )


def _ext_fixture(
    tmp_path: Path,
    *,
    track: str,
    gt_rows: list[tuple[str, str, bool]],
    win_score: int,
    ny_csv_rows: list[str],
) -> tuple[Path, Path]:
    """Synthetic SCORING-ZERO-TOL results.txt plus a ny extended measured dir."""
    lines = [f"Category 2025_{track}:", "  participating tools: ['alpha_beta_crown']"]
    for rid, true_result, valid_ce in gt_rows:
        marker = ", 'x (v)'" if valid_ce else ""
        lines.append(f"Row: ['{rid}'{marker}]")
        lines.append(f"True Result: {true_result}")
    lines.append(f"0: alpha_beta_crown score: {win_score},")
    results_txt = tmp_path / "results.txt"
    results_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ext_dir = tmp_path / "measured-ext"
    ext_dir.mkdir()
    (ext_dir / f"{track}.csv").write_text(
        "".join(row + "\n" for row in ny_csv_rows), encoding="utf-8"
    )
    return results_txt, ext_dir


def test_score_extended_folds_penalty_into_projection_and_reports_bad(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results_txt, ext_dir = _ext_fixture(
        tmp_path,
        track="lsnc_relu",
        gt_rows=[("net-prop_0", "sat", True), ("net-prop_1", "unsat", False)],
        win_score=20,
        ny_csv_rows=[
            "lsnc_relu,net.onnx,prop_0.vnnlib,unsat,1.0",
            "lsnc_relu,net.onnx,prop_1.vnnlib,unsat,1.0",
        ],
    )

    grand, bad = scorecard.score_extended(results_txt, ext_dir, 0.0)
    out = capsys.readouterr().out

    # rate 1/2 over N=2 extrapolates to +10 credit; the measured -150 penalty
    # must survive into proj_raw, and the normalized score floors at 0.
    row = next(line for line in out.splitlines() if line.startswith("lsnc_relu"))
    assert bad == 1
    assert grand == 0.0
    assert row.split()[-2:] == ["-140", "0.0"]
    assert "*** SCORECARD FAILURE: bad=1 incorrect extended" in out


def test_score_extended_prints_measured_only_norm_beside_projection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gt_rows: list[tuple[str, str, bool]] = [("net-prop_0", "unsat", False)]
    gt_rows += [(f"net-prop_{i}", "-", False) for i in range(1, 10)]
    results_txt, ext_dir = _ext_fixture(
        tmp_path,
        track="lsnc_relu",
        gt_rows=gt_rows,
        win_score=200,
        ny_csv_rows=["lsnc_relu,net.onnx,prop_0.vnnlib,unsat,1.0"],
    )

    grand, bad = scorecard.score_extended(results_txt, ext_dir, 0.0)
    out = capsys.readouterr().out

    # 1/1 sampled ok extrapolates to 100 raw (norm 50.0); measured-only is
    # 100*10/200 = 5.0 and must be printed beside the projection.
    row = next(line for line in out.splitlines() if line.startswith("lsnc_relu"))
    assert bad == 0
    assert grand == 50.0
    assert row.split()[-4:] == ["10", "5.0", "100", "50.0"]
    assert "EXTENDED-9 measured-only normalized" in out


def test_extended_sat_credit_warns_without_evidence_and_can_require_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results_txt, ext_dir = _ext_fixture(
        tmp_path,
        track="ml4acopf_2024",
        gt_rows=[("net-prop_0", "sat", True)],
        win_score=10,
        ny_csv_rows=["ml4acopf_2024,net.onnx,prop_0.vnnlib,sat,1.0"],
    )

    grand, bad = scorecard.score_extended(results_txt, ext_dir, 0.0)
    out = capsys.readouterr().out
    assert (grand, bad) == (100.0, 0)
    assert "WARNING: 1 extended sat row(s) credited WITHOUT on-disk evidence" in out
    assert "no-evidence: ml4acopf_2024 net-prop_0" in out

    grand, bad = scorecard.score_extended(
        results_txt, ext_dir, 0.0, require_evidence=True
    )
    out = capsys.readouterr().out
    assert (grand, bad) == (0.0, 0)
    assert "WARNING: 1 extended sat row(s) ZEROED by --require-evidence" in out

    # A validate_bank metadata file (<safe_stem>-<uuid4-hex>.validation.json)
    # under evidence/<track>/ restores the credit and silences the warning.
    evidence_dir = ext_dir / "evidence" / "ml4acopf_2024"
    evidence_dir.mkdir(parents=True)
    (evidence_dir / f"prop_0-{'0' * 32}.validation.json").write_text(
        "{}\n", encoding="utf-8"
    )
    grand, bad = scorecard.score_extended(
        results_txt, ext_dir, 0.0, require_evidence=True
    )
    out = capsys.readouterr().out
    assert (grand, bad) == (100.0, 0)
    assert "WARNING" not in out


def test_main_rejects_extended_with_official_artifacts() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--official-artifacts",
            "--ny-sat-status",
            "unvalidated",
            "--extended",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--extended cannot be combined with --official-artifacts" in result.stderr
    assert "ZERO-TOL denominators" in result.stderr


def test_main_rejects_require_evidence_without_extended() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--require-evidence"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--require-evidence is only valid with --extended" in result.stderr


def test_main_exits_nonzero_when_extended_bad_rows_exist(tmp_path: Path) -> None:
    results_txt, ext_dir = _ext_fixture(
        tmp_path,
        track="lsnc_relu",
        gt_rows=[("net-prop_0", "sat", True)],
        win_score=10,
        ny_csv_rows=["lsnc_relu,net.onnx,prop_0.vnnlib,unsat,1.0"],
    )
    official = tmp_path / "official"
    official.mkdir()
    measured = tmp_path / "measured"
    measured.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--official",
            str(official),
            "--measured",
            str(measured),
            "--extended",
            "--results-txt",
            str(results_txt),
            "--ext-measured",
            str(ext_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "*** SCORECARD FAILURE: bad=1 incorrect extended" in result.stdout
