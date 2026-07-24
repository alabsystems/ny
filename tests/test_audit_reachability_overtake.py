from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "audit_reachability_overtake.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_reachability_overtake", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def _raw(verdict: str) -> str:
    return {
        "holds": "unsat",
        "violated": "sat",
        "timeout": "run_instance_timeout",
        "unknown": "unknown",
    }[verdict]


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    official = tmp_path / "official"
    measured = tmp_path / "measured"
    measured.mkdir(parents=True)

    rows: dict[str, list[dict[str, object]]] = {}
    for category in audit.retro.REGULAR:
        rows[category] = [
            {
                "name": f"{category}-base",
                "truth": "holds",
                "ny": "holds",
                "tools": {"alpha_beta_crown": "holds"},
            }
        ]

    rows["metaroom_2023"] = [
        {
            "name": "metaroom-primary",
            "truth": "holds",
            "ny": "timeout",
            "tools": {"pyrat": "holds"},
        }
    ]
    # Three occurrences of the same path ensure positional multiplicity is kept.
    rows["cifar100_2024"] = [
        {
            "name": "duplicate",
            "truth": "holds",
            "ny": "timeout",
            "tools": {"nnv": "holds"},
        },
        {
            "name": "duplicate",
            "truth": "holds",
            "ny": "timeout",
            "tools": {"pyrat": "holds", "alpha_beta_crown": "holds"},
        },
        {
            "name": "duplicate",
            "truth": "holds",
            "ny": "timeout",
            "tools": {"nnv": "holds"},
        },
    ]
    rows["tinyimagenet_2024"] = [
        {
            "name": "tiny-catchup",
            "truth": "holds",
            "ny": "timeout",
            "tools": {"pyrat": "holds", "alpha_beta_crown": "holds"},
        }
    ]
    rows["cgan_2023"].append(
        {
            "name": "published-sat-is-not-an-unsat-target",
            "truth": "violated",
            "ny": "timeout",
            "tools": {"pyrat": "holds"},
        }
    )

    official_lines: dict[str, list[str]] = {
        tool: [] for tool in audit.retro.OFFICIAL_TOOLS
    }
    longtable: list[str] = []
    for category in audit.retro.REGULAR:
        measured_lines: list[str] = []
        for instance_id, row in enumerate(rows[category]):
            name = str(row["name"])
            onnx = f"benchmarks/{category}/onnx/{name}.onnx"
            vnnlib = f"benchmarks/{category}/vnnlib/{name}.vnnlib"
            tool_verdicts = row["tools"]
            assert isinstance(tool_verdicts, dict)
            for tool in audit.retro.OFFICIAL_TOOLS:
                verdict = str(tool_verdicts.get(tool, "timeout"))
                official_lines[tool].append(
                    f"{category},{onnx},{vnnlib},0,{_raw(verdict)},1\n"
                )
            measured_lines.append(
                f"{category},./onnx/{name}.onnx,./vnnlib/{name}.vnnlib,"
                f"prepared,{_raw(str(row['ny']))},1\n"
            )
            display = category.replace("_", " ")
            truth = "unsat" if row["truth"] == "holds" else "sat"
            longtable.append(f"2025 {display} & {instance_id} & \\textsc{{{truth}}}\n")
        (measured / f"{category}.csv").write_text(
            "".join(measured_lines), encoding="utf-8"
        )

    for tool, lines in official_lines.items():
        path = official / tool / "results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines), encoding="utf-8")
    latex = official / "SCORING-ZERO-TOL" / "latex"
    latex.mkdir(parents=True)
    (latex / "longtable.tex").write_text("".join(longtable), encoding="utf-8")
    return official, measured


def _summary(result: dict[str, object]) -> dict[str, dict[str, object]]:
    summaries = result["summary"]
    assert isinstance(summaries, list)
    return {item["strategy"]: item for item in summaries}


def test_classifies_overtake_catchup_and_hint_without_collapsing_rows(
    tmp_path: Path,
) -> None:
    official, measured = _fixture(tmp_path)
    result = audit.build_audit(official, measured)
    summary = _summary(result)

    assert summary["pyrat_constrained_zono_overtake"]["count"] == 1
    assert summary["pyrat_constrained_zono_overtake"]["by_category"] == {
        "metaroom_2023": 1
    }
    assert summary["pyrat_constrained_zono_overtake"]["authority"] == (
        "official_result_method_signal"
    )
    assert summary["pyrat_constrained_zono_catchup"]["count"] == 2
    assert summary["abc_unsat_catchup"]["count"] == 2
    assert summary["nnv_cp_star_only_hint"]["count"] == 2
    assert summary["nnv_cp_star_only_hint"]["authority"] == (
        "hint_only_not_proof_or_ground_truth"
    )

    hints = result["targets"]["nnv_cp_star_only_hint"]
    assert [row["instance_id"] for row in hints] == [0, 2]
    assert [row["occurrence"] for row in hints] == [0, 2]
    assert result["targets"]["pyrat_constrained_zono_catchup"][0]["occurrence"] == 1


def test_published_sat_is_excluded_from_unsat_target_queues(tmp_path: Path) -> None:
    official, measured = _fixture(tmp_path)
    result = audit.build_audit(official, measured)
    serialized_targets = json.dumps(result["targets"], sort_keys=True)
    assert "published-sat-is-not-an-unsat-target" not in serialized_targets


def test_cli_json_is_complete_and_includes_input_digests(tmp_path: Path) -> None:
    official, measured = _fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--official",
            str(official),
            "--measured",
            str(measured),
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["schema"] == audit.SCHEMA
    assert result["inputs"]["official_files"]
    assert all(len(item["sha256"]) == 64 for item in result["inputs"]["official_files"])
    assert len(result["inputs"]["measured_files"]) == len(audit.retro.REGULAR)


def test_missing_competitor_results_fail_closed(tmp_path: Path) -> None:
    official, measured = _fixture(tmp_path)
    (official / "nnv" / "results.csv").unlink()
    with pytest.raises(audit.AuditError, match="nnv results"):
        audit.build_audit(official, measured)
