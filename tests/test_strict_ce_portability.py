# Copyright 2026 Andrew Yates
# SPDX-License-Identifier: Apache-2.0 OR MIT

from __future__ import annotations

import gzip
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
REGEN = REPO_ROOT / "scripts" / "strict_ce" / "regen_witnesses.py"
VALIDATE = REPO_ROOT / "scripts" / "strict_ce" / "strict_validate.py"


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "NY_BIN",
        "NY_BROOT",
        "NY_MEASURED_DIR",
        "NY_ROOT",
        "NY_STRICT_CE_WORK_DIR",
        "VNNCOMP2026_RESULTS",
        "VNNCOMP_SCORING_DIR",
    ):
        environment.pop(name, None)
    return environment


def _fake_scoring(root: Path) -> Path:
    scoring = root / "SCORING"
    scoring.mkdir(parents=True)
    (scoring / "counterexamples.py").write_text(
        "from pathlib import Path\n"
        "def get_ce_diff(model, specification, counterexample, abs_tol, rel_tol):\n"
        "    assert Path(model).read_bytes() == b'model'\n"
        "    assert Path(specification).read_text() == '; property\\n'\n"
        "    assert Path(counterexample).is_file()\n"
        "    assert abs_tol == 0.0 and rel_tol == 0.0\n"
        "    return 'correct', 'fixture accepted'\n",
        encoding="utf-8",
    )
    return scoring


def test_strict_validate_requires_explicitly_discoverable_scoring(
    tmp_path: Path,
) -> None:
    witnesses = tmp_path / "witnesses"
    witnesses.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATE),
            str(witnesses),
            "--repo-root",
            str(tmp_path),
        ],
        env=_clean_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "official SCORING directory is unavailable" in result.stderr
    assert "--scoring-dir or VNNCOMP_SCORING_DIR" in result.stderr


def test_regen_uses_tempfile_workdir_and_env_configured_scoring(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "portable-ny"
    category = "fixture"
    category_root = (
        repo / "benchmarks" / "vnncomp2025" / "benchmarks" / category
    )
    model = category_root / "onnx" / "model.onnx"
    specification = category_root / "vnnlib" / "property.vnnlib"
    model.parent.mkdir(parents=True)
    specification.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    specification.write_text("; property\n", encoding="utf-8")

    measured = repo / "reports" / "measured" / f"{category}.csv"
    measured.parent.mkdir(parents=True)
    measured.write_text(
        "fixture,onnx/model.onnx,vnnlib/property.vnnlib,0,sat,1\n",
        encoding="utf-8",
    )
    ny_binary = repo / "target" / "release" / "ny"
    ny_binary.parent.mkdir(parents=True)
    ny_binary.write_text(
        "#!/bin/sh\nprintf 'sat\\n((X_0 0.25))\\n' > \"$6\"\n",
        encoding="utf-8",
    )
    ny_binary.chmod(0o755)

    temp_root = tmp_path / "temp-root"
    temp_root.mkdir()
    environment = _clean_environment()
    environment.update(
        {
            "NY_ROOT": str(repo),
            "TMPDIR": str(temp_root),
        }
    )
    regenerated = subprocess.run(
        [sys.executable, str(REGEN), category, "1"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert regenerated.returncode == 0, regenerated.stderr
    manifest_line = next(
        line for line in regenerated.stdout.splitlines() if line.startswith("MANIFEST ")
    )
    workdir = Path(manifest_line.removeprefix("MANIFEST "))
    assert workdir.parent == temp_root
    assert workdir.name.startswith(f"ny-strict-ce-{category}-")
    with gzip.open(workdir / "0" / "ce.gz", "rt", encoding="utf-8") as witness:
        assert witness.read() == "((X_0 0.25))\n"

    scoring = _fake_scoring(tmp_path / "official-results")
    environment["VNNCOMP_SCORING_DIR"] = str(scoring)
    validated = subprocess.run(
        [sys.executable, str(VALIDATE), str(workdir)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert validated.returncode == 0, validated.stderr
    assert "STRICT (abs_tol=0, rel_tol=0) RESULTS: {'correct': 1}" in validated.stdout
    assert "strict yield: 1/1 = 100.0%" in validated.stdout
