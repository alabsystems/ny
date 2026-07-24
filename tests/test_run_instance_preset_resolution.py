# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Regression tests for the thin VNN-COMP shell wrapper and release build.

Preset resolution and branching now live in the native ``ny vnncomp`` command,
where Rust unit tests cover full-name/base-name and newest-directory precedence.
These tests pin the shell boundary: it must preserve the organizer protocol
arguments exactly and must not reintroduce legacy preset/branching policy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORED_RUN_INSTANCE = REPO_ROOT / "run_instance.sh"
SHARED_RUN_INSTANCE = REPO_ROOT / "vnncomp_scripts" / "run_instance.sh"
BUILD_SUBMISSION_BINARY = REPO_ROOT / "vnncomp_scripts" / "build_submission_binary.sh"


def _write_fake_ny(tmp_path: Path) -> Path:
    """Create a fake ny binary that records argv/environment and emits status."""
    ny = tmp_path / "fake_ny"
    ny_args_file = tmp_path / "ny-args.txt"
    ny_env_file = tmp_path / "ny-env.txt"
    ny.write_text(
        textwrap.dedent(f"""\
            #!/bin/bash
            printf '%s\n' "$@" > "{ny_args_file}"
            printf 'AY_MILP_SMT=%s\nAY_MILP_GUB_CLIQUE=%s\nAY_MILP_STAB_ORBIT=%s\nAY_MILP_COVER_MINIMAL=%s\nAY_DUMP_QUERY_DIR=%s\nMIMALLOC_PURGE_DELAY=%s\nMIMALLOC_FUTURE_OPTION=%s\n' \
                "${{AY_MILP_SMT-<unset>}}" \
                "${{AY_MILP_GUB_CLIQUE-<unset>}}" \
                "${{AY_MILP_STAB_ORBIT-<unset>}}" \
                "${{AY_MILP_COVER_MINIMAL-<unset>}}" \
                "${{AY_DUMP_QUERY_DIR-<unset>}}" \
                "${{MIMALLOC_PURGE_DELAY-<unset>}}" \
                "${{MIMALLOC_FUTURE_OPTION-<unset>}}" > "{ny_env_file}"
            echo '{{"status": "unknown"}}'
        """),
        encoding="utf-8",
    )
    ny.chmod(0o755)
    return ny


def _setup_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create minimal ONNX and VNNLIB fixture files."""
    onnx_file = tmp_path / "model.onnx"
    vnnlib_file = tmp_path / "prop.vnnlib"
    results_file = tmp_path / "results.txt"
    onnx_file.write_bytes(b"\x08\x01")
    vnnlib_file.write_text("", encoding="utf-8")
    return onnx_file, vnnlib_file, results_file


def _run_instance(
    tmp_path: Path,
    ny_path: Path,
    category: str,
    onnx: Path,
    vnnlib: Path,
    results: Path,
    *,
    runner: Path = SHARED_RUN_INSTANCE,
    extra_env: dict[str, str] | None = None,
    timeout: str = "10",
) -> subprocess.CompletedProcess[str]:
    """Run run_instance.sh with a fake ny binary."""
    env = os.environ.copy()
    env["NY_BIN"] = str(ny_path)
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(
        [
            "bash",
            str(runner),
            "v1",
            category,
            str(onnx),
            str(vnnlib),
            str(results),
            timeout,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _captured_ny_args(tmp_path: Path) -> list[str]:
    """Read the fake ny argv capture from disk."""
    args_file = tmp_path / "ny-args.txt"
    assert args_file.is_file(), f"missing fake ny args capture: {args_file}"
    return args_file.read_text(encoding="utf-8").splitlines()


def _captured_ny_env(tmp_path: Path) -> dict[str, str]:
    """Read the solver environment observed by the fake ny process."""
    env_file = tmp_path / "ny-env.txt"
    assert env_file.is_file(), f"missing fake ny environment capture: {env_file}"
    return dict(
        line.split("=", 1) for line in env_file.read_text(encoding="utf-8").splitlines()
    )


def _write_fake_cargo(tmp_path: Path) -> None:
    """Create a fake Cargo that emits one fresh, authenticated ny artifact."""
    cargo = tmp_path / "bin" / "cargo"
    cargo.parent.mkdir(parents=True, exist_ok=True)
    cargo.write_text(
        textwrap.dedent("""\
            #!/bin/bash
            set -euo pipefail

            printf '%s\\n' "$@" > "$PWD/cargo-args.txt"
            : "${CARGO_TARGET_DIR:?missing invocation-scoped target directory}"
            artifact="$CARGO_TARGET_DIR/release/ny"
            mkdir -p "$(dirname "$artifact")"
            cat > "$artifact" <<'EOF'
            #!/bin/bash
            echo fake-ny
            EOF
            chmod +x "$artifact"
            printf '{"reason":"compiler-artifact","manifest_path":"%s","target":{"kind":["bin"],"name":"ny"},"filenames":["%s"],"executable":"%s","fresh":false}\\n' \
                "$PWD/crates/ny-cli/Cargo.toml" "$artifact" "$artifact"
            printf '{"reason":"build-finished","success":true}\\n'
        """),
        encoding="utf-8",
    )
    cargo.chmod(0o755)


def _copy_build_submission_binary(tmp_path: Path) -> Path:
    """Copy the submission build script into an isolated temp repo.

    shutil.copy preserves the source permission bits, so the copy is
    executable exactly when the checked-in script is — the same direct-exec
    contract install_tool.sh relies on. A chmod here would let this test pass
    against a committed non-executable script.
    """
    script = tmp_path / "vnncomp_scripts" / "build_submission_binary.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(BUILD_SUBMISSION_BINARY, script)
    return script


def _run_build_submission_binary(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run build_submission_binary.sh with a fake cargo shim."""
    script = _copy_build_submission_binary(tmp_path)
    _write_fake_cargo(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path / 'bin'}:{env.get('PATH', '')}"
    return subprocess.run(
        [str(script)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    "category",
    [
        # Categories with year suffix in preset filename — previously broken
        "safenlp_2024",
        "cifar100_2024",
        "dist_shift_2023",
        "acasxu_2023",
        "ml4acopf_2024",
        "tinyimagenet_2024",
        "yolo_2023",
        "traffic_signs_recognition_2023",
        # Categories without year suffix — should still work
        "nn4sys_2023",
        "relusplitter",
        "malbeware",
        "cersyve",
        "soundnessbench",
    ],
)
def test_native_vnncomp_delegation_preserves_category(
    tmp_path: Path, category: str
) -> None:
    """The shell must pass every category spelling to native policy unchanged."""
    ny = _write_fake_ny(tmp_path)
    onnx, vnnlib, results = _setup_fixtures(tmp_path)

    result = _run_instance(tmp_path, ny, category, onnx, vnnlib, results)

    assert result.returncode == 0, result.stderr
    assert _captured_ny_args(tmp_path) == [
        "vnncomp",
        "v1",
        category,
        str(onnx),
        str(vnnlib),
        str(results),
        "10",
    ]


@pytest.mark.parametrize(
    "category, timeout",
    [
        # The only two fractional per-instance timeouts in the VNN-COMP 2026 set.
        ("metaroom_2023", "210.0"),
        ("traffic_signs_recognition_2023", "480.0"),
    ],
)
def test_fractional_timeout_still_invokes_ny(
    tmp_path: Path, category: str, timeout: str
) -> None:
    """A fractional CSV timeout must not abort the wrapper before ny launches.

    The OS-level wall-clock backstop is computed with bash integer arithmetic
    (``$(( budget + 10 ))``), which aborts on a decimal point; under ``set -u``
    that left the timeout unset and the ``exec`` line failed BEFORE ny ran, so no
    verdict was ever written for metaroom_2023 (210.0) or
    traffic_signs_recognition_2023 (480.0) — two full categories scored 0. The
    wrapper now floors the budget for the backstop while forwarding the raw,
    unmodified fractional budget to ``ny vnncomp`` (which floors it internally).
    If ny is invoked at all, the wrapper survived the fractional input.
    """
    ny = _write_fake_ny(tmp_path)
    onnx, vnnlib, results = _setup_fixtures(tmp_path)

    result = _run_instance(
        tmp_path, ny, category, onnx, vnnlib, results, timeout=timeout
    )

    assert result.returncode == 0, result.stderr
    # ny actually ran (argv captured => the wrapper reached exec, did not crash),
    # and the RAW fractional budget was forwarded verbatim (ny floors internally).
    assert _captured_ny_args(tmp_path) == [
        "vnncomp",
        "v1",
        category,
        str(onnx),
        str(vnnlib),
        str(results),
        timeout,
    ]


@pytest.mark.parametrize("category", ["ml4acopf_2024", "safenlp_2024"])
def test_shell_does_not_inject_legacy_policy_flags(
    tmp_path: Path, category: str
) -> None:
    """Preset and branching policy belong solely to the native command."""
    ny = _write_fake_ny(tmp_path)
    onnx, vnnlib, results = _setup_fixtures(tmp_path)

    result = _run_instance(tmp_path, ny, category, onnx, vnnlib, results)
    ny_args = _captured_ny_args(tmp_path)

    assert result.returncode == 0, result.stderr
    assert ny_args[0] == "vnncomp"
    assert "--branching" not in ny_args, ny_args
    assert "--preset" not in ny_args, ny_args


def test_scored_root_wrapper_sanitizes_operator_environment(tmp_path: Path) -> None:
    """The organizer-facing wrapper must start ny with deterministic runtime knobs."""
    ny = _write_fake_ny(tmp_path)
    onnx, vnnlib, results = _setup_fixtures(tmp_path)
    dump_dir = tmp_path / "ay-query-dumps"

    result = _run_instance(
        tmp_path,
        ny,
        "cgan2026",
        onnx,
        vnnlib,
        results,
        runner=SCORED_RUN_INSTANCE,
        extra_env={
            # Several AY experiments test presence rather than truthiness, so
            # adversarial false-looking values must disappear too.
            "AY_MILP_SMT": "0",
            "AY_MILP_GUB_CLIQUE": "1",
            "AY_MILP_STAB_ORBIT": "0",
            "AY_MILP_COVER_MINIMAL": "false",
            "AY_DUMP_QUERY_DIR": str(dump_dir),
            "MIMALLOC_PURGE_DELAY": "0",
            "MIMALLOC_FUTURE_OPTION": "future-value",
        },
    )

    assert result.returncode == 0, result.stderr
    assert _captured_ny_env(tmp_path) == {
        "AY_MILP_SMT": "<unset>",
        "AY_MILP_GUB_CLIQUE": "<unset>",
        "AY_MILP_STAB_ORBIT": "<unset>",
        "AY_MILP_COVER_MINIMAL": "<unset>",
        "AY_DUMP_QUERY_DIR": "<unset>",
        "MIMALLOC_PURGE_DELAY": "<unset>",
        "MIMALLOC_FUTURE_OPTION": "<unset>",
    }


def test_shared_helper_preserves_operator_environment(tmp_path: Path) -> None:
    """Local developer A/B and query-capture runs retain explicit controls."""
    ny = _write_fake_ny(tmp_path)
    onnx, vnnlib, results = _setup_fixtures(tmp_path)
    dump_dir = tmp_path / "ay-query-dumps"

    result = _run_instance(
        tmp_path,
        ny,
        "cgan2026",
        onnx,
        vnnlib,
        results,
        runner=SHARED_RUN_INSTANCE,
        extra_env={
            "AY_MILP_SMT": "1",
            "AY_MILP_GUB_CLIQUE": "1",
            "AY_MILP_STAB_ORBIT": "0",
            "AY_MILP_COVER_MINIMAL": "false",
            "AY_DUMP_QUERY_DIR": str(dump_dir),
            "MIMALLOC_PURGE_DELAY": "17",
            "MIMALLOC_FUTURE_OPTION": "future-value",
        },
    )

    assert result.returncode == 0, result.stderr
    assert _captured_ny_env(tmp_path) == {
        "AY_MILP_SMT": "1",
        "AY_MILP_GUB_CLIQUE": "1",
        "AY_MILP_STAB_ORBIT": "0",
        "AY_MILP_COVER_MINIMAL": "false",
        "AY_DUMP_QUERY_DIR": str(dump_dir),
        "MIMALLOC_PURGE_DELAY": "17",
        "MIMALLOC_FUTURE_OPTION": "future-value",
    }


def test_build_submission_binary_enables_mip_feature(tmp_path: Path) -> None:
    """build_submission_binary.sh should compile ny-cli with mip support."""
    result = _run_build_submission_binary(tmp_path)

    assert result.returncode == 0, (
        f"build_submission_binary.sh failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    cargo_args = (tmp_path / "cargo-args.txt").read_text(encoding="utf-8").splitlines()
    # The fake cargo accepts every invocation, so the script's FIRST fallback
    # tier ("mip,cuda") is the one that must land here.
    assert cargo_args[:5] == ["build", "--locked", "--release", "-p", "ny-cli"]
    assert cargo_args[-2:] == ["--features", "mip,cuda"]
    assert "--message-format=json-render-diagnostics" in cargo_args
    target_index = cargo_args.index("--target-dir")
    staging = Path(cargo_args[target_index + 1])
    assert staging.parent == tmp_path / "target"
    assert staging.name.startswith(".ny-submission-build.")
    assert (tmp_path / "target" / "release" / "ny").is_file(), (
        "expected build_submission_binary.sh to create target/release/ny"
    )


def test_build_submission_binary_fails_closed_when_full_tier_fails(
    tmp_path: Path,
) -> None:
    """Competition install must not silently accept a feature-degraded binary."""
    script = _copy_build_submission_binary(tmp_path)
    cargo = tmp_path / "bin" / "cargo"
    cargo.parent.mkdir(parents=True, exist_ok=True)
    cargo.write_text(
        '#!/bin/bash\nprintf \'%s\\n\' "$@" > "$PWD/cargo-args.txt"\nexit 9\n',
        encoding="utf-8",
    )
    cargo.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path / 'bin'}:{env.get('PATH', '')}"

    result = subprocess.run(
        [str(script)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "required competition feature tier 'mip,cuda' failed" in result.stderr
    assert not (tmp_path / "target" / "release" / "ny").exists()
